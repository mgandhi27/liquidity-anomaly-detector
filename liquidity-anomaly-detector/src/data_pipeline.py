"""
data_pipeline.py
=================

End-to-end pipeline turning raw Level-2 order-book snapshots into
model-ready (sequence, label) tensors for `LiquidityAnomalyLSTM`.

Expected raw schema (one row per snapshot / tick):
    timestamp, bid_price, bid_size, ask_price, ask_size,
    bid_size_l2..lN (optional additional book levels),
    ask_size_l2..lN (optional additional book levels)

At minimum, `timestamp`, `bid_price`, `bid_size`, `ask_price`,
`ask_size` must be present; deeper levels are summed into a single
aggregate depth proxy if supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset

from .transforms import FinancialTimeTransform

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "bid_price", "bid_size", "ask_price", "ask_size")


class LOBSequenceDataset(Dataset):
    """Wraps pre-built (sequence, label) tensors for use with a DataLoader."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if len(sequences) != len(labels):
            raise ValueError(
                f"sequences ({len(sequences)}) and labels ({len(labels)}) must have the same length."
            )
        if sequences.ndim != 3:
            raise ValueError(
                f"sequences must be 3-D (n_samples, seq_len, n_features); got shape {sequences.shape}."
            )
        # .copy() guards against read-only arrays (e.g. from
        # sliding_window_view or newer pandas .to_numpy() outputs), which
        # torch.as_tensor otherwise accepts but warns about; ascontiguousarray
        # alone can be a no-op if the source is already contiguous but
        # marked read-only, so an explicit copy is used instead.
        self.sequences = torch.as_tensor(np.copy(sequences), dtype=torch.float32)
        self.labels = torch.as_tensor(np.copy(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


@dataclass
class PipelineConfig:
    """Configuration for `DataPipeline`."""

    sequence_length: int = 50
    forecast_horizon: int = 10
    depth_drop_threshold: float = 0.5  # fractional drop in depth defining a "shock"
    volatility_window: int = 20
    spread_window: int = 20
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-4
    resample_rule: Optional[str] = None  # e.g. "1s" to resample to 1-second bars
    train_frac: float = 0.7
    val_frac: float = 0.15
    batch_size: int = 64


class DataPipeline:
    """
    Cleans, engineers, and sequences raw Level-2 snapshots into
    train/validation/test DataLoaders, preserving chronological order
    (no shuffling across the time axis) to avoid lookahead leakage.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.transform = FinancialTimeTransform()
        self._feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------
    def load_raw(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Raw data file not found at '{path}'.") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Raw data file at '{path}' is empty.") from exc

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Raw data is missing required column(s): {missing}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required column(s): {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.set_index("timestamp")

        numeric_cols = list(df.columns)
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        if df[numeric_cols].isna().any().any():
            raise ValueError("Unable to fully impute missing values; raw data may be malformed.")

        if self.config.resample_rule:
            df = df.resample(self.config.resample_rule).last().ffill()

        invalid_book = (df["bid_price"] <= 0) | (df["ask_price"] <= 0) | (df["bid_price"] > df["ask_price"])
        if invalid_book.any():
            n_bad = int(invalid_book.sum())
            logger.warning("Dropping %d row(s) with non-positive or crossed quotes.", n_bad)
            df = df.loc[~invalid_book]

        if df.empty:
            raise ValueError("No valid rows remain after cleaning.")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _depth_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        bid_depth_cols = [c for c in df.columns if c.startswith("bid_size") or c.startswith("bid_depth")]
        ask_depth_cols = [c for c in df.columns if c.startswith("ask_size") or c.startswith("ask_depth")]
        if not bid_depth_cols or not ask_depth_cols:
            raise ValueError("Could not identify bid/ask size or depth columns in the DataFrame.")
        return bid_depth_cols, ask_depth_cols

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds a stationary feature matrix from cleaned raw snapshots."""
        cfg = self.config
        bid_depth_cols, ask_depth_cols = self._depth_columns(df)

        mid_price = (df["bid_price"] + df["ask_price"]) / 2.0
        total_depth = df[bid_depth_cols].sum(axis=1) + df[ask_depth_cols].sum(axis=1)
        depth_imbalance = (
            df[bid_depth_cols].sum(axis=1) - df[ask_depth_cols].sum(axis=1)
        ) / total_depth.clip(lower=1e-9)

        log_ret = self.transform.log_returns(mid_price)
        vol = self.transform.rolling_volatility(log_ret, window=cfg.volatility_window)
        spread = self.transform.rolling_spread(df["bid_price"], df["ask_price"], window=cfg.spread_window)

        log_depth = np.log1p(total_depth.clip(lower=0.0)).rename("log_total_depth")
        depth_fracdiff = self.transform.fractional_difference(
            log_depth, d=cfg.fracdiff_d, threshold=cfg.fracdiff_threshold
        )

        features = pd.concat(
            [log_ret, vol, spread, depth_imbalance.rename("depth_imbalance"), depth_fracdiff],
            axis=1,
            join="inner",
        ).dropna()

        if features.empty:
            raise ValueError(
                "Feature engineering produced an empty frame; check window sizes vs. series length."
            )

        self._feature_columns = list(features.columns)

        # Retain raw total_depth (aligned to the feature index) purely for
        # downstream label construction -- it is NOT itself a model input.
        features = features.join(total_depth.rename("total_depth"), how="left")
        return features

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    def build_target(self, features: pd.DataFrame) -> pd.Series:
        """
        Binary label: 1 if total order-book depth is on track to fall by
        more than `depth_drop_threshold` (as a fraction of current depth)
        at any point within the next `forecast_horizon` observations.

        For each t, we compute min(depth[t+1 .. t+horizon]) via a
        reversed rolling-min trick (rolling windows only look backward,
        so we reverse, roll, and reverse back to get a forward-looking
        minimum), then compare that to depth[t].
        """
        cfg = self.config
        depth = features["total_depth"]

        shifted_next = depth.shift(-1)  # value at t+1, aligned to index t
        future_min_depth = (
            shifted_next[::-1]
            .rolling(window=cfg.forecast_horizon, min_periods=cfg.forecast_horizon)
            .min()[::-1]
        )
        # future_min_depth[t] == min(depth[t+1], ..., depth[t+horizon])

        relative_drop = (depth - future_min_depth) / depth.clip(lower=1e-9)
        label = (relative_drop >= cfg.depth_drop_threshold).astype(np.float32)
        # Rows without a full forward window (the final `horizon` rows) get NaN
        # from the rolling computation upstream and are dropped by callers.
        label = label.where(future_min_depth.notna())
        return label.rename("liquidity_shock_label")

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------
    def build_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slides a fixed-length window of length `sequence_length` across
        the feature matrix; sequence i covers rows [i, i + seq_len) and
        predicts the label aligned to the final row of that window
        (i + seq_len - 1), which already encodes whether a liquidity
        shock occurs within the forward-looking `forecast_horizon`.
        """
        cfg = self.config
        feature_cols = [c for c in features.columns if c != "total_depth"]
        combined = features[feature_cols].join(labels, how="inner").dropna()

        values = combined[feature_cols].to_numpy(dtype=np.float32)
        target = combined[labels.name].to_numpy(dtype=np.float32)

        n_rows = len(combined)
        if n_rows <= cfg.sequence_length:
            raise ValueError(
                f"Not enough rows ({n_rows}) to build sequences of length {cfg.sequence_length}."
            )

        n_sequences = n_rows - cfg.sequence_length + 1
        n_features = values.shape[1]
        windows = sliding_window_view(values, window_shape=(cfg.sequence_length, n_features))
        # sliding_window_view returns a read-only view; copy so downstream
        # consumers (e.g. torch.as_tensor) get a writable, owned array.
        sequences = windows.reshape(n_sequences, cfg.sequence_length, n_features).copy()
        sequence_labels = target[cfg.sequence_length - 1 :]

        return sequences, sequence_labels

    # ------------------------------------------------------------------
    # DataLoaders (strict chronological split -- no shuffling across time)
    # ------------------------------------------------------------------
    def get_dataloaders(
        self, sequences: np.ndarray, labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        cfg = self.config
        n = len(sequences)
        if n < 10:
            raise ValueError("Not enough sequences to build train/val/test splits.")

        train_end = int(n * cfg.train_frac)
        val_end = train_end + int(n * cfg.val_frac)
        if not (0 < train_end < val_end < n):
            raise ValueError("train_frac/val_frac produce an invalid split; adjust the fractions.")

        splits = {
            "train": (sequences[:train_end], labels[:train_end]),
            "val": (sequences[train_end:val_end], labels[train_end:val_end]),
            "test": (sequences[val_end:], labels[val_end:]),
        }

        loaders = []
        for name, (seq, lab) in splits.items():
            dataset = LOBSequenceDataset(seq, lab)
            # Shuffling only within the training split's mini-batches is safe:
            # every training sequence still chronologically precedes every
            # validation/test sequence, so no lookahead leakage is introduced.
            loaders.append(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(name == "train")))
        return loaders[0], loaders[1], loaders[2]

    # ------------------------------------------------------------------
    # Convenience end-to-end entry point
    # ------------------------------------------------------------------
    def run(self, raw_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load -> clean -> engineer -> label -> sequence -> split, in one call."""
        raw = self.load_raw(raw_path)
        cleaned = self.clean(raw)
        features = self.engineer_features(cleaned)
        labels = self.build_target(features)
        sequences, sequence_labels = self.build_sequences(features, labels)
        return self.get_dataloaders(sequences, sequence_labels)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_columns)
