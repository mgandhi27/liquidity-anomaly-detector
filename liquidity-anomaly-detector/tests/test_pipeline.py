"""
tests/test_pipeline.py
=======================

Unit tests focused on shape correctness and basic numerical sanity across
the transform, model, and pipeline layers. The failure mode most worth
guarding against in sequence-model pipelines is a silent shape mismatch
(e.g. an off-by-one alignment between features and labels) rather than
an outright crash, so shape assertions are the primary emphasis here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import DataPipeline, LOBSequenceDataset, PipelineConfig
from src.engine import ModelTrainer, TrainerConfig
from src.model import LiquidityAnomalyLSTM, LSTMConfig
from src.transforms import FinancialTimeTransform


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def synthetic_lob_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2000
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.05, size=n))
    half_spread = 0.01 + 0.005 * rng.random(n)
    bid_price = mid - half_spread
    ask_price = mid + half_spread
    total_depth = 500.0 * rng.lognormal(0.0, 0.2, size=n)
    bid_size = total_depth * 0.5
    ask_size = total_depth * 0.5
    timestamps = pd.date_range("2026-01-02 09:30:00", periods=n, freq="1s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": bid_price,
            "bid_size": bid_size,
            "ask_price": ask_price,
            "ask_size": ask_size,
        }
    )


@pytest.fixture
def pipeline() -> DataPipeline:
    return DataPipeline(PipelineConfig(sequence_length=20, forecast_horizon=5, batch_size=16))


# ----------------------------------------------------------------------
# FinancialTimeTransform
# ----------------------------------------------------------------------
class TestFinancialTimeTransform:
    def test_log_returns_length_and_positivity_check(self):
        transform = FinancialTimeTransform()
        prices = pd.Series([100.0, 101.0, 99.5, 102.0])
        returns = transform.log_returns(prices)
        assert len(returns) == len(prices) - 1
        with pytest.raises(ValueError):
            transform.log_returns(pd.Series([100.0, -1.0, 102.0]))

    def test_rolling_volatility_shape(self):
        transform = FinancialTimeTransform()
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
        vol = transform.rolling_volatility(returns, window=10)
        assert len(vol) == len(returns) - 10 + 1
        assert (vol >= 0).all()

    def test_rolling_spread_rejects_crossed_book(self):
        transform = FinancialTimeTransform()
        bid = pd.Series([100.0, 101.0])
        ask = pd.Series([99.5, 101.5])  # first row is crossed (bid > ask)
        with pytest.raises(ValueError):
            transform.rolling_spread(bid, ask, window=1)

    def test_fractional_difference_d1_matches_first_difference(self):
        transform = FinancialTimeTransform()
        rng = np.random.default_rng(1)
        series = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
        fd = transform.fractional_difference(series, d=1.0, threshold=1e-5)
        manual_diff = series.diff().dropna()
        aligned = manual_diff.reindex(fd.index)
        np.testing.assert_allclose(fd.to_numpy(), aligned.to_numpy(), atol=1e-8)

    def test_fractional_difference_rejects_interior_nans(self):
        transform = FinancialTimeTransform()
        series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            transform.fractional_difference(series, d=0.5)

    def test_adf_detects_stationary_white_noise(self):
        transform = FinancialTimeTransform()
        rng = np.random.default_rng(2)
        white_noise = pd.Series(rng.normal(0, 1, 500))
        result = transform.augmented_dickey_fuller(white_noise)
        assert result["is_stationary_95pct"] is True


# ----------------------------------------------------------------------
# DataPipeline
# ----------------------------------------------------------------------
class TestDataPipeline:
    def test_clean_rejects_missing_columns(self, pipeline: DataPipeline):
        with pytest.raises(ValueError):
            pipeline.clean(pd.DataFrame({"timestamp": [1, 2, 3]}))

    def test_engineer_features_and_sequence_shapes(
        self, pipeline: DataPipeline, synthetic_lob_df: pd.DataFrame
    ):
        cleaned = pipeline.clean(synthetic_lob_df)
        features = pipeline.engineer_features(cleaned)
        labels = pipeline.build_target(features)
        sequences, sequence_labels = pipeline.build_sequences(features, labels)

        n_features = len(pipeline.feature_names)
        assert sequences.ndim == 3
        assert sequences.shape[1] == pipeline.config.sequence_length
        assert sequences.shape[2] == n_features
        assert sequences.shape[0] == len(sequence_labels)
        assert set(np.unique(sequence_labels)).issubset({0.0, 1.0})

    def test_get_dataloaders_split_proportions(
        self, pipeline: DataPipeline, synthetic_lob_df: pd.DataFrame
    ):
        cleaned = pipeline.clean(synthetic_lob_df)
        features = pipeline.engineer_features(cleaned)
        labels = pipeline.build_target(features)
        sequences, sequence_labels = pipeline.build_sequences(features, labels)
        train_loader, val_loader, test_loader = pipeline.get_dataloaders(sequences, sequence_labels)

        n_total = len(sequences)
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset)
        assert n_train + n_val + n_test == n_total
        assert n_train > n_val > 0
        assert n_test > 0

    def test_lob_sequence_dataset_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            LOBSequenceDataset(np.zeros((5, 10, 3)), np.zeros(4))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class TestLiquidityAnomalyLSTM:
    def test_forward_output_shape(self):
        config = LSTMConfig(input_size=5, hidden_size=16, num_layers=2, dropout=0.1, sequence_length=30)
        model = LiquidityAnomalyLSTM(config)
        x = torch.randn(8, 30, 5)
        out = model(x)
        assert out.shape == (8, 1)

    def test_forward_rejects_wrong_feature_dim(self):
        config = LSTMConfig(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, sequence_length=30)
        model = LiquidityAnomalyLSTM(config)
        x = torch.randn(4, 30, 6)
        with pytest.raises(ValueError):
            model(x)

    def test_predict_proba_bounded_in_unit_interval(self):
        config = LSTMConfig(input_size=4, hidden_size=8, num_layers=1, dropout=0.0, sequence_length=10)
        model = LiquidityAnomalyLSTM(config)
        x = torch.randn(6, 10, 4)
        probs = model.predict_proba(x)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            LiquidityAnomalyLSTM(LSTMConfig(input_size=0))


# ----------------------------------------------------------------------
# ModelTrainer
# ----------------------------------------------------------------------
class TestModelTrainer:
    def test_fit_runs_and_populates_history(self):
        config = LSTMConfig(input_size=3, hidden_size=8, num_layers=1, dropout=0.0, sequence_length=10)
        model = LiquidityAnomalyLSTM(config)
        trainer_config = TrainerConfig(task="classification", max_epochs=3, patience=3, verbose=False)
        trainer = ModelTrainer(model, trainer_config)

        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 10, 3)).astype(np.float32)
        y = rng.integers(0, 2, size=40).astype(np.float32)
        dataset = LOBSequenceDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        history = trainer.fit(loader, loader)
        assert len(history["train_loss"]) <= 3
        assert len(history["train_loss"]) == len(history["val_loss"])

    def test_early_stopping_triggers(self):
        config = LSTMConfig(input_size=2, hidden_size=4, num_layers=1, dropout=0.0, sequence_length=5)
        model = LiquidityAnomalyLSTM(config)
        trainer_config = TrainerConfig(
            task="regression", max_epochs=50, patience=2, min_delta=1e10, verbose=False
        )
        trainer = ModelTrainer(model, trainer_config)

        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 5, 2)).astype(np.float32)
        y = rng.normal(size=20).astype(np.float32)
        dataset = LOBSequenceDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        history = trainer.fit(loader, loader)
        # With an impossibly large min_delta, no epoch after the first counts
        # as "improved", so early stopping should trigger well before max_epochs.
        assert len(history["train_loss"]) <= trainer_config.patience + 1
