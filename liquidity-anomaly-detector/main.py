"""
main.py
=======

CLI entry point for the liquidity-anomaly-detector project.

Usage
-----
    python main.py simulate --out data/raw/lob_snapshots.csv --n-rows 20000
    python main.py train --data data/raw/lob_snapshots.csv --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import DataPipeline, PipelineConfig
from src.engine import ModelTrainer, TrainerConfig
from src.model import LiquidityAnomalyLSTM, LSTMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("liquidity_anomaly_detector")


def simulate_lob_data(n_rows: int, seed: int = 7) -> pd.DataFrame:
    """
    Generate a synthetic Level-2 snapshot series for demonstration and
    testing when real market data is unavailable. The mid-price follows a
    mean-reverting (Ornstein-Uhlenbeck-style) process; depth is modulated
    by an independent regime-switching process so that genuine
    "liquidity dry-up" episodes exist in the data for the model to learn.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0
    mid = np.zeros(n_rows)
    mid[0] = 100.0
    kappa, theta, sigma = 0.02, 100.0, 0.05
    for t in range(1, n_rows):
        mid[t] = mid[t - 1] + kappa * (theta - mid[t - 1]) * dt + sigma * rng.normal()

    # Regime-switching liquidity: occasional depth-collapse events.
    base_depth = 500.0
    regime = np.ones(n_rows)
    in_shock = False
    shock_remaining = 0
    for t in range(n_rows):
        if not in_shock and rng.random() < 0.002:
            in_shock = True
            shock_remaining = int(rng.integers(20, 80))
        if in_shock:
            regime[t] = 0.15
            shock_remaining -= 1
            if shock_remaining <= 0:
                in_shock = False
        else:
            regime[t] = 1.0
    depth_noise = rng.lognormal(mean=0.0, sigma=0.15, size=n_rows)
    total_depth = base_depth * regime * depth_noise

    half_spread = 0.01 + 0.02 * (1.0 - regime) + 0.005 * rng.random(n_rows)
    bid_price = mid - half_spread
    ask_price = mid + half_spread
    bid_size = total_depth * rng.uniform(0.4, 0.6, size=n_rows)
    ask_size = total_depth - bid_size

    timestamps = pd.date_range("2026-01-02 09:30:00", periods=n_rows, freq="1s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": bid_price,
            "bid_size": bid_size,
            "ask_price": ask_price,
            "ask_size": ask_size,
        }
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    df = simulate_lob_data(n_rows=args.n_rows, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d synthetic snapshot rows to %s", len(df), out_path)


def cmd_train(args: argparse.Namespace) -> None:
    pipeline_config = PipelineConfig(
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
    )
    pipeline = DataPipeline(pipeline_config)
    train_loader, val_loader, test_loader = pipeline.run(args.data)
    logger.info("Engineered feature columns: %s", pipeline.feature_names)

    model_config = LSTMConfig(
        input_size=len(pipeline.feature_names),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        output_size=1,
    )
    model = LiquidityAnomalyLSTM(model_config)

    checkpoint_path = args.checkpoint
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    trainer_config = TrainerConfig(
        task="classification",
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    trainer = ModelTrainer(model, trainer_config)

    history = trainer.fit(train_loader, val_loader)
    test_loss = trainer.evaluate(test_loader)
    logger.info("Final test loss: %.6f", test_loss)
    logger.info("Training history (last 5 epochs): %s", {k: v[-5:] for k, v in history.items()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Liquidity Anomaly Detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Generate synthetic L2 snapshot data.")
    sim_parser.add_argument("--out", type=str, default="data/raw/lob_snapshots.csv")
    sim_parser.add_argument("--n-rows", type=int, default=20000)
    sim_parser.add_argument("--seed", type=int, default=7)
    sim_parser.set_defaults(func=cmd_simulate)

    train_parser = subparsers.add_parser("train", help="Train the liquidity anomaly LSTM.")
    train_parser.add_argument("--data", type=str, required=True)
    train_parser.add_argument("--sequence-length", type=int, default=50)
    train_parser.add_argument("--forecast-horizon", type=int, default=10)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    train_parser.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error instead of a raw traceback
        logger.error("Command failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
