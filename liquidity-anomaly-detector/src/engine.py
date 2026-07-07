"""
engine.py
=========

Training / validation orchestration for `LiquidityAnomalyLSTM`, including
early stopping on validation loss -- essential when fitting recurrent
networks to noisy, non-stationary financial data where unchecked
training overfits to sample-specific microstructure noise rather than
generalizable liquidity dynamics.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for `ModelTrainer`."""

    task: Literal["classification", "regression"] = "classification"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cpu"
    checkpoint_path: Optional[str] = None
    verbose: bool = True


class ModelTrainer:
    """
    Handles the full optimization loop for a `LiquidityAnomalyLSTM` (or any
    `nn.Module` emitting a single raw score per sequence), with:

      * task-appropriate loss selection (BCEWithLogitsLoss / MSELoss),
      * gradient clipping to control exploding gradients in recurrent
        nets,
      * early stopping on validation loss with automatic restoration of
        the best-performing weights.
    """

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.criterion = self._build_criterion(config.task)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        self._best_val_loss: float = float("inf")
        self._epochs_no_improve: int = 0
        self._best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _build_criterion(task: str) -> nn.Module:
        if task == "classification":
            return nn.BCEWithLogitsLoss()
        if task == "regression":
            return nn.MSELoss()
        raise ValueError(f"Unsupported task '{task}'; expected 'classification' or 'regression'.")

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device).float()
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(mode=train)
        total_loss, n_obs = 0.0, 0
        previous_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(train)
        try:
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(self.device).float()
                y = self._prepare_target(y)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(x)
                if outputs.shape != y.shape:
                    raise RuntimeError(
                        f"Model output shape {tuple(outputs.shape)} does not match target shape "
                        f"{tuple(y.shape)}; check output_size / target construction."
                    )
                loss = self.criterion(outputs, y)

                if train:
                    loss.backward()
                    if self.config.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                n_obs += batch_size
        finally:
            torch.set_grad_enabled(previous_grad_state)

        if n_obs == 0:
            raise RuntimeError("Encountered an empty DataLoader during training/validation.")
        return total_loss / n_obs

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Run the full training loop with early stopping.

        Returns
        -------
        Dict[str, List[float]]
            Per-epoch train/validation loss history (only the epochs
            actually run, i.e. truncated early if early stopping fires).
        """
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            improved = val_loss < (self._best_val_loss - self.config.min_delta)
            if improved:
                self._best_val_loss = val_loss
                self._epochs_no_improve = 0
                self._best_state_dict = copy.deepcopy(self.model.state_dict())
                if self.config.checkpoint_path:
                    torch.save(self._best_state_dict, self.config.checkpoint_path)
            else:
                self._epochs_no_improve += 1

            if self.config.verbose:
                logger.info(
                    "epoch=%03d train_loss=%.6f val_loss=%.6f best_val_loss=%.6f no_improve=%d/%d",
                    epoch,
                    train_loss,
                    val_loss,
                    self._best_val_loss,
                    self._epochs_no_improve,
                    self.config.patience,
                )

            if self._epochs_no_improve >= self.config.patience:
                if self.config.verbose:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.6f).", epoch, self._best_val_loss
                    )
                break

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        return self.history

    def evaluate(self, loader: DataLoader) -> float:
        """Compute loss on a held-out loader using the current (best-restored) weights."""
        return self._run_epoch(loader, train=False)
