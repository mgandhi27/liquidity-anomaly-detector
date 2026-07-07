"""
liquidity-anomaly-detector: source package.

Exposes the primary building blocks so a training script can do, e.g.:

    from src import DataPipeline, PipelineConfig, LiquidityAnomalyLSTM, \\
        LSTMConfig, ModelTrainer, TrainerConfig
"""

from .data_pipeline import DataPipeline, LOBSequenceDataset, PipelineConfig
from .engine import ModelTrainer, TrainerConfig
from .model import LiquidityAnomalyLSTM, LSTMConfig
from .transforms import FinancialTimeTransform

__all__ = [
    "DataPipeline",
    "LOBSequenceDataset",
    "PipelineConfig",
    "ModelTrainer",
    "TrainerConfig",
    "LiquidityAnomalyLSTM",
    "LSTMConfig",
    "FinancialTimeTransform",
]
