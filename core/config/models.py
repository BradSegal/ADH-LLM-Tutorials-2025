"""
Typed configuration models for project-wide settings.

All configuration is validated through these Pydantic models before it is
used anywhere else in the codebase. This ensures that environment-specific
paths, download URLs, and cache locations fail fast if they are missing or
malformed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class PhysioNetConfig(BaseModel):
    """Paths and remote locations for the PhysioNet 2019 dataset."""

    model_config = ConfigDict(validate_assignment=True)

    raw_dir: Path = Field(..., description="Directory where raw .psv files live.")
    processed_dir: Path = Field(..., description="Directory for derived artifacts.")
    processed_file: Path = Field(
        ..., description="Location of the cached, preprocessed parquet file."
    )
    scaler_file: Path = Field(
        ..., description="Location of the cached feature scaler artifact."
    )
    download_url: HttpUrl = Field(
        ..., description="Credentialed HTTPS endpoint for dataset download."
    )
    s3_bucket: str = Field(
        default="physionet-open",
        description="Public bucket that mirrors the challenge data.",
    )
    s3_prefix: str = Field(
        default="challenge-2019/1.0.0/training",
        description="Prefix inside the bucket that holds training .psv files.",
    )

    def ensure_directories(self) -> None:
        """Ensure directories referenced by this config exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)
        self.scaler_file.parent.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """Top-level configuration for all datasets used in the notebooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    physionet_2019: PhysioNetConfig


def load_data_config(config_path: Path) -> DataConfig:
    """
    Load and validate the global data configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to `configs/data.yaml`.

    Returns
    -------
    DataConfig
        Typed configuration object ready for downstream consumption.

    Raises
    ------
    FileNotFoundError
        If the YAML file is missing.
    yaml.YAMLError
        If the YAML cannot be parsed cleanly.
    pydantic.ValidationError
        If any of the required keys are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please ensure configs/data.yaml exists."
        )

    raw_data = yaml.safe_load(config_path.read_text())
    if raw_data is None:
        raise ValueError(
            f"Configuration file {config_path} is empty. "
            "Please provide the required PhysioNet settings."
        )

    return DataConfig(**raw_data)


class TrainConfig(BaseModel):
    """
    Configuration for the training process.

    This model validates and stores all hyperparameters required for model
    training, ensuring they meet sensible constraints before the training loop
    begins (fail fast principle).
    """

    model_config = ConfigDict(validate_assignment=True)

    learning_rate: float = Field(
        ..., gt=0.0, description="Learning rate for the optimizer. Must be positive."
    )
    epochs: int = Field(
        ..., gt=0, description="Number of training epochs. Must be positive."
    )
    batch_size: int = Field(
        ...,
        gt=0,
        description="Batch size for training and evaluation. Must be positive.",
    )
    device: str = Field(
        default="cuda",
        description="Device to use for training. Either 'cuda' or 'cpu'.",
    )
    optimizer: Literal["adam", "sgd"] = Field(
        default="adam",
        description=(
            "Optimizer to use during training. Supported options: 'adam', 'sgd'."
        ),
    )
    optimizer_momentum: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Momentum factor for SGD optimizer. Ignored when optimizer != 'sgd'. "
            "Must be in the range [0.0, 1.0)."
        ),
    )
    loss: Literal["bce_logits", "cross_entropy", "mse"] = Field(
        default="bce_logits",
        description=(
            "Loss function to optimize. "
            "'bce_logits' -> nn.BCEWithLogitsLoss, "
            "'cross_entropy' -> nn.CrossEntropyLoss, "
            "'mse' -> nn.MSELoss."
        ),
    )
    scheduler: Literal["none", "step"] = Field(
        default="none",
        description=(
            "Learning rate scheduler strategy. "
            "'step' uses torch.optim.lr_scheduler.StepLR. "
            "'none' disables scheduling."
        ),
    )
    scheduler_step_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of epochs between StepLR adjustments. "
            "Required when scheduler='step'."
        ),
    )
    scheduler_gamma: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Multiplicative factor of learning rate decay for StepLR. "
            "Required when scheduler='step'. Must be between 0 and 1."
        ),
    )
    max_grad_norm: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "If provided, gradients are clipped to this max L2 norm after backward()."
        ),
    )
    train_val_split: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of patients assigned to the training set. "
            "Must be strictly between 0 and 1."
        ),
    )
    split_seed: int | None = Field(
        default=42,
        description=(
            "Seed used for deterministic train/validation splits. "
            "Set to None to allow nondeterministic splitting."
        ),
    )
    val_metrics: tuple[str, ...] = Field(
        default=("auprc",),
        description=(
            "Validation metrics to compute each epoch. "
            "Supported values: 'auprc', 'auroc'. Provide an empty tuple to "
            "disable metric computation."
        ),
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is either 'cuda' or 'cpu'."""
        allowed = {"cuda", "cpu"}
        if v not in allowed:
            raise ValueError(
                f"Device must be one of {allowed}, got '{v}'. "
                "Note: actual CUDA availability is checked at runtime by Trainer."
            )
        return v

    @model_validator(mode="after")
    def validate_scheduler_settings(self) -> TrainConfig:
        """Ensure scheduler parameters are provided / omitted correctly."""
        if self.scheduler == "step":
            if self.scheduler_step_size is None or self.scheduler_gamma is None:
                raise ValueError(
                    "scheduler_step_size and scheduler_gamma must be provided when "
                    "scheduler='step'."
                )
        else:
            if self.scheduler_step_size is not None or self.scheduler_gamma is not None:
                raise ValueError(
                    "scheduler_step_size and scheduler_gamma should not be provided "
                    "when scheduler='none'."
                )
        return self

    @field_validator("split_seed")
    @classmethod
    def validate_split_seed(cls, value: int | None) -> int | None:
        """Ensure split_seed is non-negative when provided."""
        if value is not None and value < 0:
            raise ValueError("split_seed must be a non-negative integer or None.")
        return value

    @field_validator("val_metrics")
    @classmethod
    def validate_val_metrics(cls, metrics: tuple[str, ...]) -> tuple[str, ...]:
        """Ensure requested metrics are supported."""
        allowed = {"auprc", "auroc"}
        if any(metric not in allowed for metric in metrics):
            raise ValueError(
                "Unsupported validation metric requested. "
                f"Allowed values: {', '.join(sorted(allowed))}."
            )
        return metrics


class GRUConfig(BaseModel):
    """
    Configuration for the GRU (Gated Recurrent Unit) model.

    This model validates and stores all hyperparameters required for
    instantiating and training a GRU-based sequence model for sepsis prediction.
    All parameters are validated at construction time to ensure they meet
    sensible constraints (fail fast principle).
    """

    model_config = ConfigDict(validate_assignment=True)

    input_size: int = Field(
        ...,
        gt=0,
        description="Number of input features per time step. Must be positive.",
    )
    hidden_size: int = Field(
        default=64,
        gt=0,
        description="Size of the GRU hidden state. Must be positive.",
    )
    num_layers: int = Field(
        default=1, gt=0, description="Number of stacked GRU layers. Must be positive."
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout probability applied between GRU layers. "
            "Must be in the range [0.0, 1.0). "
            "Ignored when num_layers == 1."
        ),
    )


class LSTMConfig(BaseModel):
    """
    Configuration for the LSTM (Long Short-Term Memory) model.

    This model validates and stores all hyperparameters required for
    instantiating and training an LSTM-based sequence model for sepsis prediction.
    All parameters are validated at construction time to ensure they meet
    sensible constraints (fail fast principle).
    """

    model_config = ConfigDict(validate_assignment=True)

    input_size: int = Field(
        ...,
        gt=0,
        description="Number of input features per time step. Must be positive.",
    )
    hidden_size: int = Field(
        default=64,
        gt=0,
        description="Size of the LSTM hidden state. Must be positive.",
    )
    num_layers: int = Field(
        default=1, gt=0, description="Number of stacked LSTM layers. Must be positive."
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout probability applied between LSTM layers. "
            "Must be in the range [0.0, 1.0). "
            "Ignored when num_layers == 1."
        ),
    )


class TransformerConfig(BaseModel):
    """
    Configuration for the Transformer encoder model.

    This model validates and stores all hyperparameters required for
    instantiating and training a Transformer-based sequence model for sepsis
    prediction. All parameters are validated at construction time to ensure
    they meet sensible constraints (fail fast principle).
    """

    model_config = ConfigDict(validate_assignment=True)

    input_size: int = Field(
        ...,
        gt=0,
        description=(
            "Number of input features (embedding dimension). Must be positive."
        ),
    )
    nhead: int = Field(
        default=2,
        gt=0,
        description="Number of heads in multi-head attention. Must be positive.",
    )
    num_encoder_layers: int = Field(
        default=2,
        gt=0,
        description="Number of sub-encoder-layers in the encoder. Must be positive.",
    )
    dim_feedforward: int = Field(
        default=128,
        gt=0,
        description=("Dimension of the feedforward network model. Must be positive."),
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout probability applied in attention and feedforward layers. "
            "Must be in the range [0.0, 1.0)."
        ),
    )
    max_seq_length: int = Field(
        default=5000,
        gt=0,
        description=(
            "Maximum supported sequence length for positional encodings. "
            "Increase when working with longer sequences."
        ),
    )

    @model_validator(mode="after")
    def validate_input_size_divisible_by_nhead(self) -> TransformerConfig:
        """
        Ensure input_size is divisible by nhead.

        The multi-head attention mechanism requires that the model dimension
        (input_size) can be evenly divided among the attention heads.
        """
        if self.input_size % self.nhead != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by nhead "
                f"({self.nhead}). Each attention head requires "
                f"input_size / nhead features."
            )
        return self
