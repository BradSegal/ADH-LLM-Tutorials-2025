"""
Unit tests for configuration models.

Tests the Pydantic configuration models to ensure they properly validate input
and fail fast with clear error messages when given invalid data.
"""

import pytest
from pydantic import ValidationError

from core.config.models import TrainConfig


class TestTrainConfig:
    """Test suite for TrainConfig validation."""

    def test_train_config_importable_from_package(self) -> None:
        """Test that TrainConfig can be imported from core.config package."""
        # This test ensures the __init__.py exports are correct
        from core.config import TrainConfig as PackageTrainConfig

        assert PackageTrainConfig is TrainConfig

    def test_valid_config(self) -> None:
        """Test that TrainConfig accepts valid parameters."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
            device="cuda",
        )

        assert config.learning_rate == 0.001
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.device == "cuda"
        assert config.optimizer == "adam"
        assert config.loss == "bce_logits"
        assert config.scheduler == "none"
        assert config.scheduler_step_size is None
        assert config.scheduler_gamma is None
        assert config.max_grad_norm is None
        assert config.train_val_split == 0.8
        assert config.split_seed == 42

    def test_valid_config_cpu_device(self) -> None:
        """Test that TrainConfig accepts CPU as a valid device."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
            device="cpu",
        )

        assert config.device == "cpu"

    def test_default_device_is_cuda(self) -> None:
        """Test that device defaults to 'cuda' when not specified."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
        )

        assert config.device == "cuda"
        assert config.train_val_split == 0.8
        assert config.split_seed == 42

    def test_negative_learning_rate_fails(self) -> None:
        """Test that negative learning rate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=-0.1,
                epochs=10,
                batch_size=32,
            )

        # Check that the error message mentions learning_rate
        error_str = str(exc_info.value)
        assert "learning_rate" in error_str.lower()

    def test_zero_learning_rate_fails(self) -> None:
        """Test that zero learning rate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.0,
                epochs=10,
                batch_size=32,
            )

        error_str = str(exc_info.value)
        assert "learning_rate" in error_str.lower()

    def test_negative_epochs_fails(self) -> None:
        """Test that negative epochs raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=-5,
                batch_size=32,
            )

        error_str = str(exc_info.value)
        assert "epochs" in error_str.lower()

    def test_zero_epochs_fails(self) -> None:
        """Test that zero epochs raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=0,
                batch_size=32,
            )

        error_str = str(exc_info.value)
        assert "epochs" in error_str.lower()

    def test_negative_batch_size_fails(self) -> None:
        """Test that negative batch_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=-16,
            )

        error_str = str(exc_info.value)
        assert "batch_size" in error_str.lower()

    def test_zero_batch_size_fails(self) -> None:
        """Test that zero batch_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=0,
            )

        error_str = str(exc_info.value)
        assert "batch_size" in error_str.lower()

    def test_invalid_device_fails(self) -> None:
        """Test that an invalid device string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                device="gpu",  # Invalid - should be 'cuda' or 'cpu'
            )

        error_str = str(exc_info.value)
        assert "device" in error_str.lower()

    def test_missing_required_fields_fails(self) -> None:
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            # Missing learning_rate, epochs, batch_size
            TrainConfig()  # type: ignore[call-arg]

        error_str = str(exc_info.value)
        # Should mention at least one of the missing required fields
        assert any(
            field in error_str.lower()
            for field in ["learning_rate", "epochs", "batch_size"]
        )

    def test_config_immutability_with_validation(self) -> None:
        """Test that config validates on assignment (validate_assignment=True)."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
        )

        # Try to assign an invalid value after creation
        with pytest.raises(ValidationError):
            config.learning_rate = -0.5

    def test_device_validation_message_is_helpful(self) -> None:
        """Test that device validation provides helpful error message."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                device="mps",  # Invalid device
            )

        error_str = str(exc_info.value)
        # Should mention the allowed values
        assert "cuda" in error_str.lower() or "cpu" in error_str.lower()

    def test_sgd_optimizer_accepts_momentum(self) -> None:
        """Test that SGD optimizer configuration is valid."""
        config = TrainConfig(
            learning_rate=0.01,
            epochs=5,
            batch_size=16,
            optimizer="sgd",
            optimizer_momentum=0.9,
        )

        assert config.optimizer == "sgd"
        assert config.optimizer_momentum == 0.9

    def test_train_val_split_bounds(self) -> None:
        """train_val_split must be strictly between 0 and 1."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=1,
                batch_size=8,
                train_val_split=0.0,
            )

        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=1,
                batch_size=8,
                train_val_split=1.0,
            )

        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=8,
            train_val_split=0.65,
        )
        assert config.train_val_split == 0.65

    def test_split_seed_accepts_none_and_non_negative(self) -> None:
        """split_seed can be None or non-negative."""
        config_with_none = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=8,
            split_seed=None,
        )
        assert config_with_none.split_seed is None

        config_with_value = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=8,
            split_seed=1337,
        )
        assert config_with_value.split_seed == 1337

        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=1,
                batch_size=8,
                split_seed=-5,
            )

    def test_invalid_optimizer_value_fails(self) -> None:
        """Test that unsupported optimizer names raise ValidationError."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.01,
                epochs=5,
                batch_size=16,
                optimizer="adamw",  # type: ignore[arg-type]  # Not supported
            )

    def test_scheduler_requires_parameters(self) -> None:
        """Test that scheduler='step' requires step_size and gamma."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                scheduler="step",
                scheduler_step_size=None,
                scheduler_gamma=None,
            )

    def test_scheduler_params_disallowed_when_none(self) -> None:
        """Test scheduler params cannot be provided when scheduler='none'."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                scheduler="none",
                scheduler_step_size=5,
                scheduler_gamma=0.5,
            )

    def test_invalid_scheduler_gamma_value(self) -> None:
        """Test that scheduler gamma must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                scheduler="step",
                scheduler_step_size=2,
                scheduler_gamma=1.5,
            )

    def test_max_grad_norm_must_be_positive(self) -> None:
        """Test that max_grad_norm must be positive when provided."""
        with pytest.raises(ValidationError):
            TrainConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                max_grad_norm=0.0,
            )
