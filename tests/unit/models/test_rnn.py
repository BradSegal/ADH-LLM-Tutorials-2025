"""
Unit tests for the GRU model implementation.

These tests validate the GRUModel class without requiring network access or
heavy computations. They ensure the model correctly implements the
BaseSequenceModel contract and produces outputs of the expected shape.
"""

import pytest
import torch

from core.config.models import GRUConfig
from core.models.base import BaseSequenceModel
from core.models.rnn import GRUModel


@pytest.fixture
def default_gru_config() -> GRUConfig:
    """
    Provide a default GRUConfig for testing.

    Returns
    -------
    GRUConfig
        A valid configuration with standard hyperparameters.
    """
    return GRUConfig(
        input_size=34,  # PhysioNet dataset has 34 features
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )


@pytest.fixture
def single_layer_gru_config() -> GRUConfig:
    """
    Provide a GRUConfig with a single layer (dropout should be ignored).

    Returns
    -------
    GRUConfig
        A valid configuration with num_layers=1.
    """
    return GRUConfig(
        input_size=34,
        hidden_size=32,
        num_layers=1,
        dropout=0.5,  # Should be ignored when num_layers=1
    )


class TestGRUModelInitialization:
    """Test suite for GRUModel instantiation."""

    def test_gru_model_initialization_with_default_config(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test that GRUModel can be instantiated with a valid config."""
        model = GRUModel(default_gru_config)

        assert isinstance(model, GRUModel)
        assert isinstance(model, BaseSequenceModel)
        assert model.config == default_gru_config

    def test_gru_model_initialization_with_single_layer(
        self, single_layer_gru_config: GRUConfig
    ) -> None:
        """Test that GRUModel correctly handles single-layer configuration."""
        model = GRUModel(single_layer_gru_config)

        assert isinstance(model, GRUModel)
        # Verify that dropout is set to 0 for single-layer GRU
        assert model.gru.dropout == 0.0

    def test_gru_model_has_required_components(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test that GRUModel has the expected internal components."""
        model = GRUModel(default_gru_config)

        # Model should have GRU layer
        assert hasattr(model, "gru")
        assert isinstance(model.gru, torch.nn.GRU)

        # Model should have classifier layer
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, torch.nn.Linear)

        # Verify GRU configuration
        assert model.gru.input_size == default_gru_config.input_size
        assert model.gru.hidden_size == default_gru_config.hidden_size
        assert model.gru.num_layers == default_gru_config.num_layers
        assert model.gru.batch_first is True

        # Verify classifier configuration
        assert model.classifier.in_features == default_gru_config.hidden_size
        assert model.classifier.out_features == 1


class TestGRUModelForwardPass:
    """Test suite for GRUModel forward pass."""

    def test_gru_model_forward_pass_output_shape(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test that the forward pass produces the correct output shape."""
        model = GRUModel(default_gru_config)
        model.eval()  # Set to eval mode to disable dropout

        batch_size = 8
        seq_length = 50
        input_size = default_gru_config.input_size

        # Create dummy input tensors
        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Run forward pass
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape
        assert logits.shape == (batch_size,)
        assert logits.dtype == torch.float32

    def test_gru_model_forward_pass_with_variable_sequence_lengths(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test that the model correctly handles variable-length sequences."""
        model = GRUModel(default_gru_config)
        model.eval()

        batch_size = 4
        max_seq_length = 100
        input_size = default_gru_config.input_size

        # Create dummy input with variable sequence lengths
        x = torch.randn(batch_size, max_seq_length, input_size)

        # Create masks with different sequence lengths
        # Sequence lengths: [20, 50, 75, 100]
        mask = torch.zeros(batch_size, max_seq_length, dtype=torch.bool)
        mask[0, :20] = True
        mask[1, :50] = True
        mask[2, :75] = True
        mask[3, :100] = True

        # Run forward pass
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape
        assert logits.shape == (batch_size,)
        assert not torch.isnan(logits).any(), "Output contains NaN values"
        assert not torch.isinf(logits).any(), "Output contains Inf values"

    def test_gru_model_forward_pass_single_batch(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test forward pass with a single sample (batch_size=1)."""
        model = GRUModel(default_gru_config)
        model.eval()

        seq_length = 30
        input_size = default_gru_config.input_size

        # Create single sample
        x = torch.randn(1, seq_length, input_size)
        mask = torch.ones(1, seq_length, dtype=torch.bool)

        # Run forward pass
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape
        assert logits.shape == (1,)

    def test_gru_model_forward_pass_produces_different_outputs(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test that different inputs produce different outputs."""
        model = GRUModel(default_gru_config)
        model.eval()

        batch_size = 2
        seq_length = 40
        input_size = default_gru_config.input_size

        # Create two different inputs
        x1 = torch.randn(batch_size, seq_length, input_size)
        x2 = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Run forward pass on both inputs
        with torch.no_grad():
            logits1 = model(x1, mask)
            logits2 = model(x2, mask)

        # Outputs should be different (with very high probability)
        assert not torch.allclose(logits1, logits2, atol=1e-6)

    def test_gru_model_gradient_flow(self, default_gru_config: GRUConfig) -> None:
        """Test that gradients flow correctly during backpropagation."""
        model = GRUModel(default_gru_config)
        model.train()  # Set to training mode

        batch_size = 4
        seq_length = 20
        input_size = default_gru_config.input_size

        # Create dummy input and target
        x = torch.randn(batch_size, seq_length, input_size, requires_grad=True)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        target = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

        # Forward pass
        logits = model(x, mask)

        # Compute loss and backward
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, target)
        loss.backward()

        # Check that gradients exist for model parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_gru_model_rejects_empty_sequences(
        self, default_gru_config: GRUConfig
    ) -> None:
        """Test model raises ValueError for sequences with no valid timesteps."""
        model = GRUModel(default_gru_config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_size = default_gru_config.input_size

        # Create input with all-False mask (no valid timesteps)
        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

        # Forward pass should raise ValueError
        with pytest.raises(ValueError, match="Found sequences with no valid timesteps"):
            model(x, mask)


class TestGRUConfigValidation:
    """Test suite for GRUConfig validation."""

    def test_gru_config_requires_positive_input_size(self) -> None:
        """Test that input_size must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            GRUConfig(input_size=0, hidden_size=64)

        with pytest.raises(ValueError, match="greater than 0"):
            GRUConfig(input_size=-1, hidden_size=64)

    def test_gru_config_requires_positive_hidden_size(self) -> None:
        """Test that hidden_size must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            GRUConfig(input_size=34, hidden_size=0)

        with pytest.raises(ValueError, match="greater than 0"):
            GRUConfig(input_size=34, hidden_size=-10)

    def test_gru_config_requires_positive_num_layers(self) -> None:
        """Test that num_layers must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            GRUConfig(input_size=34, hidden_size=64, num_layers=0)

    def test_gru_config_dropout_bounds(self) -> None:
        """Test that dropout must be in [0, 1)."""
        # Valid dropout values
        config1 = GRUConfig(input_size=34, hidden_size=64, dropout=0.0)
        assert config1.dropout == 0.0

        config2 = GRUConfig(input_size=34, hidden_size=64, dropout=0.99)
        assert config2.dropout == 0.99

        # Invalid: dropout >= 1.0
        with pytest.raises(ValueError):
            GRUConfig(input_size=34, hidden_size=64, dropout=1.0)

        # Invalid: dropout < 0
        with pytest.raises(ValueError):
            GRUConfig(input_size=34, hidden_size=64, dropout=-0.1)

    def test_gru_config_accepts_valid_parameters(self) -> None:
        """Test that GRUConfig accepts valid parameter combinations."""
        config = GRUConfig(input_size=34, hidden_size=128, num_layers=3, dropout=0.3)

        assert config.input_size == 34
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.dropout == 0.3
