"""
Unit tests for the Transformer model implementation.

These tests validate the TransformerModel class and PositionalEncoding module
without requiring network access or heavy computations. They ensure the model
correctly implements the BaseSequenceModel contract and produces outputs of
the expected shape.
"""

import pytest
import torch

from core.config.models import TransformerConfig
from core.models.base import BaseSequenceModel
from core.models.transformer import PositionalEncoding, TransformerModel


@pytest.fixture
def default_transformer_config() -> TransformerConfig:
    """
    Provide a default TransformerConfig for testing.

    Returns
    -------
    TransformerConfig
        A valid configuration with standard hyperparameters.
        Note: input_size must be divisible by nhead.
    """
    return TransformerConfig(
        input_size=32,  # Divisible by nhead=4
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )


@pytest.fixture
def single_layer_transformer_config() -> TransformerConfig:
    """
    Provide a TransformerConfig with a single encoder layer.

    Returns
    -------
    TransformerConfig
        A valid configuration with num_encoder_layers=1.
    """
    return TransformerConfig(
        input_size=64,  # Divisible by nhead=8
        nhead=8,
        num_encoder_layers=1,
        dim_feedforward=256,
        dropout=0.2,
    )


class TestPositionalEncoding:
    """Test suite for PositionalEncoding module."""

    def test_positional_encoding_initialization(self) -> None:
        """Test that PositionalEncoding can be instantiated."""
        pos_encoder = PositionalEncoding(d_model=32, dropout=0.1)

        assert isinstance(pos_encoder, PositionalEncoding)
        assert hasattr(pos_encoder, "pe")
        assert hasattr(pos_encoder, "dropout")

    def test_positional_encoding_buffer_shape(self) -> None:
        """Test that positional encoding buffer has correct shape."""
        d_model = 64
        max_len = 1000
        pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=max_len)

        # pe should be registered as a buffer with shape [1, max_len, d_model]
        assert pos_encoder.pe.shape == (1, max_len, d_model)

    def test_positional_encoding_forward_pass(self) -> None:
        """Test that positional encoding adds to input correctly."""
        d_model = 32
        batch_size = 4
        seq_length = 50

        pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.0)  # No dropout
        pos_encoder.eval()  # Disable dropout

        x = torch.zeros(batch_size, seq_length, d_model)

        with torch.no_grad():
            output = pos_encoder(x)

        # Output shape should match input shape
        assert output.shape == (batch_size, seq_length, d_model)

        # With dropout=0 in eval mode, output should equal pe[:, :seq_length, :]
        # (since input was all zeros)
        pe_buffer = pos_encoder.pe
        assert isinstance(pe_buffer, torch.Tensor)
        expected = pe_buffer[:, :seq_length, :]
        assert torch.allclose(output, expected.expand(batch_size, -1, -1))

    def test_positional_encoding_handles_different_sequence_lengths(self) -> None:
        """Test that positional encoding works with varying sequence lengths."""
        d_model = 16
        pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.0)
        pos_encoder.eval()

        # Test with different sequence lengths
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, d_model)
            with torch.no_grad():
                output = pos_encoder(x)
            assert output.shape == (2, seq_len, d_model)

    def test_positional_encoding_rejects_sequences_longer_than_max_len(self) -> None:
        """Test that positional encoding raises ValueError for too-long sequences."""
        d_model = 8
        max_len = 5
        pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=max_len)
        pos_encoder.eval()

        x = torch.randn(1, max_len + 1, d_model)

        with pytest.raises(
            ValueError, match="exceeds configured positional encoding capacity"
        ):
            pos_encoder(x)


class TestTransformerModelInitialization:
    """Test suite for TransformerModel instantiation."""

    def test_transformer_model_initialization_with_default_config(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that TransformerModel can be instantiated with a valid config."""
        model = TransformerModel(default_transformer_config)

        assert isinstance(model, TransformerModel)
        assert isinstance(model, BaseSequenceModel)
        assert model.config == default_transformer_config

    def test_transformer_model_has_required_components(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that TransformerModel has the expected internal components."""
        model = TransformerModel(default_transformer_config)

        # Model should have input projection
        assert hasattr(model, "input_projection")
        assert isinstance(model.input_projection, torch.nn.Linear)

        # Model should have positional encoder
        assert hasattr(model, "pos_encoder")
        assert isinstance(model.pos_encoder, PositionalEncoding)

        # Model should have transformer encoder
        assert hasattr(model, "transformer_encoder")
        assert isinstance(model.transformer_encoder, torch.nn.TransformerEncoder)

        # Model should have classifier layer
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, torch.nn.Linear)

        # Verify input projection configuration
        assert (
            model.input_projection.in_features == default_transformer_config.input_size
        )
        assert (
            model.input_projection.out_features == default_transformer_config.input_size
        )

        # Verify classifier configuration
        assert model.classifier.in_features == default_transformer_config.input_size
        assert model.classifier.out_features == 1


class TestTransformerModelForwardPass:
    """Test suite for TransformerModel forward pass."""

    def test_transformer_model_forward_pass_output_shape(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that the forward pass produces the correct output shape."""
        model = TransformerModel(default_transformer_config)
        model.eval()  # Set to eval mode to disable dropout

        batch_size = 8
        seq_length = 50
        input_size = default_transformer_config.input_size

        # Create dummy input tensors
        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Run forward pass
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape
        assert logits.shape == (batch_size,)

    def test_prepare_mask_expands_batches(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        model = TransformerModel(default_transformer_config)
        mask = torch.ones(1, 10, dtype=torch.bool)

        expanded = model.prepare_mask(mask, batch_size=4)

        assert expanded.shape == (4, 10)
        assert expanded.dtype == torch.bool

    def test_transformer_model_forward_pass_with_variable_sequence_lengths(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that the model correctly handles variable-length sequences."""
        model = TransformerModel(default_transformer_config)
        model.eval()

        batch_size = 4
        max_seq_length = 100
        input_size = default_transformer_config.input_size

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

    def test_transformer_model_forward_pass_single_batch(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test forward pass with a single sample (batch_size=1)."""
        model = TransformerModel(default_transformer_config)
        model.eval()

        seq_length = 30
        input_size = default_transformer_config.input_size

        # Create single sample
        x = torch.randn(1, seq_length, input_size)
        mask = torch.ones(1, seq_length, dtype=torch.bool)

        # Run forward pass
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape
        assert logits.shape == (1,)

    def test_transformer_model_forward_pass_produces_different_outputs(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that different inputs produce different outputs."""
        model = TransformerModel(default_transformer_config)
        model.eval()

        batch_size = 2
        seq_length = 40
        input_size = default_transformer_config.input_size

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

    def test_transformer_model_gradient_flow(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that gradients flow correctly during backpropagation."""
        model = TransformerModel(default_transformer_config)
        model.train()  # Set to training mode

        batch_size = 4
        seq_length = 20
        input_size = default_transformer_config.input_size

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

    def test_transformer_model_rejects_empty_sequences(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test model raises ValueError for sequences with no valid timesteps."""
        model = TransformerModel(default_transformer_config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_size = default_transformer_config.input_size

        # Create input with all-False mask (no valid timesteps)
        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

        # Forward pass should raise ValueError
        with pytest.raises(ValueError, match="Found sequences with no valid timesteps"):
            model(x, mask)

    def test_transformer_model_mask_inversion(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test that mask inversion is handled correctly."""
        model = TransformerModel(default_transformer_config)
        model.eval()

        batch_size = 2
        seq_length = 20
        input_size = default_transformer_config.input_size

        # Create input with partial masking
        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        mask[0, 10:] = False  # First sequence: only first 10 steps valid
        mask[1, 15:] = False  # Second sequence: only first 15 steps valid

        # Run forward pass - should not raise an error
        with torch.no_grad():
            logits = model(x, mask)

        # Verify output shape and validity
        assert logits.shape == (batch_size,)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_transformer_model_rejects_sequences_longer_than_configured_limit(
        self, default_transformer_config: TransformerConfig
    ) -> None:
        """Test model raises ValueError when sequence length exceeds max_seq_length."""
        config = default_transformer_config.model_copy(update={"max_seq_length": 10})
        model = TransformerModel(config)
        model.eval()

        batch_size = 2
        seq_length = config.max_seq_length + 1
        input_size = config.input_size

        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        with pytest.raises(
            ValueError, match="exceeds TransformerConfig.max_seq_length"
        ):
            model(x, mask)


class TestTransformerConfigValidation:
    """Test suite for TransformerConfig validation."""

    def test_transformer_config_requires_positive_input_size(self) -> None:
        """Test that input_size must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=0, nhead=4)

        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=-1, nhead=4)

    def test_transformer_config_requires_positive_nhead(self) -> None:
        """Test that nhead must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=32, nhead=0)

        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=32, nhead=-1)

    def test_transformer_config_requires_positive_num_encoder_layers(self) -> None:
        """Test that num_encoder_layers must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=32, nhead=4, num_encoder_layers=0)

    def test_transformer_config_requires_positive_dim_feedforward(self) -> None:
        """Test that dim_feedforward must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=32, nhead=4, dim_feedforward=0)

    def test_transformer_config_dropout_bounds(self) -> None:
        """Test that dropout must be in [0, 1)."""
        # Valid dropout values
        config1 = TransformerConfig(input_size=32, nhead=4, dropout=0.0)
        assert config1.dropout == 0.0

        config2 = TransformerConfig(input_size=32, nhead=4, dropout=0.99)
        assert config2.dropout == 0.99

        # Invalid: dropout >= 1.0
        with pytest.raises(ValueError):
            TransformerConfig(input_size=32, nhead=4, dropout=1.0)

        # Invalid: dropout < 0
        with pytest.raises(ValueError):
            TransformerConfig(input_size=32, nhead=4, dropout=-0.1)

    def test_transformer_config_input_size_divisible_by_nhead(self) -> None:
        """Test that input_size must be divisible by nhead."""
        # Valid: input_size divisible by nhead
        config1 = TransformerConfig(input_size=32, nhead=4)  # 32 / 4 = 8
        assert config1.input_size == 32
        assert config1.nhead == 4

        config2 = TransformerConfig(input_size=64, nhead=8)  # 64 / 8 = 8
        assert config2.input_size == 64
        assert config2.nhead == 8

        # Invalid: input_size not divisible by nhead
        with pytest.raises(ValueError, match="input_size.*must be divisible by nhead"):
            TransformerConfig(input_size=30, nhead=4)  # 30 / 4 = 7.5

    def test_transformer_config_default_handles_physionet_feature_count(self) -> None:
        """Test that the default configuration works with PhysioNet's 34 features."""
        config = TransformerConfig(input_size=34)
        assert config.nhead == 2
        assert config.input_size == 34

    def test_transformer_config_max_seq_length_must_be_positive(self) -> None:
        """Test that max_seq_length must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            TransformerConfig(input_size=32, max_seq_length=0)

        with pytest.raises(ValueError, match="input_size.*must be divisible by nhead"):
            TransformerConfig(input_size=64, nhead=5)  # 64 / 5 = 12.8

    def test_transformer_config_accepts_valid_parameters(self) -> None:
        """Test that TransformerConfig accepts valid parameter combinations."""
        config = TransformerConfig(
            input_size=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.3,
        )

        assert config.input_size == 128
        assert config.nhead == 8
        assert config.num_encoder_layers == 4
        assert config.dim_feedforward == 512
        assert config.dropout == 0.3
