"""
Integration tests for package exports.

These tests verify that all models and configurations are properly exported
from their respective package __init__.py files, ensuring that notebooks can
import them using the simplified syntax (e.g., from core.models import LSTMModel).
"""


class TestModelExports:
    """Test suite for core.models package exports."""

    def test_base_sequence_model_export(self) -> None:
        """Test that BaseSequenceModel can be imported from core.models."""
        from core.models import BaseSequenceModel

        # Verify it's an abstract base class
        assert hasattr(BaseSequenceModel, "forward")

    def test_gru_model_export(self) -> None:
        """Test that GRUModel can be imported from core.models."""
        from core.models import GRUModel

        # Verify it's a class
        assert isinstance(GRUModel, type)

    def test_lstm_model_export(self) -> None:
        """Test that LSTMModel can be imported from core.models."""
        from core.models import LSTMModel

        # Verify it's a class
        assert isinstance(LSTMModel, type)

    def test_transformer_model_export(self) -> None:
        """Test that TransformerModel can be imported from core.models."""
        from core.models import TransformerModel

        # Verify it's a class
        assert isinstance(TransformerModel, type)

    def test_all_models_exported(self) -> None:
        """Test that __all__ contains all expected models."""
        from core import models

        expected_exports = {
            "BaseSequenceModel",
            "GRUModel",
            "LSTMModel",
            "TransformerModel",
        }

        assert set(models.__all__) == expected_exports

    def test_all_models_are_base_sequence_model_subclasses(self) -> None:
        """Test that all concrete models inherit from BaseSequenceModel."""
        from core.models import BaseSequenceModel, GRUModel, LSTMModel, TransformerModel

        assert issubclass(GRUModel, BaseSequenceModel)
        assert issubclass(LSTMModel, BaseSequenceModel)
        assert issubclass(TransformerModel, BaseSequenceModel)


class TestConfigExports:
    """Test suite for core.config package exports."""

    def test_data_config_export(self) -> None:
        """Test that DataConfig can be imported from core.config."""
        from core.config import DataConfig

        # Verify it's a Pydantic model class
        assert isinstance(DataConfig, type)

    def test_physionet_config_export(self) -> None:
        """Test that PhysioNetConfig can be imported from core.config."""
        from core.config import PhysioNetConfig

        # Verify it's a Pydantic model class
        assert isinstance(PhysioNetConfig, type)

    def test_train_config_export(self) -> None:
        """Test that TrainConfig can be imported from core.config."""
        from core.config import TrainConfig

        # Verify it's a Pydantic model class
        assert isinstance(TrainConfig, type)

    def test_gru_config_export(self) -> None:
        """Test that GRUConfig can be imported from core.config."""
        from core.config import GRUConfig

        # Verify it's a Pydantic model class
        assert isinstance(GRUConfig, type)

    def test_lstm_config_export(self) -> None:
        """Test that LSTMConfig can be imported from core.config."""
        from core.config import LSTMConfig

        # Verify it's a Pydantic model class
        assert isinstance(LSTMConfig, type)

    def test_transformer_config_export(self) -> None:
        """Test that TransformerConfig can be imported from core.config."""
        from core.config import TransformerConfig

        # Verify it's a Pydantic model class
        assert isinstance(TransformerConfig, type)

    def test_load_data_config_export(self) -> None:
        """Test that load_data_config can be imported from core.config."""
        from core.config import load_data_config

        # Verify it's a callable function
        assert callable(load_data_config)

    def test_all_configs_exported(self) -> None:
        """Test that __all__ contains all expected configs."""
        from core import config

        expected_exports = {
            "DataConfig",
            "GRUConfig",
            "LSTMConfig",
            "PhysioNetConfig",
            "TrainConfig",
            "TransformerConfig",
            "load_data_config",
        }

        assert set(config.__all__) == expected_exports


class TestEndToEndModelInstantiation:
    """Test that models can be instantiated using exported classes."""

    def test_lstm_model_instantiation_via_package_import(self) -> None:
        """Test that LSTMModel can be instantiated using package imports."""
        from core.config import LSTMConfig
        from core.models import LSTMModel

        # Create config
        config = LSTMConfig(input_size=34, hidden_size=64, num_layers=2, dropout=0.1)

        # Instantiate model
        model = LSTMModel(config)

        # Verify model was created successfully
        assert model is not None
        assert model.config == config

    def test_transformer_model_instantiation_via_package_import(self) -> None:
        """Test that TransformerModel can be instantiated using package imports."""
        from core.config import TransformerConfig
        from core.models import TransformerModel

        # Create config (input_size must be divisible by nhead)
        config = TransformerConfig(
            input_size=32,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )

        # Instantiate model
        model = TransformerModel(config)

        # Verify model was created successfully
        assert model is not None
        assert model.config == config

    def test_transformer_model_instantiation_with_default_heads(self) -> None:
        """Test that TransformerModel works with default nhead for 34 features."""
        from core.config import TransformerConfig
        from core.models import TransformerModel

        config = TransformerConfig(input_size=34)  # Should use default nhead=2
        model = TransformerModel(config)

        assert model.config.nhead == 2

    def test_gru_model_instantiation_via_package_import(self) -> None:
        """Test that GRUModel can be instantiated using package imports."""
        from core.config import GRUConfig
        from core.models import GRUModel

        # Create config
        config = GRUConfig(input_size=34, hidden_size=64, num_layers=2, dropout=0.1)

        # Instantiate model
        model = GRUModel(config)

        # Verify model was created successfully
        assert model is not None
        assert model.config == config

    def test_forward_pass_with_exported_models(self) -> None:
        """Test that all models can perform forward passes using exported classes."""
        import torch

        from core.config import GRUConfig, LSTMConfig, TransformerConfig
        from core.models import GRUModel, LSTMModel, TransformerModel

        # Create dummy data
        batch_size = 4
        seq_length = 20
        input_size = 32

        x = torch.randn(batch_size, seq_length, input_size)
        mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Test GRU
        gru_config = GRUConfig(input_size=input_size)
        gru_model = GRUModel(gru_config)
        gru_model.eval()
        with torch.no_grad():
            gru_output = gru_model(x, mask)
        assert gru_output.shape == (batch_size,)

        # Test LSTM
        lstm_config = LSTMConfig(input_size=input_size)
        lstm_model = LSTMModel(lstm_config)
        lstm_model.eval()
        with torch.no_grad():
            lstm_output = lstm_model(x, mask)
        assert lstm_output.shape == (batch_size,)

        # Test Transformer (input_size must be divisible by nhead)
        transformer_config = TransformerConfig(input_size=input_size, nhead=4)
        transformer_model = TransformerModel(transformer_config)
        transformer_model.eval()
        with torch.no_grad():
            transformer_output = transformer_model(x, mask)
        assert transformer_output.shape == (batch_size,)
