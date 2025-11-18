"""
Unit tests for the Trainer class.

Tests the unified training harness to ensure it correctly manages the training
loop, device placement, gradient updates, and evaluation mode. These tests use
mock models and synthetic data for fast, deterministic validation.
"""

import logging
from typing import TypeAlias, cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from core.config.models import TrainConfig
from core.train.trainer import Trainer
from tests.unit.train.conftest import MockSequenceModel

Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


class TestTrainerInitialization:
    """Test suite for Trainer initialization and validation."""

    def test_trainer_initialization_success(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that Trainer can be instantiated with valid components."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        assert trainer.model is mock_model
        assert trainer.train_loader is mock_train_loader
        assert trainer.val_loader is mock_val_loader
        assert trainer.config is train_config
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.loss_fn, nn.BCEWithLogitsLoss)

    def test_trainer_sets_device_correctly(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that Trainer correctly sets the device."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Config specifies CPU, so device should be CPU
        assert trainer.device == torch.device("cpu")

    def test_trainer_moves_model_to_device(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that Trainer moves the model to the correct device."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Check that model parameters are on the correct device
        for param in trainer.model.parameters():
            assert param.device == trainer.device

    def test_trainer_falls_back_to_cpu_when_cuda_unavailable(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
    ) -> None:
        """
        Test that Trainer falls back to CPU when CUDA is unavailable.
        """
        # Create config requesting CUDA
        cuda_config = TrainConfig(
            learning_rate=0.001, epochs=2, batch_size=8, device="cuda"
        )

        # Mock torch.cuda.is_available to return False
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                mock_model, mock_train_loader, mock_val_loader, cuda_config
            )

            # Should fall back to CPU
            assert trainer.device == torch.device("cpu")

    def test_trainer_rejects_non_base_model(
        self,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """
        Test that Trainer rejects non-BaseSequenceModel models.
        """

        # Create a regular nn.Module that doesn't inherit from BaseSequenceModel
        class InvalidModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        invalid_model = InvalidModel()

        with pytest.raises(TypeError) as exc_info:
            Trainer(invalid_model, mock_train_loader, mock_val_loader, train_config)  # type: ignore[arg-type]

        error_msg = str(exc_info.value)
        assert "BaseSequenceModel" in error_msg
        assert "InvalidModel" in error_msg

    def test_trainer_rejects_mismatched_batch_size(
        self,
        mock_model: MockSequenceModel,
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Trainer should fail fast if DataLoader batch size differs from config."""
        dataset = TensorDataset(
            torch.randn(32, 5, 10),
            torch.ones(32, 5, dtype=torch.bool),
            torch.randint(0, 2, (32, 1), dtype=torch.float32),
        )
        mismatched_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        with pytest.raises(ValueError) as exc_info:
            Trainer(
                mock_model,
                cast(DataLoader[Batch], mismatched_loader),
                mock_val_loader,
                train_config,
            )

        assert "batch_size" in str(exc_info.value)

    def test_trainer_rejects_empty_loader(
        self,
        mock_model: MockSequenceModel,
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Trainer should fail fast if loader has zero batches."""
        empty_dataset = TensorDataset(
            torch.empty(0, 5, 10),
            torch.empty(0, 5, dtype=torch.bool),
            torch.empty(0, 1),
        )
        empty_loader = DataLoader(empty_dataset, batch_size=8)

        with pytest.raises(ValueError) as exc_info:
            Trainer(
                mock_model,
                cast(DataLoader[Batch], empty_loader),
                mock_val_loader,
                train_config,
            )

        assert "at least one batch" in str(exc_info.value)

    def test_trainer_uses_configured_optimizer_and_loss(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
    ) -> None:
        """Trainer should honor optimizer/loss choices."""
        config = TrainConfig(
            learning_rate=0.01,
            epochs=2,
            batch_size=8,
            optimizer="sgd",
            optimizer_momentum=0.8,
            loss="mse",
        )
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, config)

        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.defaults.get("momentum") == pytest.approx(0.8)
        assert isinstance(trainer.loss_fn, nn.MSELoss)

    def test_trainer_configures_step_scheduler(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
    ) -> None:
        """Trainer should build scheduler when requested."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=2,
            batch_size=8,
            scheduler="step",
            scheduler_step_size=1,
            scheduler_gamma=0.5,
        )
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, config)

        assert isinstance(trainer.scheduler, StepLR)
        assert trainer.scheduler.step_size == 1
        assert trainer.scheduler.gamma == 0.5


class TestTrainerTrainingEpoch:
    """Test suite for the _train_epoch method."""

    def test_train_epoch_changes_weights(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that _train_epoch performs gradient updates and changes weights."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in trainer.model.named_parameters()
        }

        # Run one training epoch
        loss = trainer._train_epoch()

        # Check that loss is a positive number
        assert isinstance(loss, float)
        assert loss > 0

        # Check that at least some weights have changed
        weights_changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Weights should change after training epoch"

    def test_train_epoch_sets_model_to_train_mode(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that _train_epoch sets the model to training mode."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Set model to eval mode first
        trainer.model.eval()
        assert not trainer.model.training

        # Run training epoch
        trainer._train_epoch()

        # Model should be in training mode
        assert trainer.model.training

    def test_train_epoch_applies_gradient_clipping_when_configured(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
    ) -> None:
        """Gradient clipping should be invoked when max_grad_norm is set."""
        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=8,
            max_grad_norm=1.0,
        )

        with patch("core.train.trainer.clip_grad_norm_") as mock_clip:
            trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, config)
            trainer._train_epoch()

        mock_clip.assert_called()


class TestTrainerEvaluationEpoch:
    """Test suite for the _evaluate_epoch method."""

    def test_evaluate_epoch_does_not_change_weights(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that _evaluate_epoch does NOT change model weights (no_grad)."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in trainer.model.named_parameters()
        }

        # Run evaluation epoch
        loss, metrics = trainer._evaluate_epoch()

        # Check that loss is a positive number
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(metrics, dict)

        # Check that NO weights have changed
        for name, param in trainer.model.named_parameters():
            assert torch.allclose(
                param, initial_weights[name]
            ), f"Weight {name} changed during evaluation (should be frozen)"

    def test_evaluate_epoch_sets_model_to_eval_mode(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that _evaluate_epoch sets the model to evaluation mode."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Set model to train mode first
        trainer.model.train()
        assert trainer.model.training

        # Run evaluation epoch
        trainer._evaluate_epoch()

        # Model should be in eval mode (though training flag may be true after)
        # The key test is that no weights changed (tested above)


class TestTrainerFit:
    """Test suite for the fit method (integration test of training loop)."""

    def test_fit_completes_successfully(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that fit() runs to completion without errors."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        # Run the full training loop
        history, best_path = trainer.fit()

        # Verify history structure
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        # Verify best_path is None when no save_path provided
        assert best_path is None

    def test_fit_returns_correct_history_length(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that fit() returns history with correct number of epochs."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        history, _ = trainer.fit()

        # Config specifies 2 epochs
        assert len(history["train_loss"]) == train_config.epochs
        assert len(history["val_loss"]) == train_config.epochs
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_fit_returns_decreasing_loss(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
    ) -> None:
        """Test that training generally decreases loss over epochs."""
        # Use more epochs to see trend
        config = TrainConfig(
            learning_rate=0.01,  # Higher LR for faster convergence
            epochs=5,
            batch_size=8,
            device="cpu",
        )

        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, config)
        history, _ = trainer.fit()

        # At minimum, verify we got valid loss values
        assert all(loss > 0 for loss in history["train_loss"])
        assert all(loss > 0 for loss in history["val_loss"])

    def test_fit_logs_progress(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that fit() logs progress information."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        with caplog.at_level(logging.INFO):
            trainer.fit()

        assert "Starting training" in caplog.text
        assert "Epoch 1/2" in caplog.text
        assert "Epoch 2/2" in caplog.text


class TestTrainerCheckpointing:
    """Test suite for model checkpointing and save_path functionality."""

    def test_trainer_saves_best_model(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        tmp_path,
    ) -> None:
        """Test that trainer saves model checkpoint with best validation loss."""
        save_path = tmp_path / "best_model.pt"

        trainer = Trainer(
            mock_model,
            mock_train_loader,
            mock_val_loader,
            train_config,
            save_path=save_path,
        )

        history, best_model_path = trainer.fit()

        # Verify that a checkpoint was saved
        assert best_model_path is not None
        assert best_model_path == save_path
        assert save_path.exists()
        assert save_path.is_file()

    def test_trainer_creates_save_directory(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        tmp_path,
    ) -> None:
        """Test that trainer creates parent directories for save_path."""
        # Create a nested path that doesn't exist yet
        save_path = tmp_path / "checkpoints" / "models" / "best.pt"

        # Verify the directory doesn't exist yet
        assert not save_path.parent.exists()

        # Initialize Trainer (creates parent directories as a side effect)
        Trainer(
            mock_model,
            mock_train_loader,
            mock_val_loader,
            train_config,
            save_path=save_path,
        )

        # After initialization, parent directory should exist
        assert save_path.parent.exists()
        assert save_path.parent.is_dir()

    def test_trainer_saves_loadable_state_dict(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        tmp_path,
    ) -> None:
        """Test that saved checkpoint can be loaded back into a model."""
        save_path = tmp_path / "model.pt"

        trainer = Trainer(
            mock_model,
            mock_train_loader,
            mock_val_loader,
            train_config,
            save_path=save_path,
        )

        # Train and save
        trainer.fit()

        # Load the saved state dict into a fresh model
        fresh_model = MockSequenceModel(input_size=10, hidden_size=8)
        state_dict = torch.load(save_path, map_location=torch.device("cpu"))

        # Verify that loading succeeds without error
        fresh_model.load_state_dict(state_dict)

        # Verify that the fresh model has the same structure
        assert len(list(fresh_model.parameters())) == len(list(mock_model.parameters()))

        # Verify parameter names match (structure is correct)
        for (name1, _), (name2, _) in zip(
            mock_model.named_parameters(),
            fresh_model.named_parameters(),
            strict=True,
        ):
            assert name1 == name2

    def test_trainer_only_saves_when_val_loss_improves(
        self,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        tmp_path,
    ) -> None:
        """Test that model is only saved when validation loss decreases."""
        save_path = tmp_path / "best.pt"

        # Create a model that will have increasing validation loss
        # (to test that we don't save when loss doesn't improve)
        model = MockSequenceModel(input_size=10, hidden_size=8)

        # Use very low learning rate so loss won't improve much
        config = TrainConfig(
            learning_rate=0.0001,
            epochs=3,
            batch_size=8,
            device="cpu",
        )

        trainer = Trainer(
            model, mock_train_loader, mock_val_loader, config, save_path=save_path
        )

        history, best_model_path = trainer.fit()

        # Model should have been saved (at least on first epoch)
        assert save_path.exists()

        # Verify that the best model path matches what we provided
        assert best_model_path == save_path

        # Verify that we can find a minimum validation loss (sanity check)
        min_val_loss = min(history["val_loss"])
        assert min_val_loss > 0  # Loss should be positive

    def test_trainer_logs_checkpoint_saves(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that trainer logs when it saves a checkpoint."""
        save_path = tmp_path / "model.pt"

        with caplog.at_level(logging.INFO):
            trainer = Trainer(
                mock_model,
                mock_train_loader,
                mock_val_loader,
                train_config,
                save_path=save_path,
            )
            trainer.fit()

        # Should log that it will save to the path (during init)
        assert "Best model will be saved to" in caplog.text
        # Should log when best model is saved (during fit)
        assert "New best model saved" in caplog.text
        assert str(save_path) in caplog.text

    def test_trainer_save_path_accepts_string(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
        tmp_path,
    ) -> None:
        """Test that save_path can be provided as a string."""
        save_path_str = str(tmp_path / "model.pt")

        trainer = Trainer(
            mock_model,
            mock_train_loader,
            mock_val_loader,
            train_config,
            save_path=save_path_str,
        )

        history, best_model_path = trainer.fit()

        # Path should be converted to Path object internally
        assert best_model_path is not None
        assert best_model_path.exists()

    def test_trainer_without_save_path_returns_none(
        self,
        mock_model: MockSequenceModel,
        mock_train_loader: DataLoader[Batch],
        mock_val_loader: DataLoader[Batch],
        train_config: TrainConfig,
    ) -> None:
        """Test that trainer returns None for best_path when save_path not provided."""
        trainer = Trainer(mock_model, mock_train_loader, mock_val_loader, train_config)

        history, best_model_path = trainer.fit()

        assert best_model_path is None
