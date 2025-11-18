"""
Pytest fixtures for training component tests.

This module provides reusable fixtures including mock models, mock data loaders,
and training configurations for testing the Trainer class.
"""

from typing import TypeAlias, cast

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from core.config.models import TrainConfig
from core.models.base import BaseSequenceModel

Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


class MockSequenceModel(BaseSequenceModel):
    """
    A simple mock model for testing the Trainer.

    This model implements the BaseSequenceModel contract with a minimal
    architecture suitable for fast unit tests. It performs a simple linear
    transformation on the input.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 8) -> None:
        """
        Initialize the mock model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Size of the hidden layer.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass implementing the BaseSequenceModel contract.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [Batch, SequenceLength, Features].
        mask : Tensor
            Boolean mask of shape [Batch, SequenceLength].

        Returns
        -------
        Tensor
            Output logits of shape [Batch, 1].
        """
        # Simple forward pass: take mean over sequence dimension, ignoring mask
        # In a real model, mask would be used properly
        x_mean = x.mean(dim=1)  # [Batch, Features]
        hidden = self.linear(x_mean)  # [Batch, HiddenSize]
        logits: Tensor = self.output(hidden)  # [Batch, 1]
        return logits


@pytest.fixture
def mock_model() -> MockSequenceModel:
    """
    Create a mock sequence model for testing.

    Returns
    -------
    MockSequenceModel
        A simple model that implements the BaseSequenceModel interface.
    """
    return MockSequenceModel(input_size=10, hidden_size=8)


@pytest.fixture
def mock_train_loader() -> DataLoader[Batch]:
    """
    Create a mock training DataLoader with synthetic data.

    Returns
    -------
    DataLoader
        A DataLoader yielding batches of (features, mask, labels).
    """
    # Create synthetic data: 32 samples, 5 time steps, 10 features
    batch_size = 8
    num_samples = 32
    seq_length = 5
    num_features = 10

    features = torch.randn(num_samples, seq_length, num_features)
    masks = torch.ones(num_samples, seq_length, dtype=torch.bool)
    labels = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)

    dataset = TensorDataset(features, masks, labels)
    typed_dataset = cast(Dataset[Batch], dataset)
    return DataLoader(typed_dataset, batch_size=batch_size, shuffle=True)


@pytest.fixture
def mock_val_loader() -> DataLoader[Batch]:
    """
    Create a mock validation DataLoader with synthetic data.

    Returns
    -------
    DataLoader
        A DataLoader yielding batches of (features, mask, labels).
    """
    # Create synthetic validation data: 16 samples
    batch_size = 8
    num_samples = 16
    seq_length = 5
    num_features = 10

    features = torch.randn(num_samples, seq_length, num_features)
    masks = torch.ones(num_samples, seq_length, dtype=torch.bool)
    labels = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)

    dataset = TensorDataset(features, masks, labels)
    typed_dataset = cast(Dataset[Batch], dataset)
    return DataLoader(typed_dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def train_config() -> TrainConfig:
    """
    Create a standard training configuration for tests.

    Returns
    -------
    TrainConfig
        A validated config with reasonable defaults for testing.
    """
    return TrainConfig(
        learning_rate=0.001,
        epochs=2,
        batch_size=8,
        device="cpu",  # Always use CPU for unit tests
    )
