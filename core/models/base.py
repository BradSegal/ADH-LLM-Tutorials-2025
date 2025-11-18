"""
Abstract base class for all sequence models.

This module defines the strict contract that all sequence models (GRU, LSTM,
Transformer) must implement. This enables the Trainer to be completely
model-agnostic, working with any model that inherits from BaseSequenceModel.
"""

import abc

import torch.nn as nn
from torch import Tensor


class BaseSequenceModel(nn.Module, abc.ABC):
    """
    Abstract base class for all sequence models.

    This class enforces a common interface for the forward pass, which is
    essential for the model-agnostic Trainer. All concrete sequence models
    (GRU, LSTM, Transformer) must inherit from this class and implement the
    abstract forward method.

    The forward method signature is the non-negotiable contract that enables
    the Trainer to work with any model without knowing its internal
    implementation details.
    """

    @abc.abstractmethod
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Define the forward pass for the sequence model.

        This is the core contract method that all sequence models must
        implement. The Trainer will call this method during training and
        evaluation.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape [Batch, SequenceLength, Features].
            Contains the time-series features for each patient record.
        mask : Tensor
            The boolean attention mask of shape [Batch, SequenceLength].
            True indicates valid time steps, False indicates padding that
            should be ignored.

        Returns
        -------
        Tensor
            The output logits. Shape depends on the task:
            - For binary classification with one prediction per sequence:
              [Batch]
            - For per-timestep predictions:
              [Batch, SequenceLength]

        Raises
        ------
        NotImplementedError
            If a subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the forward() method. "
            "This is required by the BaseSequenceModel contract."
        )

    def prepare_mask(self, mask: Tensor, batch_size: int) -> Tensor:
        """Ensure the provided mask matches the supplied ``batch_size``.

        Captum and other attribution methods often construct batched inputs by
        interpolating between a baseline and the real example. These batches can
        have different sizes than the original mask, so models can override this
        hook to reshape or broadcast as needed. The default implementation simply
        expands along the batch dimension.
        """

        if mask.size(0) == batch_size:
            return mask
        return mask.expand(batch_size, mask.size(1))
