"""
GRU (Gated Recurrent Unit) model for sequence prediction.

This module implements a baseline GRU model that conforms to the
BaseSequenceModel contract. It provides a simple but effective approach for
modeling time-series medical data, serving as the foundational model for the
sepsis prediction teaching module.
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor

from core.config.models import GRUConfig
from core.models.base import BaseSequenceModel

logger = logging.getLogger(__name__)


class GRUModel(BaseSequenceModel):
    """
    GRU-based sequence model for binary classification.

    This model uses a multi-layer Gated Recurrent Unit (GRU) to process
    variable-length time-series data and produces a single prediction per
    sequence. It extracts the final hidden state from each sequence (using the
    attention mask to identify the last valid timestep) and passes it through
    a linear classifier.

    The model implements the BaseSequenceModel contract, making it compatible
    with the model-agnostic Trainer.

    Parameters
    ----------
    config : GRUConfig
        Validated configuration object containing model hyperparameters
        (input_size, hidden_size, num_layers, dropout).

    Attributes
    ----------
    config : GRUConfig
        The configuration used to build this model.
    gru : nn.GRU
        The recurrent layer stack.
    classifier : nn.Linear
        Linear layer mapping final hidden state to a single logit.

    Examples
    --------
    >>> from core.config.models import GRUConfig
    >>> from core.models.rnn import GRUModel
    >>> import torch
    >>>
    >>> config = GRUConfig(input_size=34, hidden_size=64, num_layers=2, dropout=0.2)
    >>> model = GRUModel(config)
    >>> x = torch.randn(8, 50, 34)  # [batch, seq_len, features]
    >>> mask = torch.ones(8, 50, dtype=torch.bool)
    >>> logits = model(x, mask)
    >>> print(logits.shape)  # torch.Size([8])
    """

    def __init__(self, config: GRUConfig) -> None:
        """
        Initialize the GRU model with the provided configuration.

        Parameters
        ----------
        config : GRUConfig
            Validated configuration containing input_size, hidden_size,
            num_layers, and dropout.
        """
        super().__init__()
        self.config = config

        # Dropout is only applied between layers when num_layers > 1
        dropout_value = config.dropout if config.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=dropout_value,
        )

        # Linear layer to map the last hidden state to a single logit for
        # binary classification
        self.classifier = nn.Linear(config.hidden_size, 1)

        logger.info(
            "Initialized GRUModel with input_size=%d, hidden_size=%d, "
            "num_layers=%d, dropout=%.2f",
            config.input_size,
            config.hidden_size,
            config.num_layers,
            dropout_value,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for the GRU model.

        This method processes a batch of variable-length sequences through the
        GRU layers and extracts the hidden state at the last valid timestep
        for each sequence (determined by the attention mask). The final hidden
        state is then passed through a linear classifier to produce a logit.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [Batch, SequenceLength, Features].
            Contains the time-series features for each patient record.
        mask : Tensor
            Boolean attention mask of shape [Batch, SequenceLength].
            True indicates valid time steps, False indicates padding.

        Returns
        -------
        Tensor
            Output logits of shape [Batch]. Each value is a raw logit
            (not probability) for binary classification.

        Notes
        -----
        This implementation uses a simple approach to handle variable-length
        sequences: it uses the attention mask to compute sequence lengths and
        indexes the GRU output at the last valid timestep. More advanced
        implementations could use packed sequences for efficiency, but this
        approach prioritizes clarity and pedagogical value.
        """
        # Pass input through GRU layers
        # gru_out shape: [Batch, SequenceLength, HiddenSize]
        gru_out, _ = self.gru(x)

        # Calculate sequence lengths from the mask
        # For each sequence, find the index of the last valid (True) timestep
        # mask.sum(dim=1) gives the count of valid timesteps per sequence
        # Subtract 1 to convert count to zero-based index
        seq_lengths = mask.sum(dim=1) - 1

        # Validate that all sequences have at least one valid timestep
        if (seq_lengths < 0).any():
            raise ValueError(
                "Found sequences with no valid timesteps (all False in mask). "
                "All sequences must have at least one valid timestep."
            )

        # Extract the hidden state at the last valid timestep for each sequence
        # Use advanced indexing: for each item in the batch, get the output
        # at its corresponding sequence length index
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden_states = gru_out[batch_indices, seq_lengths]

        # Pass through classifier to get logits
        # Shape: [Batch, 1]
        logits: Tensor = self.classifier(last_hidden_states)

        # Squeeze to get shape [Batch] as required by the contract
        return logits.squeeze(-1)

    def prepare_mask(self, mask: Tensor, batch_size: int) -> Tensor:
        return super().prepare_mask(mask, batch_size)
