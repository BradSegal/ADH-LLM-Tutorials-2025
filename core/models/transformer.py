"""
Transformer encoder model for sequence prediction.

This module implements a Transformer encoder model that conforms to the
BaseSequenceModel contract. It uses self-attention mechanisms to capture
complex temporal dependencies in time-series medical data, providing a
state-of-the-art approach for sepsis prediction.
"""

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

from core.config.models import TransformerConfig
from core.models.base import BaseSequenceModel

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for Transformer models.

    Since the Transformer architecture does not have inherent sequential
    ordering (unlike RNNs), positional encodings are added to the input
    embeddings to inject information about the position of each timestep in
    the sequence. This implementation uses sinusoidal positional encodings
    as described in "Attention is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        The dimension of the model (embedding size).
    dropout : float
        Dropout probability to apply after adding positional encodings.
    max_len : int, optional
        Maximum sequence length to support (default: 5000).

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer applied after positional encoding.
    pe : Tensor
        Precomputed positional encoding tensor of shape [1, max_len, d_model].
        Registered as a buffer (not a trainable parameter).

    Examples
    --------
    >>> import torch
    >>> from core.models.transformer import PositionalEncoding
    >>>
    >>> pos_encoder = PositionalEncoding(d_model=512, dropout=0.1)
    >>> x = torch.randn(32, 100, 512)  # [batch, seq_len, d_model]
    >>> x_with_pos = pos_encoder(x)
    >>> print(x_with_pos.shape)  # torch.Size([32, 100, 512])
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Initialize the positional encoding module.

        Parameters
        ----------
        d_model : int
            The dimension of the model (embedding size).
        dropout : float
            Dropout probability to apply after adding positional encodings.
        max_len : int, optional
            Maximum sequence length to support (default: 5000).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape [max_len, d_model] for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: shape becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a trainable parameter)
        self.register_buffer("pe", pe)
        # Type hint for mypy - pe is registered as a buffer and accessed as an attribute
        self.pe: Tensor

        logger.debug(
            "Initialized PositionalEncoding with d_model=%d, dropout=%.2f, max_len=%d",
            d_model,
            dropout,
            max_len,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [Batch, SequenceLength, Features].

        Returns
        -------
        Tensor
            Input with added positional encodings, same shape as input.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                "Input sequence length "
                f"({seq_len}) exceeds configured positional encoding capacity "
                f"({self.pe.size(1)}). Increase TransformerConfig.max_seq_length "
                "to support longer sequences."
            )

        # Add positional encoding to input
        # pe is [1, max_len, d_model], we slice it to match sequence length
        x = x + self.pe[:, :seq_len, :]
        result: Tensor = self.dropout(x)
        return result


class TransformerModel(BaseSequenceModel):
    """
    Transformer encoder model for binary sequence classification.

    This model uses multi-head self-attention to process variable-length
    time-series data and produces a single prediction per sequence. The
    architecture consists of:
    1. Input projection layer
    2. Positional encoding
    3. Stack of Transformer encoder layers
    4. Mean pooling over valid timesteps
    5. Linear classifier

    The model implements the BaseSequenceModel contract, making it compatible
    with the model-agnostic Trainer.

    Parameters
    ----------
    config : TransformerConfig
        Validated configuration object containing model hyperparameters
        (input_size, nhead, num_encoder_layers, dim_feedforward, dropout).

    Attributes
    ----------
    config : TransformerConfig
        The configuration used to build this model.
    input_projection : nn.Linear
        Linear layer to project input features to the model dimension.
    pos_encoder : PositionalEncoding
        Positional encoding module.
    transformer_encoder : nn.TransformerEncoder
        Stack of Transformer encoder layers.
    classifier : nn.Linear
        Linear layer mapping pooled representation to a single logit.

    Examples
    --------
    >>> from core.config.models import TransformerConfig
    >>> from core.models.transformer import TransformerModel
    >>> import torch
    >>>
    >>> config = TransformerConfig(input_size=32, nhead=4, num_encoder_layers=2)
    >>> model = TransformerModel(config)
    >>> x = torch.randn(8, 50, 32)  # [batch, seq_len, features]
    >>> mask = torch.ones(8, 50, dtype=torch.bool)
    >>> logits = model(x, mask)
    >>> print(logits.shape)  # torch.Size([8])
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Transformer model with the provided configuration.

        Parameters
        ----------
        config : TransformerConfig
            Validated configuration containing input_size, nhead,
            num_encoder_layers, dim_feedforward, and dropout.
        """
        super().__init__()
        self.config = config

        # Input projection layer
        # In NLP, this would typically be an embedding layer, but for
        # continuous medical data, we use a linear projection
        self.input_projection = nn.Linear(config.input_size, config.input_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.input_size,
            dropout=config.dropout,
            max_len=config.max_seq_length,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_size,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )
        if hasattr(self.transformer_encoder, "mask_check"):
            self.transformer_encoder.mask_check = False

        # Classifier to map the pooled representation to a single logit
        self.classifier = nn.Linear(config.input_size, 1)

        logger.info(
            "Initialized TransformerModel with input_size=%d, nhead=%d, "
            "num_encoder_layers=%d, dim_feedforward=%d, dropout=%.2f",
            config.input_size,
            config.nhead,
            config.num_encoder_layers,
            config.dim_feedforward,
            config.dropout,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for the Transformer model.

        This method processes a batch of variable-length sequences through the
        Transformer encoder layers. It uses mean pooling over valid timesteps
        to create a fixed-size representation, which is then passed through a
        linear classifier to produce a logit.

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
        PyTorch's TransformerEncoder expects a padding mask where True
        indicates positions to ignore. Our mask convention is the opposite
        (True = valid), so we invert it before passing to the encoder.

        Mean pooling is used instead of taking the last timestep to better
        leverage the Transformer's ability to attend to all positions
        simultaneously. This provides a more robust representation of the
        entire sequence.
        """
        # PyTorch's TransformerEncoder expects a padding mask where True
        # indicates positions to ignore. Our mask convention is opposite.
        padding_mask = ~mask

        # Project input and add positional encoding
        # Scale by sqrt(d_model) as in "Attention is All You Need"
        seq_len = x.size(1)
        max_len = self.config.max_seq_length
        if seq_len > max_len:
            raise ValueError(
                "Input sequence length "
                f"({seq_len}) exceeds TransformerConfig.max_seq_length ({max_len}). "
                "Increase the configuration value to process longer sequences."
            )

        x = self.input_projection(x) * math.sqrt(self.config.input_size)
        x = self.pos_encoder(x)

        # Pass through Transformer encoder
        # Shape: [Batch, SequenceLength, InputSize]
        encoded_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over valid timesteps
        # Expand mask to match encoded_output dimensions for broadcasting
        # Shape: [Batch, SequenceLength, 1] -> [Batch, SequenceLength, InputSize]
        expanded_mask = mask.unsqueeze(-1).expand_as(encoded_output)

        # Zero out padded positions and sum over sequence dimension
        summed_output = torch.sum(encoded_output * expanded_mask, dim=1)

        # Count valid timesteps for each sequence
        valid_counts = mask.sum(dim=1, keepdim=True)

        # Validate that all sequences have at least one valid timestep
        if (valid_counts == 0).any():
            raise ValueError(
                "Found sequences with no valid timesteps (all False in mask). "
                "All sequences must have at least one valid timestep."
            )

        # Average over valid timesteps
        # Shape: [Batch, InputSize]
        pooled_output = summed_output / valid_counts

        # Pass through classifier to get logits
        # Shape: [Batch, 1]
        logits: Tensor = self.classifier(pooled_output)

        # Squeeze to get shape [Batch] as required by the contract
        return logits.squeeze(-1)

    def prepare_mask(self, mask: Tensor, batch_size: int) -> Tensor:
        """Ensure padding masks align with Captum's interpolated batches."""

        if mask.size(0) == batch_size:
            return mask
        # Expand along batch dimension while preserving sequence length.
        expanded = mask.expand(batch_size, mask.size(1)).clone()
        return expanded
