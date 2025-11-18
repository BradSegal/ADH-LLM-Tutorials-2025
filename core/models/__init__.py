"""
Model implementations and base abstractions.

This subpackage houses the abstract base class that defines the contract all
sequence models must adhere to, as well as concrete implementations (GRU, LSTM,
Transformer) used in the teaching notebooks.
"""

from core.models.base import BaseSequenceModel
from core.models.lstm import LSTMModel
from core.models.rnn import GRUModel
from core.models.transformer import TransformerModel

__all__ = ["BaseSequenceModel", "GRUModel", "LSTMModel", "TransformerModel"]
