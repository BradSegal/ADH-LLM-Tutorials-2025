"""
LLM utilities for the Digital Health Tutorial.

This module provides embedding generation and other LLM-related
functionality for the tutorial notebooks.
"""

from core.llm.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    generate_embeddings,
    get_embedding_model,
)
from core.llm.local_llm import OllamaSettings, setup_ollama_llm

__all__ = [
    "EmbeddingRequest",
    "EmbeddingResponse",
    "generate_embeddings",
    "get_embedding_model",
    "OllamaSettings",
    "setup_ollama_llm",
]
