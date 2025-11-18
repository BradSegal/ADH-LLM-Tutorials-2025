"""Embedding generation utilities for the Digital Health Tutorial."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingRequest(BaseModel):
    """Strict schema for embedding generation requests."""

    texts: list[str] = Field(
        ..., description="Ordered collection of text snippets to embed."
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="SentenceTransformer model identifier.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("EmbeddingRequest.texts must contain at least one item.")
        stripped = [text.strip() for text in value]
        if any(text == "" for text in stripped):
            raise ValueError("EmbeddingRequest.texts cannot include empty strings.")
        return stripped


class EmbeddingResponse(BaseModel):
    """Structured response containing embedding vectors."""

    vectors: npt.NDArray[np.float32] = Field(
        ..., description="2D array of generated embedding vectors."
    )
    model_name: str = Field(..., description="Model used to generate the embeddings.")
    dimension: int = Field(..., description="Dimensionality of each embedding vector.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("vectors")
    @classmethod
    def _validate_vector_shape(
        cls, value: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if value.ndim != 2:
            raise ValueError("EmbeddingResponse.vectors must be a 2D array.")
        if value.shape[1] <= 0:
            raise ValueError("EmbeddingResponse.vectors must have non-zero width.")
        return value


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    This function uses lru_cache to ensure that the model is only downloaded
    and loaded into memory once per session, even if called multiple times.
    This is critical for maintaining a smooth student experience in notebooks.

    Parameters
    ----------
    model_name : str, default="all-MiniLM-L6-v2"
        The name of the SentenceTransformer model to load from Hugging Face.
        Default is a lightweight, general-purpose model suitable for educational
        demonstrations.

    Returns
    -------
    SentenceTransformer
        The loaded model instance, ready for encoding text.

    Examples
    --------
    >>> model = get_embedding_model()
    Loading embedding model: all-MiniLM-L6-v2...
    >>> model = get_embedding_model()  # Uses cached version, no output

    Notes
    -----
    This function will download the model from Hugging Face on first use,
    which may take a few seconds depending on network speed. Subsequent
    calls will use the cached version.
    """
    logger.info("Loading embedding model: %s", model_name)
    print(f"Loading embedding model: {model_name}...")
    return SentenceTransformer(model_name)


def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings that satisfy the project contract."""

    model = get_embedding_model(request.model_name)
    embeddings = model.encode(request.texts, show_progress_bar=True)
    vectors = np.asarray(embeddings, dtype=np.float32)
    response = EmbeddingResponse(
        vectors=vectors,
        model_name=request.model_name,
        dimension=vectors.shape[1],
    )
    return response
