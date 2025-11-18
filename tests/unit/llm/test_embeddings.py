"""
Unit tests for the embeddings module.

Tests the embedding generation utilities with mocked SentenceTransformer
to avoid downloading models during fast unit tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.llm.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    generate_embeddings,
    get_embedding_model,
)


class TestEmbeddingRequest:
    """Validation suite for EmbeddingRequest."""

    def test_texts_are_stripped(self) -> None:
        request = EmbeddingRequest(texts=["  sample  "])
        assert request.texts == ["sample"]

    def test_empty_text_list_raises(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingRequest(texts=[])

    def test_empty_text_entry_raises(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingRequest(texts=["", "Cardiology"])


class TestEmbeddingResponse:
    """Validation suite for EmbeddingResponse."""

    def test_vectors_must_be_2d(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingResponse(
                vectors=np.array([0.1, 0.2], dtype=np.float32),
                model_name="test",
                dimension=2,
            )

    def test_vectors_must_have_positive_width(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingResponse(
                vectors=np.empty((1, 0), dtype=np.float32),
                model_name="test",
                dimension=0,
            )


class TestGetEmbeddingModel:
    """Test suite for get_embedding_model function."""

    @patch("core.llm.embeddings.SentenceTransformer")
    def test_get_embedding_model_calls_sentence_transformer(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that get_embedding_model calls SentenceTransformer constructor."""
        # Clear the cache before the test to ensure fresh call
        get_embedding_model.cache_clear()

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        result = get_embedding_model("test-model")

        mock_sentence_transformer.assert_called_once_with("test-model")
        assert result is mock_model

    @patch("core.llm.embeddings.SentenceTransformer")
    def test_get_embedding_model_uses_default_model(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that get_embedding_model uses default model when not specified."""
        # Clear the cache before the test
        get_embedding_model.cache_clear()

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        result = get_embedding_model()

        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert result is mock_model

    @patch("core.llm.embeddings.SentenceTransformer")
    def test_get_embedding_model_is_cached(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that get_embedding_model caches the model and only loads once."""
        # Clear the cache before the test
        get_embedding_model.cache_clear()

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Call twice with the same model name
        result1 = get_embedding_model("test-model")
        result2 = get_embedding_model("test-model")

        # SentenceTransformer should only be called once due to caching
        assert mock_sentence_transformer.call_count == 1
        assert result1 is result2
        assert result1 is mock_model


class TestGenerateEmbeddings:
    """Test suite for generate_embeddings function."""

    @patch("core.llm.embeddings.get_embedding_model")
    def test_generate_embeddings_calls_model_encode(
        self, mock_get_model: MagicMock
    ) -> None:
        """Test that generate_embeddings calls the model's encode method."""
        # Create a mock model
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        request = EmbeddingRequest(texts=["Heart attack", "Myocardial infarction"])
        response = generate_embeddings(request)

        # Verify get_embedding_model was called with default model
        mock_get_model.assert_called_once_with(request.model_name)

        # Verify the model's encode method was called correctly
        mock_model.encode.assert_called_once_with(request.texts, show_progress_bar=True)

        np.testing.assert_array_equal(
            response.vectors, mock_embeddings.astype(np.float32)
        )
        assert response.dimension == 3
        assert response.model_name == request.model_name

    @patch("core.llm.embeddings.get_embedding_model")
    def test_generate_embeddings_with_custom_model(
        self, mock_get_model: MagicMock
    ) -> None:
        """Test that generate_embeddings accepts a custom model name."""
        # Create a mock model
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        custom_model = "custom-embedding-model"
        request = EmbeddingRequest(texts=["Test text"], model_name=custom_model)
        response = generate_embeddings(request)

        # Verify get_embedding_model was called with custom model
        mock_get_model.assert_called_once_with(custom_model)

        # Verify the result shape
        assert response.vectors.shape == (1, 2)
        assert response.dimension == 2
        assert response.model_name == custom_model

    @patch("core.llm.embeddings.get_embedding_model")
    def test_generate_embeddings_enforces_float32(
        self, mock_get_model: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(2, 4)  # float64 by default
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        request = EmbeddingRequest(texts=["Text 1", "Text 2"])
        response = generate_embeddings(request)

        assert response.vectors.dtype == np.float32
