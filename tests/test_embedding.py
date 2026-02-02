"""
Unit tests for the embedding and indexing pipeline, validating chunk parsing,
Qdrant collection management, configuration handling, and end-to-end execution.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag import embedding


@pytest.fixture
def sample_chunk_file(tmp_path: Path):
    """
    Creates a temporary chunk file containing multiple chunks
    with metadata, including duplicates.

    Purpose:
    - Simulates chunked document input
    - Allows testing of parsing, metadata extraction,
      and deduplication logic
    """

    content = """--- CHUNK 0 ---
<!-- METADATA: {'category': 'faqs'} -->

Hello world

--- CHUNK 1 ---
<!-- METADATA: {'category': 'faqs'} -->

Hello world

--- CHUNK 2 ---
<!-- METADATA: {'category': 'policies'} -->

Another chunk
"""
    path = tmp_path / "chunks.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_chunks_parses_and_deduplicates(sample_chunk_file):
    """
    Verifies that load_chunks:
    - Parses chunk files into Document objects
    - Extracts metadata correctly
    - Deduplicates identical chunk content
    """

    docs = embedding.load_chunks(str(sample_chunk_file))

    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)

    assert len(docs) == 2
    assert docs[0].metadata["chunk_id"] == 0
    assert "category" in docs[0].metadata


def test_load_chunks_skips_empty_chunks(tmp_path):
    """
    Ensures load_chunks skips chunks that contain
    no meaningful content.

    Expected behavior:
    - Empty or whitespace-only chunks are ignored
    - No Document objects are returned
    """

    content = """--- CHUNK 0 ---
<!-- METADATA: {'a': 1} -->

"""
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")

    docs = embedding.load_chunks(str(path))
    assert docs == []


def test_recreate_collection_existing_collection():
    """
    Validates recreate_collection behavior when
    the collection already exists.

    Expected behavior:
    - Existing collection is deleted
    - Collection is recreated from scratch
    """

    client = MagicMock()
    client.collection_exists.return_value = True

    embedding.recreate_collection(client)

    client.delete_collection.assert_called_once_with(
        embedding.COLLECTION_NAME
    )
    client.create_collection.assert_called_once()


def test_recreate_collection_new_collection():
    """
    Ensures recreate_collection handles the case
    where the collection does not yet exist.

    Expected behavior:
    - No deletion occurs
    - Collection is created once
    """

    client = MagicMock()
    client.collection_exists.return_value = False

    embedding.recreate_collection(client)

    client.delete_collection.assert_not_called()
    client.create_collection.assert_called_once()


def test_main_raises_if_qdrant_url_missing(monkeypatch):
    """
    Confirms embedding.main fails fast when
    the QDRANT_URL configuration is missing.

    Expected behavior:
    - Raises ValueError
    - Clear error message is provided
    """

    monkeypatch.setattr(embedding, "QDRANT_URL", None)

    with pytest.raises(ValueError, match="QDRANT_URL is not set"):
        embedding.main()


def test_main_raises_if_chunk_file_missing(monkeypatch, tmp_path):
    """
    Ensures embedding.main raises an error when
    the expected chunk file does not exist.

    Expected behavior:
    - Raises FileNotFoundError
    """

    monkeypatch.setattr(embedding, "QDRANT_URL", "http://fake")
    monkeypatch.setattr(embedding, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(embedding, "CHUNK_FILE", "missing.txt")

    with pytest.raises(FileNotFoundError):
        embedding.main()


@patch("rag.embedding.QdrantVectorStore")
@patch("rag.embedding.FastEmbedSparse")
@patch("rag.embedding.HuggingFaceEmbeddings")
@patch("rag.embedding.QdrantClient")
@patch("rag.embedding.load_chunks")
@patch("rag.embedding.recreate_collection")
def test_main_happy_path(
    mock_recreate,
    mock_load_chunks,
    mock_qdrant_client,
    mock_dense,
    mock_sparse,
    mock_vectorstore,
    monkeypatch,
    tmp_path,
):
    """
    End-to-end happy-path test for embedding.main.

    Verifies:
    - Configuration is read correctly
    - Chunks are loaded
    - Qdrant client and vector store are initialized
    - Collection is recreated
    - Documents are embedded and stored
    """

    monkeypatch.setattr(embedding, "QDRANT_URL", "http://fake")
    monkeypatch.setattr(embedding, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(embedding, "CHUNK_FILE", "chunks.txt")

    chunk_file = tmp_path / "chunks.txt"
    chunk_file.write_text("dummy", encoding="utf-8")

    mock_load_chunks.return_value = [
        Document(page_content="x", metadata={})
    ]

    vs_instance = MagicMock()
    mock_vectorstore.return_value = vs_instance

    embedding.main()

    mock_load_chunks.assert_called_once()
    mock_qdrant_client.assert_called_once()
    mock_recreate.assert_called_once()
    vs_instance.add_documents.assert_called_once()
