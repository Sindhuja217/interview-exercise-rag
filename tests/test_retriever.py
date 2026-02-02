"""
Unit tests for the retriever module, covering hybrid retrieval,
cross-encoder reranking, and retrieval quality evaluation logic
with external dependencies fully mocked.
"""
import os
import importlib
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


@pytest.fixture
def retriever_module(monkeypatch):
    """
    Fixture to import the retriever module with all heavyweight
    external dependencies mocked out.

    Why this exists:
    - Prevents real network calls (Qdrant)
    - Avoids loading actual sentence-transformer models
    - Ensures deterministic and fast unit tests
    """

    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "fake-key")

    with patch("sentence_transformers.CrossEncoder") as mock_ce:
        mock_ce.return_value.predict = MagicMock(return_value=[1.0])

        import rag.retriever
        importlib.reload(rag.retriever)

        return rag.retriever


def test_rerank_documents_orders_and_scores(retriever_module):
    """
    Validates that:
    - Documents are reranked correctly based on model scores
    - Only top_k documents are returned
    - Relevance scores are attached to document metadata
    """

    docs = [
        Document(page_content="doc A", metadata={}),
        Document(page_content="doc B", metadata={}),
    ]

    retriever_module.RERANKER.predict = MagicMock(
        return_value=[2.0, 5.0]
    )

    ranked_docs, scores = retriever_module._rerank_documents(
        query="test",
        docs=docs,
        top_k=1,
    )

    assert len(ranked_docs) == 1
    assert ranked_docs[0].page_content == "doc B"
    assert scores == [5.0]

    assert "relevance_score" in ranked_docs[0].metadata


def test_rerank_documents_empty_input(retriever_module):
    """
    Ensures reranking behaves safely when no documents are provided:
    - No errors raised
    - Empty outputs returned
    """

    docs, scores = retriever_module._rerank_documents(
        query="test",
        docs=[],
        top_k=3,
    )

    assert docs == []
    assert scores == []


def test_evaluate_retrieval_no_docs(retriever_module):
    """
    Validates retrieval evaluation logic when nothing is retrieved:
    - Quality should be marked as 'poor'
    - Reason should clearly indicate missing documents
    """

    result = retriever_module.evaluate_retrieval([], [])

    assert result["quality"] == "poor"
    assert result["reason"] == "no_documents_retrieved"


def test_evaluate_retrieval_good_quality(retriever_module):
    """
    Tests evaluation when:
    - Multiple high-scoring documents are retrieved
    - Multiple categories are covered
    - Overall retrieval quality should be 'good'
    """

    docs = [
        Document(page_content="x", metadata={"category": "faqs"}),
        Document(page_content="y", metadata={"category": "policies"}),
    ]
    scores = [4.8, 3.9]

    result = retriever_module.evaluate_retrieval(scores, docs)

    assert result["quality"] == "good"
    assert result["num_results"] == 2
    assert "faqs" in result["categories_covered"]


def test_evaluate_retrieval_partial_quality(retriever_module):
    """
    Ensures retrieval evaluation correctly identifies
    borderline / partial relevance scenarios.
    """

    docs = [Document(page_content="x", metadata={})]
    scores = [3.4]

    result = retriever_module.evaluate_retrieval(scores, docs)

    assert result["quality"] == "partially good"


def test_retrieve_documents_empty_query(retriever_module):
    """
    Confirms that empty or whitespace-only queries
    are rejected early with a clear error.
    """

    with pytest.raises(ValueError):
        retriever_module.retrieve_documents("   ")


@patch("rag.retriever._rerank_documents")
@patch("rag.retriever.evaluate_retrieval")
@patch("rag.retriever.get_vectorstore")
def test_retrieve_documents_happy_path(
    mock_get_vs,
    mock_eval,
    mock_rerank,
    retriever_module,
):
    """
    Full happy-path integration test for retrieve_documents:

    Verifies:
    - Vector store retrieval is invoked
    - Reranking is applied
    - Retrieval quality evaluation is executed
    - Final documents and metrics are returned correctly
    """


    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="doc", metadata={})
    ]

    vs = MagicMock()
    vs.as_retriever.return_value = retriever
    mock_get_vs.return_value = vs


    mock_rerank.return_value = (
        [Document(page_content="doc", metadata={})],
        [4.2],
    )


    mock_eval.return_value = {"quality": "good"}

    docs, metrics = retriever_module.retrieve_documents("test query")

    assert len(docs) == 1
    assert metrics["quality"] == "good"


    retriever.invoke.assert_called_once()
    mock_rerank.assert_called_once()
    mock_eval.assert_called_once()
