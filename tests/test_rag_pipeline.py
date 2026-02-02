"""
Unit tests for the RAG pipeline orchestrator, validating query rewriting,
retrieval fallback logic, action overrides, and end-to-end control flow
with external dependencies mocked.
"""
import importlib
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


@pytest.fixture
def rag_pipeline(monkeypatch):
    """
    Fixture to import the RAG pipeline module with all heavyweight
    external dependencies mocked.

    Purpose:
    - Prevent real Qdrant connections
    - Avoid loading embedding models or rerankers
    - Ensure deterministic and fast unit tests
    """


    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant")


    with patch("sentence_transformers.CrossEncoder"), \
         patch("langchain_community.embeddings.HuggingFaceEmbeddings"), \
         patch("langchain_qdrant.QdrantVectorStore"), \
         patch("qdrant_client.QdrantClient"):

        import rag.rag_pipeline
        importlib.reload(rag.rag_pipeline)
        return rag.rag_pipeline


@pytest.fixture
def llm_client():
    """
    Provides a mocked LLM client used by the RAG pipeline.

    Purpose:
    - Simulates an LLM response without making real API calls
    - Returns deterministic JSON output for testing
    """

    client = MagicMock()
    client.call_text.return_value = '{"answer": "Test"}'
    return client


def make_doc(text, score):
    """
    Helper function to create a Document object
    with a predefined relevance score.

    Used to simulate reranked retrieval results.
    """

    return Document(
        page_content=text,
        metadata={"relevance_score": score},
    )


def test_resolve_ticket_happy_path(rag_pipeline, llm_client):
    """
    Tests the full happy-path execution of resolve_ticket.

    Verifies that:
    - The ticket is rewritten into search queries
    - Documents are retrieved successfully
    - A final answer is generated
    - Internal debug metadata is preserved
    """

    with patch.object(rag_pipeline, "rewrite_ticket") as mock_rewrite, \
         patch.object(rag_pipeline, "retrieve_documents") as mock_retrieve, \
         patch.object(rag_pipeline, "generate_answer") as mock_generate:

        mock_rewrite.return_value = ["query one"]

        mock_retrieve.return_value = (
            [make_doc("doc", 1.0)],
            {"quality": "good"},
        )

        mock_generate.return_value = {
            "answer": "Answer",
            "references": [],
            "action_required": "none",
        }

        result = rag_pipeline.resolve_ticket("Help", llm_client)

        assert result["answer"] == "Answer"
        assert result["_rewritten_queries"] == ["query one"]
        assert result["_retrieval_eval"]["quality"] == "good"


def test_resolve_ticket_fallback_to_original_query(rag_pipeline, llm_client):
    """
    Ensures resolve_ticket falls back to the original ticket text
    when query rewriting produces no rewritten queries.

    Verifies:
    - The original ticket is used for retrieval
    - The rewritten query list reflects the fallback behavior
    """

    with patch.object(rag_pipeline, "rewrite_ticket") as mock_rewrite, \
         patch.object(rag_pipeline, "retrieve_documents") as mock_retrieve, \
         patch.object(rag_pipeline, "generate_answer") as mock_generate:

        mock_rewrite.return_value = []

        mock_retrieve.return_value = (
            [make_doc("doc X", 1.0)],
            {"quality": "good"},
        )

        mock_generate.return_value = {
            "answer": "Answer",
            "references": [],
            "action_required": "none",
        }

        result = rag_pipeline.resolve_ticket("Original ticket", llm_client)

        mock_retrieve.assert_called_once_with("Original ticket")
        assert result["_rewritten_queries"] == ["Original ticket"]


def test_resolve_ticket_forces_follow_up_on_poor_retrieval(rag_pipeline, llm_client):
    """
    Validates safety behavior when retrieval quality is poor.

    Expected behavior:
    - The pipeline overrides any generated action
    - Forces a follow-up request instead of a confident answer
    """

    with patch.object(rag_pipeline, "rewrite_ticket") as mock_rewrite, \
         patch.object(rag_pipeline, "retrieve_documents") as mock_retrieve, \
         patch.object(rag_pipeline, "generate_answer") as mock_generate:

        mock_rewrite.return_value = ["query"]

        mock_retrieve.return_value = (
            [make_doc("doc", 0.4)],
            {"quality": "poor"},
        )

        mock_generate.return_value = {
            "answer": "Answer",
            "references": [],
            "action_required": "none",
        }

        result = rag_pipeline.resolve_ticket("Help", llm_client)

        assert result["action_required"] == "follow_up_required"
