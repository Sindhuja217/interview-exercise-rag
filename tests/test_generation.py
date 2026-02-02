"""
Unit tests for answer generation logic, validating LLM output handling,
reference selection, action inference, and strict schema enforcement.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from rag.generation import generate_answer


@pytest.fixture
def sample_docs():
    """
    Provides a minimal set of documents used as context
    for answer generation tests.

    Purpose:
    - Simulates retrieved knowledge base content
    - Includes realistic metadata for reference generation
    """

    return [
        Document(
            page_content="Reset your password using the email link.",
            metadata={
                "category": "faqs",
                "source_file": "faqs/reset.md",
                "section": "Password Reset",
            },
        )
    ]


@pytest.fixture
def valid_llm_call():
    """
    Mock LLM callable that always returns valid JSON.

    Purpose:
    - Ensures generate_answer receives well-formed output
    - Allows tests to focus on pipeline behavior, not LLM errors
    """

    return lambda prompt: json.dumps(
        {"answer": "You can reset your password via the email link."}
    )


def test_generate_answer_empty_ticket_raises(sample_docs, valid_llm_call):
    """
    Ensures generate_answer rejects empty or whitespace-only
    ticket text early.

    Expected behavior:
    - Raises ValueError
    - Error message clearly indicates invalid input
    """

    with pytest.raises(ValueError, match="ticket_text must be non-empty"):
        generate_answer("   ", sample_docs, valid_llm_call)


@patch("rag.generation.infer_action")
@patch("rag.generation.select_top_references")
def test_generate_answer_happy_path(
    mock_refs,
    mock_infer_action,
    sample_docs,
    valid_llm_call,
):
    """
    Validates the full happy-path execution of generate_answer.

    Verifies:
    - LLM output is parsed correctly
    - References are selected and attached
    - Action inference is performed
    - Final response schema is respected
    """

    mock_refs.return_value = ["faqs: Password Reset | file=faqs/reset.md"]
    mock_infer_action.return_value = {
        "action": "none",
        "confidence": 0.92,
    }

    result = generate_answer(
        ticket_text="I forgot my password",
        documents=sample_docs,
        llm_call=valid_llm_call,
    )

    assert result["answer"].startswith("You can reset your password")
    assert result["references"] == mock_refs.return_value
    assert result["action_required"] == "none"

    mock_refs.assert_called_once()
    mock_infer_action.assert_called_once()


def test_generate_answer_invalid_json_raises(sample_docs):
    """
    Ensures generate_answer fails fast when the LLM
    returns invalid (non-JSON) output.

    Expected behavior:
    - Raises ValueError
    - Error message indicates malformed LLM response
    """

    def bad_llm(_):
        return "NOT JSON"

    with pytest.raises(
        ValueError,
        match="LLM did not return valid JSON",
    ):
        generate_answer(
            ticket_text="Help",
            documents=sample_docs,
            llm_call=bad_llm,
        )


def test_generate_answer_missing_answer_key_raises(sample_docs):
    """
    Validates schema enforcement when the LLM response
    does not contain the required 'answer' field.

    Expected behavior:
    - Raises ValueError
    - Prevents incomplete responses from propagating
    """

    def llm_missing_answer(_):
        return json.dumps({"text": "hello"})

    with pytest.raises(ValueError):
        generate_answer(
            ticket_text="Help",
            documents=sample_docs,
            llm_call=llm_missing_answer,
        )


@patch("rag.generation.infer_action")
@patch("rag.generation.select_top_references")
def test_generate_answer_no_documents_context_fallback(
    mock_refs,
    mock_infer_action,
):
    """
    Ensures generate_answer gracefully handles the case
    where no documents are available for context.

    Expected behavior:
    - Answer is still generated
    - References list is empty
    - Action inference still executes
    """

    mock_refs.return_value = []
    mock_infer_action.return_value = {
        "action": "none",
        "confidence": 0.5,
    }

    llm = lambda prompt: json.dumps({"answer": "No docs found."})

    result = generate_answer(
        ticket_text="Help",
        documents=[],
        llm_call=llm,
    )

    assert result["answer"] == "No docs found."
    assert result["references"] == []


@patch("rag.generation.infer_action")
@patch("rag.generation.select_top_references")
def test_generate_answer_abstention_mapped_to_no_action(
    mock_refs,
    mock_infer_action,
    sample_docs,
    valid_llm_call,
):
    """
    Validates that high-confidence 'no action' inferences
    are mapped correctly to the final response.

    Expected behavior:
    - action_required remains 'none'
    """

    mock_refs.return_value = []
    mock_infer_action.return_value = {
        "action": "none",
        "confidence": 0.99,
    }

    result = generate_answer(
        ticket_text="Help",
        documents=sample_docs,
        llm_call=valid_llm_call,
    )

    assert result["action_required"] == "none"


@patch("rag.generation.infer_action")
@patch("rag.generation.select_top_references")
def test_generate_answer_schema_validation_failure(
    mock_refs,
    mock_infer_action,
    sample_docs,
):
    """
    Ensures strict schema validation is enforced on
    the inferred action metadata.

    Expected behavior:
    - Invalid action schema triggers a ValueError
    - Prevents unsafe or malformed pipeline output
    """

    mock_refs.return_value = ["bad-ref"]
    mock_infer_action.return_value = {
        "action": None,  
        "confidence": 0.1,
    }

    def llm(_):
        return json.dumps({"answer": "Test"})

    with pytest.raises(ValueError, match="MCP schema validation failed"):
        generate_answer(
            ticket_text="Help",
            documents=sample_docs,
            llm_call=llm,
        )
