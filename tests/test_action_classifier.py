"""
Unit tests for the action inference logic, validating deterministic
behavior, threshold handling, and confidence scoring.
"""
import importlib
import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture
def action_classifier(monkeypatch):
    """
    Fixture to import the action_classifier module with the
    SentenceTransformer model mocked BEFORE module import.

    Purpose:
    - Avoids loading a real embedding model
    - Ensures deterministic embeddings for testing
    - Guarantees consistent similarity behavior across tests
    """

    class FakeModel:
        """
        Fake embedding model that returns deterministic vectors
        for both single strings and lists of strings.
        """

        def encode(self, texts, normalize_embeddings=True):
            if isinstance(texts, str):
                return np.array([1.0, 0.0, 0.0])
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

    with patch("sentence_transformers.SentenceTransformer", return_value=FakeModel()):
        import rag.action_classifier
        importlib.reload(rag.action_classifier)
        return rag.action_classifier


def test_infer_action_empty_answer(action_classifier):
    """
    Ensures infer_action returns a safe default when
    the input answer string is empty.

    Expected behavior:
    - Action is 'no_action'
    - Confidence is 0.0
    """

    result = action_classifier.infer_action("")
    assert result == {"action": "no_action", "confidence": 0.0}


def test_infer_action_whitespace_answer(action_classifier):
    """
    Validates infer_action handles whitespace-only input
    the same as an empty string.

    Expected behavior:
    - No action inferred
    - Confidence remains zero
    """

    result = action_classifier.infer_action("   ")
    assert result == {"action": "no_action", "confidence": 0.0}


def test_infer_action_selects_best_matching_action(action_classifier):
    """
    Tests that infer_action selects the best matching action
    when similarity exceeds the threshold.

    Because embeddings are deterministic, the first prototype
    encountered with maximum similarity should be selected.
    """

    result = action_classifier.infer_action(
        "Domain suspended due to abuse report",
        threshold=0.1,
    )

    assert result["action"] in action_classifier.ACTION_PROTOTYPES
    assert isinstance(result["confidence"], float)
    assert result["confidence"] >= 0.0


def test_infer_action_below_threshold_returns_no_action(monkeypatch, action_classifier):
    """
    Ensures infer_action returns 'no_action' when the highest
    similarity score falls below the provided threshold.

    This test simulates weak semantic similarity by overriding
    the embedding output.
    """

    def fake_encode(texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([0.01, 0.0, 0.0])
        return np.array([[1.0, 0.0, 0.0] for _ in texts])

    monkeypatch.setattr(action_classifier._MODEL, "encode", fake_encode)

    result = action_classifier.infer_action(
        "Random unrelated text",
        threshold=0.99,
    )

    assert result["action"] == "no_action"


def test_infer_action_confidence_is_rounded(action_classifier):
    """
    Verifies that infer_action rounds confidence scores
    to a fixed precision for stable and predictable output.

    Expected behavior:
    - Confidence value is rounded to 3 decimal places
    """

    result = action_classifier.infer_action(
        "Billing issue refund",
        threshold=0.1,
    )

    assert round(result["confidence"], 3) == result["confidence"]
