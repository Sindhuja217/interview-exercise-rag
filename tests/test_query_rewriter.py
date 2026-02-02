"""
Unit tests for the query rewriting logic, validating input handling,
output normalization, deduplication, and query count limits.
"""
import pytest
from rag.query_rewriter import rewrite_ticket


def test_rewrite_ticket_empty_raises():
    """
    Ensures rewrite_ticket rejects empty or whitespace-only ticket text.

    Expected behavior:
    - Raises ValueError
    - Error message clearly indicates invalid input
    """

    with pytest.raises(ValueError, match="ticket_text must be non-empty"):
        rewrite_ticket("   ", llm_call=lambda _: "")


def test_rewrite_ticket_single_line():
    """
    Validates basic rewrite behavior when the LLM returns
    a single rewritten query.

    Expected behavior:
    - Single clean query is returned as a list
    """

    llm = lambda _: "domain suspension reason"

    result = rewrite_ticket(
        "My domain is suspended",
        llm_call=llm,
    )

    assert result == ["domain suspension reason"]


def test_rewrite_ticket_strips_bullets_and_numbers():
    """
    Ensures rewrite_ticket correctly cleans LLM output by:
    - Removing bullet characters
    - Removing numeric list prefixes
    - Stripping extra whitespace
    - Preserving original order
    """

    llm = lambda _: """
    - domain suspension reason
    • WHOIS verification status
    1. billing issue refund
    2. abuse complaint details
    """

    result = rewrite_ticket("Help", llm_call=llm)

    assert result == [
        "domain suspension reason",
        "WHOIS verification status",
        "billing issue refund",
        "abuse complaint details",
    ]


def test_rewrite_ticket_deduplicates_preserves_order():
    """
    Confirms rewrite_ticket removes duplicate queries
    while preserving the original order of appearance.
    """

    llm = lambda _: """
    domain suspension reason
    billing issue refund
    domain suspension reason
    """

    result = rewrite_ticket("Help", llm_call=llm)

    assert result == [
        "domain suspension reason",
        "billing issue refund",
    ]


def test_rewrite_ticket_limits_to_five():
    """
    Ensures rewrite_ticket enforces a hard limit
    on the number of rewritten queries returned.

    Expected behavior:
    - Only the first five queries are kept
    """

    llm = lambda _: """
    q1
    q2
    q3
    q4
    q5
    q6
    q7
    """

    result = rewrite_ticket("Help", llm_call=llm)

    assert result == ["q1", "q2", "q3", "q4", "q5"]


def test_rewrite_ticket_empty_llm_response_returns_empty_list():
    """
    Validates graceful handling when the LLM returns no content.

    Expected behavior:
    - No errors raised
    - Empty list returned
    """

    llm = lambda _: ""

    result = rewrite_ticket("Help", llm_call=llm)

    assert result == []
