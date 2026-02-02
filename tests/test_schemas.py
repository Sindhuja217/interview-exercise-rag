"""
Tests for Pydantic schema validation, ensuring strict field enforcement,
input constraints, and normalization for MCP responses and ticket requests.
"""
import pytest
from pydantic import ValidationError

from rag.schemas import MCPResponse, TicketRequest



def test_mcp_response_valid():
    """
    Happy path: valid MCPResponse passes validation.
    """
    resp = MCPResponse(
        answer="Your domain is active and resolving correctly.",
        references=["faqs: Domain Status | file=faqs/domain.md"],
        action_required="none",
    )

    assert resp.answer == "Your domain is active and resolving correctly."
    assert resp.references == ["faqs: Domain Status | file=faqs/domain.md"]
    assert resp.action_required == "none"


def test_mcp_response_rejects_extra_fields():
    """
    extra='forbid' should reject unknown fields.
    """
    with pytest.raises(ValidationError):
        MCPResponse(
            answer="Answer",
            references=[],
            action_required="none",
            unexpected_field="not allowed",
        )


def test_mcp_response_rejects_empty_answer():
    """
    Answer must be non-empty after stripping.
    """
    with pytest.raises(ValidationError):
        MCPResponse(
            answer="   ",
            references=[],
            action_required="none",
        )


def test_mcp_response_rejects_invalid_action():
    """
    action_required must be one of the allowed Literal values.
    """
    with pytest.raises(ValidationError):
        MCPResponse(
            answer="Answer",
            references=[],
            action_required="no_action",  
        )


def test_references_are_stripped_and_empty_removed():
    """
    References validator should strip whitespace and drop empty strings.
    """
    resp = MCPResponse(
        answer="Answer",
        references=[
            "  faqs: Reset Password | file=faqs/reset.md  ",
            "",
            "   ",
            "policies: WHOIS | file=policies/whois.md",
        ],
        action_required="none",
    )

    assert resp.references == [
        "faqs: Reset Password | file=faqs/reset.md",
        "policies: WHOIS | file=policies/whois.md",
    ]


def test_mcp_response_rejects_too_many_references():
    """
    Max 3 references allowed.
    """
    with pytest.raises(ValidationError):
        MCPResponse(
            answer="Answer",
            references=["r1", "r2", "r3", "r4"],
            action_required="none",
        )


def test_ticket_request_valid():
    """
    Valid ticket passes validation.
    """
    req = TicketRequest(ticket_text="My domain is suspended, please help.")

    assert req.ticket_text == "My domain is suspended, please help."


def test_ticket_request_rejects_short_text():
    """
    ticket_text must be at least 5 characters.
    """
    with pytest.raises(ValidationError):
        TicketRequest(ticket_text="Hi")


def test_ticket_request_rejects_extra_fields():
    """
    extra='forbid' should reject unknown fields.
    """
    with pytest.raises(ValidationError):
        TicketRequest(
            ticket_text="Valid ticket text",
            extra_field="not allowed",
        )
