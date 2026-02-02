"""
Defines Pydantic schemas for request and response validation
in the support ticket resolution API.
"""
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

ActionRequired = Literal[
    "none",
    "customer_action_required",
    "follow_up_required",
    "escalate_to_support",
    "escalate_to_abuse_team",
    "escalate_to_billing",
    "escalate_to_technical",
]


class MCPResponse(BaseModel):
    """
    Base response schema for model-generated outputs.

    Enforces strict validation and disallows unknown fields.
    """
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, max_length=5000)
    references: List[str] = Field(default_factory=list)
    action_required: ActionRequired = "none"

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, v: str) -> str:
        """
        Ensures the answer is not empty or whitespace-only.
        """
        v = v.strip()
        if not v:
            raise ValueError("answer must be non-empty")
        return v

    @field_validator("references")
    @classmethod
    def references_nonempty_strings(cls, refs: List[str]) -> List[str]:
        """
        Cleans reference strings and enforces a maximum count.
        """
        cleaned = [r.strip() for r in refs if r and r.strip()]
        if len(cleaned) > 3:
            raise ValueError("at most 3 references allowed")
        return cleaned


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticket_text: str = Field(..., min_length=5, max_length=5000)


class TicketResponse(MCPResponse):
    pass
