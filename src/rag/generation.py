"""
Generates a structured support response by combining LLM output,
retrieved document context, deterministic action inference, and
strict schema validation.
"""
import json
from typing import List, Dict
from langchain_core.documents import Document
from pydantic import ValidationError

from .schemas import MCPResponse
from .prompts import MCP_GENERATION_PROMPT
from .references import select_top_references
from .action_classifier import infer_action


def generate_answer(
    ticket_text: str,
    documents: List[Document],
    llm_call,
) -> Dict:
    """
    End-to-end MCP response generator:
    - LLM generates answer only
    - References come from top retrieved chunks
    - Action inferred deterministically (semantic)
    - Internal abstention mapped to safe external action
    """

    if not ticket_text or not ticket_text.strip():
        raise ValueError("ticket_text must be non-empty")


    context = "\n\n".join(
        doc.page_content.strip()
        for doc in documents
    ) or "No relevant documentation found."

    prompt = MCP_GENERATION_PROMPT.format(
        ticket=ticket_text.strip(),
        context=context,
    )


    raw = llm_call(prompt)

    try:
        parsed = json.loads(raw)
        answer = parsed["answer"].strip()
    except Exception:
        raise ValueError("LLM did not return valid JSON with `answer`")


    references = select_top_references(documents, k=3)
    decision = infer_action(answer)

    action_required = decision["action"]
    action_confidence = decision["confidence"]

    if action_required in ("none", "no_action"):
        action_required = "none"

    try:
        validated = MCPResponse(
            answer=answer,
            references=references,
            action_required=action_required,
        )
    except ValidationError as e:
        raise ValueError(f"MCP schema validation failed: {e}")

    return validated.model_dump()
