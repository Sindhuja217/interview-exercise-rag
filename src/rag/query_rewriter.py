"""
Rewrites raw support tickets into clean, retrieval-optimized
search queries for downstream document retrieval.
"""
from typing import List
import re

QUERY_REWRITE_PROMPT = """
You are a query rewriting assistant for a customer support knowledge base.

Rewrite the ticket into clear, retrieval-friendly search queries.

Rules:
- Use neutral, professional language
- Use support terminology (domain suspension, WHOIS, abuse, billing, etc.)
- Split multi-issue tickets into multiple queries
- Do NOT answer
- Do NOT add facts
- Output one query per line, no bullets, no numbering

Ticket:
\"\"\"{ticket}\"\"\"

Queries:
"""

def rewrite_ticket(ticket_text: str, llm_call) -> List[str]:
    """
    Rewrite a raw support ticket into one or more
    clean, retrieval-optimized search queries.
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("ticket_text must be non-empty")

    prompt = QUERY_REWRITE_PROMPT.format(ticket=ticket_text.strip())
    response = llm_call(prompt)

    lines = []
    for line in (response or "").splitlines():
        s = line.strip()
        if not s:
            continue

        s = s.lstrip("-•").strip()
        s = re.sub(r"^\d+[\.\)]\s*", "", s)

        if s:
            lines.append(s)
    seen = set()
    out = []
    for q in lines:
        if q not in seen:
            seen.add(q)
            out.append(q)

    return out[:5]
