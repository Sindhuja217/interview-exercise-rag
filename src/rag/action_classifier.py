"""
Infers follow-up or escalation actions for support responses by
comparing answer embeddings against predefined action prototypes
using semantic similarity.
"""
from typing import Literal, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ActionRequired = Literal[
    "none",
    "customer_action_required",
    "follow_up_required",
    "escalate_to_support",
    "escalate_to_abuse_team",
    "escalate_to_billing",
    "escalate_to_technical",
]

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ACTION_PROTOTYPES: Dict[ActionRequired, List[str]] = {
    "escalate_to_abuse_team": [
        "Domain suspended for phishing, malware, or spam",
        "Abuse complaint requires review by the Abuse Team",
        "Support must not manually reactivate this domain",
        "This suspension is due to a policy violation or abuse report",
    ],
    "escalate_to_billing": [
        "Billing dispute involving charges, refunds, or invoices",
        "Customer reports duplicate charge or payment failure",
        "Domain suspended due to unpaid invoice or payment issue",
        "Refund eligibility must be reviewed by Billing Team",
    ],
    "escalate_to_technical": [
        "Domain is active but DNS is not resolving",
        "Service outage or technical failure after renewal",
        "System issue where services remain offline unexpectedly",
        "Technical investigation required for infrastructure failure",
    ],
    "escalate_to_support": [
        "Domain suspended due to WHOIS verification issues",
        "Customer reports domain still suspended after completing WHOIS verification",
        "Support needs to review account status and system flags",
        "Manual review required for non-abuse domain suspension",
    ],
    "customer_action_required": [
        "Customer must verify WHOIS email address",
        "Registrant information must be updated to restore domain",
        "User needs to unlock the domain before transfer",
        "Customer must complete remediation steps",
    ],
    "follow_up_required": [
        "Reactivation will occur after review is completed",
        "Support will monitor and follow up after verification",
        "Additional review time is required before action",
    ],
    "none": [
        "General informational question about domains",
        "Explanation of policy without required action",
        "Customer is asking how the system works",
        "No action is required from support or customer",
    ],
}


_PROTOTYPE_EMBEDDINGS: Dict[ActionRequired, np.ndarray] = {
    action: _MODEL.encode(texts, normalize_embeddings=True)
    for action, texts in ACTION_PROTOTYPES.items()
}


def infer_action(
    answer: str,
    threshold: float = 0.5,
) -> Dict[str, float | str]:
    """
    Infers the most appropriate action based on the semantic
    similarity between an answer and predefined action prototypes.

    Args:
        answer:
            The generated answer text from the RAG pipeline.
        threshold:
            Minimum similarity score required to assign an action.
            Below this value, the function returns 'no_action'.

    Returns:
        A dictionary with:
        - 'action': inferred action label
        - 'confidence': similarity score rounded to 3 decimals
    """
    if not answer or not answer.strip():
        return {"action": "no_action", "confidence": 0.0}

    answer_emb = _MODEL.encode(answer, normalize_embeddings=True)

    best_action = "no_action"
    best_score = 0.0

    for action, proto_embs in _PROTOTYPE_EMBEDDINGS.items():
        sims = proto_embs @ answer_emb
        max_sim = float(np.max(sims))

        if max_sim > best_score:
            best_score = max_sim
            best_action = action

    if best_score < threshold:
        return {"action": "no_action", "confidence": round(best_score, 3)}

    return {"action": best_action, "confidence": round(best_score, 3)}
