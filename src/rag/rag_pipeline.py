"""
Implements the end-to-end RAG orchestration pipeline for resolving
support tickets, from query rewriting to answer generation.
"""

from typing import Dict, List
from langchain_core.documents import Document

from .retriever import retrieve_documents
from .query_rewriter import rewrite_ticket
from .generation import generate_answer


def resolve_ticket(ticket_text: str, llm_client) -> Dict:
    """
    End-to-end orchestrator:
      rewrite -> retrieve -> rerank -> generate (MCP JSON)
    """
    queries = rewrite_ticket(ticket_text, llm_client.call_text)
    if not queries:
        queries = [ticket_text]

    all_docs: List[Document] = []
    best_eval = {"quality": "poor"}

    for q in queries:
        docs, eval_metrics = retrieve_documents(q)
        all_docs.extend(docs)

        if eval_metrics.get("quality") == "good":
            best_eval = eval_metrics
        elif eval_metrics.get("quality") == "weak" and best_eval.get("quality") != "good":
            best_eval = eval_metrics

    dedup = {}
    for d in all_docs:
        dedup[d.page_content] = d


    global_ranked = sorted(
        dedup.values(),
        key=lambda d: d.metadata.get("relevance_score", float("-inf")),
        reverse=True,
    )

    final_docs = global_ranked[:4]

    result = generate_answer(
        ticket_text=ticket_text,
        documents=final_docs,
        llm_call=llm_client.call_text,
    )

    if best_eval.get("quality") == "poor":
        if result.get("action_required") == "none":
            result["action_required"] = "follow_up_required"

    result["_rewritten_queries"] = queries
    result["_reranked_docs"] = final_docs
    result["_retrieval_eval"] = best_eval

    return result
