"""
Exposes a FastAPI service that resolves customer support tickets
using a retrieval-augmented generation (RAG) pipeline.
"""
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging

from rag.schemas import TicketRequest, TicketResponse
from rag.rag_pipeline import resolve_ticket
from .llm_client import LLMClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("support-assistant")

app = FastAPI(
    title="Support Knowledge Assistant",
    version="1.0.0",
    description="LLM-powered RAG assistant for customer support tickets",
)

llm_client = LLMClient()

_INTERNAL_FIELDS = {
    "_rewritten_queries",
    "_reranked_docs",
    "_retrieval_eval",
}

@app.on_event("startup")
def startup_check():
    """
    Logs successful application startup.
    """
    logger.info("Support Knowledge Assistant started successfully")

@app.get("/health", summary="Service health check")
def health():
    """
    Lightweight health endpoint for monitoring.
    """
    return {"status": "ok"}

@app.get("/", summary="Root")
def root():
    """
    Returns basic service metadata and useful endpoints.
    """
    return {
        "service": "Support Knowledge Assistant",
        "status": "running",
        "version": app.version,
        "health": "/health",
        "docs": "/docs",
    }

@app.post(
    "/resolve-ticket",
    response_model=TicketResponse,
    summary="Resolve a support ticket",
    description=(
        "Processes a raw customer support ticket using a RAG pipeline "
        "and returns a grounded response, relevant references, and "
        "any required follow-up action."
    ),
)
def resolve_ticket_endpoint(request: TicketRequest):
    """
    Main API endpoint that processes a support ticket
    through the RAG pipeline and returns a structured response.
    """
    logger.info(
        "Received ticket",
        extra={"ticket_length": len(request.ticket_text)},
    )

    try:
        result = resolve_ticket(
            ticket_text=request.ticket_text,
            llm_client=llm_client,
        )
    except Exception as exc:
        logger.exception("Ticket resolution failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to resolve support ticket",
        ) from exc


    for field in _INTERNAL_FIELDS:
        result.pop(field, None)

    result.setdefault("answer", "")
    result.setdefault("references", [])
    result.setdefault("action_required", "none")

    logger.info(
        "Ticket resolved successfully",
        extra={
            "answer_length": len(result["answer"]),
            "num_references": len(result["references"]),
            "action_required": result["action_required"],
        },
    )

    return {
        "answer": result["answer"],
        "references": result["references"],
        "action_required": result["action_required"],
    }
