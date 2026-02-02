"""
Streamlit UI for the Support Knowledge Assistant.

Provides an interactive interface for submitting customer tickets,
running the RAG pipeline, inspecting retrieved documents, and
displaying answers, references, and inferred actions.
"""
import json
import streamlit as st
from typing import List

from src.rag.rag_pipeline import resolve_ticket
from src.rag.references import select_top_references
from src.rag.action_classifier import infer_action
from src.rag.llm_client import LLMClient
from langchain_core.documents import Document


st.set_page_config(
    page_title="Support Knowledge Assistant",
    page_icon="🛠️",
    layout="wide",
)

def render_doc(doc: Document, idx: int):
    """
    Render a retrieved document with metadata and truncated content
    inside a collapsible Streamlit expander.
    """
    with st.expander(f"📄 Retrieved Doc #{idx + 1}", expanded=False):
        st.markdown("**Metadata**")
        st.json(doc.metadata)
        st.markdown("**Content**")
        st.markdown(doc.page_content[:2000] + ("…" if len(doc.page_content) > 2000 else ""))


def safe_json_load(text: str):
    """
    Safely parse a JSON string, returning None on failure.
    """
    try:
        return json.loads(text)
    except Exception:
        return None


st.sidebar.title("⚙️ Settings")

show_debug = st.sidebar.checkbox("Show debug details", value=True)
auto_action = st.sidebar.checkbox("Auto-infer action_required", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Hybrid RAG · Qdrant · Cross-Encoder · MCP")

st.title("🧠 Support Ticket Assistant")
st.caption("Hybrid retrieval + grounded generation + action inference")

ticket_text = st.text_area(
    "Customer Ticket",
    placeholder="e.g. My domain was suspended due to abuse. I removed the content. Can you reactivate it?",
    height=160,
)

submit = st.button("Resolve Ticket", type="primary", use_container_width=True)

if submit:
    if not ticket_text.strip():
        st.warning("Please enter a ticket.")
        st.stop()

    llm = LLMClient()

    with st.spinner("Analyzing ticket…"):
        result = resolve_ticket(ticket_text, llm)


    answer = result.get("answer", "").strip()
    references = result.get("references", [])
    action_required = result.get("action_required", "none")

    if auto_action and action_required == "none":
        action_required = infer_action(answer)


    st.success("Ticket resolved")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("💬 Answer")
        st.markdown(answer)

    with col2:
        st.subheader("🚦 Action Required")
        st.code(action_required)

    if references:
        st.subheader("📚 References")
        for ref in references:
            st.markdown(f"- `{ref}`")


    if show_debug:
        st.markdown("---")
        st.subheader("🔍 Debug & Inspection")

        st.markdown("**Raw MCP JSON**")
        st.json(result)

        rewritten_queries = result.get("_rewritten_queries")
        if rewritten_queries:
            st.markdown("### 🔁 Query Rewriting")
            for i, q in enumerate(rewritten_queries, start=1):
                st.markdown(f"**Query {i}:** {q}")


        eval_metrics = result.get("_retrieval_eval")
        if eval_metrics:
            st.markdown("### 📊 Retrieval Quality")
            st.json(eval_metrics)

        reranked_docs: List[Document] = result.get("_reranked_docs", [])

        if reranked_docs:
            st.markdown("### 📄 Final Reranked Chunks (Post-Retrieval)")

            for i, doc in enumerate(reranked_docs):
                with st.expander(
                    f"Chunk #{i+1} · score={doc.metadata.get('relevance_score', 'n/a')}",
                    expanded=False,
                ):
                    st.markdown("**Metadata**")
                    st.json(doc.metadata)

                    st.markdown("**Content**")
                    st.markdown(
                        doc.page_content[:2500]
                        + ("…" if len(doc.page_content) > 2500 else "")
                    )
