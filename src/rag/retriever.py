"""
Implements hybrid document retrieval using dense and sparse search,
followed by cross-encoder reranking and retrieval quality evaluation.
"""
import os
from typing import List, Tuple, Dict
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

COLLECTION_NAME = "support_docs_hybrid"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_K = 6
FINAL_K = 4


if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set")

CLIENT = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
    check_compatibility=False,
)

DENSE_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=DENSE_MODEL_NAME,
    model_kwargs={"device": "cpu"},  
    encode_kwargs={"normalize_embeddings": True},
)

SPARSE_EMBEDDINGS = FastEmbedSparse(model_name="Qdrant/bm25")
RERANKER = CrossEncoder(RERANKER_MODEL)

def get_vectorstore() -> QdrantVectorStore:
    """
    Lazily initialize vector store.
    Prevents crash when collection does not yet exist.
    """
    return QdrantVectorStore(
        client=CLIENT,
        collection_name=COLLECTION_NAME,
        embedding=DENSE_EMBEDDINGS,
        sparse_embedding=SPARSE_EMBEDDINGS,
    )


def _rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int,
) -> Tuple[List[Document], List[float]]:
    """
    Rerank documents using cross-encoder and attach relevance scores.
    """
    if not docs:
        return [], []

    pairs = [(query, d.page_content) for d in docs]
    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_ranked = ranked[:top_k]

    top_docs: List[Document] = []
    top_scores: List[float] = []

    for doc, score in top_ranked:
        doc.metadata["relevance_score"] = float(score)
        top_docs.append(doc)
        top_scores.append(float(score))

    return top_docs, top_scores

def evaluate_retrieval(
    scores: List[float],
    docs: List[Document],
) -> Dict:
    """
    Weakly-supervised retrieval quality evaluation
    using cross-encoder relevance scores.
    """
    if not scores:
        return {
            "quality": "poor",
            "reason": "no_documents_retrieved",
        }

    avg_score = sum(scores) / len(scores)
    top_score = scores[0]
    score_gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]

    categories = {
        d.metadata.get("category", "unknown")
        for d in docs
    }

    if top_score > 4.0 and score_gap > 0.6:
        quality = "good"
    elif top_score > 3.0:
        quality = "partially good"
    else:
        quality = "poor"

    return {
        "quality": quality,
        "avg_relevance_score": round(avg_score, 3),
        "top_relevance_score": round(top_score, 3),
        "score_gap": round(score_gap, 3),
        "num_results": len(docs),
        "categories_covered": sorted(categories),
    }

def retrieve_documents(
    query: str,
    initial_k: int = INITIAL_K,
    final_k: int = FINAL_K,
) -> Tuple[List[Document], Dict]:
    """
    Hybrid retrieval (dense + sparse) → cross-encoder reranking
    → relevance scoring → retrieval evaluation.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Query must be a non-empty string")

    retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": initial_k}
    )

    initial_docs = retriever.invoke(q)

    final_docs, scores = _rerank_documents(
        query=q,
        docs=initial_docs,
        top_k=final_k,
    )

    eval_metrics = evaluate_retrieval(
        scores=scores,
        docs=final_docs,
    )

    return final_docs, eval_metrics
