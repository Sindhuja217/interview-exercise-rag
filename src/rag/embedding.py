"""
Loads pre-chunked support documents, generates dense and sparse embeddings,
and indexes them into a hybrid Qdrant vector store for retrieval.
"""
import os
import re
import time
import hashlib
import ast
from pathlib import Path

from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)


BASE_DIR = Path(__file__).resolve().parents[2]
CHUNK_FILE = "artifacts/langchain_chunks.txt"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  
COLLECTION_NAME = "support_docs_hybrid"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_chunks(path: str) -> List[Document]:
    """
    Parse chunked markdown output back into LangChain Documents.

    Changes vs your version:
    - Uses ast.literal_eval instead of eval (safe)
    - Adds stable chunk_id in metadata
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunk_pattern = re.compile(
        r"--- CHUNK (\d+) ---\n<!-- METADATA: ({.*?}) -->\n\n(.*?)(?=(?:--- CHUNK|\Z))",
        re.DOTALL,
    )

    docs: List[Document] = []
    seen_hashes = set()

    for chunk_idx_str, meta_str, body in chunk_pattern.findall(content):
        body = (body or "").strip()
        if not body:
            continue

        h = hashlib.sha1(body.encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        metadata = ast.literal_eval(meta_str) 
        metadata["chunk_id"] = int(chunk_idx_str)

        docs.append(Document(page_content=body, metadata=metadata))

    print(f"Loaded {len(docs)} unique chunks")
    return docs


def recreate_collection(client: QdrantClient):
    """
    Delete and recreate the Qdrant collection (idempotent).
    """
    if client.collection_exists(COLLECTION_NAME):
        print("Deleting existing collection…")
        client.delete_collection(COLLECTION_NAME)
        time.sleep(1)

    print(f"🔨 Creating hybrid collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                )
            ),
        ),
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )


def main():
    """
    Orchestrates the end-to-end embedding and indexing pipeline.

    Responsibilities:
    - Validate required environment configuration
    - Load pre-chunked documents from disk
    - Initialize and reset the Qdrant collection
    - Create dense and sparse embedding models
    - Upload documents into a hybrid Qdrant vector store

    This function performs orchestration only and delegates
    core logic to helper functions.
    """
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set")

    chunk_path = BASE_DIR / CHUNK_FILE
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

    docs = load_chunks(chunk_path)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False,
    )

    recreate_collection(client)

    dense_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
    )

    print("Uploading documents to Qdrant…")
    vectorstore.add_documents(docs)

    print("Hybrid Qdrant indexing complete")


if __name__ == "__main__":
    main()
