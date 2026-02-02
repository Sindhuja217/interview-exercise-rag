"""
Chunks markdown knowledge-base documents into structured, header-aware
segments and persists them for downstream retrieval and embedding.
"""
import os
from pathlib import Path
from typing import List, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = "data" 
OUTPUT_FILE = "artifacts/langchain_chunks.txt"


def load_paths() -> Tuple[str, str]:
    """
    Validate and construct input/output paths.
    """
    input_path = BASE_DIR / INPUT_DIR
    output_path = BASE_DIR / OUTPUT_FILE

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    return input_path, output_path


def read_markdown(path: str) -> str:
    """
    Read markdown file safely.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def discover_markdown_files(root_dir: str) -> List[str]:
    """
    Recursively find all .md files under root_dir.
    """
    md_files: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))

    if not md_files:
        raise ValueError(f"No markdown files found under {root_dir}")

    return sorted(md_files)


def infer_category(file_path: str, base_input_dir: str) -> str:
    """
    Infer category from first-level folder (faqs / policies / runbooks).
    """
    rel_path = os.path.relpath(file_path, base_input_dir)
    return rel_path.split(os.sep)[0]


def chunk_markdown(markdown_text: str) -> List[Document]:
    """
    Chunk markdown using MarkdownHeaderTextSplitter while preserving header hierarchy.

    NOTE:
    - We keep headers in the chunk content for better grounding + references.
    - Header metadata is used to build stable "references" later.
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    docs = splitter.split_text(markdown_text)

    cleaned_docs: List[Document] = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        if not content:
            continue

        cleaned_docs.append(
            Document(
                page_content=content,
                metadata=dict(doc.metadata),
            )
        )

    return cleaned_docs


def write_chunks(docs: List[Document], output_path: str) -> None:
    """
    Persist ALL chunks into a single output file (debug/audit friendly).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(f"--- CHUNK {i} ---\n")
            f.write(f"<!-- METADATA: {doc.metadata} -->\n\n")
            f.write(doc.page_content)
            f.write("\n\n")

    print(f"LangChain chunked markdown saved to: {output_path}")
    print(f" Total chunks written: {len(docs)}")


def main():
    """
    Orchestrates the end-to-end markdown chunking pipeline.

    Responsibilities:
    - Resolve and validate input/output paths
    - Discover all markdown files under the input directory
    - Read and chunk each markdown file while preserving structure
    - Enrich each chunk with metadata for traceability and retrieval
    - Persist all chunks into a single output file

    This function performs orchestration only and delegates
    actual logic to helper functions.
    """
    input_dir, output_path = load_paths()

    md_files = discover_markdown_files(input_dir)
    all_chunks: List[Document] = []

    for md_path in md_files:
        markdown = read_markdown(md_path)
        docs = chunk_markdown(markdown)

        category = infer_category(md_path, input_dir)
        source = os.path.relpath(md_path, input_dir)

    
        file_stem = os.path.splitext(os.path.basename(md_path))[0]
        default_title = file_stem.replace("_", " ").strip()

        for doc in docs:
            doc.metadata.update(
                {
                    "source_file": source,
                    "category": category,
                    "doc_title": default_title,
                }
            )
            all_chunks.append(doc)

    write_chunks(all_chunks, output_path)


if __name__ == "__main__":
    main()
