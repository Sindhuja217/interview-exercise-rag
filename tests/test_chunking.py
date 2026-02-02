"""
Unit tests for the markdown chunking pipeline, validating path resolution,
file discovery, header-aware chunking, and end-to-end orchestration.
"""
import os
import pytest
from pathlib import Path

from langchain_core.documents import Document
from rag import chunking


@pytest.fixture
def sample_markdown():
    """
    Provides a sample markdown document with nested headers.

    Purpose:
    - Used to test markdown chunking behavior
    - Includes title, sections, and subsections
    """

    return """# Title

## Section A
Content A

### Subsection
More details

## Section B
Content B
"""


@pytest.fixture
def temp_data_dir(tmp_path: Path):
    """
    Creates a temporary directory structure with a markdown file:

    data/
      faqs/
        test_doc.md

    Purpose:
    - Simulates real input directory layout
    - Used for end-to-end chunking tests
    """

    data_dir = tmp_path / "data" / "faqs"
    data_dir.mkdir(parents=True)

    md_file = data_dir / "test_doc.md"
    md_file.write_text(
        "# FAQ Title\n\n## Question\nAnswer text\n",
        encoding="utf-8",
    )

    return tmp_path


def test_load_paths_success(monkeypatch, tmp_path):
    """
    Verifies load_paths correctly resolves input and output paths
    when configuration is valid.

    Expected behavior:
    - Returns resolved input directory
    - Returns resolved output file path
    """

    monkeypatch.setattr(chunking, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(chunking, "INPUT_DIR", "data")
    monkeypatch.setattr(chunking, "OUTPUT_FILE", "out/chunks.txt")

    (tmp_path / "data").mkdir()

    input_path, output_path = chunking.load_paths()

    assert input_path.endswith("data")
    assert output_path.endswith("out/chunks.txt")


def test_load_paths_missing_input(monkeypatch, tmp_path):
    """
    Ensures load_paths raises an error when the input
    directory does not exist.

    Expected behavior:
    - Raises FileNotFoundError
    """

    monkeypatch.setattr(chunking, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(chunking, "INPUT_DIR", "data")

    with pytest.raises(FileNotFoundError):
        chunking.load_paths()


def test_read_markdown(tmp_path):
    """
    Verifies read_markdown correctly reads markdown file content.

    Expected behavior:
    - Returns file content as a string
    """

    md = tmp_path / "test.md"
    md.write_text("# Hello", encoding="utf-8")

    content = chunking.read_markdown(str(md))
    assert content == "# Hello"


def test_discover_markdown_files_success(temp_data_dir):
    """
    Ensures discover_markdown_files finds markdown files
    in nested directories.

    Expected behavior:
    - Returns a non-empty list of .md file paths
    """

    files = chunking.discover_markdown_files(str(temp_data_dir / "data"))

    assert len(files) == 1
    assert files[0].endswith(".md")


def test_discover_markdown_files_empty(tmp_path):
    """
    Verifies discover_markdown_files fails when no markdown
    files are present.

    Expected behavior:
    - Raises ValueError with clear message
    """

    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="No markdown files found"):
        chunking.discover_markdown_files(str(empty))


def test_infer_category():
    """
    Tests category inference from file path relative to base directory.

    Expected behavior:
    - Category name is extracted correctly from directory structure
    """

    base = "/root/data"
    file_path = "/root/data/faqs/test.md"

    category = chunking.infer_category(file_path, base)
    assert category == "faqs"


def test_chunk_markdown_returns_documents(sample_markdown):
    """
    Ensures chunk_markdown returns a list of Document objects.

    Expected behavior:
    - Output is a list
    - List contains one or more Document instances
    """

    docs = chunking.chunk_markdown(sample_markdown)

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_chunk_markdown_preserves_headers(sample_markdown):
    """
    Validates that markdown headers are preserved
    across chunked content.

    Expected behavior:
    - Title, section, and subsection headers appear in chunks
    """

    docs = chunking.chunk_markdown(sample_markdown)

    combined = "\n".join(d.page_content for d in docs)

    assert "# Title" in combined
    assert "## Section A" in combined
    assert "### Subsection" in combined


def test_chunk_markdown_no_empty_chunks():
    """
    Ensures chunk_markdown does not emit empty or whitespace-only chunks.

    Expected behavior:
    - All chunks contain meaningful content
    """

    docs = chunking.chunk_markdown("# Title\n\n")

    assert all(d.page_content.strip() for d in docs)


def test_write_chunks_creates_output_file(tmp_path):
    """
    Verifies write_chunks writes chunked documents
    to a properly formatted output file.

    Expected behavior:
    - Output file is created
    - Chunk markers and metadata are present
    """

    output = tmp_path / "out" / "chunks.txt"

    docs = [
        Document(
            page_content="Chunk content",
            metadata={"category": "faqs"},
        )
    ]

    chunking.write_chunks(docs, str(output))

    assert output.exists()

    text = output.read_text(encoding="utf-8")
    assert "--- CHUNK 0 ---" in text
    assert "Chunk content" in text
    assert "METADATA" in text


def test_main_end_to_end(monkeypatch, temp_data_dir):
    """
    End-to-end orchestration test for chunking.main.

    Purpose:
    - Verifies high-level pipeline wiring
    - Does not mock internal functions
    - Uses temporary filesystem only
    """

    monkeypatch.setattr(chunking, "BASE_DIR", str(temp_data_dir))
    monkeypatch.setattr(chunking, "INPUT_DIR", "data")
    monkeypatch.setattr(chunking, "OUTPUT_FILE", "out/chunks.txt")

    chunking.main()

    output = temp_data_dir / "out" / "chunks.txt"
    assert output.exists()
    assert output.read_text(encoding="utf-8").strip()
