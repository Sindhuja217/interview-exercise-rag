"""
Unit tests for reference formatting and selection logic, ensuring
stable, human-readable citations and correct top-K reference behavior.
"""
import pytest
from langchain_core.documents import Document

from rag.references import format_reference, select_top_references


def test_format_reference_with_subsection():
    """
    Verifies that format_reference correctly formats a reference string
    when both section and subsection metadata are present.

    Expected behavior:
    - Category is included as the prefix
    - Section and subsection are rendered in a hierarchical format
    - Source file path is appended at the end
    """

    doc = Document(
        page_content="content",
        metadata={
            "category": "faqs",
            "source_file": "faqs/reset.md",
            "section": "Password Reset",
            "subsection": "Email Flow",
        },
    )

    ref = format_reference(doc)

    assert ref == "faqs: Password Reset | § Email Flow | file=faqs/reset.md"


def test_format_reference_without_subsection():
    """
    Ensures format_reference behaves correctly when no subsection
    metadata is provided.

    Expected behavior:
    - Section is included
    - Subsection delimiter is omitted
    - Output remains clean and readable
    """

    doc = Document(
        page_content="content",
        metadata={
            "category": "policies",
            "source_file": "policies/privacy.md",
            "section": "Privacy Policy",
        },
    )

    ref = format_reference(doc)

    assert ref == "policies: Privacy Policy | file=policies/privacy.md"


def test_format_reference_missing_metadata_uses_defaults():
    """
    Validates fallback behavior when required metadata fields
    are missing from the document.

    Expected behavior:
    - 'unknown' category is used
    - Generic document name is applied
    - Placeholder file name is shown
    """

    doc = Document(page_content="content", metadata={})

    ref = format_reference(doc)

    assert ref == "unknown: Unknown Doc | file=unknown_file"


def test_select_top_references_respects_k():
    """
    Confirms that select_top_references:
    - Returns at most k references
    - Preserves the original document ordering
    - Formats each reference correctly
    """

    docs = [
        Document(
            page_content="a",
            metadata={
                "category": "faqs",
                "source_file": "a.md",
                "section": "A",
            },
        ),
        Document(
            page_content="b",
            metadata={
                "category": "policies",
                "source_file": "b.md",
                "section": "B",
            },
        ),
        Document(
            page_content="c",
            metadata={
                "category": "runbooks",
                "source_file": "c.md",
                "section": "C",
            },
        ),
    ]

    refs = select_top_references(docs, k=2)

    assert len(refs) == 2
    assert refs[0].startswith("faqs:")
    assert refs[1].startswith("policies:")


def test_select_top_references_less_docs_than_k():
    """
    Ensures select_top_references behaves safely when the number
    of documents is less than the requested k value.

    Expected behavior:
    - All available documents are returned
    - No padding or duplication occurs
    """

    docs = [
        Document(
            page_content="a",
            metadata={
                "category": "faqs",
                "source_file": "a.md",
                "section": "A",
            },
        )
    ]

    refs = select_top_references(docs, k=3)

    assert len(refs) == 1
