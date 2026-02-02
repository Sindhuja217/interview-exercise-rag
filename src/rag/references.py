"""
Formats and selects human-readable references from retrieved
support documents for inclusion in final responses.
"""
from typing import List
from langchain_core.documents import Document


def format_reference(doc: Document) -> str:
    """
    Stable, human-readable reference string with section disambiguation.
    """
    category = doc.metadata.get("category", "unknown")
    source_file = doc.metadata.get("source_file", "unknown_file")

    section = doc.metadata.get("section", "Unknown Doc")
    subsection = doc.metadata.get("subsection")

    parts = [f"{category}: {section}"]

    if subsection:
        parts.append(f"§ {subsection}")

    parts.append(f"file={source_file}")

    return " | ".join(parts)


def select_top_references(docs: List[Document], k: int = 3) -> List[str]:
    """
    Deterministically select top-K references from reranked docs.
    """
    refs = []
    for doc in docs[:k]:
        refs.append(format_reference(doc))
    return refs
