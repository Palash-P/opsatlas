"""Deterministic, dependency-free text normalization and chunking."""

from __future__ import annotations


def normalize_document_text(content: str) -> str:
    """Normalize line endings and surrounding whitespace before hashing/storage."""
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def chunk_text(content: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into deterministic, overlapping character chunks.

    This is an intentionally simple baseline. It prefers whitespace boundaries when
    possible, but character counts are not token counts. We will evaluate and improve
    this strategy against the golden dataset before selecting a production chunker.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = normalize_document_text(content)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            # Avoid tiny chunks by only looking for a boundary in the final 35%.
            boundary_floor = start + int(chunk_size * 0.65)
            newline_boundary = text.rfind("\n", boundary_floor, hard_end)
            space_boundary = text.rfind(" ", boundary_floor, hard_end)
            boundary = max(newline_boundary, space_boundary)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks
