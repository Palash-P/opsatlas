import pytest

from app.services.chunking import chunk_text, normalize_document_text


def test_normalization_makes_windows_and_unix_line_endings_equivalent() -> None:
    assert normalize_document_text("  first\r\nsecond\r\n") == "first\nsecond"
    assert normalize_document_text("  first\nsecond\n") == "first\nsecond"


def test_short_document_is_one_chunk() -> None:
    assert chunk_text("hello world", chunk_size=20, overlap=3) == ["hello world"]


def test_long_document_is_split_deterministically_with_overlap() -> None:
    content = ("alpha beta gamma delta " * 20).strip()
    first = chunk_text(content, chunk_size=80, overlap=10)
    second = chunk_text(content, chunk_size=80, overlap=10)

    assert first == second
    assert len(first) > 1
    assert all(0 < len(chunk) <= 80 for chunk in first)
    # Overlap means neighboring chunks share some content.
    assert any(set(first[i].split()) & set(first[i + 1].split()) for i in range(len(first) - 1))


def test_chunking_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=10)


def test_blank_content_produces_no_chunks() -> None:
    assert chunk_text(" \n \t ") == []
