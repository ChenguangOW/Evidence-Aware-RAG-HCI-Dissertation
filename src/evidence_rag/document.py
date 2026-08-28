from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Sequence


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str


def load_pdf(file_obj: BinaryIO) -> list[Page]:
    """Extract text from a text-based PDF."""
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(index, text))

    if not pages:
        raise ValueError(
            "No extractable text was found. Use a text-based PDF rather than an image-only scan."
        )
    return pages


def load_txt(file_obj: BinaryIO) -> list[Page]:
    raw = file_obj.read()
    if isinstance(raw, str):
        text = raw
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise ValueError("The TXT file is empty.")
    return [Page(1, text)]


def load_uploaded_file(uploaded_file) -> list[Page]:
    """Load a Streamlit UploadedFile or a normal binary file object."""
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    if name.endswith(".txt"):
        return load_txt(uploaded_file)
    raise ValueError("Only PDF and TXT are supported.")


def chunk_pages(
    pages: Sequence[Page],
    chunk_words: int = 180,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Overlapping word-window chunking while preserving page provenance."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require 0 <= overlap_words < chunk_words")

    chunks: list[Chunk] = []
    step = chunk_words - overlap_words

    for page in pages:
        words = page.text.split()
        if not words:
            continue

        if len(words) <= chunk_words:
            chunks.append(
                Chunk(
                    chunk_id=f"p{page.number}-c0",
                    page=page.number,
                    text=" ".join(words),
                )
            )
            continue

        c = 0
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            if len(window) < 30:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"p{page.number}-c{c}",
                    page=page.number,
                    text=" ".join(window),
                )
            )
            c += 1

    if not chunks:
        raise ValueError("No usable chunks were created.")
    return chunks
