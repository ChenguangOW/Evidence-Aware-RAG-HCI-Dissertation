from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.document import Page, chunk_pages
from evidence_rag.retrieval import TfidfRetriever
from evidence_rag.generation import generate_answer
from evidence_rag.evidence import (
    analyse_answer,
    evidence_label,
    highlight_evidence,
    select_evidence_sentences,
    split_claims,
    split_sentences,
)


def test_chunking():
    chunks = chunk_pages(
        [Page(1, "user centred design focuses on understanding users and evaluating interfaces")]
    )
    assert len(chunks) == 1
    assert chunks[0].page == 1


def test_retrieval():
    chunks = chunk_pages(
        [
            Page(1, "User centred design focuses on users, tasks and environments."),
            Page(2, "Database indexing can improve relational query performance."),
        ]
    )
    r = TfidfRetriever()
    r.build(chunks)
    result = r.retrieve("What does user centred design focus on?", top_k=1)
    assert result[0].page == 1


def test_claim_split():
    claims = split_claims("Users should be understood. Evaluation is important.")
    assert len(claims) == 2


def test_evidence_labels():
    assert evidence_label(0.50) == "Strongly Supported"
    assert evidence_label(0.20) == "Partially Supported"
    assert evidence_label(0.01) == "Unsupported / Check Source"


def test_generation_without_api():
    chunks = chunk_pages(
        [Page(1, "User-centred design focuses on users and their needs. Evaluation helps identify usability problems.")]
    )
    r = TfidfRetriever()
    r.build(chunks)
    retrieved = r.retrieve("What is user-centred design?", top_k=1)
    answer, mode = generate_answer("What is user-centred design?", retrieved)
    assert len(answer) > 10
    assert mode == "local extractive fallback"


def test_evidence_analysis():
    chunks = chunk_pages(
        [Page(3, "User-centred design starts with understanding users and their needs.")]
    )
    r = TfidfRetriever()
    r.build(chunks)
    retrieved = r.retrieve("users", top_k=1)
    analysis = analyse_answer(
        "User-centred design starts with understanding users.",
        retrieved,
    )
    assert len(analysis) == 1
    assert analysis[0].page == 3


def test_sentence_level_evidence_selects_only_relevant_sentences():
    chunk = (
        "User-centred design focuses on users, their goals, tasks, and context of use. "
        "Database indexes improve relational query performance. "
        "Design teams study users throughout an iterative design process."
    )
    evidence = select_evidence_sentences(
        "User-centred design focuses on users and uses an iterative process.",
        chunk,
    )
    assert "focuses on users" in evidence
    assert "iterative design process" in evidence
    assert "Database indexes" not in evidence
    assert len(split_sentences(evidence)) == 2


def test_sentence_level_evidence_keeps_page_and_support_label():
    chunks = chunk_pages(
        [
            Page(
                7,
                "User-centred design focuses on users and their needs. "
                "Unrelated material discusses database indexing.",
            )
        ]
    )
    r = TfidfRetriever()
    r.build(chunks)
    analysis = analyse_answer(
        "User-centred design focuses on users and their needs.",
        r.retrieve("user-centred design", top_k=1),
    )
    assert analysis[0].page == 7
    assert analysis[0].label == "Strongly Supported"
    assert "database indexing" not in analysis[0].evidence.lower()


def test_highlighting_marks_matches_and_escapes_source_html():
    highlighted = highlight_evidence(
        "Design focuses on users.",
        "Design <script>alert('x')</script> focuses on users.",
    )
    assert "<mark>Design</mark>" in highlighted
    assert "<mark>focuses</mark>" in highlighted
    assert "<mark>users</mark>" in highlighted
    assert "<mark>on</mark>" not in highlighted
    assert "<script>" not in highlighted
    assert "&lt;script&gt;" in highlighted
