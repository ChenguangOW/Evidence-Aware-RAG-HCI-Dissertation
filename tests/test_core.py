from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.document import Page, chunk_pages
from evidence_rag.retrieval import TfidfRetriever
from evidence_rag.generation import generate_answer
from evidence_rag.evidence import split_claims, evidence_label, analyse_answer


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
