from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class ClaimEvidence:
    claim: str
    page: int | None
    evidence: str
    similarity: float
    label: str


def split_claims(answer: str) -> list[str]:
    answer = re.sub(r"\s+", " ", answer.strip())
    if not answer:
        return []
    parts = re.split(r"(?<=[.!?])\s+", answer)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def evidence_label(similarity: float) -> str:
    """Heuristic HCI display label; NOT a calibrated probability."""
    if similarity >= 0.30:
        return "Strongly Supported"
    if similarity >= 0.12:
        return "Partially Supported"
    return "Unsupported / Check Source"


def align_claim_to_evidence(
    claim: str,
    retrieved: Sequence[RetrievedChunk],
) -> ClaimEvidence:
    if not retrieved:
        return ClaimEvidence(
            claim=claim,
            page=None,
            evidence="No evidence retrieved.",
            similarity=0.0,
            label="Unsupported / Check Source",
        )

    evidence_texts = [r.text for r in retrieved]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([claim] + evidence_texts)
    scores = cosine_similarity(matrix[0], matrix[1:])[0]

    idx = int(np.argmax(scores))
    score = float(scores[idx])
    best = retrieved[idx]

    return ClaimEvidence(
        claim=claim,
        page=best.page,
        evidence=best.text,
        similarity=score,
        label=evidence_label(score),
    )


def analyse_answer(
    answer: str,
    retrieved: Sequence[RetrievedChunk],
) -> list[ClaimEvidence]:
    return [align_claim_to_evidence(c, retrieved) for c in split_claims(answer)]
