from __future__ import annotations

import re
from html import escape
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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


def split_sentences(text: str) -> list[str]:
    """Split source text into displayable sentences without extra dependencies."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def select_evidence_sentences(claim: str, chunk_text: str, max_sentences: int = 2) -> str:
    """Return the one or two sentences in a chunk most relevant to a claim."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")

    sentences = split_sentences(chunk_text)
    if not sentences:
        return chunk_text.strip()
    if len(sentences) == 1:
        return sentences[0]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([claim] + sentences)
    except ValueError:
        return sentences[0]
    scores = cosine_similarity(matrix[0], matrix[1:])[0]
    ranked = np.argsort(scores)[::-1]
    best_score = float(scores[int(ranked[0])])
    selected = [int(ranked[0])]

    # Include a second sentence only when it adds meaningful claim overlap.
    if max_sentences > 1 and len(ranked) > 1 and best_score > 0:
        second = int(ranked[1])
        if float(scores[second]) >= max(0.05, best_score * 0.35):
            selected.append(second)

    return " ".join(sentences[index] for index in sorted(selected[:max_sentences]))


def highlight_evidence(claim: str, evidence: str) -> str:
    """Return escaped evidence HTML with matching claim terms marked."""
    claim_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", claim)
        if len(token) >= 3 and token.lower() not in ENGLISH_STOP_WORDS
    }
    token_pattern = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
    parts: list[str] = []
    cursor = 0
    for match in token_pattern.finditer(evidence):
        parts.append(escape(evidence[cursor:match.start()]))
        token = escape(match.group(0))
        if match.group(0).lower() in claim_terms:
            parts.append(f"<mark>{token}</mark>")
        else:
            parts.append(token)
        cursor = match.end()
    parts.append(escape(evidence[cursor:]))
    return "".join(parts)


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
        evidence=select_evidence_sentences(claim, best.text),
        similarity=score,
        label=evidence_label(score),
    )


def analyse_answer(
    answer: str,
    retrieved: Sequence[RetrievedChunk],
) -> list[ClaimEvidence]:
    return [align_claim_to_evidence(c, retrieved) for c in split_claims(answer)]
