from __future__ import annotations

import re
from typing import Sequence

from .retrieval import RetrievedChunk


SYSTEM_INSTRUCTION = """You are an academic question-answering assistant.
Answer ONLY from the supplied source passages.
Do not use outside facts.
If the sources are insufficient, explicitly say that the evidence is insufficient.
Do not invent citations.
Keep the answer concise and suitable for an MSc student.
"""


def format_context(retrieved: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source {i} | Page {item.page}]\n{item.text}"
        for i, item in enumerate(retrieved, start=1)
    )


def generate_with_openai(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    api_key: str,
    model: str,
) -> str:
    """Generate a grounded response with the OpenAI Responses API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    context = format_context(retrieved)

    prompt = f"""{SYSTEM_INSTRUCTION}

SOURCE PASSAGES
{context}

QUESTION
{question}

Return a direct answer. Do not add a reference list because the interface displays sources separately.
"""

    response = client.responses.create(model=model, input=prompt)
    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def generate_extractive(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    max_sentences: int = 3,
) -> str:
    """Local fallback so the application can run with no API key."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sentences: list[str] = []
    for item in retrieved:
        parts = re.split(r"(?<=[.!?])\s+", item.text)
        sentences.extend([p.strip() for p in parts if len(p.strip()) >= 25])

    if not sentences:
        return "The retrieved passages do not contain enough textual evidence to answer this question."

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform([question] + sentences)
    scores = cosine_similarity(matrix[0], matrix[1:])[0]

    ids = np.argsort(scores)[::-1][:max_sentences]
    chosen = [sentences[int(i)] for i in ids if scores[int(i)] > 0]

    if not chosen:
        return "The retrieved passages do not provide sufficiently relevant evidence."
    return " ".join(chosen)


def generate_answer(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    api_key: str | None = None,
    model: str = "gpt-5.6-luna",
) -> tuple[str, str]:
    if not retrieved:
        return "No relevant evidence was retrieved.", "no retrieval"

    if api_key:
        try:
            return (
                generate_with_openai(question, retrieved, api_key, model),
                f"OpenAI Responses API: {model}",
            )
        except Exception as exc:
            fallback = generate_extractive(question, retrieved)
            return (
                fallback + f"\n\n[OpenAI call failed; local fallback used: {type(exc).__name__}]",
                "local extractive fallback",
            )

    return generate_extractive(question, retrieved), "local extractive fallback"
