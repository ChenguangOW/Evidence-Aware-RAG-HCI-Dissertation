from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .document import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    page: int
    text: str
    score: float


class TfidfRetriever:
    """Portable retrieval backend used as a guaranteed fallback."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=30000,
        )
        self._matrix = None
        self._chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return "TF-IDF + cosine similarity"

    def build(self, chunks: Sequence[Chunk]) -> None:
        from sklearn.preprocessing import normalize
        self._chunks = list(chunks)
        if not self._chunks:
            raise ValueError("No chunks to index.")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])
        self._matrix = normalize(self._matrix)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity
        if self._matrix is None:
            raise RuntimeError("Index not built.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        k = min(max(top_k, 1), len(self._chunks))
        ids = np.argsort(scores)[::-1][:k]

        return [
            RetrievedChunk(
                chunk_id=self._chunks[int(i)].chunk_id,
                page=self._chunks[int(i)].page,
                text=self._chunks[int(i)].text,
                score=float(scores[int(i)]),
            )
            for i in ids
        ]


class SentenceTransformerFaissRetriever:
    """Preferred dissertation backend: Sentence-BERT embeddings + FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return f"Sentence-Transformers + FAISS ({self.model_name})"

    def _ensure_dependencies(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "Sentence-Transformers/FAISS are not installed. "
                "Install requirements-full.txt or choose the TF-IDF backend."
            ) from exc
        return SentenceTransformer, faiss

    def build(self, chunks: Sequence[Chunk]) -> None:
        SentenceTransformer, faiss = self._ensure_dependencies()
        self._chunks = list(chunks)
        if not self._chunks:
            raise ValueError("No chunks to index.")

        self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(
            [c.text for c in self._chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or self._model is None:
            raise RuntimeError("Index not built.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        q = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        k = min(max(top_k, 1), len(self._chunks))
        scores, ids = self._index.search(q, k)

        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            chunk = self._chunks[int(idx)]
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return out


def make_retriever(backend: str):
    backend = backend.lower().strip()
    if backend in {"sbert", "sentence-transformers", "faiss"}:
        return SentenceTransformerFaissRetriever()
    if backend in {"tfidf", "tf-idf"}:
        return TfidfRetriever()
    raise ValueError(f"Unknown retriever backend: {backend}")
