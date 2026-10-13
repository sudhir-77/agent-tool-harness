"""Tool retrieval: BM25, FAISS embeddings, and hierarchical app-then-tool."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from harness.registry import ToolSpec


# No trailing word boundary: the slot often ends on a quote, which is not a word char.
_SLOT = re.compile(r"\b[a-z_]+=(?:'[^']*'|-?\d+)")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieval_query(text: str) -> str:
    """Drop key=value slots so field names like 'note' do not vote for an app."""
    cleaned = _SLOT.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return up to k tool names, best first."""


def _rank_by_score(tools: list[ToolSpec], scores: list[float] | np.ndarray) -> list[int]:
    return sorted(range(len(tools)), key=lambda i: (-float(scores[i]), tools[i].name))


class BM25Retriever:
    name = "bm25"

    def __init__(self, tools: list[ToolSpec]) -> None:
        self.tools = list(tools)
        docs = [tokenize(tool.retrieval_text) for tool in self.tools]
        self._bm25 = BM25Okapi(docs)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self._bm25.get_scores(tokenize(retrieval_query(query)))
        order = _rank_by_score(self.tools, scores)
        return [self.tools[i].name for i in order[:k]]


class HierarchicalRetriever:
    """Rank apps with BM25, then rank tools inside the top two apps."""

    name = "hierarchical"

    def __init__(self, tools: list[ToolSpec]) -> None:
        self.tools = list(tools)
        app_terms: dict[str, list[str]] = {}
        for tool in self.tools:
            app_terms.setdefault(tool.app, [])
            app_terms[tool.app].append(tool.app)
            app_terms[tool.app].extend(tokenize(tool.phrase))
        self.apps = sorted(app_terms)
        app_docs = [sorted(set(app_terms[app])) for app in self.apps]
        self._app_bm25 = BM25Okapi(app_docs)
        docs = [tokenize(tool.retrieval_text) for tool in self.tools]
        self._tool_bm25 = BM25Okapi(docs)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q = tokenize(retrieval_query(query))
        app_scores = self._app_bm25.get_scores(q)
        app_order = sorted(range(len(self.apps)), key=lambda i: (-float(app_scores[i]), self.apps[i]))
        top_apps = {self.apps[i] for i in app_order[:2]}
        tool_scores = self._tool_bm25.get_scores(q)
        candidates = [i for i, tool in enumerate(self.tools) if tool.app in top_apps]
        candidates.sort(key=lambda i: (-float(tool_scores[i]), self.tools[i].name))
        return [self.tools[i].name for i in candidates[:k]]


class FaissEmbeddingRetriever:
    name = "embed"

    def __init__(self, tools: list[ToolSpec], vectors: np.ndarray) -> None:
        import faiss

        faiss.omp_set_num_threads(4)
        self.tools = list(tools)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.tools):
            raise ValueError("embedding matrix must have one row per tool")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search_vector(self, vector: np.ndarray, k: int = 5) -> list[str]:
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._index.d:
            raise ValueError("query dimension does not match the index")
        import faiss

        faiss.normalize_L2(query)
        take = min(k, len(self.tools))
        scores, ids = self._index.search(query, take)
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if int(i) >= 0]
        pairs.sort(key=lambda pair: (-round(pair[1], 6), self.tools[pair[0]].name))
        return [self.tools[i].name for i, _ in pairs]


class SentenceTransformerEncoder:
    """Frozen MiniLM encoder. Documents are encoded once; queries are cached."""

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.device = device
        self.model = SentenceTransformer(self.MODEL_ID, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self.encode_documents([text])[0]
        self._cache[text] = vector
        return vector


class EmbeddingRetriever:
    """FAISS retriever that encodes queries with a provided encoder."""

    name = "embed"

    def __init__(self, tools: list[ToolSpec], encoder: SentenceTransformerEncoder | _ArrayEncoder) -> None:
        self.tools = list(tools)
        self.encoder = encoder
        vectors = encoder.encode_documents([tool.retrieval_text for tool in self.tools])
        self._faiss = FaissEmbeddingRetriever(tools, vectors)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        vector = self.encoder.encode_query(retrieval_query(query))
        return self._faiss.search_vector(vector, k=k)


class _ArrayEncoder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class FrozenRetriever:
    """Replay rankings captured before a heavy encoder is released."""

    def __init__(self, name: str, mapping: dict[str, list[str]]) -> None:
        self.name = name
        self._mapping = {key: list(value) for key, value in mapping.items()}

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return list(self._mapping.get(query, []))[:k]
