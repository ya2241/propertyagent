"""Two retrievers over property descriptions: plain keyword filtering vs vector search.

The default embedder is a local LSA model (TF-IDF + truncated SVD, numpy only) so the
project runs offline. Set EMBEDDER=st to use sentence-transformers instead.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

import numpy as np

from .data import Property

STOPWORDS = set(
    "a an and are as at be but by for from has have i in is it its me my near of on or our so that the "
    "their there this to we with you your want need looking something somewhere place home".split()
)


def stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")) if t not in STOPWORDS and len(t) > 1]


class Retriever(Protocol):
    def rank(self, query: str, candidate_ids: list[str], k: int) -> list[tuple[str, float]]: ...


class KeywordRetriever:
    """Baseline: a property matches if it shares tokens with the query; score = share of query tokens hit."""

    def __init__(self, properties: list[Property]):
        self._tokens = {p.id: set(tokenize(p.text)) for p in properties}

    def rank(self, query, candidate_ids, k):
        q = set(tokenize(query))
        if not q:
            return []
        scored = [(pid, len(q & self._tokens[pid]) / len(q)) for pid in candidate_ids]
        return sorted([s for s in scored if s[1] > 0], key=lambda s: -s[1])[:k]


class LSAEmbedder:
    def __init__(self, dims: int = 12):
        self.dims = dims

    def fit(self, docs: list[str]) -> "LSAEmbedder":
        toks = [tokenize(d) for d in docs]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in toks for t in d}))}
        df = np.zeros(len(self.vocab))
        for d in toks:
            for t in set(d):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        X = np.vstack([self._tfidf(d) for d in toks])
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.proj = vt[: self.dims].T
        return self

    def _tfidf(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                v[self.vocab[t]] += 1
        v = np.log1p(v) * self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.vstack([self._tfidf(tokenize(t)) @ self.proj for t in texts])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(name)

    def fit(self, docs):
        return self

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class VectorRetriever:
    def __init__(self, properties: list[Property], embedder=None):
        self.ids = [p.id for p in properties]
        if embedder is None:
            embedder = SentenceTransformerEmbedder() if os.getenv("EMBEDDER") == "st" else LSAEmbedder()
        self.embedder = embedder.fit([p.text for p in properties])
        self.matrix = self.embedder.embed([p.text for p in properties])

    def rank(self, query, candidate_ids, k):
        if not tokenize(query):
            return []
        qv = self.embedder.embed([query])[0]
        sims = {pid: float(self.matrix[i] @ qv) for i, pid in enumerate(self.ids)}
        scored = sorted(((pid, sims[pid]) for pid in candidate_ids), key=lambda s: -s[1])
        return scored[:k]


def build_retriever(mode: str, properties: list[Property]) -> Retriever:
    return KeywordRetriever(properties) if mode == "keyword" else VectorRetriever(properties)
