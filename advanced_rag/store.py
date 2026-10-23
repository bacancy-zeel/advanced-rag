"""Qdrant in local mode, with two vectors per chunk.

`dense` is BGE (meaning) and `bm25` is a sparse term vector (exact words,
with IDF applied by Qdrant at query time). Both are computed in-process by
fastembed, so indexing a 200-page report costs no API calls.
"""
from __future__ import annotations

import atexit
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from . import config
from .chunking import Chunk


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    config.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    db = QdrantClient(path=str(config.QDRANT_DIR))
    # Closed explicitly: left to the garbage collector, local mode tries to
    # close while the interpreter is shutting down and prints a traceback.
    atexit.register(db.close)
    return db


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL, cache_dir=config.EMBED_CACHE_DIR)


@lru_cache(maxsize=1)
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(config.RERANK_MODEL, cache_dir=config.EMBED_CACHE_DIR)


def _sparse(vector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices.tolist(),
                               values=vector.values.tolist())


def build(collection: str, chunks: list[Chunk]) -> int:
    """Replace `collection` with `chunks`. Rebuilding is cheap enough that an
    index is never patched in place: a parser change means a fresh build."""
    db = client()
    if db.collection_exists(collection):
        db.delete_collection(collection)
    db.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(
            size=config.EMBED_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    texts = [c.embed_text() for c in chunks]
    dense = dense_model().embed(texts, batch_size=config.EMBED_BATCH)
    sparse = sparse_model().embed(texts, batch_size=config.EMBED_BATCH)
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={"dense": d.tolist(), "bm25": _sparse(s)},
            payload={
                "text": chunk.text, "page": chunk.page, "kind": chunk.kind,
                "section": chunk.section, "heading": chunk.heading,
                "parent": chunk.parent, "parent_id": chunk.parent_id,
                **chunk.meta,
            },
        )
        for chunk, d, s in zip(chunks, dense, sparse)
    ]
    for start in range(0, len(points), 256):
        db.upsert(collection, points[start:start + 256])
    return len(points)


def query_vectors(question: str) -> tuple[list[float], models.SparseVector]:
    """BGE's query side adds its retrieval instruction prefix; passages are
    embedded without it. Mixing the two up costs several points of recall."""
    dense = next(iter(dense_model().query_embed(question))).tolist()
    sparse = _sparse(next(iter(sparse_model().query_embed(question))))
    return dense, sparse
