"""Retrieval: dense, or hybrid fused with RRF, then an optional rerank.

Each stage is a switch so the evaluation can add techniques one at a time and
measure what each one changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import models

from . import config, store


@dataclass
class Settings:
    hybrid: bool = True
    rerank: bool = True
    # Hand the model the chunk's parent section instead of the chunk itself.
    parents: bool = False
    k: int = config.TOP_K


@dataclass
class Hit:
    id: str
    text: str
    page: int
    heading: str
    kind: str
    score: float
    payload: dict = field(repr=False, default_factory=dict)
    rank: int = 0

    @property
    def context(self) -> str:
        """What the model reads: the parent section when there is one."""
        return self.payload.get("parent") or self.text


def _filter(where: dict | None) -> models.Filter | None:
    if not where:
        return None
    return models.Filter(must=[
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in where.items()
    ])


def _candidates(collection: str, question: str, settings: Settings,
                where: dict | None, limit: int) -> list[Hit]:
    dense, sparse = store.query_vectors(question)
    flt = _filter(where)
    db = store.client()
    if settings.hybrid:
        # RRF only looks at ranks, so a cosine similarity and a BM25 score
        # never need a common scale.
        result = db.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=flt,
                                limit=config.PREFETCH),
                models.Prefetch(query=sparse, using="bm25", filter=flt,
                                limit=config.PREFETCH),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        )
    else:
        result = db.query_points(collection, query=dense, using="dense",
                                 query_filter=flt, limit=limit, with_payload=True)
    return [
        Hit(id=str(p.id), text=p.payload["text"], page=p.payload["page"],
            heading=p.payload.get("heading", ""), kind=p.payload.get("kind", "text"),
            score=p.score, payload=p.payload)
        for p in result.points
    ]


def search(collection: str, question: str, settings: Settings | None = None,
           where: dict | None = None) -> list[Hit]:
    settings = settings or Settings()
    pool = config.RERANK_POOL if settings.rerank else settings.k * 3
    hits = _candidates(collection, question, settings, where, pool)
    # A filter is a guess about where the answer lives. When it leaves too
    # little to choose from, fill the pool from the whole document rather
    # than answer from two passages.
    if where and len(hits) < settings.k:
        seen = {h.id for h in hits}
        hits += [h for h in _candidates(collection, question, settings, None, pool)
                 if h.id not in seen]

    if settings.rerank and hits:
        # The cross-encoder reads question and passage together, which is
        # what a vector comparison cannot do.
        scores = store.reranker().rerank(
            question, [f"{h.heading}\n{h.text}" for h in hits])
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        hits.sort(key=lambda h: h.score, reverse=True)

    out, parents_seen = [], set()
    for hit in hits:
        parent = hit.payload.get("parent_id")
        if settings.parents and parent:
            # Two children of one section would hand the model the same
            # parent twice; the first one stands for both.
            if parent in parents_seen:
                continue
            parents_seen.add(parent)
        out.append(hit)
        if len(out) == settings.k:
            break
    for rank, hit in enumerate(out, start=1):
        hit.rank = rank
    return out
