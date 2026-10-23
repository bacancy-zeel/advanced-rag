"""The three documents' pipelines, and the naive baseline they are measured against.

The naive pipeline is the textbook one: the PDF reader's plain text, cut
every 1,000 characters, dense search only. Each document's advanced pipeline
swaps in its own parser and switches on the retrieval stages that measured
well on it (see `evaluate.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pymupdf

from . import config, route, store
from .chunking import Chunk, fixed_size
from .generate import answer
from .parsers import gep, gpo, imac
from .retrieve import Hit, Settings, search


@dataclass
class Pipeline:
    key: str
    title: str
    parse: Callable[[object], list[Chunk]]
    settings: Settings
    router: Callable[[str], dict] | None = None
    after_parse: Callable[[list[Chunk]], object] | None = None

    @property
    def collection(self) -> str:
        return self.key


@dataclass
class Result:
    question: str
    answer: str
    hits: list[Hit]
    where: dict = field(default_factory=dict)


def naive_parse(path) -> list[Chunk]:
    chunks = []
    for number, page in enumerate(pymupdf.open(path), start=1):
        for piece in fixed_size(page.get_text()):
            if piece.strip():
                chunks.append(Chunk(text=piece, page=number))
    return chunks


TITLES = {
    "gpo": "U.S. Government Publishing Office (GPO) 2018 Annual Report",
    "gep": "World Bank, Global Economic Prospects, June 2026",
    "imac": "Apple iMac Product Environmental Report (October 2024)",
}

PIPELINES = {
    "gpo": Pipeline("gpo", TITLES["gpo"], gpo.parse,
                    Settings(hybrid=True, rerank=True)),
    "gep": Pipeline("gep", TITLES["gep"], gep.parse,
                    Settings(hybrid=True, rerank=True, parents=True),
                    router=route.route, after_parse=route.save_countries),
    "imac": Pipeline("imac", TITLES["imac"], imac.parse,
                     Settings(hybrid=True, rerank=False)),
}


def naive(key: str) -> Pipeline:
    return Pipeline(f"{key}_naive", TITLES[key], naive_parse,
                    Settings(hybrid=False, rerank=False))


def get(key: str, use_naive: bool = False) -> Pipeline:
    return naive(key) if use_naive else PIPELINES[key]


def ingest(pipeline: Pipeline, doc_key: str) -> int:
    chunks = pipeline.parse(config.DOCUMENTS[doc_key])
    if pipeline.after_parse:
        pipeline.after_parse(chunks)
    return store.build(pipeline.collection, chunks)


def retrieve(pipeline: Pipeline, question: str, settings: Settings | None = None,
             use_router: bool = True) -> tuple[list[Hit], dict]:
    where = pipeline.router(question) if pipeline.router and use_router else {}
    return search(pipeline.collection, question, settings or pipeline.settings, where), where


def ask(pipeline: Pipeline, question: str) -> Result:
    hits, where = retrieve(pipeline, question)
    return Result(question, answer(question, hits, pipeline.title), hits, where)
