"""A local web page for the three pipelines: pick a document, ask, compare.

    python -m advanced_rag serve --port 8001   ->  http://localhost:8001

`GET /` is the page; `POST /api/ask` is the JSON endpoint it calls, also
browsable at `/docs`. Endpoints are plain `def`: embedding, reranking and the
Gemini call all block, and FastAPI runs sync endpoints in a worker thread.

Qdrant's local mode lets one process open the index at a time, so stop the
server before running `ingest` or `eval`, and the other way round.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import markdown
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config, pipeline, route, store
from .llm import LLMError

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "templates" / "index.html").read_text()

app = FastAPI(title="Advanced RAG challenge",
              description="Ask the GPO, World Bank and iMac reports, with the "
                          "naive baseline or the advanced pipeline.")


@app.on_event("startup")
def warm_up() -> None:
    """Open the index and load the three local models before the first
    question, so it is not the one that waits ~15s. If another process holds
    the index, the server fails here, at launch, instead of on a request."""
    store.client()
    store.dense_model(), store.sparse_model(), store.reranker()


class Ask(BaseModel):
    doc: Literal["gpo", "gep", "imac"]
    question: str = Field(min_length=3, max_length=500)
    naive: bool = False


def _examples(doc: str) -> list[dict]:
    items = json.loads((HERE / "questions" / f"{doc}.json").read_text())
    return [{"q": i["q"], "sample": i.get("sample", False)} for i in items]


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE


@app.get("/api/documents")
def documents() -> list[dict]:
    return [{"key": key, "title": pipeline.TITLES[key], "examples": _examples(key)}
            for key in config.DOCUMENTS]


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    pipe = pipeline.get(body.doc, body.naive)
    try:
        result = pipeline.ask(pipe, body.question.strip())
    except LLMError as error:
        raise HTTPException(502, f"The answer model failed: {error}") from error
    except RuntimeError as error:
        # Most often the index lock: another process (ingest/eval) holds it.
        raise HTTPException(503, str(error).splitlines()[0]) from error
    return {
        "answer": result.answer,
        "answer_html": markdown.markdown(result.answer),
        "filter": route.describe(result.where) if pipe.router else None,
        "pipeline": "naive" if body.naive else "advanced",
        "sources": [{
            "rank": h.rank, "page": h.page, "kind": h.kind,
            "heading": h.heading, "score": round(h.score, 3), "text": h.context,
        } for h in result.hits],
    }
