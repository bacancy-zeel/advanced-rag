"""Measure each pipeline on hand-checked questions.

Retrieval: a passage counts as the answer passage when what the model would
read from it (the parent section, if any) contains every evidence string.
MRR is the mean of 1/rank of the first such passage (0 when none is in the
top k); Hit@k is the share of questions with one in the top k.

Answers: an answer passes when it contains every required term (a term may
list alternatives as "a|b"), or, for a question the document cannot answer,
when it says so.

Techniques are switched on one at a time over the same index, so each row
shows what one change bought.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import replace

from . import config, pipeline
from .llm import LLMError
from .retrieve import Hit, Settings

QUESTIONS_DIR = config.ROOT / "questions"
RESULTS_DIR = config.ROOT / "results"


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("₂", "2")
    text = re.sub(r"\[\^\d+\]", "", text)
    return re.sub(r"\s+", " ", text)


def _contains(text: str, term: str) -> bool:
    return any(_norm(alt) in _norm(text) for alt in term.split("|"))


def answer_rank(hits: list[Hit], evidence: list[str]) -> int:
    for hit in hits:
        if all(_contains(hit.context, e) for e in evidence):
            return hit.rank
    return 0


def answer_ok(item: dict, text: str) -> bool:
    if item.get("refuse"):
        return "could not find" in text.lower() or "does not" in text.lower()
    return all(_contains(text, term) for term in item["answer"])


def load(doc: str) -> list[dict]:
    return json.loads((QUESTIONS_DIR / f"{doc}.json").read_text())


def variants(doc: str) -> list[tuple[str, pipeline.Pipeline, Settings, bool]]:
    """(label, pipeline, settings, use router) per row of the table."""
    adv = pipeline.get(doc)
    base = replace(adv.settings, hybrid=False, rerank=False)
    rows = [
        ("naive: plain text, 1000-char chunks, dense", pipeline.naive(doc),
         Settings(hybrid=False, rerank=False), False),
        ("parsed + structure-aware chunks, dense", adv, base, False),
        ("+ hybrid (dense + BM25, RRF)", adv, replace(base, hybrid=True), False),
        ("+ cross-encoder rerank", adv, replace(base, hybrid=True, rerank=True), False),
    ]
    if adv.router:
        rows.append(("+ metadata routing (region/chapter filter)", adv,
                     replace(base, hybrid=True, rerank=True), True))
    rows.append(("advanced (shipped config)", adv, adv.settings, True))
    return rows


def retrieval(doc: str) -> list[dict]:
    items = [q for q in load(doc) if not q.get("refuse")]
    rows = []
    for label, pipe, settings, use_router in variants(doc):
        ranks = []
        for item in items:
            hits, _ = pipeline.retrieve(pipe, item["q"], settings, use_router)
            ranks.append(answer_rank(hits, item["evidence"]))
        rows.append({
            "variant": label,
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
            "hit": round(sum(1 for r in ranks if r) / len(ranks), 3),
            "ranks": ranks,
        })
    return rows


def answers(doc: str, pause: float = 0.0) -> list[dict]:
    out = []
    for item in load(doc):
        row = {"q": item["q"], "sample": item.get("sample", False)}
        for name, use_naive in (("naive", True), ("advanced", False)):
            pipe = pipeline.get(doc, use_naive)
            try:
                result = pipeline.ask(pipe, item["q"])
                text = result.answer
                rank = answer_rank(result.hits, item["evidence"]) if item["evidence"] else None
            except LLMError as error:
                text, rank = f"ERROR: {error}", None
            row[name] = {"answer": text, "ok": answer_ok(item, text), "rank": rank}
            time.sleep(pause)
        out.append(row)
    return out


def report(doc: str, retr: list[dict], ans: list[dict] | None) -> str:
    lines = [f"## {pipeline.TITLES[doc]}", "",
             f"Retrieval, {len(retr[0]['ranks'])} answerable questions, top {config.TOP_K}:", "",
             "| Pipeline | MRR | Hit@5 |", "|---|---|---|"]
    lines += [f"| {r['variant']} | {r['mrr']:.2f} | {r['hit']:.0%} |" for r in retr]
    if ans:
        naive_ok = sum(r["naive"]["ok"] for r in ans)
        adv_ok = sum(r["advanced"]["ok"] for r in ans)
        lines += ["", f"Answers: naive {naive_ok}/{len(ans)}, advanced {adv_ok}/{len(ans)}", "",
                  "| Question | Naive | Advanced |", "|---|---|---|"]
        for r in ans:
            mark = lambda v: "pass" if v["ok"] else "**fail**"
            star = " (sample)" if r["sample"] else ""
            lines.append(f"| {r['q']}{star} | {mark(r['naive'])} | {mark(r['advanced'])} |")
    return "\n".join(lines)


def run(docs: list[str], with_answers: bool, pause: float) -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    sections = []
    for doc in docs:
        retr = retrieval(doc)
        ans = answers(doc, pause) if with_answers else None
        (RESULTS_DIR / f"{doc}.json").write_text(json.dumps(
            {"retrieval": retr, "answers": ans}, indent=2, ensure_ascii=False))
        sections.append(report(doc, retr, ans))
    text = "\n\n".join(sections)
    if with_answers and len(docs) == len(config.DOCUMENTS):
        (RESULTS_DIR / "scorecard.md").write_text("# Scorecard\n\n" + text + "\n")
    return text

