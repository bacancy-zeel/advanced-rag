"""Grounded answers: cite the numbered passages, or refuse.

A refusal is a correct answer when the document does not hold the fact. The
numbered passages make every claim in an answer checkable against a page.
"""
from __future__ import annotations

from .llm import generate
from .retrieve import Hit

NOT_FOUND = "I could not find this in the document."

SYSTEM = f"""You answer questions about one document: {{title}}.
Use ONLY the numbered passages below. They are excerpts of that document,
each labelled with its page and section.

Rules:
- Cite every fact with its passage number in brackets, e.g. "6.3 percent [2]";
  for several passages write "[1][4]".
- Quote numbers exactly as the passage gives them, with their units and the
  year or period they refer to. Do not round, convert or recompute them,
  except to add or subtract figures that the passages themselves give, and
  then show the arithmetic.
- Keep facts from different sections apart. If passages describe different
  regions, products or years, answer only for the one asked about.
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
  If they contain part of it, give that part and say what is missing.
- Never use knowledge from outside the passages.
- Be concise: a direct answer first, then a sentence or two of support."""


def context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        label = f"[{hit.rank}] page {hit.page}"
        if hit.heading:
            label += f" · {hit.heading}"
        if hit.kind != "text":
            label += f" · {hit.kind}"
        blocks.append(f"{label}\n{hit.context}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, hits: list[Hit], title: str) -> str:
    if not hits:
        return NOT_FOUND
    prompt = f"PASSAGES\n\n{context(hits)}\n\nQUESTION\n{question}"
    return generate(prompt, system=SYSTEM.format(title=title))
