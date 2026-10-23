"""Reading order for multi-column PDF pages.

PyMuPDF returns blocks in content-stream order, which on a designed page is
whatever order the layout tool wrote them in. On two-column reports that
interleaves the columns, so a sentence can end in the left column and resume
three paragraphs later. Here a page is cut into horizontal bands at every
block that spans both columns, and inside a band the left column is read top
to bottom before the right one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf


@dataclass
class Line:
    text: str
    font: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]
    spans: list[dict] = field(repr=False, default_factory=list)


@dataclass
class Block:
    lines: list[Line]
    bbox: tuple[float, float, float, float]

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def _is_bold(span: dict) -> bool:
    font = span["font"]
    return bool(span["flags"] & 16) or any(
        mark in font for mark in ("Bold", "Black", "-Md", "Heavy", "Semibold"))


def page_blocks(page: pymupdf.Page, top: float = 0, bottom: float | None = None,
                ) -> list[Block]:
    """Text blocks between `top` and `bottom` (to drop running headers and
    footers), with their lines and the first span's font, in reading order."""
    bottom = page.rect.height if bottom is None else bottom
    blocks = []
    for raw in page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if raw.get("type") != 0:
            continue
        x0, y0, x1, y1 = raw["bbox"]
        if y1 <= top or y0 >= bottom:
            continue
        lines = []
        for l in raw["lines"]:
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            first = max(spans, key=lambda s: len(s["text"].strip()))
            lines.append(Line(
                text="".join(s["text"] for s in l["spans"]).strip(),
                font=first["font"], size=round(first["size"], 1),
                bold=all(_is_bold(s) for s in spans),
                bbox=tuple(l["bbox"]), spans=l["spans"],
            ))
        if lines:
            blocks.append(Block(lines, (x0, y0, x1, y1)))
    return reading_order(blocks, page.rect.width)


def reading_order(blocks: list[Block], width: float) -> list[Block]:
    mid = width / 2
    spanning = lambda b: b.bbox[0] < mid - 20 and b.bbox[2] > mid + 20

    ordered, band = [], []

    def flush():
        left = [b for b in band if (b.bbox[0] + b.bbox[2]) / 2 < mid]
        right = [b for b in band if (b.bbox[0] + b.bbox[2]) / 2 >= mid]
        ordered.extend(sorted(left, key=lambda b: b.bbox[1]))
        ordered.extend(sorted(right, key=lambda b: b.bbox[1]))
        band.clear()

    for block in sorted(blocks, key=lambda b: b.bbox[1]):
        if spanning(block):
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered
