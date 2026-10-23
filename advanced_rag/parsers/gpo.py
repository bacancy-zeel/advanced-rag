"""Task 1: GPO 2018 Annual Report, the plain-text baseline.

Ordinary prose on two-column pages. What it needs is the basics done
properly: read the columns in order, drop the running footer, heal
hyphenated line breaks, know which section a paragraph sits in, and cut
chunks at sentence boundaries.
"""
from __future__ import annotations

import re

import pymupdf

from ..chunking import Chunk, pack
from ..layout import Block, page_blocks
from ..textfix import clean, join_lines

# The running footer ("2 0 1 8 A N N U A L R E P O R T | page") sits below
# this line on every page.
FOOTER_Y = 740
CHUNK_CHARS = 800


def _is_heading(block: Block) -> bool:
    text = block.text.strip()
    if not text or len(text) > 110 or len(block.lines) > 3:
        return False
    if re.fullmatch(r"[\d\s.,$%()\-–]+", text):  # a figure's number label
        return False
    # Table-of-contents leaders and bold table rows ("Total assets 840,741")
    if "...." in text or re.search(r"\d,\d{3}", text):
        return False
    big = max(l.size for l in block.lines) >= 14
    return big or (all(l.bold for l in block.lines) and not text.endswith((".", ",", ";")))


def _is_aside(block: Block) -> bool:
    return max(l.size for l in block.lines) < 9


def parse(path) -> list[Chunk]:
    doc = pymupdf.open(path)
    # (section, page, paragraph) in reading order.
    paragraphs: list[tuple[str, int, str]] = []
    # Photo captions and sidebar labels are set in small type. Read in place
    # they land mid-sentence ("over 35 | <caption> | million title
    # retrievals", and the sentence often resumes on the next page), so they
    # wait until the body text reaches a sentence end.
    asides: list[tuple[str, int, str]] = []
    section, drop_cap = "Front matter", ""

    def ends_sentence(text: str) -> bool:
        return text.endswith((".", "!", "?", ":", "”", '"'))

    for number, page in enumerate(doc, start=1):
        for block in page_blocks(page, bottom=FOOTER_Y):
            # A drop cap is its own block: "T" here, "he U.S. Government" next.
            if re.fullmatch(r"[A-Z]", block.text.strip()):
                drop_cap = block.text.strip()
                continue
            if _is_heading(block):
                paragraphs.extend(asides)
                asides.clear()
                section = clean(" ".join(l.text for l in block.lines))
                continue
            text = clean(drop_cap + join_lines([l.text for l in block.lines]))
            drop_cap = ""
            if len(text) <= 2:
                continue
            if _is_aside(block):
                asides.append((section, number, text))
                continue
            # A paragraph broken by a column or page turn resumes in a new
            # block; glue it back so its sentence is whole again.
            prev = paragraphs[-1] if paragraphs else None
            if (prev and prev[0] == section and not ends_sentence(prev[2])
                    and text[:1].islower()):
                paragraphs[-1] = (section, prev[1], f"{prev[2]} {text}")
            else:
                paragraphs.append((section, number, text))
            if ends_sentence(text):
                paragraphs.extend(asides)
                asides.clear()
    paragraphs.extend(asides)

    chunks, run = [], []

    def flush():
        if not run:
            return
        name = run[0][0]
        for page, text in pack([(p, t) for _, p, t in run], CHUNK_CHARS):
            chunks.append(Chunk(text=text, page=page, section=[name],
                                meta={"section": name}))
        run.clear()

    for item in paragraphs:
        if run and item[0] != run[0][0]:
            flush()
        run.append(item)
    flush()
    return chunks
