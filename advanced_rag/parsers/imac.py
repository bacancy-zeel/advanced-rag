"""Task 3: Apple iMac Product Environmental Report, short and visual.

Designed like a brochure: facts sit in tables, call-outs, a bar chart and
numbered endnotes. Text extraction gets every word but loses which value
belongs to which column, which caveat belongs to which claim, and what the
chart adds up to. Each of those is rebuilt here:

- Tables: every table sits under a dark header bar. The bars are found in
  the page's vector drawings and rows are rebuilt from word positions, so
  each value is stored next to its row label and column heading.
- Footnotes: superscript markers are matched to the endnotes on pages 11-12,
  and the endnote text travels inside the chunk that makes the claim.
- Charts: a vision model describes each page's visual content once, at
  index time (see `vision.py`), and that description is indexed too.
- One chunk per element (a call-out, a paragraph, a table), never a
  character-count cut through the middle of one.
"""
from __future__ import annotations

import re

import pymupdf

from ..chunking import Chunk, pack
from ..layout import Block, Line, page_blocks
from ..textfix import clean, join_lines
from .. import vision

TITLE = "iMac Product Environmental Report"
ENDNOTE_PAGES = (11, 12)
FOOTER_Y = 770
COPYRIGHT_Y = 715
ELEMENT_CHARS = 1200


# ---------------------------------------------------------------- footnotes

def _is_marker(span: dict, line_size: float) -> bool:
    text = span["text"].strip()
    # Superscript only: the "2" in CO2e is a subscript, set just as small.
    return bool(re.fullmatch(r"\d{1,2}(,\s?\d{1,2})*", text)) and bool(
        span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT)


def _line_text(line: Line) -> tuple[str, list[int]]:
    """A line's text with superscript markers written as [^n], and the
    endnote numbers it cites."""
    size = max(s["size"] for s in line.spans if s["text"].strip())
    out, notes = "", []
    for span in line.spans:
        if _is_marker(span, size):
            numbers = [int(n) for n in re.findall(r"\d+", span["text"])]
            notes += numbers
            out = out.rstrip() + "".join(f"[^{n}]" for n in numbers) + " "
        else:
            out += span["text"]
    return out.strip(), notes


def _endnotes(doc: pymupdf.Document, skip: list[pymupdf.Rect]) -> dict[int, str]:
    notes: dict[int, list[str]] = {}
    current = None
    for number in ENDNOTE_PAGES:
        page = doc[number - 1]
        for block in page_blocks(page, bottom=COPYRIGHT_Y):
            for line in block.lines:
                if any(pymupdf.Rect(line.bbox).intersects(r) for r in skip):
                    continue
                raw = "".join(s["text"] for s in line.spans)
                m = re.match(r"^\s*(\d{1,2})\s*\t\s*(.*)", raw)
                if m:
                    current = int(m.group(1))
                    notes[current] = [m.group(2)]
                elif current and line.text != "Endnotes":
                    notes[current].append(line.text)
    return {n: clean(join_lines(lines)) for n, lines in notes.items()}


def _with_notes(text: str, cited: list[int], notes: dict[int, str]) -> str:
    seen = [n for i, n in enumerate(cited) if n in notes and n not in cited[:i]]
    if not seen:
        return text
    return text + "\n" + "\n".join(f"[^{n}] {notes[n]}" for n in seen)


# ------------------------------------------------------------------- tables

def _header_bars(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Dark filled rectangles wide enough to head a table, with the cells of
    one bar (drawn as separate rectangles) merged back together."""
    rects = sorted(
        (d["rect"] for d in page.get_drawings()
         if d.get("fill") and sum(d["fill"]) / 3 < 0.35 and d["rect"].width > 80),
        key=lambda r: (round(r.y0), r.x0))
    bars: list[pymupdf.Rect] = []
    for rect in rects:
        near = [i for i, b in enumerate(bars) if (b + (-2, -2, 2, 2)).intersects(rect)]
        if near:
            bars[near[0]] = bars[near[0]] | rect
        else:
            bars.append(pymupdf.Rect(rect))
    return bars


def _table(page: pymupdf.Page, bar: pymupdf.Rect,
           stop_y: float) -> tuple[str, pymupdf.Rect, list[int]]:
    """The table under `bar` as text, the area it covers (so its words are
    not indexed a second time as loose text), and the endnotes it cites."""
    lines = [l for b in page_blocks(page) for l in b.lines]
    inside = lambda l: bar.x0 - 5 <= l.bbox[0] and l.bbox[2] <= bar.x1 + 5
    header = [l for l in lines if inside(l)
              and bar.y0 - 1 <= (l.bbox[1] + l.bbox[3]) / 2 <= bar.y1 + 1]
    body, last_y = [], bar.y1
    for line in sorted((l for l in lines if inside(l) and l.bbox[1] >= bar.y1 - 1),
                       key=lambda l: l.bbox[1]):
        # A table ends at the next bar, a gap, or a line as wide as the
        # table: that is a paragraph after it, not a row.
        if line.bbox[1] >= stop_y or line.bbox[1] - last_y > 24 \
                or line.bbox[2] - line.bbox[0] > 0.6 * bar.width:
            break
        body.append(line)
        last_y = max(last_y, line.bbox[3])

    # Column starts come from the header cells that have nothing beneath
    # them in the header; a cell spanning the others ("Power consumption
    # for iMac" over "100V | 115V | 230V") titles the table instead.
    def spans_others(h):
        return any(o is not h and o.bbox[1] > h.bbox[1] + 2
                   and h.bbox[0] + 15 < o.bbox[0] < h.bbox[2] for o in header)
    leaf = sorted([h for h in header if not spans_others(h)], key=lambda h: h.bbox[0])
    columns: list[list] = []
    for h in leaf:
        if columns and h.bbox[0] - columns[-1][0] < 15:
            columns[-1][1].append(h.text)       # a header cell set on two lines
        else:
            columns.append([h.bbox[0], [h.text]])
    names = [clean(" ".join(texts)) for _, texts in columns]
    title = " ".join(clean(h.text) for h in header if spans_others(h))

    rows: list[list[Line]] = []
    for line in body:
        centre = (line.bbox[1] + line.bbox[3]) / 2
        if rows and abs(centre - (rows[-1][0].bbox[1] + rows[-1][0].bbox[3]) / 2) < 3:
            rows[-1].append(line)
        else:
            rows.append([line])

    cited, out = [], []
    for row in rows:
        cells: dict[int, str] = {}
        for line in sorted(row, key=lambda l: l.bbox[0]):
            text, refs = _line_text(line)
            cited += refs
            col = max([i for i, (x, _) in enumerate(columns) if x <= line.bbox[0] + 8],
                      default=0)
            cells[col] = clean(f"{cells.get(col, '')} {text}")
        label = cells.pop(0, "")
        if not cells:
            out.append(label)           # a note under the table
        elif len(names) == 2:
            out.append(f"{label} ({names[1]}): {cells.get(1, '')}")
        else:
            out.append(f"{label}: " + "; ".join(
                f"{names[c] if c < len(names) else ''} = {v}" for c, v in sorted(cells.items())))
    head = f"Table: {title + ' — ' if title else ''}{' | '.join(names)}"
    region = pymupdf.Rect(bar.x0, bar.y0, bar.x1, last_y)
    return head + "\n" + "\n".join(out), region, cited


# ----------------------------------------------------------------- elements

def _page_title(blocks: list[Block]) -> str:
    big = [l for b in blocks for l in b.lines if l.size >= 20]
    if not big:
        return ""
    size = max(l.size for l in big)
    return clean(" ".join(l.text for l in big if l.size == size))


def _elements(blocks: list[Block], skip: list[pymupdf.Rect]):
    """Text elements in reading order. On a brochure page one call-out is
    often several blocks (its big number, its heading, each line of its
    text), so blocks stacked closely in the same column are one element."""
    kept = [b for b in blocks
            if not any(pymupdf.Rect(b.bbox).intersects(r) for r in skip)]
    # Union blocks that sit in the same column with less than a line's gap
    # between them, wherever the reading order happened to put them.
    parent = list(range(len(kept)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, a in enumerate(kept):
        for j, b in enumerate(kept):
            ra, rb = pymupdf.Rect(a.bbox), pymupdf.Rect(b.bbox)
            overlap = min(ra.x1, rb.x1) - max(ra.x0, rb.x0)
            if i != j and overlap > 0.3 * min(ra.width, rb.width) \
                    and -2 <= rb.y0 - ra.y1 < 16:
                parent[root(j)] = root(i)
    members: dict[int, list[Block]] = {}
    for i, block in enumerate(kept):
        members.setdefault(root(i), []).append(block)
    groups = sorted((sorted(g, key=lambda b: b.bbox[1]) for g in members.values()),
                    key=lambda g: (round(g[0].bbox[1] / 20), g[0].bbox[0]))

    out = []
    for group in groups:
        parts, cited = [], []
        for line in (l for b in group for l in b.lines):
            text, refs = _line_text(line)
            parts.append(text)
            cited += refs
        text = clean(join_lines(parts))
        # Page numbers, the running footer, and bare labels ("Transportation",
        # a chart segment's name) that say nothing until the chart is described
        if len(text) < 40 and not re.search(r"\d", text):
            continue
        if re.fullmatch(r"iMac \| Product Environmental Report \d{1,2}", text):
            continue
        out.append((text, cited))
    return out


def parse(path, use_vision: bool = True) -> list[Chunk]:
    doc = pymupdf.open(path)
    tables: dict[int, list[tuple[str, pymupdf.Rect, list[int]]]] = {}
    for number, page in enumerate(doc, start=1):
        bars = _header_bars(page)
        for i, bar in enumerate(bars):
            stop = bars[i + 1].y0 if i + 1 < len(bars) else page.rect.height
            tables.setdefault(number, []).append(_table(page, bar, stop))

    regions = {n: [r for _, r, _ in t] for n, t in tables.items()}
    notes = _endnotes(doc, regions.get(ENDNOTE_PAGES[1], []))

    chunks: list[Chunk] = []
    for number, page in enumerate(doc, start=1):
        blocks = page_blocks(page, bottom=FOOTER_Y)
        title = _page_title(blocks) if number not in ENDNOTE_PAGES else "Endnotes"
        section = [TITLE, title] if title else [TITLE]
        for text, region, cited in tables.get(number, []):
            chunks.append(Chunk(text=_with_notes(text, cited, notes), page=number,
                                section=section, kind="table"))
        if number in ENDNOTE_PAGES:
            continue
        for text, cited in _elements(blocks, regions.get(number, [])):
            for page_no, piece in pack([(number, text)], ELEMENT_CHARS):
                chunks.append(Chunk(text=_with_notes(piece, cited, notes),
                                    page=page_no, section=section))
        if use_vision:
            description = vision.describe(path, number)
            if description:
                cited = [int(n) for n in re.findall(r"\[\^(\d+)\]", description)]
                chunks.append(Chunk(text=_with_notes(description, cited, notes),
                                    page=number, section=section, kind="chart"))

    # Each endnote is also searchable on its own, for questions about a caveat.
    for n, text in sorted(notes.items()):
        chunks.append(Chunk(text=f"Endnote {n}: {text}", page=ENDNOTE_PAGES[0],
                            section=[TITLE, "Endnotes"], kind="footnote"))
    return chunks
