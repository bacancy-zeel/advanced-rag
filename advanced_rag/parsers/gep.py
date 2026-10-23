"""Task 2: World Bank Global Economic Prospects, June 2026, a long structured report.

Six regions all talk about "growth forecasts", so the words match everywhere
and only a passage's location tells them apart. This parser recovers that
location and makes it searchable:

- Chapters come from the PDF outline. Headings inside a chapter come from
  fonts: body text is Garamond and headings are Helvetica, at sizes that
  give their level.
- Every chunk carries its heading path (chapter > region > section) as
  metadata and in front of the text it is embedded with, plus a `region`
  code and a `chapter` key that a question can be filtered on.
- Boxed articles are kept apart from the chapter text they interrupt.
- Tables and figures are captured as one chunk each, with table rows rebuilt
  from word positions, so their numbers do not break up the prose.
- Prose is indexed as small children for precise matching, each pointing to
  the ~900-token parent section the model reads.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

import pymupdf

from ..chunking import Chunk, pack_sentences
from ..layout import Line, page_blocks
from ..textfix import clean, join_lines

HEADER_Y, FOOTER_Y = 55, 735
CHILD_CHARS, PARENT_CHARS = 1000, 3600

# The codes the report itself uses, so "SAR" in a passage and "South Asia" in
# a question resolve to the same filter value.
REGIONS = {
    "East Asia and Pacific": "EAP",
    "Europe and Central Asia": "ECA",
    "Latin America and the Caribbean": "LAC",
    "Middle East, North Africa, Afghanistan, and Pakistan": "MNA",
    "South Asia": "SAR",
    "Sub-Saharan Africa": "SSA",
}

_CAPTION = re.compile(r"^(TABLE|FIGURE) [A-Z]*\d")
_BOX = re.compile(r"^BOX (\d+\.\d+)\s+(.*?)(\s*\(continued\))?\s*$")


@dataclass
class _Group:
    """A table or figure: its caption line and the small type under it."""
    caption: Line
    page: int
    section: list[str]
    meta: dict
    lines: list[Line] = field(default_factory=list)


def _chapters(doc: pymupdf.Document) -> list[tuple[int, str, str]]:
    """(first page, chapter key, title) from the outline, in page order."""
    out = [(1, "front", "Front matter")]
    for _, title, page in doc.get_toc():
        title = " ".join(title.split())
        if title.startswith("Executive Summary"):
            out.append((page, "summary", "Executive Summary"))
        elif m := re.match(r"Ch\. (\d) (.*)", title):
            out.append((page, m.group(1), f"Chapter {m.group(1)}: {m.group(2)}"))
        elif title.startswith("Statistical"):
            out.append((page, "appendix", "Statistical Appendix"))
    return sorted(out)


def _chapter_at(chapters, page: int) -> tuple[str, str]:
    current = chapters[0]
    for entry in chapters:
        if entry[0] <= page:
            current = entry
    return current[1], current[2]


def _level(line: Line) -> int | None:
    """Heading level from the font, or None for body text."""
    font, size = line.font, line.size
    if "Helvetica" in font and "Cond" in font and size >= 20:
        return 2                                  # Executive Summary, Foreword
    if "Helvetica" in font and line.bold and size >= 14:
        return 2                                  # Outlook, South Asia
    if font.startswith("HelveticaLTStd-Roman") and 10.5 <= size <= 11.5:
        return 3                                  # Recent developments
    if "Garamond" in font and "BoldItalic" in font and size >= 10.5:
        return 4                                  # run-in topic heading
    return None


def _table_rows(lines: list[Line]) -> list[str]:
    """Rebuild a table's rows: cells whose vertical centres agree are one
    row, read left to right. Extracted in stream order the same table
    comes out column by column, with every number cut off from its label."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2):
        centre = (line.bbox[1] + line.bbox[3]) / 2
        if rows and abs(centre - (rows[-1][0].bbox[1] + rows[-1][0].bbox[3]) / 2) < 3:
            rows[-1].append(line)
        else:
            rows.append([line])
    out = []
    for row in rows:
        cells = [clean(l.text) for l in sorted(row, key=lambda l: l.bbox[0])]
        cells = [c for c in cells if c]
        if cells:
            out.append(" | ".join(cells))
    return out


def _group_text(group: _Group) -> str:
    caption = clean(group.caption.text)
    if caption.startswith("TABLE"):
        body = "\n".join(_table_rows(group.lines))
    else:
        # A chart's tick labels ("-2", "2025e") mean nothing without the
        # bars; keep its subtitle, panel titles, sources and notes.
        kept = [l for l in sorted(group.lines, key=lambda l: (l.bbox[1], l.bbox[0]))
                if l.size >= 6 and not re.fullmatch(r"[\d\s.,–\-%ef/]+", l.text.strip())]
        body = join_lines([l.text for l in kept])
    return clean(caption) + "\n" + body


def parse(path) -> list[Chunk]:
    doc = pymupdf.open(path)
    chapters = _chapters(doc)
    width = doc[0].rect.width

    paragraphs: list[tuple[tuple, int, str, dict]] = []  # (path, page, text, meta)
    groups: list[_Group] = []
    heads = {2: "", 3: "", 4: ""}
    box: dict | None = None
    region, chapter_key = "", ""

    def section_path(chapter_title):
        parts = [chapter_title]
        if region:
            parts.append(next(n for n, c in REGIONS.items() if c == region))
        if box:
            parts.append(box["title"])
            parts += [box[i] for i in (3, 4) if box.get(i)]
        else:
            parts += [heads[i] for i in (2, 3, 4) if heads[i]]
        return tuple(p for p in parts if p)

    for number, page in enumerate(doc, start=1):
        key, chapter_title = _chapter_at(chapters, number)
        if key != chapter_key:
            chapter_key, region, box = key, "", None
            heads = {2: "", 3: "", 4: ""}
        blocks = page_blocks(page, top=HEADER_Y, bottom=FOOTER_Y)
        lines = [(i, l) for i, b in enumerate(blocks) for l in b.lines]
        # A box runs page to page under a repeated "BOX 1.1 ... (continued)"
        # header; a page without one is back in the chapter.
        if not any(_BOX.match(l.text) and l.bold for _, l in lines):
            box = None
        page_groups: list[_Group] = []
        pending: list[str] = []  # body lines of the paragraph being read
        last_head, last_block, region_text = None, None, ""

        def meta():
            return {"chapter": chapter_key, "region": region,
                    "box": box["id"] if box else ""}

        def flush():
            if pending:
                text = clean(join_lines(pending))
                pending.clear()
                if len(text) > 3 and "....." not in text:
                    paragraphs.append((section_path(chapter_title), number, text, meta()))

        for block_no, line in lines:
            if block_no != last_block:
                flush()  # a new block is a new paragraph
                last_block = block_no
            text = clean(line.text)
            if not text:
                continue
            if (m := _BOX.match(text)) and line.bold:
                flush()
                if not box or box["id"] != m.group(1):
                    box = {"id": m.group(1), "title": f"Box {m.group(1)} {m.group(2)}"}
                continue
            if _CAPTION.match(text) and line.bold:
                flush()
                group = _Group(line, number, list(section_path(chapter_title)), meta())
                for name, code in REGIONS.items():
                    if name in text or re.search(rf"\b{code}\b", text):
                        group.meta["region"] = code
                page_groups.append(group)
                continue
            if line.size < 9:
                # Small type is a figure or table's content. It belongs to
                # the nearest caption above it and not to its right.
                owners = [g for g in page_groups
                          if g.caption.bbox[1] <= line.bbox[1] + 1
                          and g.caption.bbox[0] - 10 <= line.bbox[0]]
                if owners:
                    # Prefer a right-column caption for right-column type,
                    # then the closest caption above.
                    max(owners, key=lambda g: (g.caption.bbox[0] > width / 2 - 20,
                                               g.caption.bbox[1])).lines.append(line)
                continue
            level = _level(line)
            if level:
                flush()
                if chapter_key == "2" and level == 2:
                    # Region titles can be set on two lines ("Latin America
                    # and" / "the Caribbean"); a prefix of one is enough.
                    name = f"{region_text} {text}" if last_head == 2 else text
                    match = [code for full, code in REGIONS.items()
                             if full.startswith(name)]
                    if match:
                        region, region_text = match[0], name
                        heads = {2: "", 3: "", 4: ""}
                        last_head = 2
                        continue
                if level == 2 and text == "References":
                    region = ""
                target = box if box else heads
                # A heading set on two lines arrives as two lines.
                if last_head == level and target.get(level):
                    target[level] = f"{target[level]} {text}"
                else:
                    target[level] = text
                    for deeper in range(level + 1, 5):
                        target[deeper] = ""
                last_head = level
                continue
            last_head = None
            pending.append(line.text)
        flush()
        groups.extend(page_groups)

    return _prose_chunks(paragraphs) + _group_chunks(groups)


def _merge_continuations(paragraphs):
    """A paragraph cut by a column or page turn resumes lowercase in the
    next block; glue it back so its sentences are whole."""
    out = []
    for item in paragraphs:
        if out and out[-1][0] == item[0] and item[2][:1].islower() \
                and not out[-1][2].endswith((".", "?", "!", ":")):
            path, page, text, meta = out[-1]
            out[-1] = (path, page, f"{text} {item[2]}", meta)
        else:
            out.append(item)
    return out


def _prose_chunks(paragraphs) -> list[Chunk]:
    chunks, run = [], []

    def flush():
        if not run:
            return
        path, meta = list(run[0][0]), run[0][3]
        for parent_sents in pack_sentences([(p, t) for _, p, t, _ in run],
                                           PARENT_CHARS, overlap=0):
            parent_text = " ".join(s for _, s in parent_sents)
            parent_id = str(uuid.uuid4())
            # Children overlap by a sentence; their parent is what the model reads.
            for child in pack_sentences(
                    [(p, s) for p, s in parent_sents], CHILD_CHARS, overlap=1):
                chunks.append(Chunk(
                    text=" ".join(s for _, s in child), page=child[0][0],
                    section=path, meta=dict(meta), parent=parent_text,
                    parent_id=parent_id))
        run.clear()

    for item in _merge_continuations(paragraphs):
        if run and item[0] != run[0][0]:
            flush()
        run.append(item)
    flush()
    return chunks


def _group_chunks(groups: list[_Group]) -> list[Chunk]:
    chunks = []
    for group in groups:
        text = _group_text(group)
        if len(text) < 40:
            continue
        kind = "table" if text.startswith("TABLE") else "figure"
        chunks.append(Chunk(text=text[:4000], page=group.page, kind=kind,
                            section=group.section, meta=dict(group.meta)))
    return chunks
