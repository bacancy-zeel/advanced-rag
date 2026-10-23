"""Sentence-aligned chunks, with optional parents.

A chunk that ends mid-sentence can lose exactly the number a question asks
about, so chunks here are packed from whole sentences, and the last sentence
of one chunk opens the next. Parsers decide what a section is; this module
only decides where inside a section to cut.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

# Abbreviations whose period is not a sentence end.
_ABBREV = ("U.S", "e.g", "i.e", "etc", "Fig", "No", "vs", "Inc", "Mr", "Ms",
           "Dr", "St", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
           "Sept", "Oct", "Nov", "Dec", "approx", "al", "Ch", "pp", "U.N")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[A-Z0-9“\"(•])")


@dataclass
class Chunk:
    text: str
    page: int
    section: list[str] = field(default_factory=list)
    kind: str = "text"
    meta: dict = field(default_factory=dict)
    # The passage handed to the model when this chunk is retrieved. Empty
    # means the chunk itself; a parent section otherwise.
    parent: str = ""
    parent_id: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def heading(self) -> str:
        return " > ".join(self.section)

    def embed_text(self) -> str:
        """Heading path + text. A passage that only ever says "SAR" still
        lands near a question about South Asia once its heading travels with
        it. The stored text is unchanged; only the vector sees this."""
        return f"{self.heading}\n{self.text}" if self.section else self.text


def sentences(text: str) -> list[str]:
    parts, out = _SENTENCE_END.split(text), []
    for part in parts:
        if out and (out[-1].endswith(tuple(a + "." for a in _ABBREV))
                    or re.search(r"\b[A-Z]\.$", out[-1])):  # "Brett M. Kavanaugh"
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return [s.strip() for s in out if s.strip()]


def pack_sentences(paragraphs: list[tuple[int, str]], max_chars: int,
                   overlap: int = 1) -> list[list[tuple[int, str]]]:
    """Greedy sentence packing up to `max_chars`, carrying `overlap`
    sentences into the next chunk. Paragraphs come as (page, text) and every
    sentence keeps its page. A paragraph break is not a forced cut, so short
    paragraphs share a chunk."""
    sents = [(page, s) for page, p in paragraphs for s in sentences(p)]
    chunks, current = [], []

    def size(items):
        return sum(len(s) + 1 for _, s in items)

    for item in sents:
        if current and size(current) + len(item[1]) > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            # A long sentence carried over would make the next chunk mostly
            # repeat; start clean instead.
            if size(current) + len(item[1]) > max_chars:
                current = []
        current.append(item)
    if current:
        chunks.append(current)
    return chunks


def pack(paragraphs: list[tuple[int, str]], max_chars: int,
         overlap: int = 1) -> list[tuple[int, str]]:
    """`pack_sentences`, joined: (page of the first sentence, text)."""
    return [(group[0][0], " ".join(s for _, s in group))
            for group in pack_sentences(paragraphs, max_chars, overlap)]


def fixed_size(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    """The naive baseline: cut every `size` characters, wherever that lands."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]
