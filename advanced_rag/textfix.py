"""Repairs for text damaged on its way out of a PDF.

To a search engine "Te" and "The" are different words, and "sustainabil- ity"
is two words that are neither. Fixing these before chunking is cheap and helps
BM25 most, since it only ever matches exact tokens.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_WORDLIST = Path("/usr/share/dict/words")

# Damage seen in the World Bank report, whose font loses the "Th" ligature.
# Some of these ("Tat", "Tis") are real words in a big dictionary, so the
# dictionary test below would leave them alone; they are listed here instead.
_KNOWN_TH = {
    "Te": "The", "Tis": "This", "Tese": "These", "Tey": "They", "Tat": "That",
    "Teir": "Their", "Tere": "There", "Tus": "Thus", "Tose": "Those",
    "Terefore": "Therefore", "Tird": "Third", "Tailand": "Thailand",
}


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    if not _WORDLIST.exists():
        return frozenset()
    return frozenset(w.strip().lower() for w in _WORDLIST.read_text().splitlines())


def _repair_th(match: re.Match) -> str:
    word = match.group(0)
    if word in _KNOWN_TH:
        return _KNOWN_TH[word]
    words = _words()
    rest = word[1:]
    # "Treshold" -> "Threshold": fix only when the T-word is not a word and
    # the Th-word is, so "Trade" and "Total" are never touched.
    if words and word.lower() not in words and ("th" + rest).lower() in words:
        return "Th" + rest
    return word


def fix_ligatures(text: str) -> str:
    """'ﬁ' -> 'fi' and friends, plus a dropped 'Th' at the start of a word."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\bT[a-z]+\b", _repair_th, text)


def join_lines(lines: list[str]) -> str:
    """Join a block's lines into one paragraph, healing hyphenated breaks.

    'produc-' + 'tion' becomes 'production', but 'low-' + 'income' keeps its
    hyphen: a break inside a word is healed, a break after a whole word is
    taken to be a real hyphen.
    """
    out = ""
    for line in (l.strip() for l in lines):
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            head = re.search(r"(\w+)-$", out)
            tail = re.match(r"\w+", line)
            joined = (head.group(1) + tail.group(0)).lower() if head and tail else ""
            # Join when the result is a word, or when the first half is not
            # one ("sustainabil-" + "ity"); keep "low-" + "income".
            if not _words() or joined in _words() or head.group(1).lower() not in _words():
                out = out[:-1] + line
                continue
            out += line
        elif out.endswith(("—", "/", "-")):  # "Vance-" + "Cooks"
            out += line
        else:
            out = f"{out} {line}" if out else line
    return re.sub(r"\s+", " ", out).strip()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", fix_ligatures(text)).strip()
