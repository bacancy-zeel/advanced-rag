"""Query understanding for the World Bank report: a question that names a
place or a chapter becomes a metadata filter.

Six regions all report "growth forecasts", so a search for South Asia's
outlook also finds East Asia's. A filter removes the other five before
ranking even starts. The rules are deliberately conservative, because a
wrong filter hides the answer, which is worse than no filter at all:

- A chapter is filtered only when the question names it.
- A region is filtered when the question names it, its code, or a country
  that the report's own regional tables list under it. That map is how
  "Pakistan" finds Middle East, North Africa, Afghanistan, and Pakistan,
  which the World Bank groups it with and a reader may not expect.
- "Middle East" alone is not a region here: in this report it is mostly the
  conflict, not the region.
"""
from __future__ import annotations

import json
import re

from . import config
from .chunking import Chunk
from .parsers.gep import REGIONS

COUNTRIES_FILE = config.CACHE_DIR / "gep-countries.json"

# Matched ignoring case; the codes themselves ("SAR") are matched as written.
_NAMES = {
    "EAP": [r"east asia"],
    "ECA": [r"europe and central asia", r"central asia"],
    "LAC": [r"latin america", r"caribbean"],
    "MNA": [r"middle east,? (and )?north africa", r"north africa", r"\bmena\b"],
    "SAR": [r"south asia"],
    "SSA": [r"sub-?saharan"],
}
_CHAPTERS = {
    "summary": [r"executive summary"],
    "appendix": [r"statistical appendix"],
    **{str(n): [rf"\bchapter {n}\b", rf"\bch\.? ?{n}\b"] for n in range(1, 5)},
}


def save_countries(chunks: list[Chunk]) -> dict[str, str]:
    """Country -> region code, from the per-region forecast tables."""
    countries = {}
    for chunk in chunks:
        region = chunk.meta.get("region")
        if chunk.kind != "table" or not region or "economy forecasts" not in chunk.text:
            continue
        for row in chunk.text.splitlines()[1:]:
            name = re.sub(r"\s+\d$", "", row.split(" | ")[0]).strip()
            if " | " in row and re.match(r"[A-Z][A-Za-z.,'’ -]+$", name) \
                    and not name.startswith(("Calendar", "Fiscal", "Memo", "Source")):
                countries[name] = region
    COUNTRIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    COUNTRIES_FILE.write_text(json.dumps(countries, indent=1, sort_keys=True))
    return countries


def _countries() -> dict[str, str]:
    return json.loads(COUNTRIES_FILE.read_text()) if COUNTRIES_FILE.exists() else {}


def route(question: str) -> dict:
    """The filter for `question`, e.g. {"region": "SAR"}, or {} for none."""
    where = {}
    for key, patterns in _CHAPTERS.items():
        if any(re.search(p, question, re.I) for p in patterns):
            where["chapter"] = key
            break
    regions = {code for code, patterns in _NAMES.items()
               if re.search(rf"\b{code}\b", question)
               or any(re.search(p, question, re.I) for p in patterns)}
    for country, code in _countries().items():
        # "Egypt, Arab Rep." is asked about as "Egypt".
        short = country.split(",")[0]
        if re.search(rf"\b{re.escape(short)}\b", question):
            regions.add(code)
    # Two regions in one question is a comparison; a filter on either would
    # hide the other.
    if len(regions) == 1:
        where["region"] = regions.pop()
        where.pop("chapter", None)   # regions only exist in chapter 2 anyway
    return where


def describe(where: dict) -> str:
    names = {code: name for name, code in REGIONS.items()}
    parts = []
    if "region" in where:
        parts.append(f"region = {names[where['region']]}")
    if "chapter" in where:
        parts.append(f"chapter = {where['chapter']}")
    return ", ".join(parts) or "none"
