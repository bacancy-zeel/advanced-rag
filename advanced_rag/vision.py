"""Page descriptions from a vision model, for facts that only a chart shows.

The labels on the iMac carbon chart are extractable; that 445 kg of baseline
emissions minus 99 kg of reductions leaves 346 kg is not. One Gemini call per
page, at index time, cached on disk so rebuilding the index costs nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

from . import config
from .llm import generate, image_part

PROMPT = """This is page {page} of a corporate report. Describe ONLY the
facts carried by its charts and call-outs (big numbers with a caption).

- For a chart: name it, then list every bar or segment with its label and
  value, any totals, and how the values relate (for example "baseline A
  minus reductions B equals C"). Say which segments are larger or smaller
  when the chart shows it but prints no number, and never invent a number
  that is not printed.
- For a call-out: the number and what it counts.
- Keep footnote numbers as [^n].
- Skip body paragraphs and tables (they are extracted separately), and
  skip decorative navigation graphics such as a wheel of section names.
- Plain sentences, no markdown.

If nothing is left to describe, reply exactly: NONE"""


def _cache_path(pdf: Path) -> Path:
    digest = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()[:12]
    return config.CACHE_DIR / f"vision-{Path(pdf).stem}-{digest}.json"


def describe(pdf: Path, page: int) -> str:
    path = _cache_path(pdf)
    cache = json.loads(path.read_text()) if path.exists() else {}
    if str(page) not in cache:
        png = pymupdf.open(pdf)[page - 1].get_pixmap(dpi=150).tobytes("png")
        text = generate([image_part(png), PROMPT.format(page=page)])
        cache[str(page)] = "" if text.strip().upper().startswith("NONE") else text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2))
    return cache[str(page)]
