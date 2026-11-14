"""Small shared types for evidence gathered from web pages."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Page:
    url: str                          # final URL after redirects
    title: str
    text: str                         # visible text, whitespace-normalised, scripts/styles removed
    published: Optional[dt.date] = None   # best-effort publication/last-modified date; None if unknown


@dataclass(frozen=True)
class Judgement:
    # "supports" | "contradicts" | "not_stated" | "error".
    # "supports"/"contradicts" are only ever returned AFTER mechanical verification
    # (verbatim quote found in the page; for "contradicts", claim and page values differ).
    verdict: str
    quote: str = ""                   # verbatim passage from the page backing the verdict ("" if none)
    reason: str = ""                  # one plain-language sentence
