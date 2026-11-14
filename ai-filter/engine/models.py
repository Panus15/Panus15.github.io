"""Shared data types. Every module speaks in these and nothing else.

The pipeline is  text -> [Claim] -> checkers -> [Finding] -> verdict.

THE ONE RULE EVERY CHECKER MUST KEEP: a failure of *our* machinery (network
down, timeout, rate limit, sandbox error, unparseable input) is UNVERIFIED,
never CONTRADICTED. CONTRADICTED means "we found positive evidence the claim is
wrong" and nothing else. A filter that cries wolf whenever the wifi drops
teaches its user to ignore it, which is worse than no filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ClaimKind(str, Enum):
    URL = "url"                # a link:                 data={"url": str}
    DOI = "doi"                # a DOI:                  data={"doi": str}   (bare, no prefix, lower-case)
    REFERENCE = "reference"    # a cited work:           data={"title": str, "authors": [str], "year": int|None}
    QUOTE = "quote"            # a quoted statement:     data={"quote": str, "attribution": str|None}
    ARITHMETIC = "arithmetic"  # "a op b = c":           data={"expression": str, "claimed": str}
    CODE = "code"              # a fenced code block:    data={"language": str, "code": str}
    VAGUE = "vague"            # unsourced appeal:       data={"phrase": str}


class Status(str, Enum):
    VERIFIED = "verified"          # positive evidence the claim holds
    UNVERIFIED = "unverified"      # could not confirm either way (includes our own failures)
    CONTRADICTED = "contradicted"  # positive evidence the claim is wrong


@dataclass(frozen=True)
class Claim:
    id: str                         # unique within one run: "c1", "c2", ... in document order
    kind: ClaimKind
    text: str                       # the exact source span, so a report can point at it
    start: int                      # character offsets into the original text; text == source[start:end]
    end: int
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding:
    claim: Claim
    status: Status
    module: str                     # "citations" | "recompute" | "sandbox" | "vague" | ...
    reason: str                     # one plain-language sentence a non-expert can act on
    evidence: tuple[str, ...] = ()  # URLs / API results / program output backing the reason
