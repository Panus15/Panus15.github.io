"""Registry of TRUSTED sources (engine/sources.json), in two tiers.

* "official": governments, regulators, central banks, statistics offices,
  intergovernmental bodies, and the wire/broadcast news outlets already listed.
* "reference": a deliberately SHORT list of broad general references that are
  NOT official but are widely used and edited under policy (Wikipedia,
  Britannica). They exist so facts no official site covers (history, sport,
  culture, ...) can still be checked. engine.official treats them with extra
  caution (they can support, never contradict).

A wrong domain here grants trust to someone else's site, so the file is
validated strictly on load and any problem raises ValueError naming the
offending entry. Matching is done by `engine.search.host_matches`, never by
substring tests. Every lookup defaults to the "official" tier only, so adding a
reference entry never widens what an existing caller trusts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from engine.search import host_matches

KINDS = ("government", "intergovernmental", "statistics", "central_bank", "regulator", "news", "academic", "reference")
TIERS = ("official", "reference")
TOPICS = ("law", "finance", "economy", "statistics", "health", "science", "climate", "energy",
          "government", "elections", "trade", "news", "technology", "education", "general")
_COUNTRY = re.compile(r"^[A-Z]{2,3}$")
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_ENTRY_KEYS = {"domain", "name", "kind", "tier", "country", "topics"}

_IMPLIED = {
    "law": ("law", "government"),
    "price": ("finance", "economy", "trade"),
    "number": ("statistics", "economy", "health"),
}
_OFFICIAL_KINDS = ("intergovernmental", "statistics", "central_bank", "regulator", "government")


@dataclass(frozen=True)
class Source:
    domain: str
    name: str
    kind: str
    country: str
    topics: tuple[str, ...]
    tier: str = "official"


def _check_tiers(tiers: object) -> tuple[str, ...]:
    if isinstance(tiers, str):
        tiers = (tiers,)
    try:
        out = tuple(tiers)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError("tiers must be a sequence of tier names") from None
    for t in out:
        if t not in TIERS:
            raise ValueError(f"unknown tier {t!r}; must be one of {list(TIERS)}")
    return out


def _order_key(index: int, s: Source):
    # news last; then Thai sources; then official kinds before academic; file order breaks ties
    kind_rank = 0 if s.kind in _OFFICIAL_KINDS else 1
    return (s.kind == "news", s.country != "TH", kind_rank, index)


@dataclass(frozen=True)
class Registry:
    sources: tuple[Source, ...]

    def all_domains(self, tiers: Sequence[str] = ("official",)) -> list[str]:
        ts = _check_tiers(tiers)
        return [s.domain for s in self.sources if s.tier in ts]

    def domains_for(self, category: str, topics: Sequence[str] = (),
                    tiers: Sequence[str] = ("official",)) -> list[str]:
        """Domains for a kind of fact, restricted to `tiers` (default official only).
        When no topic matches, falls back to every non-news domain WITHIN those tiers."""
        ts = _check_tiers(tiers)
        wanted = set(_IMPLIED.get(str(category).lower(), ())) | {str(t).lower() for t in topics}
        ranked = [(i, s) for i, s in sorted(enumerate(self.sources), key=lambda p: _order_key(*p))
                  if s.tier in ts]
        hits = [s.domain for _, s in ranked if wanted & set(s.topics)]
        if hits:
            return hits
        return [s.domain for _, s in ranked if s.kind != "news"]

    def _match(self, url: str) -> Optional[Source]:
        best: Optional[Source] = None
        for s in self.sources:
            if host_matches(url, [s.domain]) and (best is None or len(s.domain) > len(best.domain)):
                best = s
        return best

    def tier_of(self, url: str) -> Optional[str]:
        """The tier of the registry entry whose domain `url` belongs to (longest
        match), or None if the URL is not on a registered domain."""
        s = self._match(url)
        return s.tier if s else None

    def is_trusted(self, url: str, tiers: Sequence[str] = ("official",)) -> bool:
        ts = _check_tiers(tiers)
        return host_matches(url, self.all_domains(ts))

    def describe(self, url: str) -> Optional[str]:
        s = self._match(url)
        return s.name if s else None


def _valid_domain(d: str) -> bool:
    if not d.isascii() or len(d) > 253 or d != d.lower():
        return False
    labels = d.split(".")
    return len(labels) >= 2 and all(_LABEL.match(l) for l in labels) and not labels[-1].isdigit()


def _parse(data: object) -> Registry:
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError('registry must be an object with a "sources" list')
    seen: set[str] = set()
    out: list[Source] = []
    for i, e in enumerate(data["sources"]):
        label = f"entry #{i}" + (f" ({e['domain']!r})" if isinstance(e, dict) and isinstance(e.get("domain"), str) else "")

        def bad(msg: str):
            raise ValueError(f"{label}: {msg}")

        if not isinstance(e, dict):
            bad("not an object")
        if set(e) != _ENTRY_KEYS:
            bad(f"keys must be exactly {sorted(_ENTRY_KEYS)}")
        d, name, kind, country, topics = e["domain"], e["name"], e["kind"], e["country"], e["topics"]
        tier = e["tier"]
        if not isinstance(d, str) or not _valid_domain(d):
            bad("domain must be a lower-case hostname with a dot (no scheme, path, wildcard, port or IP)")
        if d in seen:
            bad("duplicate domain")
        seen.add(d)
        if not isinstance(name, str) or not name.strip():
            bad("name must be a non-empty string")
        if kind not in KINDS:
            bad(f"kind must be one of {list(KINDS)}")
        if tier not in TIERS:
            bad(f"tier must be one of {list(TIERS)}")
        if (kind == "reference") != (tier == "reference"):
            bad('kind "reference" and tier "reference" must go together')
        if not isinstance(country, str) or not _COUNTRY.match(country):
            bad("country must be a 2-3 letter upper-case code")
        if not isinstance(topics, list) or not topics:
            bad("topics must be a non-empty list")
        for t in topics:
            if t not in TOPICS:
                bad(f"unknown topic {t!r}")
        if len(set(topics)) != len(topics):
            bad("duplicate topics")
        out.append(Source(d, name, kind, country, tuple(topics), tier))
    return Registry(tuple(out))


def load_registry(path: Optional[str] = None) -> Registry:
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read registry {path!r}: {e.strerror or e}") from None
    except ValueError as e:
        raise ValueError(f"registry {path!r} is not valid JSON: {e}") from None
    return _parse(data)
