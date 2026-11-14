"""The web-search seam. (Protocol only here; Brave implementation, fake, and the
trusted-domain registry are added by the module owner.)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


class SearchError(Exception):
    """Search could not be performed (no key, HTTP error, quota, bad response)."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchClient(Protocol):
    def search(self, query: str, *, count: int = 5,
               domains: Optional[Sequence[str]] = None) -> list[SearchResult]:
        """Web results for `query`. If `domains` is given, ONLY results whose host is
        one of those domains or a subdomain of one are returned. Raises SearchError."""
        ...
