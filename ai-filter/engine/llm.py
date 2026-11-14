"""The LLM seam. (Protocol only here; the Anthropic implementation, a fake for
tests, and the call budget are added by the module owner.)

Rules every user of an LLM in this package follows:

* The model is an ASSISTANT to the checkers, never the judge of last resort.
  Anything it returns is validated mechanically (a quote must appear verbatim in
  the source, a span must exist in the answer) before it can change a Finding.
* Text from the answer and from fetched pages is UNTRUSTED DATA. It is wrapped in
  delimiters and the system prompt says to treat it as data. That reduces prompt
  injection; it does not eliminate it, which is why the mechanical checks exist.
* Any failure (error, refusal, budget, bad JSON) raises LLMError -> callers turn
  it into UNVERIFIED, never CONTRADICTED.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(Exception):
    """The LLM could not give a usable answer (API error, refusal, budget, invalid JSON)."""


class LLM(Protocol):
    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2000) -> dict[str, Any]:
        """Return a dict conforming to `schema` (JSON Schema) or raise LLMError."""
        ...
