"""The LLM seam: the Protocol, the Anthropic implementation (official SDK, lazily
imported, optional), a call budget, a scripted fake for tests, and a small
JSON-Schema validator used on every model answer.

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

from typing import Any, Optional, Protocol


class LLMError(Exception):
    """The LLM could not give a usable answer (API error, refusal, budget, invalid JSON)."""


class LLM(Protocol):
    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2000) -> dict[str, Any]:
        """Return a dict conforming to `schema` (JSON Schema) or raise LLMError."""
        ...


# ---------------------------------------------------------------------------
# Phase 2 implementation: schema validator, Anthropic client, budget, fake.
# ---------------------------------------------------------------------------

import copy
import json
import threading
from typing import Callable, Union

DEFAULT_MODEL = "claude-opus-5-5"

# claude-opus-5-5 always thinks (thinking cannot be disabled) and thinking tokens
# count toward max_tokens. Callers size `max_tokens` for the JSON they want back;
# we add this much on top so a thinking-heavy turn is not reported as "truncated".
THINKING_HEADROOM = 8000

# Beta header for server-side refusal fallbacks, `fallbacks: "default"` scalar form.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Models this module sends `fallbacks: "default"` to when `fallbacks=None` (auto).
# (claude-haiku-5-5 400s on a `fallbacks` field, so it is deliberately absent.)
_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})

_MISSING_SDK = "the 'anthropic' package is required for --ai: pip install anthropic"
_AUTH_MSG = "authentication failed — check ANTHROPIC_API_KEY or `ant auth login`"


# ----- tiny JSON-Schema validator -------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(v: Any, t: str) -> bool:
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "null":
        return v is None
    if t == "number":
        return _is_number(v)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool) or (isinstance(v, float) and v.is_integer())
    return True  # unknown type name: ignored, like any unknown keyword


def _json_eq(a: Any, b: Any) -> bool:
    """Equality where True != 1 (Python's == would conflate them)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    return a == b


def _check(v: Any, s: Any, path: str, errs: list[str]) -> None:
    if not isinstance(s, dict):
        return
    t = s.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        if not any(isinstance(n, str) and _type_ok(v, n) for n in names):
            errs.append(f"{path}: wrong type (expected {'|'.join(str(n) for n in names)})")
            return
    if "enum" in s and isinstance(s["enum"], list) and not any(_json_eq(v, e) for e in s["enum"]):
        errs.append(f"{path}: not one of the allowed values")
    if isinstance(v, str):
        if isinstance(s.get("minLength"), int) and len(v) < s["minLength"]:
            errs.append(f"{path}: shorter than minLength {s['minLength']}")
        if isinstance(s.get("maxLength"), int) and len(v) > s["maxLength"]:
            errs.append(f"{path}: longer than maxLength {s['maxLength']}")
    if _is_number(v):
        if _is_number(s.get("minimum")) and v < s["minimum"]:
            errs.append(f"{path}: below minimum {s['minimum']}")
        if _is_number(s.get("maximum")) and v > s["maximum"]:
            errs.append(f"{path}: above maximum {s['maximum']}")
    if isinstance(v, list):
        if isinstance(s.get("minItems"), int) and len(v) < s["minItems"]:
            errs.append(f"{path}: fewer than minItems {s['minItems']}")
        if isinstance(s.get("maxItems"), int) and len(v) > s["maxItems"]:
            errs.append(f"{path}: more than maxItems {s['maxItems']}")
        if isinstance(s.get("items"), dict):
            for i, item in enumerate(v):
                _check(item, s["items"], f"{path}[{i}]", errs)
    if isinstance(v, dict):
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        for name in s.get("required") or []:
            if isinstance(name, str) and name not in v:
                errs.append(f"{path}: missing required property '{name}'")
        for k, val in v.items():
            if k in props:
                _check(val, props[k], f"{path}.{k}", errs)
            else:
                ap = s.get("additionalProperties")
                if ap is False:
                    errs.append(f"{path}: unexpected property '{k}'")
                elif isinstance(ap, dict):
                    _check(val, ap, f"{path}.{k}", errs)


def validate_schema(value: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty list == valid). Never includes `value` itself."""
    errs: list[str] = []
    _check(value, schema, "$", errs)
    return errs


def _validate_or_raise(value: Any, schema: dict[str, Any]) -> None:
    errs = validate_schema(value, schema)
    if errs:
        raise LLMError("output does not match the schema: " + "; ".join(errs[:3]))


# ----- schema sanitising for the API ----------------------------------------

# The API's structured-output grammar rejects these; we strip them from the copy we
# SEND and still enforce them locally against the original schema.
_UNSUPPORTED_KEYS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                               "multipleOf", "minLength", "maxLength", "minItems", "maxItems",
                               "uniqueItems", "pattern"})


def _api_schema(s: Any) -> Any:
    if isinstance(s, list):
        return [_api_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {k: _api_schema(v) for k, v in s.items() if k not in _UNSUPPORTED_KEYS}
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False   # required by the API for every object
    return out


# ----- Anthropic implementation ----------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_token(x: Any) -> str:
    s = x if isinstance(x, str) else ""
    return s if 0 < len(s) <= 40 and all(c.isalnum() or c in "_-" for c in s) else ""


def _map_sdk_error(e: BaseException) -> LLMError:
    """SDK exception -> LLMError carrying ONLY class name / status code.

    Never str(e), e.message, e.body, e.response or e.request: they can echo
    headers or request details, and the key must never reach a message.
    Classification is by class name through the MRO so this module never has to
    import the SDK at call time (and works with the test fakes).
    """
    names = {c.__name__ for c in type(e).__mro__}
    status = getattr(e, "status_code", None)
    status = status if isinstance(status, int) and not isinstance(status, bool) else None
    cls = type(e).__name__
    if "AuthenticationError" in names:
        return LLMError(_AUTH_MSG)
    if "PermissionDeniedError" in names:
        return LLMError("Anthropic API permission denied (HTTP 403) — check the key's access to this model")
    if "RateLimitError" in names:
        return LLMError("Anthropic API rate limited (HTTP 429)")
    if "APITimeoutError" in names:
        return LLMError("Anthropic API request timed out (APITimeoutError)")
    if "APIConnectionError" in names:
        return LLMError("could not reach the Anthropic API (APIConnectionError)")
    if "APIStatusError" in names or status is not None:
        return LLMError(f"Anthropic API error ({cls}, HTTP {status if status is not None else '?'})")
    return LLMError(f"Anthropic API call failed ({cls})")


def _parse_json(text: str) -> Any:
    def _no_const(name: str) -> Any:
        raise ValueError(name)          # NaN / Infinity are not JSON
    return json.loads(text, parse_constant=_no_const)


class AnthropicLLM:
    """`LLM` backed by the official `anthropic` SDK (an OPTIONAL dependency).

    Request shape (claude-opus-5-5): messages.create with `system`, one user message,
    `output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}`.
    No temperature/top_p/top_k, no prefill, no `thinking` key (always on, adaptive),
    no tool_choice. With `fallbacks` on (default for models that support it) the same
    call goes through `client.beta.messages.create(betas=[FALLBACK_BETA],
    fallbacks="default")` so a classifier decline is retried server-side.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any = None,
                 effort: str = "medium", fallbacks: Optional[bool] = None):
        self.model = model
        self.effort = effort
        self.fallbacks = (model in _FALLBACK_MODELS) if fallbacks is None else bool(fallbacks)
        if client is None:
            try:
                import anthropic                      # lazy: optional dependency
            except ImportError:
                raise LLMError(_MISSING_SDK) from None
            try:
                client = anthropic.Anthropic()        # SDK resolves credentials itself
            except Exception as e:
                raise LLMError(f"could not create the Anthropic client ({type(e).__name__}) — "
                               "check ANTHROPIC_API_KEY or `ant auth login`") from None
        self._client = client

    def __repr__(self) -> str:                         # never repr the client
        return f"AnthropicLLM(model={self.model!r}, effort={self.effort!r})"

    def _request(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens + THINKING_HEADROOM,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": _api_schema(copy.deepcopy(schema))}},
        )
        if self.fallbacks:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2000) -> dict[str, Any]:
        kw = self._request(system, user, schema, max_tokens)
        try:
            api = self._client.beta.messages if self.fallbacks else self._client.messages
            resp = api.create(**kw)
        except LLMError:
            raise
        except Exception as e:
            raise _map_sdk_error(e) from None
        return self._read(resp, schema)

    @staticmethod
    def _read(resp: Any, schema: dict[str, Any]) -> dict[str, Any]:
        # stop_reason FIRST: a refusal's content is empty or partial.
        reason = _get(resp, "stop_reason")
        if reason == "refusal":
            cat = _safe_token(_get(_get(resp, "stop_details"), "category"))
            raise LLMError("model refused the request" + (f" (category: {cat})" if cat else ""))
        if reason == "max_tokens":
            raise LLMError("output truncated")
        if reason != "end_turn":
            raise LLMError(f"unexpected stop_reason: {_safe_token(reason) or 'unknown'}")
        blocks = list(_get(resp, "content") or [])
        # Only text after the last fallback switch point (non-streaming omits the
        # declined partial anyway; this is belt and braces).
        start = 0
        for i, b in enumerate(blocks):
            if _get(b, "type") == "fallback":
                start = i + 1
        text = "".join(_get(b, "text") or "" for b in blocks[start:]
                       if _get(b, "type") == "text" and isinstance(_get(b, "text"), str))
        if not text.strip():
            raise LLMError("model returned no text")
        try:
            data = _parse_json(text)
        except ValueError:
            raise LLMError("model output was not valid JSON") from None
        if not isinstance(data, dict):
            raise LLMError("model output was not a JSON object")
        _validate_or_raise(data, schema)
        return data


# ----- call budget -------------------------------------------------------------

class BudgetedLLM:
    """Wraps an LLM and refuses calls past `max_calls`. Failed inner calls still count."""

    def __init__(self, inner: LLM, max_calls: int):
        if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative int")
        self.inner = inner
        self.max_calls = max_calls
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self._calls)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2000) -> dict[str, Any]:
        with self._lock:
            if self._calls >= self.max_calls:
                raise LLMError("LLM call budget exhausted")
            self._calls += 1
        return self.inner.complete_json(system=system, user=user, schema=schema, max_tokens=max_tokens)


# ----- test double ------------------------------------------------------------------

class FakeLLM:
    """Scripted LLM for tests. `script`: list consumed in order, or callable (system, user, schema) -> dict.

    An Exception instance in the list (or returned by the callable) is raised.
    Every call is recorded in `.history`. An exhausted list raises LLMError so a test
    that forgot a response fails loudly. Scripted dicts are validated against the
    schema (unless validate=False) so tests cannot script outputs the real client
    would have rejected.
    """

    def __init__(self, script: Union[list, Callable[[str, str, dict], Any]], *, validate: bool = True):
        self._script = script
        self._validate = validate
        self._i = 0
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any],
                      max_tokens: int = 2000) -> dict[str, Any]:
        with self._lock:
            self.history.append({"system": system, "user": user, "schema": schema, "max_tokens": max_tokens})
            if not callable(self._script):
                if self._i >= len(self._script):
                    raise LLMError("FakeLLM: script exhausted")
                item = self._script[self._i]
                self._i += 1
        if callable(self._script):
            item = self._script(system, user, schema)
        if isinstance(item, BaseException):
            raise item
        if not isinstance(item, dict):
            raise LLMError("FakeLLM: scripted item must be a dict or an Exception instance")
        if self._validate:
            _validate_or_raise(item, schema)
        return copy.deepcopy(item)


def make_llm(model: Optional[str] = None, *, max_calls: int = 40) -> BudgetedLLM:
    """AnthropicLLM (default model unless given) behind a call budget."""
    return BudgetedLLM(AnthropicLLM(model) if model else AnthropicLLM(), max_calls)
