"""Tests for engine.llm: AnthropicLLM (against a hand-written fake SDK client),
BudgetedLLM, FakeLLM and the schema validator. Offline; the `anthropic` package
is never imported (and need not be installed).

Run: python3 tests/test_llm.py
"""
import json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm import (AnthropicLLM, BudgetedLLM, FakeLLM, LLMError, make_llm, validate_schema,
                        FALLBACK_BETA, THINKING_HEADROOM)

SECRET = "sk-ant-api03-SECRETSECRETSECRET"
NS = types.SimpleNamespace

SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]},
                                           "n": {"type": "integer", "minimum": 0, "maximum": 10}},
          "required": ["verdict"], "additionalProperties": False}


# ---- fake SDK client (shapes per the claude-api guide) ---------------------------

def text_block(t): return NS(type="text", text=t)
def response(content=None, stop_reason="end_turn", stop_details=None):
    return NS(content=content if content is not None else [], stop_reason=stop_reason,
              stop_details=stop_details, model="claude-opus-5-5", usage=NS(iterations=[]))


class _Messages:
    def __init__(self, outer, kind): self.outer, self.kind = outer, kind
    def create(self, **kw):
        self.outer.calls.append((self.kind, kw))
        r = self.outer.result
        if isinstance(r, BaseException):
            raise r
        return r


class FakeClient:
    def __init__(self, result): 
        self.result, self.calls = result, []
        self.messages = _Messages(self, "messages")
        self.beta = NS(messages=_Messages(self, "beta"))
    def __repr__(self): return f"FakeClient(api_key={SECRET})"   # must never reach an error


def ok(obj): return response([text_block(json.dumps(obj))])
def llm_for(result, **kw): 
    c = FakeClient(result); return AnthropicLLM(client=c, **kw), c
def call(llm, **kw):
    return llm.complete_json(system="SYS", user="USER", schema=kw.pop("schema", SCHEMA), **kw)
def raises(fn, needle=None):
    try: fn()
    except LLMError as e:
        if needle: assert needle in str(e), str(e)
        return e
    raise AssertionError("LLMError not raised")


# ---- AnthropicLLM: success + exact request ------------------------------------------

def test_success_returns_parsed_dict():
    llm, c = llm_for(ok({"verdict": "yes", "n": 3}))
    assert call(llm) == {"verdict": "yes", "n": 3}

def test_default_request_shape():
    llm, c = llm_for(ok({"verdict": "no"}))
    call(llm, max_tokens=500)
    kind, kw = c.calls[0]
    assert kind == "beta"                                   # fallbacks need the beta endpoint
    assert set(kw) == {"model", "max_tokens", "system", "messages", "output_config", "betas", "fallbacks"}, set(kw)
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 500 + THINKING_HEADROOM
    assert kw["system"] == "SYS"
    assert kw["messages"] == [{"role": "user", "content": "USER"}]   # no assistant prefill
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] == [FALLBACK_BETA]
    assert kw["fallbacks"] == "default"
    oc = kw["output_config"]
    assert oc["effort"] == "medium"
    assert oc["format"]["type"] == "json_schema"
    sent = oc["format"]["schema"]
    assert sent["properties"]["verdict"] == {"type": "string", "enum": ["yes", "no"]}
    assert sent["additionalProperties"] is False

def test_forbidden_parameters_never_sent():
    llm, c = llm_for(ok({"verdict": "no"}), effort="high")
    call(llm)
    kw = c.calls[0][1]
    for bad in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "tools", "api_key", "prefill", "budget_tokens"):
        assert bad not in kw, bad
    assert kw["output_config"]["effort"] == "high"
    assert kw["messages"][-1]["role"] == "user"

def test_unsupported_schema_keywords_stripped_from_request_but_enforced_locally():
    llm, c = llm_for(ok({"verdict": "yes", "n": 99}))
    raises(lambda: call(llm), "maximum")                          # local check still fires
    sent = c.calls[0][1]["output_config"]["format"]["schema"]
    assert sent["properties"]["n"] == {"type": "integer"}         # minimum/maximum removed
    assert "minimum" in SCHEMA["properties"]["n"]                 # caller's schema untouched

def test_fallbacks_off_uses_plain_endpoint():
    llm, c = llm_for(ok({"verdict": "no"}), fallbacks=False)
    call(llm)
    kind, kw = c.calls[0]
    assert kind == "messages" and "betas" not in kw and "fallbacks" not in kw

def test_fallbacks_auto_off_for_models_without_support():
    llm, c = llm_for(ok({"verdict": "no"}), model="claude-haiku-5-5")
    call(llm)
    assert c.calls[0][0] == "messages" and "fallbacks" not in c.calls[0][1]
    assert c.calls[0][1]["model"] == "claude-haiku-5-5"

def test_thinking_and_fallback_blocks_are_skipped():
    content = [NS(type="thinking", thinking="", signature="x"), text_block("stale partial"),
               NS(type="fallback"), NS(type="thinking", thinking="", signature="y"),
               text_block('{"verdict": "no"}')]
    llm, _ = llm_for(response(content))
    assert call(llm) == {"verdict": "no"}

def test_dict_shaped_blocks_work_too():
    llm, _ = llm_for({"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"verdict":"yes"}'}]})
    assert call(llm) == {"verdict": "yes"}


# ---- AnthropicLLM: failure paths ------------------------------------------------------

def test_refusal_with_category_checked_before_content():
    r = response([text_block('{"verdict":"yes"}')], stop_reason="refusal", stop_details=NS(category="cyber", explanation="x"))
    e = raises(lambda: call(llm_for(r)[0]), "refus")
    assert "cyber" in str(e)

def test_refusal_without_details():
    e = raises(lambda: call(llm_for(response([], stop_reason="refusal", stop_details=None))[0]), "refus")
    assert "category" not in str(e)

def test_refusal_category_is_sanitised():
    r = response([], stop_reason="refusal", stop_details=NS(category="x\n" + SECRET * 5))
    e = raises(lambda: call(llm_for(r)[0]), "refus")
    assert SECRET not in str(e)

def test_max_tokens():
    raises(lambda: call(llm_for(response([text_block('{"verdict"')], stop_reason="max_tokens"))[0]), "output truncated")

def test_unexpected_stop_reasons():
    for sr in ("tool_use", "pause_turn", "stop_sequence", None, "weird"):
        raises(lambda: call(llm_for(response([text_block('{"verdict":"yes"}')], stop_reason=sr))[0]), "stop_reason")

def test_no_text_block():
    raises(lambda: call(llm_for(response([NS(type="thinking", thinking="")]))[0]), "no text")

def test_non_json_text():
    raises(lambda: call(llm_for(response([text_block("Sure! Here you go")]))[0]), "not valid JSON")
    raises(lambda: call(llm_for(response([text_block('{"verdict": NaN}')]))[0]), "not valid JSON")

def test_json_but_not_an_object():
    raises(lambda: call(llm_for(response([text_block('["yes"]')]))[0]), "JSON object")

def test_json_violating_schema():
    raises(lambda: call(llm_for(ok({"verdict": "maybe"}))[0]), "allowed")
    raises(lambda: call(llm_for(ok({}))[0]), "required")
    raises(lambda: call(llm_for(ok({"verdict": "yes", "extra": 1}))[0]), "unexpected")


class _Resp:
    headers = {"x-api-key": SECRET, "authorization": "Bearer " + SECRET}

def test_sdk_exceptions_map_to_llmerror_without_leaking():
    class APIError(Exception): pass
    class APIStatusError(APIError): pass
    class APIConnectionError(APIError): pass
    APITimeoutError = type("APITimeoutError", (APIConnectionError,), {})
    AuthenticationError = type("AuthenticationError", (APIStatusError,), {})
    PermissionDeniedError = type("PermissionDeniedError", (APIStatusError,), {})
    RateLimitError = type("RateLimitError", (APIStatusError,), {})
    InternalServerError = type("InternalServerError", (APIStatusError,), {})

    def mk(cls, status=None):
        e = cls(f"boom {SECRET}"); e.message = f"boom {SECRET}"; e.response = _Resp(); e.body = {"k": SECRET}
        if status: e.status_code = status
        return e
    cases = [(mk(AuthenticationError, 401), "authentication failed"),
             (mk(PermissionDeniedError, 403), "403"),
             (mk(RateLimitError, 429), "429"),
             (mk(InternalServerError, 529), "529"),
             (mk(APITimeoutError), "timed out"),
             (mk(APIConnectionError), "reach"),
             (mk(ValueError), "ValueError"),
             (mk(RuntimeError), "RuntimeError")]
    for exc, needle in cases:
        llm, c = llm_for(exc)
        e = raises(lambda: call(llm), needle)
        for text in (str(e), repr(e), repr(e.args)):
            assert SECRET not in text, (exc, text)
        assert e.__cause__ is None and e.__suppress_context__            # no chained original in tracebacks
    e = raises(lambda: call(llm_for(mk(AuthenticationError, 401))[0]))
    assert "ANTHROPIC_API_KEY" in str(e) and "ant auth login" in str(e)

def test_llm_repr_does_not_include_client():
    llm, _ = llm_for(ok({"verdict": "no"}))
    assert SECRET not in repr(llm) and SECRET not in str(llm)


# ---- missing / present SDK -----------------------------------------------------------

def _with_module(name, mod, fn):
    had = name in sys.modules; old = sys.modules.get(name)
    sys.modules[name] = mod
    try: return fn()
    finally:
        if had: sys.modules[name] = old
        else: sys.modules.pop(name, None)

def test_missing_sdk_raises_clear_llmerror():
    e = _with_module("anthropic", None, lambda: raises(lambda: AnthropicLLM()))   # None => ImportError
    assert str(e) == "the 'anthropic' package is required for --ai: pip install anthropic"

def test_injected_client_never_imports_sdk():
    def go():
        AnthropicLLM(client=FakeClient(ok({"verdict": "no"})))   # would raise if it tried to import
    _with_module("anthropic", None, go)

def test_sdk_client_built_without_explicit_credentials():
    built = []
    class Anthropic:
        def __init__(self, *a, **k): built.append((a, k))
    _with_module("anthropic", NS(Anthropic=Anthropic), lambda: AnthropicLLM())
    assert built == [((), {})]                                   # no api_key passed

def test_sdk_client_construction_failure_is_llmerror():
    class Anthropic:
        def __init__(self): raise RuntimeError(f"no key {SECRET}")
    e = _with_module("anthropic", NS(Anthropic=Anthropic), lambda: raises(lambda: AnthropicLLM()))
    assert SECRET not in str(e) and "RuntimeError" in str(e)


# ---- BudgetedLLM ----------------------------------------------------------------------

def test_budget_counts_and_blocks_without_calling_inner():
    inner = FakeLLM([{"verdict": "yes"}, {"verdict": "no"}, {"verdict": "yes"}])
    b = BudgetedLLM(inner, 2)
    assert (b.calls, b.remaining) == (0, 2)
    call(b); assert (b.calls, b.remaining) == (1, 1)
    call(b); assert (b.calls, b.remaining) == (2, 0)
    raises(lambda: call(b), "LLM call budget exhausted")
    assert len(inner.history) == 2 and b.calls == 2             # third call never reached inner

def test_budget_failed_inner_calls_still_count():
    b = BudgetedLLM(FakeLLM([LLMError("x"), {"verdict": "yes"}]), 2)
    raises(lambda: call(b), "x")
    assert b.calls == 1
    call(b)
    raises(lambda: call(b), "budget")

def test_budget_zero_and_bad_values():
    b = BudgetedLLM(FakeLLM([]), 0)
    raises(lambda: call(b), "budget")
    for bad in (-1, 1.5, True, None):
        try: BudgetedLLM(FakeLLM([]), bad)
        except ValueError: pass
        else: raise AssertionError(bad)

def test_budget_passes_arguments_through():
    inner = FakeLLM([{"verdict": "yes"}])
    BudgetedLLM(inner, 1).complete_json(system="S", user="U", schema=SCHEMA, max_tokens=77)
    assert inner.history[0] == {"system": "S", "user": "U", "schema": SCHEMA, "max_tokens": 77}

def test_make_llm():
    def go():
        class Anthropic:
            def __init__(self): pass
        m = make_llm(max_calls=3)
        assert isinstance(m, BudgetedLLM) and m.max_calls == 3 and m.inner.model == "claude-opus-5-5"
        assert make_llm("claude-sonnet-5-5").inner.model == "claude-sonnet-5-5"
    _with_module("anthropic", NS(Anthropic=type("Anthropic", (), {})), go)
    raises(lambda: _with_module("anthropic", None, make_llm), "pip install anthropic")


# ---- FakeLLM ----------------------------------------------------------------------------

def test_fake_consumes_in_order_and_records_history():
    f = FakeLLM([{"verdict": "yes"}, {"verdict": "no"}])
    assert call(f, max_tokens=5) == {"verdict": "yes"}
    assert call(f) == {"verdict": "no"}
    assert f.history == [{"system": "SYS", "user": "USER", "schema": SCHEMA, "max_tokens": 5},
                         {"system": "SYS", "user": "USER", "schema": SCHEMA, "max_tokens": 2000}]

def test_fake_exhausted_fails_loudly_and_is_recorded():
    f = FakeLLM([])
    raises(lambda: call(f), "FakeLLM: script exhausted")
    assert len(f.history) == 1

def test_fake_raises_exception_items():
    f = FakeLLM([LLMError("boom"), RuntimeError("r"), {"verdict": "no"}])
    raises(lambda: call(f), "boom")
    try: call(f)
    except RuntimeError: pass
    else: raise AssertionError
    assert call(f) == {"verdict": "no"}

def test_fake_callable_script():
    f = FakeLLM(lambda s, u, sch: {"verdict": "yes"} if u == "USER" else LLMError("nope"))
    assert call(f) == {"verdict": "yes"}
    raises(lambda: f.complete_json(system="a", user="other", schema=SCHEMA), "nope")
    assert len(f.history) == 2

def test_fake_validates_scripted_output_unless_disabled():
    raises(lambda: call(FakeLLM([{"verdict": "maybe"}])), "allowed")
    assert call(FakeLLM([{"verdict": "maybe"}], validate=False)) == {"verdict": "maybe"}
    raises(lambda: call(FakeLLM(["not a dict"])), "dict")

def test_fake_returns_copy():
    item = {"verdict": "yes"}
    out = call(FakeLLM([item])); out["verdict"] = "no"
    assert item["verdict"] == "yes"


# ---- validator -----------------------------------------------------------------------------

def V(value, schema): return validate_schema(value, schema) == []

def test_validator_types():
    cases = {"object": ({}, [1]), "array": ([], {}), "string": ("a", 1), "boolean": (True, 1),
             "null": (None, 0), "number": (1.5, "1"), "integer": (3, 1.5)}
    for t, (good, bad) in cases.items():
        assert V(good, {"type": t}), t
        assert not V(bad, {"type": t}), t
    assert not V(True, {"type": "integer"}) and not V(True, {"type": "number"})   # bool is not a number
    assert V(2.0, {"type": "integer"}) and V(1, {"type": "number"})
    assert V(None, {"type": ["string", "null"]}) and not V(1, {"type": ["string", "null"]})

def test_validator_properties_required_additional():
    s = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
    assert V({"a": "x"}, s)
    assert not V({}, s) and not V({"a": 1}, s) and not V({"a": "x", "b": 1}, s)
    assert V({"a": "x", "b": 1}, {"type": "object", "properties": {"a": {"type": "string"}}})   # extras fine by default
    assert V({"b": 1}, {"type": "object", "additionalProperties": {"type": "integer"}})
    assert not V({"b": "s"}, {"type": "object", "additionalProperties": {"type": "integer"}})

def test_validator_enum():
    assert V("a", {"enum": ["a", "b"]}) and not V("c", {"enum": ["a", "b"]})
    assert not V(1, {"enum": [True]}) and not V(True, {"enum": [1]})     # 1 != True
    assert V(None, {"enum": [None]}) and V([1, {"x": 2}], {"enum": [[1, {"x": 2}]]})

def test_validator_items_and_item_counts():
    s = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}
    assert V([1], s) and V([1, 2], s)
    assert not V([], s) and not V([1, 2, 3], s) and not V([1, "x"], s)

def test_validator_string_lengths():
    s = {"type": "string", "minLength": 2, "maxLength": 3}
    assert V("ab", s) and V("abc", s) and not V("a", s) and not V("abcd", s)

def test_validator_numeric_bounds():
    s = {"type": "number", "minimum": 0, "maximum": 1}
    assert V(0, s) and V(1, s) and V(0.5, s) and not V(-0.1, s) and not V(1.1, s)

def test_validator_nested_paths_and_unknown_keywords():
    s = {"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "object",
         "properties": {"q": {"type": "string", "format": "uri", "x-whatever": 1, "pattern": "^never$"}}}}},
         "$comment": "ignored", "title": "t"}
    assert V({"xs": [{"q": "anything"}]}, s)                 # format/pattern/unknown ignored
    errs = validate_schema({"xs": [{"q": 5}]}, s)
    assert errs and "$.xs[0].q" in errs[0]

def test_validator_errors_never_echo_values():
    errs = validate_schema({"a": SECRET}, {"type": "object", "properties": {"a": {"type": "integer"}}})
    assert errs and SECRET not in " ".join(errs)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback; traceback.print_exc()
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
