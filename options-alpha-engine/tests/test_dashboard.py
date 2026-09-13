"""Tests for the dashboard page (tools/rotation_dashboard.py).

WHY THIS FILE EXISTS. The dashboard is the only part of this engine a human
actually looks at, and it had no tests at all — 476 of them passed around it. It
is also the one module where a defect is invisible to Python: the page is a
~500-line string of HTML/CSS/JS, and a single bad character in the script makes
the ENTIRE page render blank while every Python test still passes. That is not
hypothetical. It happened while this panel was being written:

    c => ({'&': '&amp;', ... , BACKSLASH + QUOTE : '&quot;'}[c])

`TEMPLATE` is a NON-RAW Python triple-quoted string, so Python consumed the
backslash and shipped a bare quote to the browser — a syntax error that killed the whole
script. The test below that scans the template region for backslashes is aimed at
exactly that root cause, and it is portable; the one that runs node is better but
can only run where node exists.

The second thing tested here is that the options panel renders its own ABSENCE. A
panel that disappears when its data is missing looks identical to one that had
nothing to say, which is the opposite of true: the variance edge is the only half
of this engine with oracle-tested machinery behind it.

Run: python3 tests/test_dashboard.py
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from engine import pricing, volforecast
from engine.data import SyntheticAdapter
from tools.rotation_dashboard import (TEMPLATE, build_payload, demo_world,
                                      ledger_panel, options_panel, render)

#: One temp dir for the process. The first draft cached paths INSIDE a
#: TemporaryDirectory that a later test had already deleted, so four tests read a
#: "missing chain" and two of them still passed — a fixture bug that presents as a
#: product bug is worth the comment.
_TMP = tempfile.mkdtemp(prefix="dash-test-")
_FIXTURE: dict = {}


def _chain_files(*, dense=True, dte=30, iv_mult=1.35, days=600):
    """A chain written in the JsonFileAdapter schema, priced at a vol ABOVE the
    forecast so the variance edge has something to find."""
    key = (dense, dte, iv_mult, days)
    if key in _FIXTURE:
        return _FIXTURE[key]
    prices = SyntheticAdapter().price_history("SPY", days=days)
    spot = prices[-1]
    iv = volforecast.har_rv_forecast(prices) * iv_mult
    T, r = dte / 365.0, 0.03
    quotes, k = [], round(spot * (0.78 if dense else 0.97))
    step = max(round(spot * (0.01 if dense else 0.05)), 1)
    while k <= spot * (1.22 if dense else 1.03):
        for kind in ("call", "put"):
            mid = pricing.price(spot, k, T, r, 0.0, iv, kind)
            if mid < 0.02:
                continue
            half = max(mid * 0.01, 0.01)
            quotes.append({"expiry_days": dte, "strike": float(k), "kind": kind,
                           "bid": round(mid - half, 2), "ask": round(mid + half, 2)})
        k += step
    cp = os.path.join(_TMP, f"chain_{int(dense)}_{dte}_{days}.json")
    pp = os.path.join(_TMP, f"px_{days}.json")
    with io.open(cp, "w", encoding="utf-8") as fh:
        json.dump({"symbol": "SPY", "spot": spot, "r": r, "q": 0.0,
                   "asof": "2026-09-12", "quotes": quotes}, fh)
    with io.open(pp, "w", encoding="utf-8") as fh:
        json.dump(prices, fh)
    _FIXTURE[key] = (cp, pp)
    return cp, pp


# --------------------------------------------------------------------------
# The bug class that Python cannot see
# --------------------------------------------------------------------------

def test_the_template_contains_no_backslash_python_would_eat():
    """TEMPLATE is a non-raw triple-quoted string. Every backslash in it is a
    PYTHON escape first and a JS/CSS character second, so `\\"` reaches the
    browser as `"` and can close a string the author meant to keep open. This
    already shipped once and blanked the entire page."""
    src = io.open(os.path.join(ROOT, "tools/rotation_dashboard.py"),
                  encoding="utf-8").read()
    i = src.index('TEMPLATE = """')
    j = src.index('"""', i + 14)
    region = src[i + 14:j]
    bad = [(n, ln.strip()) for n, ln in enumerate(region.split("\n"), 1)
           if "\\" in ln]
    assert not bad, ("backslash in the template at " + "; ".join(
        f"line {n}: {ln[:70]}" for n, ln in bad) + " — make TEMPLATE a raw string "
        "(r\"\"\") or rewrite without the escape")


def test_the_page_script_parses_when_a_js_engine_is_available():
    """The general version of the test above: actually parse the script the page
    ships. Runs only where node exists, and says so when it does not, because a
    check that silently does nothing is worse than one that is absent."""
    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        print("    (not checked: no node on PATH — the portable backslash guard "
              "above is the only protection here)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5,
                                    tail=12, horizon=21, run_test=False))
        start = page.index("<script>", page.index("</style>"))
        js = page[start + len("<script>"):page.rindex("</script>")]
        f = os.path.join(tmp, "page.js")
        io.open(f, "w", encoding="utf-8").write(js)
        r = subprocess.run([node, "--check", f], capture_output=True, text=True)
        assert r.returncode == 0, ("the page ships a script node cannot parse:\n"
                                  + (r.stderr or r.stdout)[:900])


def test_every_template_token_is_filled():
    payload = build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                            horizon=21, run_test=False)
    page = render(payload)
    assert "@@" not in page
    assert page.lstrip().startswith("<!doctype html>")


# --------------------------------------------------------------------------
# The options panel renders its own absence
# --------------------------------------------------------------------------

def test_with_no_chain_the_panel_says_why_and_how_rather_than_vanishing():
    o = options_panel("", "")
    assert o["available"] is False
    assert "no option chain" in o["reason"]
    assert "run_live" in o["how"] and "--chain-json" in o["how"]


def test_a_missing_chain_file_is_named_not_swallowed():
    o = options_panel("/nonexistent/path/chain.json", "")
    assert o["available"] is False and "does not exist" in o["reason"]


def test_a_chain_without_price_history_says_the_p_side_is_missing():
    cp, _ = _chain_files()
    o = options_panel(cp, "")
    assert o["available"] is False
    assert "price bars" in o["reason"], o["reason"]
    assert "--price-json" in o["how"]


def test_a_corrupt_chain_file_is_a_data_outcome_not_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.json")
        io.open(bad, "w", encoding="utf-8").write("{not json at all")
        o = options_panel(bad, "")
        assert o["available"] is False and o["reason"]


def test_the_payload_always_carries_an_options_key():
    """If it were None the page would skip the panel, and a skipped panel is
    indistinguishable from one with nothing to say."""
    payload = build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                            horizon=21, run_test=False)
    assert isinstance(payload["options"], dict)
    assert payload["options"]["available"] is False


# --------------------------------------------------------------------------
# The available panel, and the property that it cannot misstate its risk
# --------------------------------------------------------------------------

def test_a_dense_rich_chain_produces_a_placeable_order():
    cp, pp = _chain_files()
    o = options_panel(cp, pp, symbol="SPY", dte=30, equity=250_000.0)
    assert o["available"] is True, o.get("reason")
    assert o["volSide"] == "SELL VOL", o["volSide"]
    assert o["placeable"] is True, o["refusals"]
    assert o["contracts"] > 0 and len(o["legs"]) >= 2
    assert o["qVol"] > o["pVol"], "the fixture was priced rich on purpose"


def test_the_panel_cannot_print_more_risk_than_the_budget():
    """The same theorem the ticket is held to, checked on what the PAGE shows —
    because the number a human acts on is the rendered one."""
    cp, pp = _chain_files()
    for eq in (50_000.0, 100_000.0, 250_000.0, 1_000_000.0):
        for frac in (0.005, 0.01, 0.02, 0.05):
            o = options_panel(cp, pp, symbol="SPY", equity=eq,
                              max_risk_frac=frac)
            if not o.get("placeable"):
                continue
            assert o["maxLossTotal"] <= frac * eq + 1e-6, (eq, frac, o["maxLossTotal"])
            assert o["maxLossTotal"] == o["contracts"] * o["maxLossPerContract"]


def test_a_small_account_refuses_by_name_rather_than_showing_a_tiny_order():
    cp, pp = _chain_files()
    o = options_panel(cp, pp, symbol="SPY", equity=2_000.0)
    assert o["available"] is True
    assert o["placeable"] is False
    assert o["refusals"] and all(r["code"] and r["detail"] for r in o["refusals"])


def test_the_evidence_label_reaches_the_panel():
    cp, pp = _chain_files()
    o = options_panel(cp, pp, symbol="SPY")
    assert "settled 0 trades" in o["evidence"], o["evidence"]


def test_the_requested_tenor_and_the_listed_one_are_both_reported():
    """`--dte 30` on a chain that lists only 30 must not silently become 30; on
    one that lists something else the move has to be visible."""
    cp, pp = _chain_files(dte=45)
    o = options_panel(cp, pp, symbol="SPY", dte=30)
    assert o["requestedDte"] == 30 and o["dte"] == 45


def test_the_risk_budget_the_page_prints_is_the_one_that_was_asked_for():
    """The panel echoes equity and the risk fraction so a reader can check the
    arithmetic. A hardcoded echo would show a budget the order was not sized
    against, which is the most convincing way to be wrong."""
    cp, pp = _chain_files()
    for eq, frac in ((100_000.0, 0.02), (250_000.0, 0.005), (1_000_000.0, 0.05)):
        o = options_panel(cp, pp, symbol="SPY", equity=eq, max_risk_frac=frac)
        assert o["equity"] == eq and o["maxRiskFrac"] == frac, (eq, frac, o["equity"],
                                                               o["maxRiskFrac"])
        if o.get("placeable"):
            assert o["maxLossTotal"] <= o["maxRiskFrac"] * o["equity"] + 1e-6


def test_the_page_actually_calls_the_panel_renderer():
    """Everything else here tests the DATA. If the page never calls drawOptions
    the panel is an empty box and no Python test can tell."""
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    assert "function drawOptions()" in page
    body = page[page.rindex("function drawOptions()"):]
    assert "drawOptions();" in body, "defined but never invoked"


def test_untrusted_text_from_a_chain_file_cannot_inject_markup():
    """The symbol comes from a JSON file someone hands you. It reaches the page
    through a template literal, so an unescaped quote or angle bracket would close
    an attribute or open a tag."""
    import re
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                               horizon=21, run_test=False))
    fn = page[page.index("function esc("):page.index("function money(")]
    cls = re.search(r"replace\(/\[([^\]]+)\]/g", fn)
    assert cls, fn
    for ch in "&<>\"":
        assert ch in cls.group(1), f"esc() does not escape {ch!r}: {cls.group(1)!r}"
    # and every interpolation of chain-supplied text goes through it
    draw = page[page.index("function drawOptions()"):page.index("function drawTest()")]
    for field in ("o.symbol", "o.structure", "o.volReason", "o.expiryDate",
                  "r.code", "r.detail", "i.name", "i.reading", "o.evidence"):
        assert f"esc({field})" in draw, f"{field} reaches the page unescaped"


def test_the_options_panel_comes_before_the_rotation_panels():
    """Ordering is the point, not decoration. The one study this repo ran on real
    data answered NO for the sector quadrant, so a page that leads with it teaches
    the reader to act on its weakest number."""
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    assert page.index('id="opt"') < page.index('id="rrg"') < page.index('id="testcard"')


def test_every_colour_the_panel_uses_is_defined_in_the_palette():
    """A var(--name) that does not exist renders as nothing at all — an invisible
    number, which is worse than a wrong one. This has already bitten once."""
    import re
    used = set(re.findall(r"var\(--([a-z0-9-]+)\)", TEMPLATE))
    defined = set(re.findall(r"--([a-z0-9-]+)\s*:", TEMPLATE))
    missing = sorted(used - defined)
    assert not missing, f"undefined CSS variables: {missing}"


# --------------------------------------------------------------------------
# The forward-test panel — the only one that can show PERSISTENCE
# --------------------------------------------------------------------------

_LEDGER: list = []


def _ledger_file():
    """A real ledger, recorded and settled offline. Built once per process."""
    if _LEDGER:
        return _LEDGER[0]
    from engine.signal_backtest import synthetic_chain_series
    from models.baseline import BaselineDensityForecaster
    from tools.paper_trade import paper_trade_series
    ladder = tuple(round(-0.20 + 0.025 * i, 3) for i in range(17))
    prices = SyntheticAdapter(seed=7).price_history("SPY", days=700)
    led = paper_trade_series(prices, synthetic_chain_series(dte=21, ladder=ladder,
                                                           min_px=0.005),
                             BaselineDensityForecaster(), dte=21, warmup=120, step=9)
    path = os.path.join(_TMP, "ledger.jsonl")
    led.save(path)
    _LEDGER.append(path)
    return path


def test_with_no_ledger_the_panel_says_the_clock_has_not_started():
    f = ledger_panel("")
    assert f["available"] is False
    assert f["recorded"] == 0 and f["settled"] == 0
    assert "graded" in f["reason"]
    assert "paper_trade record" in f["how"]


def test_a_missing_ledger_file_names_itself():
    f = ledger_panel(os.path.join(_TMP, "not-there.jsonl"))
    assert f["available"] is False and "does not exist" in f["reason"]
    assert f["recorded"] == 0


def test_an_empty_ledger_file_is_zero_recorded_not_an_error():
    """A file that exists with nothing in it is the state right after someone sets
    up the job and before the first run. It must read as 0, not as broken."""
    p = os.path.join(_TMP, "empty.jsonl")
    io.open(p, "w", encoding="utf-8").write("")
    f = ledger_panel(p)
    assert f["available"] is False and f["recorded"] == 0
    assert "0 recorded" in f["reason"], f["reason"]


def test_a_corrupt_ledger_line_is_reported_not_silently_skipped():
    p = os.path.join(_TMP, "corrupt.jsonl")
    io.open(p, "w", encoding="utf-8").write('{"id":0,"status":"open"}\n{not json\n')
    f = ledger_panel(p)
    assert f["available"] is False and f["reason"]


def test_a_real_ledger_reaches_the_panel_with_its_counts_and_series():
    f = ledger_panel(_ledger_file(), dte_hint=21)
    assert f["available"] is True
    assert f["recorded"] > 0 and f["settled"] > 0
    assert f["recorded"] == f["settled"] + f["open"], (
        "recorded must equal settled plus open, or one of the three is wrong")
    assert len(f["series"]) == f["recorded"]
    for d in f["series"]:
        assert set(d) >= {"date", "pVol", "qVol", "vrp", "traded", "settled",
                          "coverage", "pnl"}


def test_the_panel_counts_agree_with_the_ledger_itself():
    """The panel must not become a second, drifting source of truth for the two
    numbers the whole project is waiting on."""
    from tools.paper_trade import PaperLedger
    path = _ledger_file()
    rep = PaperLedger.load(path).report()
    f = ledger_panel(path, dte_hint=21)
    assert (f["recorded"], f["settled"], f["open"]) == (rep.n_recorded,
                                                        rep.n_settled, rep.n_open)
    assert f["nTrades"] == rep.n_trades and f["totalPnl"] == rep.total_pnl
    assert f["verdict"] == rep._verdict()


def test_the_signal_fired_count_and_the_graded_count_are_kept_separate():
    """They differ — a date the signal traded stays ungraded until its horizon
    elapses — and showing them under one label reads as an inconsistency."""
    f = ledger_panel(_ledger_file(), dte_hint=21)
    fired = sum(1 for d in f["series"] if d["traded"])
    assert fired >= f["nTrades"], (fired, f["nTrades"])
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    draw = page[page.index("function drawForward()"):page.index("function drawTest()")]
    assert "Signal fired" in draw and "Graded trades" in draw
    assert "graded so far" in draw


def test_dates_missing_from_the_source_do_not_render_a_broken_range():
    """A synthetic chain carries no asof, so every entry's date is empty. " - "
    is a broken range; saying the dates are absent is the actual state."""
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    draw = page[page.index("function drawForward()"):page.index("function drawTest()")]
    assert "dates not recorded" in draw
    f = ledger_panel(_ledger_file(), dte_hint=21)
    assert f["firstDate"] == "", "the fixture is supposed to have no dates"


def test_the_lag_to_a_first_score_is_shown_while_nothing_has_settled():
    """The honest cost of an out-of-sample test. Zero settled with no explanation
    looks like a broken panel rather than a clock that has not run long enough."""
    path = os.path.join(_TMP, "young.jsonl")
    src = [l for l in io.open(_ledger_file(), encoding="utf-8")][:3]
    out = []
    for line in src:
        e = json.loads(line)
        e["status"] = "open"
        e.pop("trade_pnl", None)
        out.append(json.dumps(e))
    io.open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    f = ledger_panel(path, dte_hint=21)
    assert f["available"] is True and f["settled"] == 0
    assert f["needMoreRecords"] > 0, f
    assert f["needMoreRecords"] == max(f["dte"] - f["recorded"], 0)


def test_the_forward_panel_sits_between_the_decision_and_the_rotation_chart():
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    assert page.index('id="opt"') < page.index('id="fwd"') < page.index('id="rrg"')
    assert "function drawForward()" in page
    body = page[page.rindex("function drawForward()"):]
    assert "drawForward();" in body, "defined but never invoked"


def test_thin_coverage_dates_are_flagged_rather_than_averaged_in_silently():
    """A date whose wings did not reach +/-10% has a Q biased LOW, which inflates
    the apparent variance premium in exactly the regimes that matter."""
    f = ledger_panel(_ledger_file(), dte_hint=21)
    assert any("coverage" in d for d in f["series"])
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    draw = page[page.index("function drawForward()"):page.index("function drawTest()")]
    assert "biased LOW" in draw and "d.coverage" in draw


def test_a_date_that_never_recorded_its_coverage_is_not_treated_as_clean():
    """An UNMARKED point on the chart reads as "checked and good". A date missing
    the flag was never checked, so defaulting it to trustworthy is a claim nobody
    made — it is marked, with its own reason."""
    p = os.path.join(_TMP, "nocov.jsonl")
    rows = []
    for line in list(io.open(_ledger_file(), encoding="utf-8"))[:4]:
        e = json.loads(line)
        e.pop("coverage_ok", None)
        rows.append(json.dumps(e))
    io.open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    f = ledger_panel(p, dte_hint=21)
    assert f["available"] is True
    assert all(d["coverage"] is False for d in f["series"]), (
        "a missing coverage flag was read as coverage confirmed")
    assert all(d["coverageKnown"] is False for d in f["series"])
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    draw = page[page.index("function drawForward()"):page.index("function drawTest()")]
    assert "did not record whether" in draw, "the two reasons read the same"


def test_a_losing_ledger_is_printed_as_losing():
    """abs() on a total is the single cheapest way to turn a failing forward test
    into a passing-looking one, and the positive fixture cannot catch it."""
    p = os.path.join(_TMP, "losing.jsonl")
    rows = []
    for line in io.open(_ledger_file(), encoding="utf-8"):
        e = json.loads(line)
        if e.get("trade_pnl") is not None:
            e["trade_pnl"] = -abs(e["trade_pnl"]) - 1.0
        rows.append(json.dumps(e))
    io.open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    f = ledger_panel(p, dte_hint=21)
    assert f["nTrades"] > 0
    assert f["totalPnl"] < 0, f["totalPnl"]


def test_the_lag_line_disappears_once_something_has_settled():
    """Telling an operator to keep waiting when the score already exists is worse
    than showing nothing: it hides the answer behind a countdown.

    The short ledger is the case that matters. With 3 settled entries at a 21-day
    horizon the arithmetic `horizon - recorded` still returns 18, so a countdown
    that is not gated on `settled == 0` would demand 18 more records from someone
    already holding a graded result."""
    f = ledger_panel(_ledger_file(), dte_hint=21)
    assert f["settled"] > 0 and f["recorded"] > f["dte"]
    assert f["needMoreRecords"] == 0, f["needMoreRecords"]

    short = os.path.join(_TMP, "short_settled.jsonl")
    rows = [l for l in io.open(_ledger_file(), encoding="utf-8")
            if json.loads(l).get("status") == "settled"][:3]
    io.open(short, "w", encoding="utf-8").writelines(rows)
    g = ledger_panel(short, dte_hint=21)
    assert (g["recorded"], g["settled"]) == (3, 3), (g["recorded"], g["settled"])
    assert g["dte"] - g["recorded"] == 18, "the fixture no longer tests the gate"
    assert g["needMoreRecords"] == 0, (
        f"asked for {g['needMoreRecords']} more records while already holding "
        f"{g['settled']} graded results")


def test_the_payload_always_carries_a_ledger_key():
    payload = build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                            horizon=21, run_test=False)
    assert isinstance(payload["ledger"], dict)
    assert payload["ledger"]["available"] is False


def test_an_absent_date_range_is_guarded_in_the_page_itself():
    """The fixture has no dates, so an unguarded range renders as a bare dash. The
    guard is pinned in the source because a portable test cannot run the DOM."""
    page = render(build_payload(*demo_world(n=700), window=63, mom_lag=5, tail=12,
                                horizon=21, run_test=False))
    draw = page[page.index("function drawForward()"):page.index("function drawTest()")]
    assert "f.firstDate && f.lastDate" in draw, (
        "the date range is printed unconditionally")


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
