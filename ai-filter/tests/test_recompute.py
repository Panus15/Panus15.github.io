"""Tests for engine/recompute.py. Offline, stdlib only.

Run: python3 tests/test_recompute.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import Claim, ClaimKind, Status
from engine import recompute

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED
_n = [0]


def claim(expr, claimed, kind=ClaimKind.ARITHMETIC):
    _n[0] += 1
    return Claim(f"c{_n[0]}", kind, f"{expr} = {claimed}", 0, 0, {"expression": expr, "claimed": claimed})


def run(expr, claimed):
    out = recompute.check([claim(expr, claimed)])
    assert len(out) == 1 and out[0].module == "recompute"
    return out[0]


def st(expr, claimed):
    return run(expr, claimed).status


def test_float_traps_are_exact():
    assert st("0.1+0.2", "0.3") == V
    assert st("0.1 + 0.7", "0.8") == V
    assert st("1.1 * 1.1", "1.21") == V
    assert st("0.1*3", "0.3") == V


def test_basic_and_wrong_sums():
    assert st("2+2", "4") == V
    assert st("2+2", "5") == C
    assert st("17 * 23", "391") == V
    assert st("17 * 23", "392") == C
    assert st("1000000 * 1000000", "1000000000000") == V
    assert st("1 - 3", "-2") == V
    r = run("2+2", "5")
    assert "4" in r.reason and "5" in r.reason


def test_reason_shows_both_values():
    r = run("1200 * 5%", "60")
    assert r.status == V and "60" in r.reason and "matches" in r.reason
    r = run("7 * 6", "41")
    assert "42" in r.reason and "41" in r.reason


def test_percent_forms():
    assert st("1200 * 5%", "60") == V
    assert st("15% * 2000", "300") == V
    assert st("15% * 2000", "30") == C
    assert st("50 / 200", "25%") == V
    assert st("50 / 200", "26%") == C
    assert st("50 / 200", "25") == C          # unit mismatch: 0.25 is not 25
    assert st("5%", "0.05") == V
    assert st("5%", "5%") == V
    assert st("(1+1)%", "0.02") == V
    assert st("200 * 12.5%", "25") == V


def test_additive_percent_has_two_readings():
    assert st("100 + 10%", "110") == V        # as people mean it
    assert st("100 + 10%", "100.1") == V      # literal reading
    assert st("200 - 25%", "150") == V
    assert st("100 + 10%", "111") == C        # neither reading
    assert "percentage of the base" in run("100 + 10%", "110").reason


def test_precedence_and_associativity():
    assert st("-2^2", "-4") == V
    assert st("-2^2", "4") == C
    assert st("(-2)^2", "4") == V
    assert st("2^3^2", "512") == V
    assert st("2^3^2", "64") == C
    assert st("2 + 3 * 4", "14") == V
    assert st("(2 + 3) * 4", "20") == V
    assert st("2^-2", "0.25") == V
    assert st("10 - 2 - 3", "5") == V
    assert st("100 / 10 / 2", "5") == V
    assert st("2 * -3", "-6") == V
    assert st("--3", "3") == V
    assert st("2**3", "8") == V


def test_rounding_policy():
    assert run("100/3", "33.3").status == V and "rounded" in run("100/3", "33.3").reason
    assert st("100/3", "33.33") == V
    assert st("100/3", "33.4") == C
    assert st("100/3", "33.2") == C
    assert st("2/3", "0.67") == V             # rounded
    r = run("2/3", "0.66")                    # truncated
    assert r.status == U and "only if truncated" in r.reason   # not stamped as checked
    assert st("2/3", "0.68") == C
    assert st("2/3", "0.6") == U              # truncation to 1 place
    assert st("1/8", "0.125") == V            # exact
    assert st("1/8", "0.13") == V             # round half up
    assert st("1/8", "0.12") == U             # truncation
    r = run("10/4", "2")                      # d=0: round-half-up gives 3, truncation 2
    assert r.status == U and "only if truncated" in r.reason
    assert st("10/4", "3") == V               # rounded
    assert st("10/4", "4") == C
    assert st("-10/4", "-3") == V             # half away from zero
    assert st("-10/4", "-2") == U
    assert st("1/3", "33%") == V              # percent: 33.33.. rounds to 33
    assert st("2/3", "66%") == U              # truncated
    assert st("2/3", "67%") == V
    assert st("2/3", "70%") == C
    assert st("1/3", "0.3") == V
    assert st("1/3", "0.4") == C


def test_claimed_formats():
    assert st("500 * 2", "1,000") == V
    assert st("500 * 2", "1000.0") == V
    assert st("500 * 2", "+1000") == V
    assert st("500 * 2", " 1000 ") == V
    assert st("1 - 3", "−2") == V
    assert st("5/10", "50 %") == V
    assert st("2+2", "four") == U
    assert st("2+2", "") == U
    assert st("2+2", "4 apples") == U
    assert st("2+2", "1e3") == U
    assert st("2+2", None) == U
    assert st("2+2", "4" * 500) == U


def test_division_by_zero_is_contradicted():
    r = run("5/0", "0")
    assert r.status == C and "division by zero is undefined" in r.reason.lower()
    assert st("1/(2-2)", "1") == C
    assert st("0^-1", "1") == C
    assert st("0^0", "1") == V
    assert st("0/5", "0") == V


def test_fractional_exponents_are_approximate():
    r = run("2^0.5", "1.414")
    assert r.status == V and "approximate" in r.reason
    assert st("2^0.5", "1.5") == C
    assert st("4^0.5", "2") == V
    assert st("64^(1/3)", "4") == V
    assert st("(-8)^0.5", "2") == U
    assert st("0^0.5", "0") == U


def test_dos_guards_are_fast_and_unverified():
    for expr in ("2^10000", "9^9^9", "2**9999999", "9^9^9^9", "10^1001", "(2^1000)^1000",
                 "99999999999^999", "2^0.5^9999999", "2^100000.5"):
        t = time.time()
        r = run(expr, "1")
        assert time.time() - t < 1.0, expr
        assert r.status == U, (expr, r)
    assert st("2^1000", "1") == C             # within limits: really computed
    assert st("1^100000", "1") == V           # trivial base is cheap
    assert st("(-1)^100001", "-1") == V
    assert st("1" * 201, "1") == U            # too long
    assert st("1+" * 99 + "1", "100") == V    # 199 chars, ok
    assert st("(" * 150 + "1" + ")" * 150, "1") == U     # length guard
    assert st("(" * 90 + "1" + ")" * 90, "1") == U       # depth guard
    assert st("-" * 190 + "1", "1") == U


def test_garbage_is_unverified_never_contradicted():
    for expr in ("", "   ", "2 +", "+", "* 3", "()", "(1", "1)", "5%%", "1 2", "2 3 + 4", "1.2.3",
                 "abc", "2 x 3", "1,000 + 1", "2(3)", "%", "^2", "3 ^", "1 / ", "((", "5 % 3",
                 "\x00", "١+٢"):
        r = run(expr, "3")
        assert r.status == U, (expr, r)
    assert run("2 +", "4").reason.lower().startswith("could not parse")


def test_injection_attempts_never_execute():
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pwned_marker")
    attempts = ["__import__('os').system('touch %s')" % marker,
                "__import__('os')",
                "open('%s','w')" % marker,
                "exec('1')", "eval('1')", "().__class__", "1;2", "1\n+2\nimport os",
                "[x for x in range(10**9)]", "9**9**9", "10**10**10", "2**9999999"]
    for a in attempts:
        t = time.time()
        r = run(a, "1")
        assert time.time() - t < 1.0
        assert r.status == U, (a, r)
    assert not os.path.exists(marker)


def test_non_arithmetic_claims_ignored_and_order_kept():
    other = claim("2+2", "4", kind=ClaimKind.VAGUE)
    a, b = claim("1+1", "2"), claim("1+1", "3")
    out = recompute.check([other, a, other, b])
    assert [f.claim.id for f in out] == [a.id, b.id]
    assert [f.status for f in out] == [V, C]
    assert recompute.check([]) == []


def test_missing_data_is_unverified():
    c = Claim("x", ClaimKind.ARITHMETIC, "", 0, 0, {})
    assert recompute.check([c])[0].status == U
    c = Claim("x", ClaimKind.ARITHMETIC, "", 0, 0, {"expression": 5, "claimed": "5"})
    assert recompute.check([c])[0].status == U


def test_huge_but_allowed_values_format_without_crashing():
    r = run("10^1000", "1")                   # 1001 digits, > 3000 bits: scientific in the reason
    assert r.status == C and "e+1000" in r.reason
    r = run("9^999", "1")
    assert r.status == C and len(r.reason) < 400


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
