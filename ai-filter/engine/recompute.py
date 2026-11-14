"""Recompute checker: re-does the arithmetic in ARITHMETIC claims, exactly.

check(claims) -> list[Finding], one per ARITHMETIC claim (input order, other
kinds ignored), module="recompute".

How it works
------------
* The expression is parsed by a small hand-written recursive-descent parser into
  a tree of whitelisted node types (number, group, percent, unary +/-, and the
  binary operators + - * / ^).  Nothing is ever passed to eval/exec/ast.
* Evaluation uses fractions.Fraction, so 0.1 + 0.2 == 0.3 exactly.
* Grammar / precedence (loosest to tightest):
      + -   <   * /   <   unary + -   <   ^ (right associative)   <   postfix %
  so -2^2 = -4, 2^3^2 = 2^(3^2) = 512, 2^-1 = 0.5, and `5%` is 5/100.
  `**` is accepted as a spelling of `^`, and x / the division sign / the true
  minus sign are normalised.  `%%` (chained percent) is a parse failure.
* "100 + 10%" is ambiguous (100.1 literally, 110 as people mean it).  When a
  percent literal is the right operand of + or - we try BOTH readings and call
  the claim verified if either matches; it is contradicted only if neither does.

Comparison policy (claimed value vs. computed value)
----------------------------------------------------
Let `claimed` be numeric text, d = the number of decimal places written in its
numeric part.
* If claimed ends with `%` it is a percentage figure: we compare computed * 100
  to the number written (so 50 / 200 = 25% compares 25 with 25, and d refers to
  the percentage figure).  Otherwise computed is compared to the number as is.
* VERIFIED if the value equals the claimed number exactly, OR rounded half away
  from zero to d places equals it ("rounded").
* UNVERIFIED if the claim matches ONLY when truncated rather than rounded
  (2/3 = 0.66, 10/4 = 2).  Some people do truncate, so it is not proven wrong,
  but by ordinary rounding it IS wrong, and a filter must not stamp that as
  checked.  The reason says so.
* Otherwise CONTRADICTED, showing the computed value.
* Division by zero anywhere is CONTRADICTED ("division by zero is undefined"):
  no number can be right.
* Fractional exponents (2^0.5) are computed with a float (base must be > 0) and
  the value is cleaned to 12 significant digits; the reason says "approximate".
  Non-positive base with a fractional exponent is UNVERIFIED.
* UNVERIFIED (never CONTRADICTED, see models.py): expression longer than 200
  chars, unparseable, too deeply nested, intermediate result above ~10,000
  digits, an integer exponent with |exp| > 1000 (except bases 0, 1, -1), a
  claimed value that is not a plain number, or any internal error.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Sequence

from .models import Claim, ClaimKind, Finding, Status

MODULE = "recompute"
MAX_EXPR_LEN = 200
MAX_EXPONENT = 1000
MAX_BITS = 33_220            # ~10,000 decimal digits
MAX_DEPTH = 40               # nesting of parentheses / unary / exponent chains
MAX_CLAIMED_LEN = 100
_STR_BITS = 3_000            # above this we print scientific notation, never str(int)


class _ParseError(Exception):
    pass


class _TooLarge(Exception):
    pass


class _DivZero(Exception):
    pass


class _Unsupported(Exception):
    def __init__(self, why: str) -> None:
        super().__init__(why)
        self.why = why


# ---------------------------------------------------------------- parsing ---

_NUMBER = re.compile(r"\d+\.?\d*|\.\d+")


def _tokenize(expr: str) -> list:
    expr = (expr.replace("×", "*").replace("÷", "/")
                .replace("−", "-").replace("**", "^"))
    toks: list = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch in " \t":
            i += 1
            continue
        if ch in "+-*/^()%":
            toks.append(ch)
            i += 1
            continue
        m = _NUMBER.match(expr, i)
        if not m or ch not in "0123456789.":
            raise _ParseError(f"unexpected {ch!r}")
        text = m.group(0)
        ip, _, fp = text.partition(".")
        toks.append(("num", Fraction(int((ip + fp) or "0"), 10 ** len(fp))))
        i = m.end()
    return toks


class _Parser:
    def __init__(self, toks: list) -> None:
        self.t = toks
        self.i = 0
        self.depth = 0
        self.relative_possible = False

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _next(self):
        tok = self._peek()
        if tok is None:
            raise _ParseError("unexpected end")
        self.i += 1
        return tok

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _ParseError("too deeply nested")

    def parse(self):
        if not self.t:
            raise _ParseError("empty")
        node = self._additive()
        if self.i != len(self.t):
            raise _ParseError("trailing tokens")
        return node

    def _additive(self):
        left = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            right = self._term()
            if right[0] == "pct":
                self.relative_possible = True
            left = ("bin", op, left, right)
        return left

    def _term(self):
        left = self._unary()
        while self._peek() in ("*", "/"):
            op = self._next()
            right = self._unary()
            left = ("bin", op, left, right)
        return left

    def _unary(self):
        if self._peek() in ("+", "-"):
            op = self._next()
            self._enter()
            operand = self._unary()
            self.depth -= 1
            return ("neg" if op == "-" else "pos", operand)
        return self._power()

    def _power(self):
        base = self._postfix()
        if self._peek() == "^":
            self._next()
            self._enter()
            exp = self._unary()          # right associative; allows 2^-3
            self.depth -= 1
            return ("bin", "^", base, exp)
        return base

    def _postfix(self):
        node = self._primary()
        if self._peek() == "%":
            self._next()
            node = ("pct", node)
            if self._peek() == "%":
                raise _ParseError("chained percent")
        return node

    def _primary(self):
        tok = self._next()
        if isinstance(tok, tuple):
            return tok
        if tok == "(":
            self._enter()
            node = self._additive()
            if self._next() != ")":
                raise _ParseError("expected )")
            self.depth -= 1
            return ("group", node)
        raise _ParseError(f"unexpected {tok!r}")


# ------------------------------------------------------------- evaluation ---

class _Evaluator:
    def __init__(self, relative: bool) -> None:
        self.relative = relative
        self.approx = False

    @staticmethod
    def _chk(v: Fraction) -> Fraction:
        if v.numerator.bit_length() > MAX_BITS or v.denominator.bit_length() > MAX_BITS:
            raise _TooLarge()
        return v

    def ev(self, node) -> Fraction:
        kind = node[0]
        if kind == "num":
            return self._chk(node[1])
        if kind == "group":
            return self.ev(node[1])
        if kind == "pct":
            return self._chk(self.ev(node[1]) / 100)
        if kind == "neg":
            return -self.ev(node[1])
        if kind == "pos":
            return self.ev(node[1])
        _, op, ln, rn = node
        if op in "+-" and self.relative and rn[0] == "pct":
            left = self.ev(ln)
            p = self.ev(rn[1]) / 100
            return self._chk(left * (1 + p if op == "+" else 1 - p))
        left = self.ev(ln)
        right = self.ev(rn)
        if op == "+":
            return self._chk(left + right)
        if op == "-":
            return self._chk(left - right)
        if op == "*":
            return self._chk(left * right)
        if op == "/":
            if right == 0:
                raise _DivZero()
            return self._chk(left / right)
        return self._power(left, right)

    def _power(self, b: Fraction, e: Fraction) -> Fraction:
        if e.denominator == 1:
            n = e.numerator
            if b == 0:
                if n < 0:
                    raise _DivZero()
                return Fraction(1 if n == 0 else 0)
            if b == 1:
                return Fraction(1)
            if b == -1:
                return Fraction(1 if n % 2 == 0 else -1)
            if abs(n) > MAX_EXPONENT:
                raise _TooLarge()
            if n != 0 and max(b.numerator.bit_length(), b.denominator.bit_length()) * abs(n) > MAX_BITS:
                raise _TooLarge()
            return self._chk(b ** n)
        # fractional exponent: float, approximate, positive base only
        if b <= 0:
            raise _Unsupported("a fractional power of a non-positive number cannot be checked exactly")
        if abs(e) > MAX_EXPONENT:
            raise _TooLarge()
        if b == 1:
            return Fraction(1)
        try:
            r = float(b) ** float(e)
        except OverflowError:
            raise _TooLarge()
        if not math.isfinite(r) or r > 1e300:
            raise _TooLarge()
        if r < 1e-300:
            raise _Unsupported("the result is too small to check")
        self.approx = True
        return self._chk(Fraction(f"{r:.12g}"))


# -------------------------------------------------------------- formatting ---

def _decimal(x: Fraction, places: int) -> str:
    """x rounded half away from zero to `places` decimals, as text."""
    sign = "-" if x < 0 else ""
    a = abs(x) * 10 ** places
    q = (2 * a.numerator + a.denominator) // (2 * a.denominator)
    s = str(q).rjust(places + 1, "0")
    body = s[:-places] + "." + s[-places:] if places else s
    return sign + body if q else body


def _sci(x: Fraction) -> str:
    sign = "-" if x < 0 else ""
    num, den = abs(x.numerator), x.denominator
    lg = math.log10(num) - math.log10(den)
    e = math.floor(lg)
    mant = 10 ** (lg - e)
    return f"about {sign}{mant:.4g}e{e:+d}"


def _fmt(x: Fraction) -> str:
    if x.numerator.bit_length() > _STR_BITS or x.denominator.bit_length() > _STR_BITS:
        return _sci(x)
    if x.denominator == 1:
        return str(x.numerator) if x.numerator.bit_length() <= 200 else _sci(x)
    d, twos, fives = x.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d == 1 and max(twos, fives) <= 20:
        return _decimal(x, max(twos, fives))
    if abs(x) < Fraction(1, 10 ** 6):
        return _sci(x)
    return "about " + _decimal(x, 10)


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- claimed ---

_CLAIMED = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
_THOUSANDS = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?")


def _parse_claimed(text: str):
    """-> (number as Fraction, decimals d, is_percent) or None."""
    if not isinstance(text, str) or len(text) > MAX_CLAIMED_LEN:
        return None
    s = text.strip().replace("−", "-")
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    if _THOUSANDS.fullmatch(s):
        s = s.replace(",", "")
    if not _CLAIMED.fullmatch(s):
        return None
    neg = s.startswith("-")
    body = s.lstrip("+-")
    ip, _, fp = body.partition(".")
    val = Fraction(int((ip + fp) or "0"), 10 ** len(fp))
    return (-val if neg else val), len(fp), pct


def _compare(computed: Fraction, num: Fraction, d: int, pct: bool):
    """'exact' | 'round' | 'trunc' | None.  See module docstring."""
    value = computed * 100 if pct else computed
    if value == num:
        return "exact"
    scale = 10 ** d
    target = num * scale
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    a = abs(scaled)
    rounded = (2 * a.numerator + a.denominator) // (2 * a.denominator)
    truncated = a.numerator // a.denominator
    if sign * rounded == target:
        return "round"
    if sign * truncated == target:
        return "trunc"
    return None


# ------------------------------------------------------------------- main ---

def _mk(claim: Claim, status: Status, reason: str, evidence: tuple = ()) -> Finding:
    return Finding(claim, status, MODULE, reason, evidence)


def _places(d: int) -> str:
    return "a whole number" if d == 0 else f"{d} decimal place{'s' if d != 1 else ''}"


def _check_one(claim: Claim) -> Finding:
    expr = claim.data.get("expression")
    claimed = claim.data.get("claimed")
    if not isinstance(expr, str) or not expr.strip():
        return _mk(claim, Status.UNVERIFIED, "Could not parse the expression (it is missing or empty).")
    shown = _clip(expr, 80)
    if len(expr) > MAX_EXPR_LEN:
        return _mk(claim, Status.UNVERIFIED, f"The expression is too long to check ({len(expr)} characters).")

    try:
        parser = _Parser(_tokenize(expr))
        tree = parser.parse()
    except (_ParseError, RecursionError, ValueError):
        return _mk(claim, Status.UNVERIFIED, f"Could not parse the expression \"{shown}\".")

    try:
        ev = _Evaluator(relative=False)
        computed = ev.ev(tree)
    except _DivZero:
        return _mk(claim, Status.CONTRADICTED, "Division by zero is undefined, so this has no numeric answer.",
                   (f"expression: {shown}",))
    except _TooLarge:
        return _mk(claim, Status.UNVERIFIED, f"\"{shown}\" is too large to check.")
    except _Unsupported as e:
        return _mk(claim, Status.UNVERIFIED, f"Could not check \"{shown}\": {e.why}.")
    except RecursionError:
        return _mk(claim, Status.UNVERIFIED, f"Could not parse the expression \"{shown}\" (too deeply nested).")

    parsed = _parse_claimed(claimed)
    if parsed is None:
        return _mk(claim, Status.UNVERIFIED,
                   f"The claimed value \"{_clip(str(claimed), 40)}\" is not a plain number, so it was not compared.")
    num, d, pct = parsed
    claimed_txt = _clip(str(claimed).strip(), 40)

    # candidate readings: (value, approximate?, note)
    candidates = [(computed, ev.approx, "")]
    if parser.relative_possible:
        try:
            ev2 = _Evaluator(relative=True)
            alt = ev2.ev(tree)
            if alt != computed:
                candidates.append((alt, ev2.approx,
                                   " (reading \"+ x%\" / \"- x%\" as a percentage of the base amount)"))
        except (_DivZero, _TooLarge, _Unsupported, RecursionError):
            pass

    for value, approx, note in candidates:
        how = _compare(value, num, d, pct)
        if how is None:
            continue
        shown_val = _fmt(value) if not pct else f"{_fmt(value)} (i.e. {_fmt(value * 100)}%)"
        tail = " Fractional powers are computed approximately." if approx else ""
        ev_t = (f"computed: {shown_val}", f"claimed: {claimed_txt}")
        if how == "exact":
            return _mk(claim, Status.VERIFIED,
                       f"{shown} = {shown_val}, matches the claimed {claimed_txt}{note}.{tail}",
                       ev_t)
        if how == "round":
            return _mk(claim, Status.VERIFIED,
                       f"{shown} = {shown_val}, matches the claimed {claimed_txt} when rounded to {_places(d)}{note}.{tail}",
                       ev_t)
        return _mk(claim, Status.UNVERIFIED,
                   f"{shown} = {shown_val}, which matches the claimed {claimed_txt} only if truncated "
                   f"rather than rounded to {_places(d)}; by normal rounding it is off{note}.{tail}", ev_t)

    value, approx, _ = candidates[0]
    shown_val = _fmt(value) if not pct else f"{_fmt(value)} (i.e. {_fmt(value * 100)}%)"
    extra = ""
    if len(candidates) > 1:
        extra = f" (reading a trailing \"+ x%\" as a percentage of the base amount gives {_fmt(candidates[1][0])}, also not a match)"
    tail = " (fractional powers are computed approximately)" if approx else ""
    return _mk(claim, Status.CONTRADICTED,
               f"{shown} works out to {shown_val}, not the claimed {claimed_txt}{extra}{tail}.",
               (f"computed: {shown_val}", f"claimed: {claimed_txt}"))


def check(claims: Sequence[Claim]) -> list[Finding]:
    out: list[Finding] = []
    for claim in claims:
        if claim.kind != ClaimKind.ARITHMETIC:
            continue
        try:
            out.append(_check_one(claim))
        except Exception as e:  # our own failure -> UNVERIFIED, never CONTRADICTED
            out.append(_mk(claim, Status.UNVERIFIED,
                           f"Could not check this arithmetic (internal error: {type(e).__name__})."))
    return out
