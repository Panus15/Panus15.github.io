"""Before you buy an option in your broker's app: is it priced above the vol the
stock has actually been showing?

    python -m tools.option_check IREN --iv 121.42 --days 24
    python -m tools.option_check IREN --iv 121.42 --days 24 --spot 16.39 \\
        --strike 16 --price 2.41 --kind call
    python -m tools.option_check IREN --iv 121.42 --days 24 --prices-csv iren.csv

WHY THIS EXISTS. The person this repo is for can trade US options in one app, and
that app lets them BUY calls and puts. Everything else here assumes a chain feed
and the freedom to sell; neither is available to them. What IS available is the
screen every broker shows -- an implied vol, a strike, a premium -- and a free
daily price history. That is enough to answer the question a buyer most needs
answered and least often asks: am I paying for more movement than this stock has
been making?

WHAT IT COMPARES. The IV on the screen is the market's price for movement over
the option's life. Against it goes `volforecast.term_vol`, the same horizon-aware
forecast the rest of the engine uses (HAR-RV, mean-reverting toward the 252-day
level), on the stock's own closes. Their ratio is printed, with the market's
one-sd move beside the forecast's, and -- when a strike and premium are given --
the break-even, the move needed to reach it, and the premium a buyer would pay
at the forecast vol instead of the market's.

WHAT IT IS NOT. It is not a signal and it does not say the stock will go up or
down. The bands that turn the ratio into a word (OVERPRICED / FAIR / CHEAP) are
reading aids, NOT measured thresholds: nothing in this repo has shown that a
ratio above 1.25 loses money for a buyer. The forward ledger is the only thing
that can show that, and it has not yet. Single names also carry scheduled jumps
-- earnings above all -- that a history of closes cannot see, which is why every
report ends by telling you to check the earnings date yourself.

Prices here are close-to-close; the probability it prints assumes a lognormal
move at the FORECAST vol with no drift, which understates fat tails on exactly
the kind of stock this is usually run on. Read it as a rough guide.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing, volforecast

#: Fewest closes the forecast is allowed to run on. HAR-RV needs a month of daily
#: variance to regress on and the term structure anchors to a 252-day level; below
#: this the "forecast" would be mostly its own fallbacks.
MIN_CLOSES = 120

#: Reading aids, not measured edges -- see the module docstring.
RICH_RATIO = 1.25
CHEAP_RATIO = 0.85


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def assess(closes, *, iv: float, days: int, spot: float | None = None,
           strike: float | None = None, premium: float | None = None,
           kind: str = "call", r: float = 0.04) -> dict:
    """Compare a quoted implied vol (as a FRACTION, 1.2142 = 121.42%) with the
    horizon forecast from `closes`. Returns every number the report prints."""
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not iv or iv <= 0:
        raise ValueError("iv must be positive (as a fraction: 1.2142 for 121.42%)")
    if not days or days <= 0:
        raise ValueError("days to expiry must be positive")
    closes = [float(c) for c in closes if c and c > 0]
    if len(closes) < MIN_CLOSES:
        raise ValueError(f"need at least {MIN_CLOSES} daily closes, got {len(closes)}")
    if (strike is None) != (premium is None):
        raise ValueError("give both --strike and --price, or neither")

    T = days / 365.0
    s = float(spot) if spot else closes[-1]
    fvol = volforecast.term_vol(closes, T)
    rv21 = volforecast.close_to_close_vol(closes, 21)
    ratio = iv / fvol
    band = ("OVERPRICED" if ratio >= RICH_RATIO
            else "CHEAP" if ratio <= CHEAP_RATIO else "FAIR")

    out = {"spot": s, "days": days, "T": T, "iv": iv, "forecast_vol": fvol,
           "realized_21d": rv21, "ratio": ratio, "gap_points": (iv - fvol) * 100,
           "band": band,
           "market_move": s * iv * math.sqrt(T),
           "forecast_move": s * fvol * math.sqrt(T)}

    if strike is not None:
        K, p = float(strike), float(premium)
        be = K + p if kind == "call" else K - p
        fair = pricing.price(s, K, T, r, 0.0, fvol, kind)
        sig = fvol * math.sqrt(T)
        # P(finish beyond break-even) under a driftless lognormal at the forecast vol
        d = (math.log(be / s) + 0.5 * sig * sig) / sig if be > 0 else -math.inf
        p_beyond = (1.0 - _ncdf(d)) if kind == "call" else _ncdf(d)
        out.update({"kind": kind, "strike": K, "premium": p, "breakeven": be,
                    "move_needed": be / s - 1.0, "fair_at_forecast": fair,
                    "overpay_per_contract": (p - fair) * 100.0,
                    "p_beyond_breakeven": p_beyond})
    return out


def report(a: dict, symbol: str) -> str:
    L = [f"{symbol}  ราคาหุ้น {a['spot']:.2f}  เหลือ {a['days']} วัน",
         "",
         f"  ตลาดตั้งความผันผวน (IV)        {a['iv']:7.1%}",
         f"  หุ้นผันผวนจริง 21 วันล่าสุด     {a['realized_21d']:7.1%}",
         f"  ระบบคาดช่วงอายุออปชั่นนี้      {a['forecast_vol']:7.1%}",
         f"  IV / ที่คาด                    {a['ratio']:7.2f}x   ({a['gap_points']:+.1f} vol points)",
         "",
         f"  ตลาดคิดว่าหุ้นจะวิ่ง ±{a['market_move']:.2f} (1 sd)  "
         f"ส่วนที่หุ้นเคยวิ่งจริง ±{a['forecast_move']:.2f}"]
    if "strike" in a:
        side = "ซื้อ Call" if a["kind"] == "call" else "ซื้อ Put"
        L += ["",
              f"  {side} strike {a['strike']:g} ราคา {a['premium']:.2f}",
              f"    จุดคุ้มทุนตอนหมดอายุ        {a['breakeven']:.2f}  "
              f"(หุ้นต้องวิ่ง {a['move_needed']:+.1%})",
              f"    ราคาที่ควรเป็น (ตามที่คาด)   {a['fair_at_forecast']:.2f}",
              f"    จ่ายเกิน/ต่ำกว่าต่อสัญญา      {a['overpay_per_contract']:+.0f} ดอลลาร์",
              f"    โอกาสจบเลยจุดคุ้มทุน (คร่าวๆ) {a['p_beyond_breakeven']:.0%}"]
    word = {"OVERPRICED": "ออปชั่นแพงกว่าที่หุ้นเคยวิ่งจริง — ฝั่งซื้อเสียเปรียบ",
            "CHEAP": "ออปชั่นถูกกว่าที่หุ้นเคยวิ่งจริง — ฝั่งซื้อได้เปรียบ",
            "FAIR": "ราคาใกล้เคียงกับที่หุ้นเคยวิ่งจริง"}[a["band"]]
    L += ["",
          f"  อ่านว่า: {a['band']} — {word}",
          "  (เป็นตัวช่วยอ่าน ไม่ใช่สัญญาณที่พิสูจน์แล้ว)",
          "  !! เช็กวันประกาศงบเองเสมอ: ถ้าประกาศงบก่อนหมดอายุ IV สูงเพราะเหตุนั้น"]
    return "\n".join(L)


def _closes_from_csv(path: str) -> list:
    from tools.merge_csv import read_series
    rows = read_series(path)
    return [rows[d] for d in sorted(rows)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("symbol")
    ap.add_argument("--iv", type=float, required=True,
                    help="implied vol as shown in the app, in PERCENT (121.42)")
    ap.add_argument("--days", type=int, required=True,
                    help="calendar days left to expiry")
    ap.add_argument("--spot", type=float, help="current stock price (default: last close)")
    ap.add_argument("--strike", type=float)
    ap.add_argument("--price", type=float, help="the option's premium per share")
    ap.add_argument("--kind", default="call", choices=["call", "put"])
    ap.add_argument("--prices-csv", dest="prices_csv",
                    help="read closes from this date,close CSV instead of fetching")
    ap.add_argument("--cache", default="prices",
                    help="where fetched closes are kept (default: prices/)")
    a = ap.parse_args(argv)
    sym = a.symbol.upper()

    if a.prices_csv:
        closes = _closes_from_csv(a.prices_csv)
    else:
        from tools.fetch_prices import fetch_one_auto
        path = os.path.join(a.cache, f"{sym}.csv")
        got = fetch_one_auto(sym, path, force=True)
        if got["status"].split()[0] not in ("fetched", "cached"):
            print(f"could not get price history for {sym}: {got['status']}")
            return 1
        closes = _closes_from_csv(path)

    try:
        res = assess(closes, iv=a.iv / 100.0, days=a.days, spot=a.spot,
                     strike=a.strike, premium=a.price, kind=a.kind)
    except ValueError as e:
        print(f"refused: {e}")
        return 1
    print(report(res, sym))
    return 0


if __name__ == "__main__":
    sys.exit(main())
