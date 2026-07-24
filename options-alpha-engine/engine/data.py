"""Data model + adapters.

`OptionQuote` / `OptionChain` are the interface the whole engine speaks. Real
data (Polygon, ORATS, CBOE, IBKR) plugs in by implementing `MarketDataAdapter`
and returning these objects — nothing downstream changes.

A deterministic synthetic generator is included so the demo runs out of the box
with zero data vendor and zero API key. It is NOT a market simulator; it exists
so the plumbing is exercisable end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .pricing import price


@dataclass
class OptionQuote:
    expiry_days: int      # calendar days to expiry
    strike: float
    kind: str             # "call" or "put"
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def T(self) -> float:
        return self.expiry_days / 365.0


@dataclass
class OptionChain:
    symbol: str
    spot: float
    r: float
    q: float
    quotes: list[OptionQuote]
    asof: str | None = None   # ISO snapshot date of this chain (look-ahead contract)
    # LOOK-AHEAD CONTRACT: any price history fed to a P-forecast alongside this
    # chain MUST be truncated to <= asof. The physical model and the Q-snapshot
    # have to be aligned to the same instant, or the future leaks into the edge.
    # Enforced by convention in Phase-1 (synthetic data is a single instant);
    # becomes a hard assertion once real timestamped data lands.


class MarketDataAdapter(Protocol):
    """Implement this against a real vendor to go live."""

    def price_history(self, symbol: str, days: int) -> list[float]:
        ...

    def option_chain(self, symbol: str) -> OptionChain:
        ...


class SyntheticAdapter:
    """Deterministic synthetic data so the pipeline is runnable with no vendor.

    Prices follow a fixed pseudo-random walk (seeded, no external randomness),
    and the option chain is generated with a deliberate volatility *skew* and a
    small rich/cheap dislocation so the mispricing scanner has something to find.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    def _lcg(self):
        # Minimal linear-congruential generator -> reproducible without `random`.
        x = self._seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x / 0x7FFFFFFF

    def price_history(self, symbol: str, days: int = 260) -> list[float]:
        gen = self._lcg()
        s = 100.0
        prices = [s]
        # Regime with ~18% annual vol and slight upward drift.
        daily_vol = 0.18 / math.sqrt(252)
        drift = 0.06 / 252
        for _ in range(days):
            u = next(gen)
            # Box-Muller-ish shock from a uniform (good enough for a fixture).
            z = (u - 0.5) * 2 * math.sqrt(3)  # unit-variance uniform shock
            s *= math.exp(drift - 0.5 * daily_vol ** 2 + daily_vol * z)
            prices.append(s)
        return prices

    def option_chain(self, symbol: str, spot: float | None = None) -> OptionChain:
        gen = self._lcg()
        spot = spot if spot is not None else self.price_history(symbol)[-1]
        r, q = 0.05, 0.0
        quotes: list[OptionQuote] = []
        for dte in (7, 30, 60):
            T = dte / 365.0
            for moneyness in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
                strike = round(spot * (1 + moneyness), 0)
                # True vol surface: base + smile + term structure.
                base_vol = 0.17 + 0.9 * moneyness ** 2 + 0.01 * (dte / 30)
                # Inject a small rich/cheap dislocation the scanner should catch.
                dislocation = 0.03 if (dte == 30 and abs(moneyness) < 0.03) else 0.0
                market_vol = base_vol + dislocation
                for kind in ("call", "put"):
                    fair = price(spot, strike, T, r, q, market_vol, kind)
                    if fair < 0.05:
                        continue
                    # Realistic-ish spread: wider for cheap/OTM contracts.
                    half_spread = max(0.02, 0.02 * fair) + 0.01
                    quotes.append(OptionQuote(
                        expiry_days=dte,
                        strike=strike,
                        kind=kind,
                        bid=round(max(fair - half_spread, 0.01), 2),
                        ask=round(fair + half_spread, 2),
                    ))
        return OptionChain(symbol=symbol, spot=spot, r=r, q=q, quotes=quotes)
