"""Macro regime gate — economic state as a FORWARD-LOOKING vol-risk veto.

The engine already has two regime gates: the backward-looking price gate
(edge.regime_stressed — realised vol accelerating) and the news gate
(news_signal — a burst of negative/dispersed headlines). This adds the third
leg the user asked for: the MACROECONOMIC state. Like the other two it is NOT a
directional bet — it does not say "rates up so stocks down". It says "the macro
backdrop is one in which a volatility spike is more likely, so stop SELLING vol",
which is the only thing a variance-premium engine can honestly use macro for.

Four battle-tested stress signals, each a leading indicator of a vol regime:
  * yield-curve inversion   (10y - 3m < 0): the most reliable recession lead.
  * credit-spread blowout   (HY-IG or similar, in bp): funding/solvency stress
                            shows up in credit before it shows up in equity vol.
  * VIX term backwardation  (near VIX > far VIX): the market pricing acute,
                            imminent stress rather than a calm forward path.
  * rapid tightening        (policy-rate change per month): a hiking shock is a
                            classic prelude to a vol regime break.

It composes with the other gates WITHOUT changing signatures: news_signal.
event_risk gains an optional macro snapshot and ORs this in, so a caller passes
the combined stressed flag to edge.compare exactly as before. Sentiment-style
honesty caveat applies doubly: macro is a coarse, slow overlay — it gates the
vol signal, it never sizes or directs a trade on its own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MacroSnapshot:
    """A dated snapshot of the macro backdrop. All rates/spreads in DECIMAL
    (0.045 = 4.5%); ``credit_spread`` in decimal too (0.05 = 500bp)."""
    asof: str
    short_rate: float = 0.0        # e.g. 3-month T-bill yield
    long_rate: float = 0.0         # e.g. 10-year yield
    credit_spread: float = 0.0     # e.g. HY option-adjusted spread
    rate_change_1m: float = 0.0    # change in the policy rate over the last month
    vix: float | None = None       # spot VIX (front-month), in vol points (e.g. 18.0)
    vix_3m: float | None = None    # ~3-month VIX, in vol points

    @property
    def curve_slope(self) -> float:
        """10y - 3m. Negative = inverted (a recession lead)."""
        return self.long_rate - self.short_rate

    @property
    def vix_term_slope(self) -> float | None:
        """far - near VIX. Negative = backwardation = acute near-term stress."""
        if self.vix is None or self.vix_3m is None:
            return None
        return self.vix_3m - self.vix


def macro_regime_gate(
    snap: MacroSnapshot,
    *,
    inversion_thresh: float = -0.001,     # curve slope below this (-10bp) = inverted
    credit_thresh: float = 0.06,          # credit spread above this (600bp) = stress
    vix_backwardation_thresh: float = -0.5,  # near - far VIX steeper than this
    tightening_thresh: float = 0.005,     # +50bp/month policy move = hiking shock
) -> tuple[bool, str]:
    """Return (elevated_macro_risk, reason). Fires when the macro backdrop favours
    a vol spike — the engine should then stop selling vol. Reasons are ORed; the
    first that trips is reported."""
    if snap.curve_slope <= inversion_thresh:
        return True, f"yield curve inverted (10y-3m = {snap.curve_slope*100:+.0f}bp)"
    if snap.credit_spread >= credit_thresh:
        return True, f"credit spreads wide ({snap.credit_spread*100:.0f}bp)"
    ts = snap.vix_term_slope
    if ts is not None and ts <= vix_backwardation_thresh:
        return True, f"VIX term backwardation (3m-spot = {ts:+.1f})"
    if snap.rate_change_1m >= tightening_thresh:
        return True, f"rapid tightening ({snap.rate_change_1m*100:+.0f}bp/1m)"
    return False, "macro benign"


def macro_stress_at(snapshot_at, **gate_kwargs):
    """Build a ``stress_at(t, trailing) -> bool`` callable for
    engine.signal_backtest.run_signal_backtest from a ``snapshot_at(t) ->
    MacroSnapshot | None`` lookup. Skips it causes are counted under
    ``skip_reasons['news']`` (the forward-gate bucket)."""
    def _stress(t, trailing):
        snap = snapshot_at(t)
        return snap is not None and macro_regime_gate(snap, **gate_kwargs)[0]
    return _stress


@dataclass
class FileMacroAdapter:
    """OFFLINE macro source: load dated MacroSnapshots from a JSON file.

    JSON: list of objects with the MacroSnapshot fields (asof required). Use
    ``at(date)`` to get the most recent snapshot on or before a decision date
    (no look-ahead), or ``latest()`` for the newest."""
    path: str

    def _load(self) -> list[MacroSnapshot]:
        import json
        with open(self.path) as fh:
            rows = json.load(fh)
        snaps = [MacroSnapshot(**{k: v for k, v in row.items()
                                  if k in MacroSnapshot.__dataclass_fields__}) for row in rows]
        return sorted(snaps, key=lambda s: s.asof)

    def at(self, asof: str) -> MacroSnapshot | None:
        prior = [s for s in self._load() if s.asof <= asof]
        return prior[-1] if prior else None

    def latest(self) -> MacroSnapshot | None:
        snaps = self._load()
        return snaps[-1] if snaps else None
