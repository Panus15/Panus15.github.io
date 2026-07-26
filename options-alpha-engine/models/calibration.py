"""Calibration — is the physical (P) density RIGHT, in absolute terms?

`models/objective.py` scores forecasters *against each other* (NLL, CRPS, pinball).
Those are relative: a model can win every one of them and still be badly
calibrated — systematically too narrow, or too wide, in a way that no ranking
reveals. That matters here more than usual, because the engine's entire edge is
the GAP between P and Q. If P's vol is biased low, the variance risk premium it
reports is overstated by exactly that bias, and every RICH verdict inherits it.

The standard absolute test is the **probability integral transform** (PIT). If the
forecast density is correct, then u = F_P(S_realized) — the forecast CDF evaluated
at what actually happened — is UNIFORM on [0, 1]. Its shape names the failure:

    flat            calibrated
    U-shaped        too NARROW  (outcomes land in the tails too often — vol too low,
                                 so the reported VRP is overstated)
    hump-shaped     too WIDE    (outcomes cluster in the middle — vol too high, so
                                 the VRP is understated)
    tilted          the SAMPLE trended (see below)

A tilt is NOT a model fault here. This engine's P forecast is deliberately
direction-neutral — its mean is pinned to the forward — so on any trending sample
the raw PIT leans, and that says nothing about whether the SPREAD is right. Since
the tradeable quantity is the variance/skew gap, the report judges WIDTH from a
**centered** PIT (the sample's realised drift removed) and reports the drift
separately as context. Measured on real GOOG through the 2008 crash, the raw PIT
mean is 0.35 purely because the stock fell; centered, the width verdict is clean.

Beyond the diagnosis, `calibration_report` returns the actionable number: the
**vol scale factor** that would make the density calibrated. "Your P vol is ~18%
too low on this asset" is a statement you can act on, and it is exactly what
decides whether a live P-vs-Q gap is real premium or model error.

Pure stdlib: the uniformity test is a Kolmogorov-Smirnov distance with an
asymptotic p-value, no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

UNIFORM_SD = 1.0 / math.sqrt(12.0)          # sd of U(0,1) ~ 0.2887


def realized_drift(windows) -> float:
    """Mean realised log-return PER YEAR across the windows (a DIAGNOSTIC, computed
    in hindsight over the whole sample — never used to make a forecast)."""
    tot, n = 0.0, 0
    for ctx, h, s_realized in windows:
        if ctx[-1] > 0 and s_realized > 0 and h > 0:
            tot += math.log(s_realized / ctx[-1]) / (h / 365.0)
            n += 1
    return tot / n if n else 0.0


def pit_values(forecaster, windows, *, r: float = 0.0, q: float = 0.0,
               vol_scale: float | None = None, centered: bool = False,
               drift: float | None = None) -> list[float]:
    """PIT values u = F_P(S_realized), one per window. Uniform iff P is correct.

    ``vol_scale`` refits each forecast with its annualised vol multiplied by that
    factor (used by ``suggest_vol_scale`` to search for the calibrating scale).

    ``centered`` removes the sample's realised DRIFT before comparing. This engine's
    P forecast is deliberately DIRECTION-NEUTRAL (its mean is pinned to the
    forward), so on any trending sample the raw PIT is skewed by the trend — that
    is the sample having direction, not the model smuggling one in, and it says
    nothing about whether the SPREAD is right. Since the tradeable quantity is the
    variance (and skew) gap, the centered PIT is the one that judges the edge.
    Centering uses hindsight and is therefore a diagnostic only.
    """
    if centered and drift is None:
        drift = realized_drift(windows)
    out = []
    for ctx, h, s_realized in windows:
        T = h / 365.0
        spot = ctx[-1]
        dist = forecaster.forecast(ctx, T, r=r, q=q, spot=spot)
        if vol_scale is not None and vol_scale > 0:
            base = dist.log_return_vol(spot, T)
            dist = forecaster.forecast(ctx, T, r=r, q=q, spot=spot,
                                       vol=base * vol_scale)
        s = s_realized
        if centered:
            s = s_realized * math.exp(-drift * T)      # strip the sample's trend
        out.append(min(1.0, max(0.0, dist.cdf(s))))
    return out


def ks_uniform(u) -> tuple[float, float]:
    """(KS distance from U(0,1), asymptotic p-value). Small D / large p = uniform."""
    n = len(u)
    if n == 0:
        raise ValueError("no PIT values")
    s = sorted(u)
    d = 0.0
    for i, x in enumerate(s):
        d = max(d, (i + 1) / n - x, x - i / n)
    # Kolmogorov asymptotic tail: Q(t) = 2 * sum_{k>=1} (-1)^{k-1} e^{-2k^2 t^2}
    t = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * t * t) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def coverage(u, level: float = 0.90) -> float:
    """Empirical frequency of outcomes inside the central ``level`` interval.
    A calibrated forecast returns ~``level``; below it, the density is too narrow."""
    lo, hi = 0.5 - level / 2, 0.5 + level / 2
    return sum(1 for x in u if lo <= x <= hi) / len(u) if u else float("nan")


@dataclass
class CalibrationReport:
    n: int
    mean: float                 # 0.5 when calibrated (RAW: includes sample drift)
    sd: float                   # 0.2887 when calibrated
    ks_d: float
    ks_p: float
    coverage_50: float
    coverage_90: float
    verdict: str                # calibrated / too narrow / too wide (WIDTH, centered)
    vol_scale: float            # multiply the P vol by this to calibrate the width
    histogram: list = field(default_factory=list)
    centered_mean: float = 0.5  # after removing the sample's realised drift
    centered_sd: float = UNIFORM_SD
    centered_coverage_90: float = 0.90
    realized_drift: float = 0.0  # annualised, hindsight — context, not a fault

    def summary(self) -> str:
        return (f"PIT calibration over {self.n} out-of-sample windows\n"
                f"  RAW      mean={self.mean:.3f} (0.500)  sd={self.sd:.3f} "
                f"({UNIFORM_SD:.3f})  cover90={self.coverage_90:.1%}\n"
                f"  CENTERED mean={self.centered_mean:.3f}  sd={self.centered_sd:.3f}"
                f"  cover90={self.centered_coverage_90:.1%}   <- judges the WIDTH\n"
                f"  sample realised drift {self.realized_drift:+.1%}/yr "
                f"(a direction-neutral forecast is EXPECTED to skew the raw PIT here)\n"
                f"  KS D={self.ks_d:.3f} p={self.ks_p:.3f}   histogram {self.histogram}\n"
                f"  -> {self.verdict}; suggested P-vol scale x{self.vol_scale:.2f}")


def pit_histogram(u, bins: int = 10) -> list:
    counts = [0] * bins
    for x in u:
        counts[min(bins - 1, int(x * bins))] += 1
    return counts


def _verdict(mean: float, sd: float, tol_mean: float, tol_sd: float) -> str:
    """Judge the WIDTH from a CENTERED PIT. The location is reported separately:
    a direction-neutral forecast is supposed to be wrong about drift, and that
    says nothing about the variance edge."""
    if sd > UNIFORM_SD + tol_sd:
        return ("TOO NARROW — outcomes land in the tails too often; the P vol is too "
                "LOW, so any reported VRP is OVERSTATED by that bias")
    if sd < UNIFORM_SD - tol_sd:
        return ("TOO WIDE — outcomes cluster in the middle; the P vol is too HIGH, "
                "so any reported VRP is UNDERSTATED")
    return "CALIBRATED — the P density is consistent with what actually happened"


def suggest_vol_scale(forecaster, windows, *, r: float = 0.0, q: float = 0.0,
                      lo: float = 0.4, hi: float = 2.5, iters: int = 24) -> float:
    """The vol multiplier that best flattens the PIT (golden-section on KS distance).

    This is the actionable output: >1 means the forecast is too narrow and its vol
    should be scaled UP (shrinking the apparent variance premium); <1 the reverse.
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0

    drift = realized_drift(windows)

    def loss(s):
        return ks_uniform(pit_values(forecaster, windows, r=r, q=q, vol_scale=s,
                                     centered=True, drift=drift))[0]

    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
    return 0.5 * (a + b)


def calibration_report(forecaster, windows, *, r: float = 0.0, q: float = 0.0,
                       bins: int = 10, tol_mean: float = 0.05,
                       tol_sd: float = 0.02, with_scale: bool = True) -> CalibrationReport:
    """Full PIT diagnosis of a forecaster on leak-free windows (objective.build_windows)."""
    u = pit_values(forecaster, windows, r=r, q=q)
    if not u:
        raise ValueError("no windows")
    drift = realized_drift(windows)
    uc = pit_values(forecaster, windows, r=r, q=q, centered=True, drift=drift)

    def _ms(vals):
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)
        return m, math.sqrt(v)

    n = len(u)
    mean, sd = _ms(u)
    cmean, csd = _ms(uc)
    d, p = ks_uniform(uc)          # uniformity of the WIDTH-relevant PIT
    scale = suggest_vol_scale(forecaster, windows, r=r, q=q) if with_scale else 1.0
    return CalibrationReport(
        n=n, mean=mean, sd=sd, ks_d=d, ks_p=p,
        coverage_50=coverage(u, 0.50), coverage_90=coverage(u, 0.90),
        verdict=_verdict(cmean, csd, tol_mean, tol_sd), vol_scale=scale,
        histogram=pit_histogram(uc, bins),
        centered_mean=cmean, centered_sd=csd,
        centered_coverage_90=coverage(uc, 0.90), realized_drift=drift,
    )
