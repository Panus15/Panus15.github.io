"""Phase-1 baseline distributional forecaster — THIS SHIPS FIRST.

It emits a `MixtureLogNormal` for the terminal price S_T at horizon T, with:
  - total volatility set by a **HAR-RV** realised-vol forecast (Corsi, 2009), and
  - a **data-driven negative skew** from a two-component (calm + crash) mixture.

Why a baseline at all: it is the fair, interpretable benchmark the future
TFT-encoder + MDN head MUST exceed out-of-sample before it earns production.
Both emit the SAME `MixtureLogNormal`, so the neural model is a drop-in for
`forecast()` and nothing downstream changes.

Skew is NOT a hardcoded constant (an earlier version's fatal flaw — a mixture
whose components all scale with total vol has scale-invariant, i.e. constant,
skew, making it a strawman for the skew edge). Here the mixture SHAPE responds to:
  * the leverage effect  — skew grows more negative when forecast vol is high;
  * horizon attenuation  — skew decays toward 0 as T grows (central-limit pull);
  * realised skew        — seeded from the recent 3rd moment of returns.
So `log_return_skew()` genuinely varies with vol, horizon, and data.

This is a PHYSICAL (P-measure) forecast — what will actually happen. Its mean is
deliberately pinned to the forward S*exp((r-q)T) for direction-neutrality; the
tradeable edge lives in variance and skew (both mean-invariant), never in a
smuggled directional bet. Compare P to the market's risk-neutral Q via edge.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine import volforecast

from .density import LogNormalComponent, MixtureLogNormal

REF_VOL = 0.20             # vol at which the base skew applies
REF_T = 30.0 / 365.0       # horizon at which the base skew applies


@dataclass
class BaselineDensityForecaster:
    """Turns a price history into a distributional forecast of S_T.

    The LEVEL of vol is data-driven (HAR-RV); the SHAPE (skew/tails) responds to
    vol, horizon, and realised skew via the multipliers below.
    """
    stress_weight: float = 0.15       # probability mass in the crash component
    calm_sd_mult: float = 0.90        # calm component sd (multiple of scale)
    stress_sd_mult: float = 2.30      # stress component sd (fatter)
    base_drop_mult: float = 1.10      # crash gap at REF_VOL / REF_T (horizon-sd units)
    leverage_exp: float = 0.50        # how strongly skew steepens with vol
    horizon_exp: float = 0.25         # how strongly skew attenuates with T
    realized_gain: float = 1.5        # how strongly realised skew feeds the shape
    use_realized_skew: bool = True
    use_term_structure: bool = True   # mean-revert vol toward its long-run level
    kappa: float = 2.77               # variance mean-reversion (~3-month half-life)

    def _drop_mult(self, prices, vol: float, T: float) -> float:
        """Effective crash-gap multiplier — vol-, horizon- and data-dependent.

        This is what breaks scale-invariance: the RATIO of the crash gap to the
        component spreads changes with regime, so mixture skewness is no longer
        constant. Clamped so between-component variance never exceeds the total.
        """
        m = self.base_drop_mult
        m *= (vol / REF_VOL) ** self.leverage_exp          # leverage effect
        m *= (REF_T / T) ** self.horizon_exp               # horizon attenuation
        if self.use_realized_skew:
            rs = volforecast.realized_skew(prices)
            m *= 1.0 + self.realized_gain * max(0.0, -rs)  # steeper if history is crash-skewed
        # Keep between-variance = w_c*w_s*m^2 strictly below the total (m < 1/sqrt(w_c w_s)).
        w_s = self.stress_weight
        hard_cap = 0.95 / math.sqrt((1.0 - w_s) * w_s)
        return max(0.2, min(m, hard_cap))

    def forecast(
        self,
        prices,
        T: float,
        *,
        r: float = 0.0,
        q: float = 0.0,
        spot: float | None = None,
        vol: float | None = None,
        bars=None,
    ) -> MixtureLogNormal:
        """Forecast the S_T distribution T years ahead.

        vol : override the annualised vol; default uses HAR-RV on ``prices``.
        bars : optional [(o,h,l,c), ...] aligned with ``prices``. When supplied,
            the volatility that sets the whole density comes from the RANGE
            rather than from one squared close-to-close return per day. Measured
            against a known latent variance over 40 simulated worlds this cut
            21-day forecast RMSE 8.725 -> 7.460 vol points, paired 90% interval
            on the squared-error difference [+1.16e-03, +2.95e-03]. A caller with
            only closes passes nothing and gets exactly what it always got.
        """
        if T <= 0:
            raise ValueError("T must be positive")
        spot = spot if spot is not None else prices[-1]
        if vol is None:
            # Term structure matters: a FLAT annualised vol across maturities makes
            # the longest expiry look richest against any upward-sloping implied
            # curve — an artifact, not an edge (found on the first real chain).
            vol = (volforecast.term_vol(prices, T, kappa=self.kappa, bars=bars)
                   if self.use_term_structure
                   else volforecast.har_rv_forecast(prices, bars=bars))

        s_tot = vol * math.sqrt(T)                 # target sd of ln(S_T) over the horizon
        w_s = self.stress_weight
        w_c = 1.0 - w_s
        drop = self._drop_mult(prices, vol, T) * s_tot

        # Between-component variance = w_c*w_s*drop^2 (derivation in tests).
        between = w_c * w_s * drop * drop
        # Within-component scale so total log-variance hits the target s_tot^2.
        denom = w_c * self.calm_sd_mult ** 2 + w_s * self.stress_sd_mult ** 2
        sigma0 = math.sqrt(max(s_tot * s_tot - between, 1e-12) / denom)
        s_c = self.calm_sd_mult * sigma0
        s_s = self.stress_sd_mult * sigma0

        # Fix the shape (calm at 0, stress at -drop), then shift so the mixture
        # mean equals the forward. The shift is variance/skew-neutral.
        raw = w_c * math.exp(0.0 + 0.5 * s_c ** 2) + w_s * math.exp(-drop + 0.5 * s_s ** 2)
        forward = spot * math.exp((r - q) * T)
        shift = math.log(forward) - math.log(raw)

        return MixtureLogNormal([
            LogNormalComponent(w_c, shift, s_c),
            LogNormalComponent(w_s, shift - drop, s_s),
        ])
