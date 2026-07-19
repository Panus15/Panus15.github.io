"""Phase-1 baseline distributional forecaster — THIS SHIPS FIRST.

It emits a `MixtureLogNormal` for the terminal price S_T at horizon T, with:
  - total volatility set by a **HAR-RV** realised-vol forecast (Corsi, 2009), and
  - a fixed equity-style **negative skew** built from a two-component mixture:
    a high-weight "calm" component plus a low-weight, lower-mean "stress"
    component (the crash tail).

Why a baseline at all: it is the fair, interpretable, hard-to-beat benchmark
that the future TFT-encoder + MDN head MUST exceed out-of-sample before it earns
production. Both emit the SAME `MixtureLogNormal`, so the neural model is a
drop-in replacement for `forecast()` — nothing downstream changes.

This is a PHYSICAL (P-measure) forecast: it predicts what will actually happen.
Compare it to the market's risk-neutral (Q) law via models/edge.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine import volforecast

from .density import LogNormalComponent, MixtureLogNormal


@dataclass
class BaselineDensityForecaster:
    """Turns a price history into a distributional forecast of S_T.

    Parameters shape the fixed skew/kurtosis of the mixture; the LEVEL of vol is
    data-driven (HAR-RV). Defaults encode a mild, realistic equity crash-skew.
    """
    stress_weight: float = 0.15      # probability mass in the crash component
    calm_sd_mult: float = 0.90       # calm component sd, as a multiple of scale
    stress_sd_mult: float = 2.30     # stress component sd (fatter)
    stress_drop_mult: float = 1.10   # how far the stress mean sits below calm (in horizon-sd units)

    def forecast(
        self,
        prices,
        T: float,
        *,
        r: float = 0.0,
        q: float = 0.0,
        spot: float | None = None,
        vol: float | None = None,
    ) -> MixtureLogNormal:
        """Forecast the S_T distribution T years ahead.

        vol : override the annualised vol; default uses HAR-RV on ``prices``.
        The mixture is centred so E[S_T] = spot * exp((r - q) T) (the forward) —
        we do NOT smuggle in a directional bet; the edge is in vol and shape.
        """
        if T <= 0:
            raise ValueError("T must be positive")
        spot = spot if spot is not None else prices[-1]
        vol = vol if vol is not None else volforecast.har_rv_forecast(prices)

        s_tot = vol * math.sqrt(T)                 # target sd of ln(S_T) over the horizon
        w_s = self.stress_weight
        w_c = 1.0 - w_s
        drop = self.stress_drop_mult * s_tot       # log-distance calm mean -> stress mean

        # Between-component variance is fixed once weights and drop are set:
        #   between = w_c * w_s * drop^2   (see derivation in tests)
        # Clamp drop so the between-variance never exceeds the target.
        between = w_c * w_s * drop * drop
        if between >= s_tot * s_tot:
            drop = 0.9 * s_tot / math.sqrt(w_c * w_s)
            between = w_c * w_s * drop * drop

        # Solve the within-component scale so total log-variance hits the target.
        denom = w_c * self.calm_sd_mult ** 2 + w_s * self.stress_sd_mult ** 2
        sigma0 = math.sqrt(max(s_tot * s_tot - between, 1e-12) / denom)
        s_c = self.calm_sd_mult * sigma0
        s_s = self.stress_sd_mult * sigma0

        # Levels: fix the shape (calm at 0, stress at -drop), then shift so the
        # mixture mean equals the forward price. Shift is variance/skew-neutral.
        raw = w_c * math.exp(0.0 + 0.5 * s_c ** 2) + w_s * math.exp(-drop + 0.5 * s_s ** 2)
        forward = spot * math.exp((r - q) * T)
        shift = math.log(forward) - math.log(raw)

        return MixtureLogNormal([
            LogNormalComponent(w_c, shift, s_c),
            LogNormalComponent(w_s, shift - drop, s_s),
        ])
