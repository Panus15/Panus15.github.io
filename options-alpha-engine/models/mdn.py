"""Trainable Mixture Density Network (MDN) density forecaster — the PRODUCTION
core, a strict drop-in for the Phase-1 HAR baseline (models/baseline.py).

WHAT THIS IS
------------
`SequenceMDNForecaster.forecast(prices, T, *, r, q, spot)` emits the SAME
`density.MixtureLogNormal` object the baseline does, with the SAME call
signature, so nothing downstream (pricing, edge, backtest, objective gate)
changes. It is the MDN half of the spec's "TFT encoder -> MDN head" target
(see models/MODEL_SPEC.md sect. 1 & 6): a Gaussian-mixture head whose outputs
map exactly onto `LogNormalComponent(weight, mu, sigma)` --
  * softmax  -> mixture weights w_i,
  * linear   -> component means m_i (of the log-return),
  * softplus + hard FLOOR -> component sigmas s_i.
The mixture is over the LOG-RETURN  R = ln(S_T / spot); a component with
per-horizon mean m_i and sigma s_i becomes `LogNormalComponent(w_i,
mu=ln(spot)+m_i, sigma=s_i)`, i.e. ln(S_T) ~ Normal(ln(spot)+m_i, s_i).

TEMPORAL ENCODER = FIXED RANDOM FEATURES (deliberate, documented choice)
-----------------------------------------------------------------------
Pure-stdlib backprop through a full recurrent/attention encoder is fragile and
slow. Instead the temporal encoder here is a FIXED, seeded random-feature map
(a random projection of hand-built vol/momentum/skew/horizon features, passed
through tanh) -- a legitimate "random features" / reservoir model. ONLY the MDN
head (three linear maps) is gradient-trained. This guarantees the head trains
robustly with exact closed-form gradients, while keeping the emitted type
identical. The PRODUCTION path swaps the fixed encoder for a trained TFT/GRU
that feeds the SAME MDN head and emits the SAME `MixtureLogNormal` -- no
downstream change. The head's training math (below) is encoder-agnostic.

HEAD TRAINING (Bishop, 1994) + PITFALL GUARDS
---------------------------------------------
The head minimises the mean negative log-likelihood (NLL) of realised
log-returns over `objective.build_windows(...)` by SGD, using the standard MDN
responsibilities gamma_i = w_i N(R; m_i, s_i) / sum_j w_j N(R; m_j, s_j) and the
closed-form gradients of NLL w.r.t. the head's pre-activations (derived and
implemented in `_grads`). Two guards flagged by the adversarial review
(MODEL_SPEC.md sect. 4.6 / 5) are implemented:

  * HARD SIGMA FLOOR -- every emitted sigma is `sigma_floor + softplus(...)`,
    so a component can never collapse sigma -> 0 to drive NLL -> -inf.
  * WEIGHT-ENTROPY REGULARISER -- a small `+entropy_reg * sum_i w_i ln w_i`
    penalty pulls the weights toward uniform, preventing mode collapse /
    weight starvation (a dead component whose weight -> 0).

The emitted `MixtureLogNormal` is therefore always well-posed: weights are
positive and normalised, every sigma >= sigma_floor > 0.

HORIZON HANDLING
----------------
One network serves all horizons. It predicts PER-UNIT (annualised, per
sqrt-year) mixture parameters and the emission scales them to the horizon T:
sqrt(T) is an input feature AND an explicit multiplier -- component sigma is
`sigma_floor + softplus(raw_i) * sqrt(T)` and mean is `raw_mean_i * sqrt(T)`.
So sigma scales like sqrt(T) (the diffusive law) while the network may still
modulate the per-unit level with the sqrt(T) feature (term structure).
`forecast()` at any T is thus self-consistent, and training pools all horizons.

This is a PHYSICAL (P-measure) forecast, like the baseline. Nothing here is
risk-neutral; compare P to Q via models/edge.py.
"""

from __future__ import annotations

import math
import random

from engine import volforecast

from .density import LogNormalComponent, MixtureLogNormal

LOG_SQRT2PI = 0.5 * math.log(2.0 * math.pi)

REF_VOL = 0.20              # reference annual vol used to centre the vol features
REF_T = 30.0 / 365.0        # reference horizon used to centre the sqrt(T) feature
_N_SKIP = 4                 # raw features passed straight to the head (skip-connection)


# --- small numeric helpers (stdlib only) ----------------------------------
def _softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:  # = d softplus / dx
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _dot(a, b) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


class SequenceMDNForecaster:
    """Drop-in MDN distributional forecaster of the terminal price S_T.

    Same interface as `BaselineDensityForecaster`:
        forecast(prices, T, *, r=0.0, q=0.0, spot=None) -> MixtureLogNormal
    plus `fit(prices, ...)` which gradient-trains the MDN head on realised
    log-returns. The temporal encoder is a fixed seeded random-feature map; only
    the head is trained (see module docstring).
    """

    def __init__(
        self,
        n_components: int = 3,
        hidden: int = 16,
        context: int = 63,
        sigma_floor: float = 0.02,
        entropy_reg: float = 1e-3,
        seed: int = 0,
        *,
        grad_clip: float = 5.0,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")
        self.n_components = n_components
        self.hidden = hidden
        self.context = context
        self.sigma_floor = sigma_floor
        self.entropy_reg = entropy_reg
        self.seed = seed
        self.grad_clip = grad_clip
        self.history: list[float] = []       # per-epoch mean train NLL (filled by fit)

        rng = random.Random(seed)

        # --- FIXED random-feature encoder (never trained) -----------------
        # Random projection of the F_raw hand-built features -> `hidden` tanh
        # units. Scale ~ 1/sqrt(F_raw) keeps pre-activations O(1).
        self._f_raw = 9
        scale = 1.0 / math.sqrt(self._f_raw)
        self._W_enc = [
            [rng.gauss(0.0, scale) for _ in range(self._f_raw)] for _ in range(hidden)
        ]
        self._b_enc = [rng.gauss(0.0, 0.5) for _ in range(hidden)]

        # phi = [tanh hidden units] ++ [N_SKIP raw skip features] ++ [1.0 bias]
        self._phi_dim = hidden + _N_SKIP + 1
        self._bias_idx = self._phi_dim - 1

        # --- TRAINABLE MDN head: three linear maps head-out = W @ phi -------
        # A -> softmax logits, M -> means, B -> raw sigmas (softplus + floor).
        P = self._phi_dim
        self.A = [[rng.gauss(0.0, 0.05) for _ in range(P)] for _ in range(n_components)]
        self.M = [[rng.gauss(0.0, 0.02) for _ in range(P)] for _ in range(n_components)]
        self.B = [[rng.gauss(0.0, 0.05) for _ in range(P)] for _ in range(n_components)]
        # Bias the raw sigma so the UNTRAINED per-unit sigma ~ 0.24 (slightly
        # wide vs typical realised vol, leaving clear room for fit to tighten).
        # softplus(-1.26) ~ 0.245.
        for i in range(n_components):
            self.B[i][self._bias_idx] = -1.26 + rng.gauss(0.0, 0.02)

    # ------------------------------------------------------------------ #
    #  Feature encoder (fixed)                                            #
    # ------------------------------------------------------------------ #
    def _safe_c2c(self, prices, window: int, fallback: float) -> float:
        try:
            rets = volforecast.log_returns(prices)
            w = min(window, len(rets))
            if w < 2:
                return fallback
            return volforecast.close_to_close_vol(prices, w)
        except (ValueError, ZeroDivisionError):
            return fallback

    def _raw_features(self, prices, T: float) -> list[float]:
        """Hand-built, standardised feature vector (length self._f_raw = 9)."""
        try:
            v_ewma = volforecast.ewma_vol(prices)
        except (ValueError, ZeroDivisionError):
            v_ewma = REF_VOL
        try:
            v_har = volforecast.har_rv_forecast(prices)
        except (ValueError, ZeroDivisionError):
            v_har = v_ewma
        v_short = self._safe_c2c(prices, 5, v_ewma)
        v_long = self._safe_c2c(prices, 21, v_ewma)
        skew = volforecast.realized_skew(prices)

        rets = volforecast.log_returns(prices)
        sig_d = max(v_ewma / math.sqrt(volforecast.TRADING_DAYS), 1e-6)
        mom_s = sum(rets[-5:]) / (sig_d * math.sqrt(5)) if len(rets) >= 1 else 0.0
        mom_l = sum(rets[-21:]) / (sig_d * math.sqrt(21)) if len(rets) >= 1 else 0.0
        last = rets[-1] / sig_d if rets else 0.0

        return [
            v_har / REF_VOL - 1.0,                     # 0  vol level (HAR)
            v_ewma / REF_VOL - 1.0,                    # 1  vol level (EWMA)
            v_short / REF_VOL - 1.0,                   # 2  short realised vol
            v_long / REF_VOL - 1.0,                    # 3  long realised vol
            _clamp(skew, -3.0, 3.0),                   # 4  realised skew
            _clamp(mom_s, -4.0, 4.0),                  # 5  standardised 5d move
            _clamp(mom_l, -4.0, 4.0),                  # 6  standardised 21d move
            _clamp(last, -6.0, 6.0),                   # 7  last standardised shock
            math.sqrt(T) / math.sqrt(REF_T) - 1.0,     # 8  horizon feature
        ]

    def _features(self, prices, T: float) -> list[float]:
        raw = self._raw_features(prices, T)
        phi = []
        for j in range(self.hidden):
            phi.append(math.tanh(_dot(self._W_enc[j], raw) + self._b_enc[j]))
        # skip-connection: give the head direct linear access to the key drivers
        # (vol level, horizon, skew, momentum) instead of only tanh-mangled ones.
        phi.extend((raw[0], raw[8], raw[4], raw[6]))
        phi.append(1.0)                                # bias term
        return phi

    # ------------------------------------------------------------------ #
    #  Forward pass -> mixture parameters                                 #
    # ------------------------------------------------------------------ #
    def _forward(self, prices, T: float):
        phi = self._features(prices, T)
        sqrtT = math.sqrt(T)
        alpha = [_dot(self.A[i], phi) for i in range(self.n_components)]
        mu_raw = [_dot(self.M[i], phi) for i in range(self.n_components)]
        rho = [_dot(self.B[i], phi) for i in range(self.n_components)]

        # softmax weights (stable)
        amax = max(alpha)
        ex = [math.exp(a - amax) for a in alpha]
        ssum = sum(ex)
        w = [e / ssum for e in ex]

        su = [_softplus(r) for r in rho]               # per-unit (annualised) sigma
        m = [mu_raw[i] * sqrtT for i in range(self.n_components)]        # horizon mean
        s = [self.sigma_floor + su[i] * sqrtT for i in range(self.n_components)]  # horizon sigma, floored
        return {
            "phi": phi, "sqrtT": sqrtT, "alpha": alpha, "rho": rho,
            "w": w, "su": su, "m": m, "s": s,
        }

    # ------------------------------------------------------------------ #
    #  Public interface  (drop-in for BaselineDensityForecaster)          #
    # ------------------------------------------------------------------ #
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
        """Forecast the S_T distribution T years ahead as a MixtureLogNormal.

        Identical signature to `BaselineDensityForecaster.forecast`. `r`/`q` are
        accepted for interface symmetry (this is a physical, direction-neutral
        forecast; the learned means capture drift from data). `vol`, if given,
        rescales the predicted per-unit sigmas so their weighted average equals
        `vol` -- a convenience override matching the baseline's `vol=` kwarg.
        """
        if T <= 0:
            raise ValueError("T must be positive")
        spot = spot if spot is not None else prices[-1]
        fwd = self._forward(prices, T)
        w, su, m, s = fwd["w"], fwd["su"], fwd["m"], fwd["s"]
        sqrtT = fwd["sqrtT"]

        if vol is not None:
            avg_su = sum(w[i] * su[i] for i in range(self.n_components))
            if avg_su > 1e-9:
                factor = vol / avg_su
                s = [self.sigma_floor + su[i] * factor * sqrtT for i in range(self.n_components)]

        ln_spot = math.log(spot)
        comps = [
            LogNormalComponent(w[i], ln_spot + m[i], s[i])
            for i in range(self.n_components)
        ]
        return MixtureLogNormal(comps)

    # ------------------------------------------------------------------ #
    #  Loss + closed-form gradients (Bishop 1994 MDN responsibilities)    #
    # ------------------------------------------------------------------ #
    def _nll(self, fwd, R: float) -> float:
        """NLL of a realised log-return R under the mixture, + entropy penalty."""
        w, m, s = fwd["w"], fwd["m"], fwd["s"]
        logs = []
        for i in range(self.n_components):
            z = (R - m[i]) / s[i]
            logN = -LOG_SQRT2PI - math.log(s[i]) - 0.5 * z * z
            logs.append(math.log(max(w[i], 1e-300)) + logN)
        lmax = max(logs)
        lse = lmax + math.log(sum(math.exp(l - lmax) for l in logs))
        ent = sum(w[i] * math.log(max(w[i], 1e-300)) for i in range(self.n_components))
        return -lse + self.entropy_reg * ent

    def _grads(self, fwd, R: float):
        """Gradients of the (NLL + entropy) loss w.r.t. head weights A, M, B.

        Uses the standard MDN responsibilities gamma_i and the closed-form
        derivatives of NLL w.r.t. the head pre-activations:
          d/d alpha_i  = (w_i - gamma_i)  + entropy_reg * w_i (ln w_i - Hbar)
          d/d mu_raw_i = gamma_i (m_i - R)/s_i^2 * sqrt(T)
          d/d rho_i    = gamma_i (1 - ((R-m_i)/s_i)^2)/s_i * sigmoid(rho_i) * sqrt(T)
        then chained through the linear head:  dW[i][f] = d(pre_i) * phi[f].
        Returns (nll_value, dA, dM, dB).
        """
        n = self.n_components
        phi, sqrtT = fwd["phi"], fwd["sqrtT"]
        w, m, s, rho = fwd["w"], fwd["m"], fwd["s"], fwd["rho"]

        logs = []
        for i in range(n):
            z = (R - m[i]) / s[i]
            logN = -LOG_SQRT2PI - math.log(s[i]) - 0.5 * z * z
            logs.append(math.log(max(w[i], 1e-300)) + logN)
        lmax = max(logs)
        lse = lmax + math.log(sum(math.exp(l - lmax) for l in logs))
        gamma = [math.exp(l - lse) for l in logs]
        ent = sum(w[i] * math.log(max(w[i], 1e-300)) for i in range(n))
        nll = -lse + self.entropy_reg * ent

        Hbar = sum(w[i] * math.log(max(w[i], 1e-300)) for i in range(n))
        g_alpha, g_mu, g_rho = [0.0] * n, [0.0] * n, [0.0] * n
        cl = self.grad_clip
        for i in range(n):
            ga = (w[i] - gamma[i]) + self.entropy_reg * w[i] * (
                math.log(max(w[i], 1e-300)) - Hbar
            )
            diff = R - m[i]
            gm = gamma[i] * (-diff) / (s[i] * s[i]) * sqrtT
            gs = gamma[i] * (1.0 - (diff * diff) / (s[i] * s[i])) / s[i]
            gr = gs * _sigmoid(rho[i]) * sqrtT
            g_alpha[i] = _clamp(ga, -cl, cl)
            g_mu[i] = _clamp(gm, -cl, cl)
            g_rho[i] = _clamp(gr, -cl, cl)

        dA = [[g_alpha[i] * phi[f] for f in range(self._phi_dim)] for i in range(n)]
        dM = [[g_mu[i] * phi[f] for f in range(self._phi_dim)] for i in range(n)]
        dB = [[g_rho[i] * phi[f] for f in range(self._phi_dim)] for i in range(n)]
        return nll, dA, dM, dB

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        prices,
        *,
        horizons=(7, 30, 60),
        epochs: int = 80,
        lr: float = 0.04,
        r: float = 0.0,
        q: float = 0.0,
        max_windows: int | None = None,
        verbose: bool = False,
    ) -> "SequenceMDNForecaster":
        """Train the MDN head by FULL-BATCH gradient descent on mean NLL of
        realised log-returns.

        Windows come from `objective.build_windows(prices, context=self.context,
        horizons=horizons)` (leak-free). Full-batch (mean) gradients are used
        rather than per-sample SGD: the MDN's `1/sigma^2` mean-gradient is
        notoriously unstable under noisy single-sample updates, whereas
        averaging over the batch cancels the noise and converges smoothly.
        Returns self; `self.history` holds the per-epoch mean training NLL.
        """
        from . import objective  # local import: avoid import cycles at module load

        windows = objective.build_windows(prices, context=self.context, horizons=horizons)
        if not windows:
            raise ValueError("no training windows -- need more prices or smaller context")

        rng = random.Random(self.seed + 1)
        if max_windows is not None and len(windows) > max_windows:
            windows = rng.sample(windows, max_windows)

        # Pre-compute (log-return target R) per window; features are recomputed
        # each step from ctx (cheap, and keeps the encoder the single source).
        samples = [(ctx, h / 365.0, math.log(s_real / ctx[-1])) for ctx, h, s_real in windows]
        N = len(samples)

        n = self.n_components
        P = self._phi_dim
        self.history = []
        for ep in range(epochs):
            gA = [[0.0] * P for _ in range(n)]
            gM = [[0.0] * P for _ in range(n)]
            gB = [[0.0] * P for _ in range(n)]
            total = 0.0
            for ctx, T, R in samples:
                fwd = self._forward(ctx, T)
                nll, dA, dM, dB = self._grads(fwd, R)
                total += nll
                for i in range(n):
                    gAi, gMi, gBi = gA[i], gM[i], gB[i]
                    dAi, dMi, dBi = dA[i], dM[i], dB[i]
                    for f in range(P):
                        gAi[f] += dAi[f]
                        gMi[f] += dMi[f]
                        gBi[f] += dBi[f]
            step = lr / N
            for i in range(n):
                Ai, Mi, Bi = self.A[i], self.M[i], self.B[i]
                gAi, gMi, gBi = gA[i], gM[i], gB[i]
                for f in range(P):
                    Ai[f] -= step * gAi[f]
                    Mi[f] -= step * gMi[f]
                    Bi[f] -= step * gBi[f]
            mean_nll = total / N
            self.history.append(mean_nll)
            if verbose:
                print(f"epoch {ep + 1:3d}  mean train NLL(log-return) = {mean_nll:.5f}")
        return self
