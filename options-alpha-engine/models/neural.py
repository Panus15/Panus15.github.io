"""Trained neural encoder + MDN head — the production upgrade path.

The stdlib `models/mdn.py` trains only the MDN HEAD on top of a FIXED random-
feature encoder. This module learns the encoder END TO END: engineered temporal
features -> a trained MLP encoder -> the MDN head, all optimised jointly on the
negative log-likelihood via backprop. It is the concrete step toward a TFT/GRU
encoder; swap the MLP here for a recurrent/attention stack and nothing else
changes — it still emits the same `models.density.MixtureLogNormal`.

This is the ONE module that needs numpy (the stdlib core never imports it, so
the engine still runs dependency-free). Same drop-in interface as the baseline:
`forecast(prices, T, *, r=0, q=0, spot=None, vol=None) -> MixtureLogNormal`.

Guards carried over from the review: a hard sigma floor (NLL can't be driven to
-inf) and a weight-entropy regulariser (prevents mode collapse). Full-batch
gradient descent — per-sample SGD destabilises the 1/sigma^2 mean gradient.
"""

from __future__ import annotations

import math

import numpy as np

from engine import volforecast

from .density import LogNormalComponent, MixtureLogNormal


def _features(ctx) -> list[float]:
    """Engineered temporal features from a price context window."""
    har = volforecast.har_rv_forecast(ctx)
    ewma = volforecast.ewma_vol(ctx)
    ctc = volforecast.close_to_close_vol(ctx, min(21, len(ctx) - 1))
    rs = volforecast.realized_skew(ctx)
    mom5 = math.log(ctx[-1] / ctx[-6]) if len(ctx) > 6 else 0.0
    mom21 = math.log(ctx[-1] / ctx[-22]) if len(ctx) > 22 else 0.0
    return [har, ewma, ctc, rs, mom5, mom21]


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralMDNForecaster:
    """MLP encoder + Gaussian-mixture-density head, trained end to end.

    Predicts the distribution of the STANDARDISED log-return z = R / sqrt(T)
    (per-calendar-year units); `forecast` rescales the mixture by sqrt(T) and
    recentres on ln(spot), so components are LogNormal in S_T.
    """

    def __init__(self, n_components=3, hidden=12, context=63,
                 sigma_floor=0.03, entropy_reg=1e-3, seed=0):
        self.K = n_components
        self.H = hidden
        self.context = context
        self.sigma_floor = sigma_floor
        self.entropy_reg = entropy_reg
        self.rng = np.random.default_rng(seed)
        self._init_params(len(_features([100.0] * (context))))
        self._feat_mean = None
        self._feat_std = None
        self.history: list[float] = []

    def _init_params(self, d):
        r = self.rng
        s = 0.3
        self.W1 = r.standard_normal((d, self.H)) * s
        self.b1 = np.zeros(self.H)
        self.Wpi = r.standard_normal((self.H, self.K)) * s
        self.bpi = np.zeros(self.K)
        self.Wmu = r.standard_normal((self.H, self.K)) * s
        self.bmu = np.zeros(self.K)
        self.Ws = r.standard_normal((self.H, self.K)) * s
        self.bs = np.zeros(self.K)

    # --- forward ----------------------------------------------------------- #
    def _forward(self, X):
        pre = X @ self.W1 + self.b1
        h = np.tanh(pre)
        logits = h @ self.Wpi + self.bpi
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        pi = ex / ex.sum(axis=1, keepdims=True)
        mu = h @ self.Wmu + self.bmu
        raw = h @ self.Ws + self.bs
        sigma = self.sigma_floor + _softplus(raw)
        return pre, h, logits, pi, mu, raw, sigma

    def _standardize(self, X):
        return (X - self._feat_mean) / self._feat_std

    # --- training ---------------------------------------------------------- #
    def fit(self, prices, *, horizons=(7, 30, 60), epochs=200, lr=0.05, max_windows=400):
        from . import objective
        windows = objective.build_windows(prices, context=self.context, horizons=horizons)
        if len(windows) > max_windows:
            step = len(windows) // max_windows
            windows = windows[::step]
        X = np.array([_features(c) for c, _, _ in windows], dtype=float)
        z = np.array([math.log(s / c[-1]) / math.sqrt(h / 365.0)
                      for c, h, s in windows], dtype=float).reshape(-1, 1)
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        N = len(Xs)

        for _ in range(epochs):
            pre, h, logits, pi, mu, raw, sigma = self._forward(Xs)
            # responsibilities
            diff = z - mu
            comp_log = -0.5 * (diff / sigma) ** 2 - np.log(sigma) - 0.5 * math.log(2 * math.pi)
            log_pi = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            log_mix = log_pi + comp_log
            m = log_mix.max(axis=1, keepdims=True)
            lse = m + np.log(np.exp(log_mix - m).sum(axis=1, keepdims=True))
            nll = -lse.mean()
            self.history.append(float(nll))
            gamma = np.exp(log_mix - lse)                      # (N,K) responsibilities

            # grads w.r.t. head pre-activations
            ent = self.entropy_reg
            dlogits = (pi - gamma) + ent * pi * (np.log(pi + 1e-12) -
                        (pi * np.log(pi + 1e-12)).sum(axis=1, keepdims=True))
            dmu = gamma * (mu - z) / sigma ** 2
            dsigma = gamma * (1.0 - (diff / sigma) ** 2) / sigma
            draw = dsigma * _sigmoid(raw)

            gWpi = h.T @ dlogits / N; gbpi = dlogits.mean(0)
            gWmu = h.T @ dmu / N;     gbmu = dmu.mean(0)
            gWs = h.T @ draw / N;     gbs = draw.mean(0)
            dh = dlogits @ self.Wpi.T + dmu @ self.Wmu.T + draw @ self.Ws.T
            dpre = dh * (1.0 - h ** 2)
            gW1 = Xs.T @ dpre / N; gb1 = dpre.mean(0)

            for p, g in ((self.Wpi, gWpi), (self.bpi, gbpi), (self.Wmu, gWmu),
                         (self.bmu, gbmu), (self.Ws, gWs), (self.bs, gbs),
                         (self.W1, gW1), (self.b1, gb1)):
                p -= lr * g
        return self

    # --- inference (drop-in) ---------------------------------------------- #
    def forecast(self, prices, T, *, r=0.0, q=0.0, spot=None, vol=None) -> MixtureLogNormal:
        spot = spot if spot is not None else prices[-1]
        ctx = list(prices[-self.context:])
        X = np.array([_features(ctx)], dtype=float)
        if self._feat_mean is not None:
            X = self._standardize(X)
        _, _, _, pi, mu, _, sigma = self._forward(X)
        pi, mu, sigma = pi[0], mu[0], sigma[0]
        root = math.sqrt(T)
        comps = []
        ln_s = math.log(spot)
        for k in range(self.K):
            comps.append(LogNormalComponent(
                float(pi[k]), ln_s + float(mu[k]) * root, max(float(sigma[k]) * root, 1e-6)))
        dist = MixtureLogNormal(comps)
        if vol is not None:  # optional: rescale total vol to an override, keep shape
            cur = dist.log_return_vol(spot, T)
            if cur > 0:
                f = vol / cur
                comps = [LogNormalComponent(c.weight, ln_s + (c.mu - ln_s) * f, c.sigma * f)
                         for c in dist.components]
                dist = MixtureLogNormal(comps)
        return dist
