"""GRU sequence encoder + MDN head — the recurrent upgrade path.

`models/neural.py` learns an MLP over *engineered* features (a HAR vol, an EWMA,
a skew, ...). This module instead consumes the raw return SEQUENCE and learns the
temporal structure itself with a Gated Recurrent Unit (Cho et al., 2014), trained
end to end by backpropagation-through-time onto the mixture negative
log-likelihood. It is the concrete "real TFT/GRU encoder" step: a recurrent stack
in place of the flat MLP, emitting the SAME `models.density.MixtureLogNormal`, so
it is a drop-in for the baseline and is judged by the exact same promotion gate.

Honesty, stated plainly: more capacity is not more edge. HAR-RV is a genuine wall,
and on the data we have the gate has already rejected a trained MDN on the crash
tail. This encoder EARNS production only if it beats the baseline out-of-sample on
both aggregate NLL and the decomposed left tail — the gate decides, not the author.
It exists so that, when a real, long, crash-containing option-data history is
available, the capability is ready and honestly arbitrated.

numpy-only (like neural.py); the stdlib core never imports it. The BPTT gradients
are verified against finite differences in tests/test_gru.py — the one honest way
to trust a hand-derived recurrent backward pass.
"""

from __future__ import annotations

import math

import numpy as np

from .density import LogNormalComponent, MixtureLogNormal


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _seq_features(ctx, seq_len: int):
    """Last ``seq_len`` steps of [log-return, |log-return|] from a price context.
    Left-padded with zeros if the context is shorter. Shape (seq_len, 2)."""
    rets = [math.log(ctx[i] / ctx[i - 1]) for i in range(1, len(ctx))]
    rets = rets[-seq_len:]
    pad = seq_len - len(rets)
    seq = [[0.0, 0.0]] * pad + [[r, abs(r)] for r in rets]
    return np.array(seq, dtype=float)


class GRUMDNForecaster:
    """A single-layer GRU over the return sequence -> Gaussian-mixture head.

    Predicts the distribution of the standardised log-return z = R / sqrt(T);
    ``forecast`` rescales by sqrt(T) and recentres on ln(spot) so the mixture is
    LogNormal in S_T — identical output contract to the baseline and neural heads.
    """

    def __init__(self, n_components=3, hidden=8, seq_len=40, context=63,
                 sigma_floor=0.03, entropy_reg=1e-3, seed=0):
        self.K = n_components
        self.H = hidden
        self.L = seq_len
        self.context = context
        self.sigma_floor = sigma_floor
        self.entropy_reg = entropy_reg
        self.D = 2
        self.rng = np.random.default_rng(seed)
        self._init_params()
        self._x_mean = None
        self._x_std = None

    def _init_params(self):
        r, H, D, K = self.rng, self.H, self.D, self.K
        g = 1.0 / math.sqrt(H)
        # GRU gates: update (z), reset (r), candidate (n).
        self.Wz = r.standard_normal((D, H)) * g; self.Uz = r.standard_normal((H, H)) * g; self.bz = np.zeros(H)
        self.Wr = r.standard_normal((D, H)) * g; self.Ur = r.standard_normal((H, H)) * g; self.br = np.zeros(H)
        self.Wn = r.standard_normal((D, H)) * g; self.Un = r.standard_normal((H, H)) * g; self.bn = np.zeros(H)
        # MDN head.
        s = 0.3
        self.Wpi = r.standard_normal((H, K)) * s; self.bpi = np.zeros(K)
        self.Wmu = r.standard_normal((H, K)) * s; self.bmu = np.zeros(K)
        self.Ws = r.standard_normal((H, K)) * s;  self.bs = np.zeros(K)

    def _param_list(self):
        return [self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br,
                self.Wn, self.Un, self.bn, self.Wpi, self.bpi,
                self.Wmu, self.bmu, self.Ws, self.bs]

    # --- GRU forward over the sequence, caching activations for BPTT --------- #
    def _encode(self, X):
        """X: (N, L, D) -> final hidden H_T: (N, H), plus a cache for backward."""
        N = X.shape[0]
        h = np.zeros((N, self.H))
        cache = []
        for t in range(self.L):
            x = X[:, t, :]
            z = _sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
            rr = _sigmoid(x @ self.Wr + h @ self.Ur + self.br)
            hr = rr * h
            n = np.tanh(x @ self.Wn + hr @ self.Un + self.bn)
            h_new = (1.0 - z) * n + z * h
            cache.append((x, h, z, rr, hr, n, h_new))
            h = h_new
        return h, cache

    def _head(self, H):
        logits = H @ self.Wpi + self.bpi
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        pi = ex / ex.sum(axis=1, keepdims=True)
        mu = H @ self.Wmu + self.bmu
        raw = H @ self.Ws + self.bs
        sigma = self.sigma_floor + (np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0))
        return logits, pi, mu, raw, sigma

    # --- loss + full gradient (head backward, then BPTT) -------------------- #
    def _loss_and_grads(self, X, z_tgt):
        N = X.shape[0]
        H_T, cache = self._encode(X)
        logits, pi, mu, raw, sigma = self._head(H_T)

        diff = z_tgt - mu
        comp_log = -0.5 * (diff / sigma) ** 2 - np.log(sigma) - 0.5 * math.log(2 * math.pi)
        log_pi = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        log_mix = log_pi + comp_log
        m = log_mix.max(axis=1, keepdims=True)
        lse = m + np.log(np.exp(log_mix - m).sum(axis=1, keepdims=True))
        nll = float(-lse.mean())
        gamma = np.exp(log_mix - lse)

        ent = self.entropy_reg
        dlogits = (pi - gamma) + ent * pi * (np.log(pi + 1e-12) -
                    (pi * np.log(pi + 1e-12)).sum(axis=1, keepdims=True))
        dmu = gamma * (mu - z_tgt) / sigma ** 2
        dsigma = gamma * (1.0 - (diff / sigma) ** 2) / sigma
        draw = dsigma * _sigmoid(raw)

        gWpi = H_T.T @ dlogits / N; gbpi = dlogits.mean(0)
        gWmu = H_T.T @ dmu / N;     gbmu = dmu.mean(0)
        gWs = H_T.T @ draw / N;     gbs = draw.mean(0)
        dH = (dlogits @ self.Wpi.T + dmu @ self.Wmu.T + draw @ self.Ws.T)  # grad into GRU final state

        # BPTT — accumulate gate-param grads walking back through time.
        gWz = np.zeros_like(self.Wz); gUz = np.zeros_like(self.Uz); gbz = np.zeros_like(self.bz)
        gWr = np.zeros_like(self.Wr); gUr = np.zeros_like(self.Ur); gbr = np.zeros_like(self.br)
        gWn = np.zeros_like(self.Wn); gUn = np.zeros_like(self.Un); gbn = np.zeros_like(self.bn)
        dh = dH
        for t in range(self.L - 1, -1, -1):
            x, h_prev, z, rr, hr, n, _ = cache[t]
            dz = dh * (h_prev - n)
            dn = dh * (1.0 - z)
            dh_prev = dh * z                              # direct path h_t = ... + z*h_{t-1}

            da_n = dn * (1.0 - n ** 2)                    # tanh'
            gWn += x.T @ da_n / N; gUn += hr.T @ da_n / N; gbn += da_n.mean(0)
            dhr = da_n @ self.Un.T
            dr = dhr * h_prev
            dh_prev = dh_prev + dhr * rr                  # through hr = r ⊙ h_{t-1}

            da_r = dr * rr * (1.0 - rr)                   # sigmoid'
            gWr += x.T @ da_r / N; gUr += h_prev.T @ da_r / N; gbr += da_r.mean(0)
            dh_prev = dh_prev + da_r @ self.Ur.T

            da_z = dz * z * (1.0 - z)
            gWz += x.T @ da_z / N; gUz += h_prev.T @ da_z / N; gbz += da_z.mean(0)
            dh_prev = dh_prev + da_z @ self.Uz.T

            dh = dh_prev

        grads = [gWz, gUz, gbz, gWr, gUr, gbr, gWn, gUn, gbn,
                 gWpi, gbpi, gWmu, gbmu, gWs, gbs]
        return nll, grads

    # --- training ---------------------------------------------------------- #
    def fit(self, prices, *, horizons=(7, 30, 60), epochs=150, lr=0.05, max_windows=400):
        from . import objective
        windows = objective.build_windows(prices, context=self.context, horizons=horizons)
        if len(windows) > max_windows:
            windows = windows[::len(windows) // max_windows]
        X = np.array([_seq_features(c, self.L) for c, _, _ in windows])       # (N,L,D)
        z = np.array([math.log(s / c[-1]) / math.sqrt(h / 365.0)
                      for c, h, s in windows]).reshape(-1, 1)
        self._x_mean = X.reshape(-1, self.D).mean(axis=0)
        self._x_std = X.reshape(-1, self.D).std(axis=0) + 1e-8
        Xs = (X - self._x_mean) / self._x_std
        self.history = []
        params = self._param_list()
        for _ in range(epochs):
            nll, grads = self._loss_and_grads(Xs, z)
            self.history.append(nll)
            for p, g in zip(params, grads):
                p -= lr * g
        return self

    # --- inference (drop-in) ---------------------------------------------- #
    def forecast(self, prices, T, *, r=0.0, q=0.0, spot=None, vol=None) -> MixtureLogNormal:
        spot = spot if spot is not None else prices[-1]
        ctx = list(prices[-self.context:])
        X = _seq_features(ctx, self.L)[None, :, :]
        if self._x_mean is not None:
            X = (X - self._x_mean) / self._x_std
        H_T, _ = self._encode(X)
        _, pi, mu, _, sigma = self._head(H_T)
        pi, mu, sigma = pi[0], mu[0], sigma[0]
        root = math.sqrt(T)
        ln_s = math.log(spot)
        comps = [LogNormalComponent(float(pi[k]), ln_s + float(mu[k]) * root,
                                    max(float(sigma[k]) * root, 1e-6)) for k in range(self.K)]
        dist = MixtureLogNormal(comps)
        if vol is not None:
            cur = dist.log_return_vol(spot, T)
            if cur > 0:
                f = vol / cur
                dist = MixtureLogNormal([
                    LogNormalComponent(c.weight, ln_s + (c.mu - ln_s) * f, c.sigma * f)
                    for c in dist.components])
        return dist
