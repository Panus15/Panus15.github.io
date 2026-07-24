"""Phase-2 NLP sentiment — news as a FORWARD-LOOKING vol-regime signal.

This is deliberately NOT a "good news = buy" directional bet. In a variance-risk-
premium engine the useful thing sentiment gives you is EARLY WARNING of a vol
regime change: the HAR-RV regime gate is backward-looking and only reacts once
realised vol has already spiked (it sells vol loudest right into a developing
crash). A burst of negative / high-dispersion news is a LEADING indicator of the
realised-vol spike, so it lets the engine stop selling vol BEFORE the move — see
models/news_signal.py.

Design mirrors the rest of the engine:
  * stdlib CORE — LexiconSentimentScorer (Loughran-McDonald-style finance word
    lists) runs with zero dependencies and is fully unit-tested.
  * optional FinBERT backend — FinBERTScorer uses transformers if installed;
    SAME interface, a drop-in. It is never imported by the stdlib core.

Sentiment is an EXPERIMENTAL OVERLAY. Per the review, it is noisy and must not
drive capital on its own — it gates/tilts the vol signal, it does not replace it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date

# Compact Loughran-McDonald-style finance sentiment lexicons (curated subset).
_POSITIVE = {
    "gain", "gains", "gained", "profit", "profits", "profitable", "growth", "grew",
    "beat", "beats", "upgrade", "upgraded", "strong", "strength", "surge", "surged",
    "rally", "rallied", "record", "outperform", "outperformed", "bullish", "robust",
    "exceeded", "exceeds", "improve", "improved", "improvement", "boost", "boosted",
    "recovery", "rebound", "rebounded", "optimistic", "upside", "buyback", "momentum",
    "breakthrough", "approval", "approved", "win", "wins", "awarded", "opportunity",
    "confidence", "accelerate", "expansion", "positive", "raised", "raise",
}
_NEGATIVE = {
    "loss", "losses", "decline", "declined", "drop", "dropped", "plunge", "plunged",
    "fall", "fell", "weak", "weakness", "downgrade", "downgraded", "miss", "missed",
    "bearish", "recession", "crisis", "default", "bankruptcy", "bankrupt", "lawsuit",
    "sued", "investigation", "probe", "fraud", "warning", "warned", "cut", "cuts",
    "layoff", "layoffs", "slump", "slowdown", "deficit", "risk", "risks", "concern",
    "concerns", "fear", "fears", "selloff", "crash", "volatile", "volatility",
    "uncertainty", "uncertain", "halt", "halted", "recall", "delay", "delayed",
    "negative", "plummet", "plummeted", "tumble", "tumbled", "slashed", "downturn",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed", "lack", "lacks",
             "lacking", "hardly", "nt", "cannot", "cant"}
_WORD = re.compile(r"[a-z']+")


class SentimentScorer:
    """Interface: text -> sentiment in [-1, 1] (positive good, negative bad)."""

    def score(self, text: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class LexiconSentimentScorer(SentimentScorer):
    """Stdlib finance-sentiment scorer with simple negation handling."""

    def __init__(self, negation_window: int = 3):
        self.negation_window = negation_window

    def score(self, text: str) -> float:
        tokens = _WORD.findall(text.lower())
        pos = neg = 0
        neg_flag = 0  # counts down over the negation window
        for tok in tokens:
            if tok in _NEGATORS:
                neg_flag = self.negation_window + 1
            polarity = 1 if tok in _POSITIVE else -1 if tok in _NEGATIVE else 0
            if polarity != 0:
                if neg_flag > 0:               # a negator recently -> flip polarity
                    polarity = -polarity
                if polarity > 0:
                    pos += 1
                else:
                    neg += 1
            if neg_flag > 0:
                neg_flag -= 1
        if pos + neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


class FinBERTScorer(SentimentScorer):
    """Optional FinBERT backend (ProsusAI/finbert). Requires `transformers`.

    Same interface as the lexicon scorer, so it is a drop-in. Never imported by
    the stdlib core; raises a clear error if transformers/the model is missing.
    """

    def __init__(self, model: str = "ProsusAI/finbert"):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:  # pragma: no cover - optional path
            raise RuntimeError("FinBERT needs `transformers` (pip install transformers "
                               "torch); or use LexiconSentimentScorer") from e
        self._pipe = pipeline("sentiment-analysis", model=model)

    def score(self, text: str) -> float:  # pragma: no cover - needs model weights
        out = self._pipe(text[:512])[0]
        label, conf = out["label"].lower(), float(out["score"])
        if label == "positive":
            return conf
        if label == "negative":
            return -conf
        return 0.0


@dataclass
class NewsItem:
    date: str            # ISO date
    symbol: str
    headline: str
    body: str = ""

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.body}".strip()


@dataclass
class SentimentFeature:
    score: float          # recency-weighted mean sentiment in [-1, 1]
    volume: float         # recency-decayed count of items
    dispersion: float     # weighted std of item scores (disagreement / uncertainty)
    most_negative: float  # the single most negative item (tail-risk headline)
    n_items: int


def aggregate_sentiment(items: list[NewsItem], scorer: SentimentScorer, *,
                        asof: date, halflife_days: float = 3.0) -> SentimentFeature:
    """Collapse a stream of news for one name into a single dated feature.

    Recent items count more (exponential decay). `dispersion` — the spread of
    opinion — is itself informative: high disagreement often precedes a vol move.
    """
    if not items:
        return SentimentFeature(0.0, 0.0, 0.0, 0.0, 0)
    scores, weights = [], []
    for it in items:
        try:
            age = max(0, (asof - date.fromisoformat(it.date)).days)
        except ValueError:
            age = 0
        w = 0.5 ** (age / halflife_days)
        scores.append(scorer.score(it.text))
        weights.append(w)
    wsum = sum(weights) or 1e-9
    mean = sum(s * w for s, w in zip(scores, weights)) / wsum
    var = sum(w * (s - mean) ** 2 for s, w in zip(scores, weights)) / wsum
    return SentimentFeature(
        score=mean,
        volume=wsum,
        dispersion=math.sqrt(max(var, 0.0)),
        most_negative=min(scores),
        n_items=len(items),
    )


# --------------------------------------------------------------------------- #
# News ingestion adapters
# --------------------------------------------------------------------------- #
@dataclass
class FileNewsAdapter:
    """OFFLINE news source: load NewsItems from a JSON or CSV file.

    JSON: list of {date, symbol, headline, body?}.
    CSV : header date,symbol,headline[,body].
    """
    path: str

    def load(self, symbol: str | None = None) -> list[NewsItem]:
        import csv
        import json
        items: list[NewsItem] = []
        if self.path.endswith(".json"):
            with open(self.path) as fh:
                for row in json.load(fh):
                    items.append(NewsItem(row["date"], row.get("symbol", ""),
                                          row["headline"], row.get("body", "")))
        else:
            with open(self.path, newline="") as fh:
                for row in csv.DictReader(fh):
                    items.append(NewsItem(row["date"], row.get("symbol", ""),
                                          row["headline"], row.get("body", "")))
        if symbol:
            items = [it for it in items if it.symbol == symbol]
        return items


class LiveNewsAdapter:
    """Live news stub (NewsAPI / RSS / vendor). Needs a key + network.

    Wire your provider in `fetch`; kept minimal because this sandbox has no
    egress. Returns NewsItem objects so the aggregation path is unchanged.
    """

    def __init__(self, api_key: str | None = None):
        import os
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")

    def fetch(self, symbol: str, *, days: int = 3) -> list[NewsItem]:  # pragma: no cover
        raise RuntimeError("LiveNewsAdapter.fetch is a stub — plug in a provider "
                           "(NewsAPI/RSS) and set NEWS_API_KEY; sandbox has no network")
