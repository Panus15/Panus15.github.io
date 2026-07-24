"""Tests for the Phase-2 NLP sentiment overlay (stdlib core).

The FinBERT backend is not exercised (needs model weights); the lexicon scorer,
aggregation, news gate, and file adapter are fully tested offline.
Run: python3 tests/test_sentiment.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.news_signal import event_risk, news_risk_gate
from models.sentiment import (FileNewsAdapter, LexiconSentimentScorer, NewsItem,
                              aggregate_sentiment)

SC = LexiconSentimentScorer()


def test_lexicon_polarity():
    assert SC.score("Company beats estimates, profit surges to record high") > 0.5
    assert SC.score("Shares plunge on fraud probe and bankruptcy fears") < -0.5
    assert SC.score("The meeting is scheduled for Tuesday") == 0.0     # neutral


def test_negation_flips_polarity():
    pos = SC.score("earnings beat expectations")
    neg = SC.score("earnings did not beat expectations")
    assert pos > 0 and neg < pos                    # negation pulls it down/negative


def test_aggregate_recency_weighting():
    asof = date(2026, 7, 19)
    # An old positive story and a fresh negative one -> feature leans negative.
    items = [
        NewsItem("2026-07-05", "X", "record profit and strong growth"),   # 14d old
        NewsItem("2026-07-19", "X", "stock plunges on default and crisis"),  # today
    ]
    feat = aggregate_sentiment(items, SC, asof=asof, halflife_days=3.0)
    assert feat.score < 0                            # recent negative dominates
    assert feat.n_items == 2 and feat.most_negative < 0


def test_dispersion_detects_disagreement():
    asof = date(2026, 7, 19)
    same = [NewsItem("2026-07-19", "X", "profit growth beat") for _ in range(4)]
    split = [NewsItem("2026-07-19", "X", "profit growth beat"),
             NewsItem("2026-07-19", "X", "crisis crash plunge"),
             NewsItem("2026-07-19", "X", "record surge rally"),
             NewsItem("2026-07-19", "X", "fraud lawsuit bankruptcy")]
    d_same = aggregate_sentiment(same, SC, asof=asof).dispersion
    d_split = aggregate_sentiment(split, SC, asof=asof).dispersion
    assert d_split > d_same                          # disagreement -> higher dispersion


def test_news_gate_fires_on_negative_regime():
    asof = date(2026, 7, 19)
    bad = [NewsItem("2026-07-19", "X", "plunge crisis default fear selloff")
           for _ in range(4)]
    feat = aggregate_sentiment(bad, SC, asof=asof)
    fired, reason = news_risk_gate(feat)
    assert fired, reason


def test_news_gate_ignores_thin_news():
    asof = date(2026, 7, 19)
    one = [NewsItem("2026-07-19", "X", "crisis crash plunge")]
    fired, _ = news_risk_gate(aggregate_sentiment(one, SC, asof=asof), min_volume=3)
    assert not fired                                 # one story is not a regime


def test_event_risk_combines_price_and_news():
    # Price gate alone fires.
    assert event_risk(True, None)[0] is True
    # Neither fires when calm + no news.
    assert event_risk(False, None)[0] is False


def test_file_news_adapter_json(tmp_path=None):
    import json
    import tempfile
    rows = [{"date": "2026-07-19", "symbol": "SPX", "headline": "market rallies to record"},
            {"date": "2026-07-19", "symbol": "AAPL", "headline": "probe and lawsuit"}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "news.json")
        with open(p, "w") as f:
            json.dump(rows, f)
        items = FileNewsAdapter(p).load(symbol="SPX")
    assert len(items) == 1 and items[0].symbol == "SPX"


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
