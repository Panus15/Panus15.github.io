#!/usr/bin/env python3
"""Run the filter over a labelled corpus and report where it errs.

    python3 eval/run_eval.py              # live network for needs_network cases
    python3 eval/run_eval.py --offline    # skip needs_network cases (what CI can run)
    python3 eval/run_eval.py --ai         # also enable --ai (needs credentials; spends money)

Each expectation names a claim by a substring of its text and the status it must get
(or "absent": it must NOT be flagged at all). Outcomes:

  FALSE ALARM  expected verified/absent, got contradicted (or flagged when it should be silent)
  MISS         expected contradicted, got verified or nothing
  SOFT         any other mismatch (e.g. verified <-> unverified); cheaper than the two above
  OK           matches

Exit code 1 if there is any FALSE ALARM or MISS. A claim that cannot be found counts as absent.
"""
from __future__ import annotations

import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import pipeline
from engine.net import OfflineFetcher, UrllibFetcher


def classify(expected: str, got: str) -> str:
    if expected == got:
        return "OK"
    if expected in ("verified", "absent") and got == "contradicted":
        return "FALSE ALARM"
    if expected == "absent" and got in ("verified", "unverified"):
        return "SOFT"            # flagged something we wanted silent (noise, not an accusation)
    if expected == "contradicted" and got in ("verified", "absent"):
        return "MISS"
    return "SOFT"


def got_status(findings, exp):
    hits = [f for f in findings if exp["find"] in f.claim.text
            and (("kind" not in exp) or f.claim.kind.value == exp["kind"])]
    if not hits:
        return "absent", None
    order = {"contradicted": 0, "unverified": 1, "verified": 2}   # report the worst finding for the claim
    f = min(hits, key=lambda h: order[h.status.value])
    return f.status.value, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "corpus.json"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--ai", action="store_true")
    ap.add_argument("--only", help="run only case ids containing this text")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    cases = json.load(open(a.corpus, encoding="utf-8"))["cases"]
    ai = None
    if a.ai:
        import cli
        ai, err = cli._build_ai(argparse.Namespace(model=None, max_llm_calls=60))
        if err:
            sys.exit(f"eval: {err}")
    tally = {"OK": 0, "SOFT": 0, "FALSE ALARM": 0, "MISS": 0}
    bad = []
    skipped = 0
    for case in cases:
        if a.only and a.only not in case["id"]:
            continue
        if a.offline and case.get("needs_network"):
            skipped += 1
            continue
        fetcher = OfflineFetcher() if a.offline else UrllibFetcher()
        t0 = time.time()
        findings = pipeline.run(case["text"], fetcher, timeout=20, ai=ai)
        for exp in case["expect"]:
            got, f = got_status(findings, exp)
            verdict = classify(exp["status"], got)
            tally[verdict] += 1
            if verdict != "OK":
                bad.append((verdict, case["id"], exp, got, f))
            if a.verbose or verdict != "OK":
                why = f.reason[:90] if f else "(no finding)"
                print(f"{verdict:11} {case['id']:30} {exp['find'][:24]:24} want {exp['status']:12} got {got:12} {why}")
    n = sum(tally.values())
    print(f"\n{n} expectations, {skipped} cases skipped (offline). " +
          "  ".join(f"{k}: {v}" for k, v in tally.items()))
    sys.exit(1 if tally["FALSE ALARM"] or tally["MISS"] else 0)


if __name__ == "__main__":
    main()
