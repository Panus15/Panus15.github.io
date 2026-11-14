"""Roll many Findings up into one overall verdict.

The verdict is deliberately conservative in ONE direction. It is allowed to say
"review this" too often; it is never allowed to say "fine" about something it did
not actually check:

  DO_NOT_USE          at least one claim is CONTRADICTED by evidence
  REVIEW              nothing contradicted, but something could not be confirmed
  NO_CHECKABLE_CLAIMS the filter found nothing it knows how to check. This is NOT
                      a pass: an answer with no links, numbers or code is simply
                      outside what this filter can see.
  NO_PROBLEMS_FOUND   every checked claim was verified. Still not "true": a link
                      that opens does not prove the page says what the AI said.

There is no "OK" or "SAFE" value on purpose.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Sequence

from engine.models import Finding, Status


class Verdict(str, Enum):
    DO_NOT_USE = "do_not_use"
    REVIEW = "review"
    NO_CHECKABLE_CLAIMS = "no_checkable_claims"
    NO_PROBLEMS_FOUND = "no_problems_found"


def verdict(findings: Sequence[Finding]) -> Verdict:
    statuses = {f.status for f in findings}
    if Status.CONTRADICTED in statuses:
        return Verdict.DO_NOT_USE
    if Status.UNVERIFIED in statuses:
        return Verdict.REVIEW
    if not findings:
        return Verdict.NO_CHECKABLE_CLAIMS
    return Verdict.NO_PROBLEMS_FOUND


def counts(findings: Sequence[Finding]) -> dict[Status, int]:
    c = Counter(f.status for f in findings)
    return {s: c.get(s, 0) for s in Status}


# --------------------------------------------------------------------------- risk (second view)
#
# The verdict above is a GATE (it decides the exit code). Risk is a READING AID: how much
# of the answer needs a human's attention, on topics where nothing could be verified or
# contradicted. It is a transparent weighted count, not a probability, and THE WEIGHTS
# ARE GUESSES until they are calibrated on real labelled answers (eval/). They live in
# this one table so that calibration is a data change, not a code change.

CUE_WEIGHTS: dict[str, int] = {
    "placeholder_url": 3,            # a template address cannot be a source
    "unsourced_quote": 3,            # fabricated quotations are a known failure
    "inconsistent_across_samples": 3,  # independent answers disagree: the model was guessing
    "absolute_claim": 2,
    "named_expert_claim": 2,
    "count_mismatch": 2,
    "question_asks_for_sources": 2,
    "time_sensitive": 1,
    "ai_self_disclosure": 1,
    "consistency_unavailable": 0,    # says nothing about the answer
}
DEFAULT_UNVERIFIED_WEIGHT = 1        # any other UNVERIFIED finding (vague appeal, unchecked link, ...)
CONTRADICTED_WEIGHT = 10             # positive evidence outweighs any pile of cues
RISK_HIGH, RISK_MEDIUM = 8, 3        # score thresholds


def _weight(f: Finding) -> int:
    if f.status is Status.CONTRADICTED:
        return CONTRADICTED_WEIGHT
    if f.status is Status.VERIFIED:
        return 0
    return CUE_WEIGHTS.get(f.tag, DEFAULT_UNVERIFIED_WEIGHT)


def risk(findings: Sequence[Finding]) -> dict:
    """{"score": int, "level": "high"|"medium"|"low"|"none", "top": [Finding, ...]} (top = heaviest, max 5)."""
    score = sum(_weight(f) for f in findings)
    contradicted = any(f.status is Status.CONTRADICTED for f in findings)
    level = ("high" if contradicted or score >= RISK_HIGH else
             "medium" if score >= RISK_MEDIUM else "low" if score > 0 else "none")
    top = sorted((f for f in findings if _weight(f) > 0), key=lambda f: -_weight(f))[:5]
    return {"score": score, "level": level, "top": top}


# Process exit codes, so the CLI can gate a script or CI step.
EXIT_CODE = {
    Verdict.NO_PROBLEMS_FOUND: 0,
    Verdict.NO_CHECKABLE_CLAIMS: 0,
    Verdict.REVIEW: 1,
    Verdict.DO_NOT_USE: 2,
}
