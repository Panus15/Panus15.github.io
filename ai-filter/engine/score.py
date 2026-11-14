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


# Process exit codes, so the CLI can gate a script or CI step.
EXIT_CODE = {
    Verdict.NO_PROBLEMS_FOUND: 0,
    Verdict.NO_CHECKABLE_CLAIMS: 0,
    Verdict.REVIEW: 1,
    Verdict.DO_NOT_USE: 2,
}
