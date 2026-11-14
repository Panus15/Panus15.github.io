"""Glue: text -> claims -> every checker -> one ordered list of Findings.

Two guarantees live here, because no single checker can give them:

1. COVERAGE. Every extracted claim ends with exactly one Finding. A claim no
   checker picked up (a kind that has no checker yet) becomes UNVERIFIED rather
   than silently vanishing -- otherwise adding an extractor rule without a
   checker would make answers look cleaner than they are.

2. ISOLATION. A checker that crashes costs you that checker's findings, turned
   into UNVERIFIED "could not check", never the whole report and never a
   CONTRADICTED.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from engine import citations, extract, recompute, sandbox, vague
from engine.models import Claim, ClaimKind, Finding, Status
from engine.net import Fetcher


def run(text: str, fetcher: Optional[Fetcher] = None, *, run_code: bool = False,
        timeout: float = 10.0) -> list[Finding]:
    claims = extract.extract(text)
    findings: list[Finding] = []

    def guarded(name: str, kinds: set[ClaimKind], fn: Callable[[list[Claim]], list[Finding]]):
        mine = [c for c in claims if c.kind in kinds]
        if not mine:
            return
        try:
            findings.extend(fn(mine))
        except Exception as e:  # a checker bug must not take the run down
            findings.extend(Finding(c, Status.UNVERIFIED, name,
                                    f"could not check ({type(e).__name__})") for c in mine)

    guarded("citations", {ClaimKind.URL, ClaimKind.DOI, ClaimKind.REFERENCE},
            lambda cs: citations.check(cs, fetcher, timeout=timeout))
    guarded("recompute", {ClaimKind.ARITHMETIC}, recompute.check)
    guarded("sandbox", {ClaimKind.CODE}, lambda cs: sandbox.check(cs, run=run_code))

    covered = {f.claim.id for f in findings}
    for c in claims:
        if c.id not in covered:
            findings.append(Finding(c, Status.UNVERIFIED, "pipeline",
                                    f"no checker handled this {c.kind.value} claim"))

    try:
        findings.extend(vague.scan(text))
    except Exception as e:
        findings.append(Finding(Claim("v0", ClaimKind.VAGUE, "", 0, 0), Status.UNVERIFIED,
                                "vague", f"could not scan for unsourced claims ({type(e).__name__})"))

    findings.sort(key=lambda f: (f.claim.start, f.claim.end, f.claim.id))
    return findings
