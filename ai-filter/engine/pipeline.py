"""Glue: text -> claims -> every checker -> one ordered list of Findings.

Guarantees that live here, because no single checker can give them:

1. COVERAGE. Every extracted claim ends with exactly one Finding. A claim no
   checker picked up becomes UNVERIFIED rather than silently vanishing.

2. ISOLATION. A checker that crashes costs you that checker's findings, turned
   into UNVERIFIED "could not check", never the whole report and never a
   CONTRADICTED.

3. OPT-IN AI. Everything that sends text to a third party (the LLM, the search
   engine) runs only when `ai` is passed. Without it this module is the phase-1
   pipeline: stdlib, offline-capable, nothing leaves the machine except the
   link/DOI lookups.

4. HONEST DEGRADATION. If AI was requested but a part of it cannot run (no search
   key, LLM error, budget exhausted) that part produces visible UNVERIFIED
   findings. The verdict then says "review", never "no problems".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from engine import citations, extract, recompute, sandbox, vague
from engine.llm import LLM, LLMError
from engine.models import Claim, ClaimKind, Finding, Status
from engine.net import Fetcher, OfflineFetcher, UrllibFetcher
from engine.search import SearchClient


@dataclass(frozen=True)
class AIConfig:
    llm: LLM
    search: Optional[SearchClient] = None     # None -> official-source checks are reported as unavailable
    max_fact_claims: int = 10
    samples: int = 3                          # independent re-answers for the consistency check (needs a question)
    general_fallback: bool = True             # consult general references (Wikipedia, ...) when no official source decides


def _ai_note(kind_id: str, reason: str) -> Finding:
    """A finding about the filter itself ("AI extraction unavailable"), not about a claim."""
    return Finding(Claim(kind_id, ClaimKind.FACT, "", 0, 0), Status.UNVERIFIED, "ai", reason)


def _inside(inner: Claim, outer: Claim) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def _note(name: str, e: Exception) -> Finding:
    """A finding about the filter itself: one of its checkers broke. Never an accusation."""
    return Finding(Claim(f"{name}0", ClaimKind.CUE, "", 0, 0), Status.UNVERIFIED, name,
                   f"the {name} check could not run ({type(e).__name__})")


def run(text: str, fetcher: Optional[Fetcher] = None, *, run_code: bool = False,
        timeout: float = 10.0, ai: Optional[AIConfig] = None,
        question: Optional[str] = None) -> list[Finding]:
    claims = extract.extract(text)
    findings: list[Finding] = []
    # One fetcher for the whole run: citations and the support check then share
    # connection behaviour, and tests inject a single fake.
    fetcher = fetcher or UrllibFetcher()

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
            lambda cs: citations.check(cs, fetcher, timeout=timeout, text=text))
    guarded("recompute", {ClaimKind.ARITHMETIC}, recompute.check)
    guarded("sandbox", {ClaimKind.CODE}, lambda cs: sandbox.check(cs, run=run_code))

    # Topic-independent checks: they need no knowledge base, so they apply to every kind of answer.
    # Each is imported INSIDE its own try: a module that fails even to import (not only to run)
    # costs that module, never the whole report.
    offline = isinstance(fetcher, OfflineFetcher)

    def _cue():
        from engine import cues
        return cues.scan(text, question=question)

    def _computable():
        from engine import computable
        return computable.scan(text)

    def _codecheck():
        from engine import codecheck
        return codecheck.scan(text, claims, fetcher=fetcher, registries=not offline)

    for name, fn in (("cue", _cue), ("computable", _computable), ("codecheck", _codecheck)):
        try:
            findings.extend(fn())
        except Exception as e:
            findings.append(_note(name, e))

    if ai is not None:
        findings = _run_ai(text, claims, findings, fetcher, ai, question)

    covered = {f.claim.id for f in findings}
    for c in claims:
        if c.id not in covered:
            findings.append(Finding(c, Status.UNVERIFIED, "pipeline",
                                    f"no checker handled this {c.kind.value} claim"))

    # A citation proven fake must not count as "a source" for a nearby unsourced claim.
    fake = [(f.claim.start, f.claim.end) for f in findings
            if f.status is Status.CONTRADICTED
            and f.claim.kind in (ClaimKind.URL, ClaimKind.DOI, ClaimKind.REFERENCE)]
    try:
        findings.extend(vague.scan(text, ignore_spans=fake))
    except Exception as e:
        findings.append(Finding(Claim("v0", ClaimKind.VAGUE, "", 0, 0), Status.UNVERIFIED,
                                "vague", f"could not scan for unsourced claims ({type(e).__name__})"))

    findings.sort(key=lambda f: (f.claim.start, f.claim.end, f.claim.id))
    return findings


def _run_ai(text: str, claims: list[Claim], findings: list[Finding], fetcher: Fetcher,
            ai: AIConfig, question: Optional[str] = None) -> list[Finding]:
    # Imported here so that phase-1 use never needs (or loads) any AI module.
    from engine import ai_extract, consistency, official, support

    # (a) does the page behind each link actually say what the answer says?
    try:
        findings = support.refine(text, findings, llm=ai.llm, fetcher=fetcher)
    except Exception as e:  # keep the phase-1 findings if refinement breaks
        findings = findings + [_ai_note("a0s", f"could not check whether pages support their claims ({type(e).__name__})")]

    # (b) AI-proposed factual claims, verified against official sources.
    try:
        facts = ai_extract.extract_facts_ex(text, ai.llm)
    except LLMError as e:
        return findings + [_ai_note("a0", f"AI claim extraction unavailable ({e})")]
    except Exception as e:
        return findings + [_ai_note("a0", f"AI claim extraction failed ({type(e).__name__})")]

    fact_claims = [c for c in facts.claims
                   if not any(_inside(c, o) for o in claims
                              if o.kind in (ClaimKind.ARITHMETIC, ClaimKind.CODE))]
    out = list(findings)
    if facts.truncated:
        out.append(_ai_note("a0t", "the answer was too long to analyse completely for factual claims"))
    if not fact_claims:
        return out
    if ai.search is None:
        out += [Finding(c, Status.UNVERIFIED, "official",
                        "not checked: no web-search key configured (set BRAVE_API_KEY)") for c in fact_claims]
    else:
        try:
            out += official.check(fact_claims, llm=ai.llm, search=ai.search, fetcher=fetcher,
                                  max_claims=ai.max_fact_claims, general_fallback=ai.general_fallback)
        except Exception as e:
            out += [Finding(c, Status.UNVERIFIED, "official",
                            f"could not check ({type(e).__name__})") for c in fact_claims]
    # Re-ask the question: needs no search key and no knowledge base, so it works on any topic.
    if question and question.strip():
        try:
            out += consistency.check(question, text, ai.llm, n_samples=ai.samples, facts=fact_claims)
        except Exception as e:
            out.append(_note("consistency", e))
    return out
