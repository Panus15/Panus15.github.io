# Evaluating the filter

```
python3 eval/run_eval.py                # all cases; the needs_network ones hit live Crossref / doi.org
python3 eval/run_eval.py --offline      # what CI runs
python3 eval/run_eval.py -v --only doi  # show every expectation for matching cases
python3 eval/run_eval.py --ai           # include the AI checks (needs credentials, spends money)
```

`corpus.json` holds answers and, for each claim, the status it must get — `verified`,
`unverified`, `contradicted`, or `absent` (must not be flagged). Mismatches are graded:

| Outcome | Meaning | Cost |
|---|---|---|
| FALSE ALARM | a true/innocent claim was called contradicted | breaks trust — the worst failure |
| MISS | a planted error was called verified or ignored | the filter's whole job |
| SOFT | any other mismatch (e.g. verified vs unverified) | noise or caution |

The process exits 1 on any FALSE ALARM or MISS.

## What the current numbers mean — and do not

The corpus is **synthetic**: answers written by the tool's author with hand-set labels. A
perfect score shows the filter does what it was designed to do on those cases. It says
nothing about how often it errs on real AI output, because the author wrote both the
tests and the answers and will not think of the ways a model actually goes wrong.

Running it against the real network did find three bugs the offline tests could not:
every link reported "domain does not exist" behind a proxy; a real DOI pasted onto a
fabricated reference verified it; a fake DOI next to an unsourced claim made the claim
look sourced.

## Adding real answers (the step that makes the numbers meaningful)

1. Take answers from an AI you actually use, on topics you can check. Aim for 20+, long ones included.
2. Add each to `cases` with `"needs_network": true` if it has links/DOIs/references.
3. For every checkable claim, label what is *true* (you verify it by hand) — not what you expect the filter to say.
4. Run it. Read every FALSE ALARM and MISS; each is either a filter bug or a wrong label.
5. Put real, redacted answers in a separate file (`--corpus mine.json`) if they are private.

Once there are ~20 real answers, the FALSE ALARM and MISS counts per module are the numbers
to tune against.
