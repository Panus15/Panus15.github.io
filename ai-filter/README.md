# ai-filter

Fact-checks the text of an AI answer. **Phase 1**: everything that can be checked
mechanically, with no AI in the loop. Standard library only.

```
python3 cli.py check answer.md              # report on stdout
python3 cli.py check answer.md --json       # for scripts
python3 cli.py check answer.md --offline    # never touch the network
python3 cli.py check answer.md --run-code   # ALSO execute Python blocks (read below)
python3 cli.py check answer.md --ai         # phase 2: AI claim extraction + official sources + "does the page say it?"
cat answer.txt | python3 cli.py check -
```

Exit code: `0` no problems found / nothing checkable · `1` needs review · `2` do not use · `3` bad input.

## What it checks

| Module | Looks at | Verdict logic |
|---|---|---|
| `citations` | URLs, DOIs, cited papers (Crossref, doi.org, OpenAlex) | dead link / unregistered DOI / real title with wrong authors → ❌. Not found in a database → ⚠️ only |
| `recompute` | `a × b = c` and "N% of M is K" | exact arithmetic (Fractions); wrong → ❌; right only by truncation → ⚠️ |
| `sandbox` | fenced Python blocks | does not parse → ❌. Parses but not run → ⚠️. With `--run-code`: raises / wrong stated output → ❌ |
| `vague` | "studies show", "มีการศึกษาพบว่า", "ผู้เชี่ยวชาญระบุว่า", over-precise figures with no source | ⚠️ only — it can say "needs a source", never "false" |

## Phase 2: `--ai` (opt-in)

**Sends data to third parties.** The answer text goes to the Anthropic API; search queries derived from it go to Brave Search; fetched pages go to Anthropic for judging. Nothing is sent unless you pass `--ai`, and the CLI prints a notice when you do.

Setup: `pip install anthropic` (the only non-stdlib dependency, imported lazily), working Anthropic credentials (`ANTHROPIC_API_KEY` or `ant auth login`), and `BRAVE_API_KEY` for official-source search (without it, factual claims are reported as "not checked"). `--model` picks the Claude model (default `claude-opus-5-5`); `--max-llm-calls` (default 40) is a hard cap.

| Module | What it adds |
|---|---|
| `ai_extract` | the LLM proposes check-worthy facts (names, numbers, dates, laws, prices); each must quote a span that really exists in the answer or it is dropped |
| `official` | searches only the trusted-domain registry (`sources.json`, 120 official sites), fetches the pages, and has them judged. Fast-changing facts need a recent dated page |
| `support` | for a link that opens, checks the page actually contains what the sentence around it claims. A link that opens but lacks the claim is downgraded to ⚠️ |

**The LLM never decides alone.** Its output is validated in code (`engine/judge.py`): a quote must appear verbatim on the page; every number in the claim must be in the quote; the negation must agree; a contradiction needs two differing values that really sit in the claim and the quote. Failing any rule gives "not stated" (⚠️), never ✅/❌. Answer text and page text are treated as untrusted (hidden HTML text is stripped, tags are escaped), so a page saying "ignore your instructions" changes nothing.

Not verified live: the Claude call was checked against the real SDK's request validation and a local fake server, not against the live API (no credentials in the build environment). The first real run is the live test; if the API rejects structured outputs together with `fallbacks`, pass `fallbacks=False` to `AnthropicLLM`.

## Rules the code is built around

1. **Our failure is never the AI's fault.** Network down, timeout, rate limit, sandbox error → ⚠️, never ❌. `❌` always means positive evidence.
2. **Absence is not evidence.** A paper missing from OpenAlex is ⚠️ (books and Thai-language works are often unindexed).
3. **No "OK" verdict.** The best result is "no problems found in what was checked". A link that opens does not prove the page says what the AI claims. An answer with nothing checkable is *not* a pass.
4. **Every claim is accounted for.** A claim no checker handles surfaces as ⚠️ rather than vanishing.
5. **Untrusted input.** Answer text drives network requests and report output. `engine/net.py` refuses non-http(s) and any host resolving to a private/loopback/link-local address, at every redirect hop; `engine/report.py` neutralises Markdown from claim text and remote titles.

## `--run-code` warning

Runs code taken from the answer on this machine, in a subprocess with time/memory/file limits, a scrubbed environment, a temp working directory and a socket guard. That is defence against accidents, **not a security boundary against hostile code**. Off by default; use Docker/gVisor for untrusted input.

## Tests

```
for t in tests/test_*.py; do python3 "$t" || echo FAIL $t; done
```

All offline (checkers take an injected `Fetcher`; tests use `FakeFetcher`). `tests/test_pipeline.py` runs an answer with planted defects and true controls end to end.

## Not yet (phases 3–4)

Repeat-asking and cross-model comparison (modules 4–5), calibration of the scoring weights, a web UI. Known gaps in phase 2: negation is checked only by presence of a negation word, not scope; the trusted-domain list is hand-curated and incomplete (Thai courts and the Attorney General's office were left out as unverified).
