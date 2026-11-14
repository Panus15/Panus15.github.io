# ai-filter

Fact-checks the text of an AI answer. **Phase 1**: everything that can be checked
mechanically, with no AI in the loop. Standard library only.

```
python3 cli.py check answer.md              # report on stdout
python3 cli.py check answer.md --json       # for scripts
python3 cli.py check answer.md --offline    # never touch the network
python3 cli.py check answer.md --run-code   # ALSO execute Python blocks (read below)
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

## Not yet (phases 2–4)

AI-based claim extraction, search against official sources (module 2), checking that a page *supports* the claim, repeat-asking and cross-model comparison (modules 4–5), web UI.
