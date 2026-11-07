# mutation

```bash
python3 mutation/run.py                 # every target
python3 mutation/run.py verdict data    # named targets only
```

**242 deliberate defects across 10 targets. All 242 caught.**

| target | file | defects |
|---|---|---|
| `verdict` | `engine/verdict.py` | 45 |
| `data` | `engine/data.py` | 26 |
| `analyse` | `tools/analyse.py` | 25 |
| `levers` | `engine/levers.py` | 25 |
| `holdout` | `engine/holdout.py` | 23 |
| `report` | `tools/report.py` | 23 |
| `backtest` | `engine/backtest.py` | 21 |
| `stats` | `engine/stats.py` | 20 |
| `costs` | `engine/costs.py` | 18 |
| `patterns` | `models/patterns.py` | 16 |

## Why this is in the repository

It did not used to be. The sweeps lived in a scratchpad directory outside the
tree, the README quoted a number from them, and when the container was rebuilt
they were gone — leaving a quality claim that nobody, including the person who
made it, could reproduce. That is precisely the kind of assertion the rest of
this engine refuses to accept, so it does not get a pass here either.

Moving them in cost five surviving mutants immediately, because the tables were
rewritten from the source rather than recovered. Each was a real gap:

- **`h&s completion looks ahead`** — the neckline-break scan could start AT the
  right shoulder instead of after it. The right shoulder is a swing high and
  cannot be known until later bars print, so that is look-ahead. Every existing
  fixture had a shoulder closing above the neckline, which cannot tell the two
  versions apart.
- **`swing bars reduced to one`** — the test asserted `SWING_BARS <= i < len(bars) - SWING_BARS`,
  which is satisfied at any width, including 1. Scaling a fixture by the constant
  under test passes for every value of it.
- **`closeness not relative`** — `EQUAL_TOL` is a fraction, but on EURUSD at 1.10
  a 0.15% tolerance and an absolute 0.0015 are nearly the same number. Only a
  JPY-scale pair distinguishes them.
- **`span ceiling removed`** — no fixture had a shape wider than `MAX_SPAN`.
- **`negative markup allowed`** — negative `round_turn_pips` was refused and
  tested; negative `swap_markup_annual`, a broker that pays you to hold, was
  refused and not.

## How it protects your working tree

It rewrites source files in place, so:

1. **A target with uncommitted changes is skipped by name.** It will not put
   work at risk to measure itself.
2. **The original is copied to `mutation/.restore/` before the first mutation**
   and removed only once the file is back. `finally` covers exceptions and
   Ctrl-C; it does **not** run through a `SIGKILL` — and that is not
   hypothetical. A run of this harness was killed mid-sweep and left
   `if False:` sitting in `holdout.py`, disabling a guard in a way that passes
   its own test suite by construction. Every run now repairs anything it finds
   in `.restore/` first and prints that it did.
3. **Every run forces `PYTHONDONTWRITEBYTECODE=1`.** A stale `.pyc` once
   inverted a sweep in this project and reported mutants killed that were not.

## How a defect is written

```python
("ambiguity ignored", "    if res.ambiguous_frac > AMBIGUITY_WARN_FRAC:", "    if False:")
```

A label, the exact source snippet, and what to replace it with. The snippet must
match **exactly once**: zero or several matches is reported as `ANCHOR` and
**fails the run** rather than being skipped. A harness that quietly skipped
stale anchors would report a shrinking, still-perfect score while its coverage
drained away.

## Reading a survivor

A survivor means the test suite cannot tell the real code from a broken version
of it — so the line that was changed is untested, whatever a coverage report
says about it. Two outcomes are legitimate:

- **Fix the test.** Usually it was asserting something weaker than it looked.
  The zero-on-the-axis test in `report.py` ended in `or zero_y < min(ys)`, a
  clause satisfied by zero falling off the chart: the exact failure it was
  written to catch.
- **Record it as an EQUIVALENT mutant.** Some changes cannot alter behaviour.
  Judging `gross.net` instead of `net.net` in `verdict.py` is one:
  `mean(gross) − mean(cost)` *is* `mean(net)`, verified to 1e-14 on random
  inputs. Those are documented in the module and removed from the table, never
  left in as a permanent "known survivor".
