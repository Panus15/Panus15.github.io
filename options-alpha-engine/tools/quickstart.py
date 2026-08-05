"""Cold clone to a dashboard on screen, in one command.

Every failure this file guards against actually happened to a user on Windows,
and none of them were interesting: a placeholder path pasted literally, two
commands merged onto one line, `cd options-alpha-engine` run from inside
options-alpha-engine, and a vendor bot-check that left no output file so the
next command died on FileNotFoundError instead of on the real cause.

The fix is structural rather than documentary. Every step runs from THIS FILE'S
directory, not the shell's, so a wrong working directory cannot happen. The
steps run in one process, so they cannot be merged or reordered. And no step
starts until the previous one produced the file it promised — a run that cannot
finish says which step stopped it and why, instead of failing two steps later
with an unrelated error.

    python3 -m tools.quickstart              # update, fetch, render, open
    python3 -m tools.quickstart --demo       # no network: generated fixture
    python3 -m tools.quickstart --no-update  # skip git

On Windows, START.bat is this file with the console kept open.
"""

import argparse
import os
import subprocess
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import fetch_prices, rotation_dashboard          # noqa: E402

BRANCH = "claude/ml-options-trading-engine-b471vm"
RESULT = "RESULT.txt"

# The pre-registered rotation parameters. Spelled out here rather than left to a
# default so the one-click path cannot quietly run a different experiment from
# the documented one; a test asserts these still equal the CLI's own defaults,
# because a launcher that drifts is worse than no launcher.
PARAMS = {"window": 63, "mom_lag": 5, "tail": 12, "horizon": 21}


def project_dir() -> str:
    """Where the code lives — never where the shell happens to be."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_console_safe() -> None:
    """Stop a punctuation mark from ending the run.

    print() encodes through the console's codepage, and every verdict this tool
    prints contains an em dash. On a console whose codepage cannot represent one
    — which is most of them outside western Europe — print raises
    UnicodeEncodeError from inside the success path, and the user sees a stack
    trace at the exact moment the answer was ready. Ask for utf-8; if the
    console refuses, degrade the character rather than the run.
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass                              # redirected, or already fine


def _git(args, cwd, timeout=120):
    try:
        p = subprocess.run(["git"] + args, cwd=cwd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"{type(e).__name__}: {e}"


def update(cwd: str, branch: str = BRANCH) -> str:
    """Pull, but only onto the branch this work lives on.

    `git pull origin <branch>` while another branch is checked out merges that
    branch into whatever you are standing on. Refusing is the safe answer: a
    stale checkout renders an old chart, a bad merge loses work.
    """
    rc, head = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if rc:
        return f"skipped — not a git checkout ({head.splitlines()[0][:60]})"
    if head != branch:
        return f"skipped — on branch '{head}', not '{branch}'"
    rc, out = _git(["pull", "origin", branch], cwd)
    last = out.splitlines()[-1] if out else ""
    return ("updated: " + last) if rc == 0 else ("could not pull: " + last[:80])


def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")
    print("-" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true",
                    help="render a generated world; needs no network")
    ap.add_argument("--no-update", action="store_true", help="skip git pull")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--source", default="auto", help="auto, yahoo or stooq")
    ap.add_argument("--base-url", default="", help="fetch from a local mirror")
    ap.add_argument("--csv", default="sectors.csv")
    ap.add_argument("--out", default="rotation.html")
    a = ap.parse_args(argv)
    make_console_safe()

    here = project_dir()
    csv_path = os.path.join(here, a.csv)
    out_path = os.path.join(here, a.out)
    total = 3 if a.demo else 4

    print("=" * 60)
    print("  Options Alpha Engine — sector rotation")
    print("=" * 60)
    print(f"  running from: {here}")

    step(1, total, "checking for updates")
    print("  " + ("skipped (--no-update)" if a.no_update else update(here)))

    if a.demo:
        px, bench, dates = rotation_dashboard.demo_world()
        print("\n  --demo: using a generated world, nothing downloaded")
    else:
        step(2, total, "downloading prices")
        argv2 = ["--out", csv_path, "--source", a.source]
        if a.base_url:
            argv2 += ["--base-url", a.base_url]
        if fetch_prices.main(argv2) != 0 or not os.path.exists(csv_path):
            print("\n" + "=" * 60)
            print("  STOPPED at step 2: no price file was written.")
            print("  Nothing below this line ran. The reason is printed above —")
            print("  it is about the data vendor, not about your setup.")
            print("=" * 60)
            return 1

        step(3, total, "reading the price file")
        try:
            px, bench, dates = rotation_dashboard.load_csv(csv_path)
        except (ValueError, OSError) as e:
            print(f"  STOPPED: {a.csv} is unreadable: {e}")
            return 1
        print(f"  {len(px)} symbols, {len(dates)} bars, "
              f"{dates[0]} .. {dates[-1]}")

    step(total, total, "building the dashboard")
    payload = rotation_dashboard.build_payload(px, bench, dates, **PARAMS)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(rotation_dashboard.render(payload))
    print(f"  wrote {a.out}  ({len(payload['points'])} sectors, "
          f"{payload['bars']} bars)")

    lines = []
    if payload["test"]:
        lines = ["panel: " + payload["test"]["panelVerdict"],
                 "book : " + payload["test"]["bookVerdict"]]
        with open(os.path.join(here, RESULT), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    for ln in lines:
        print("  " + ln)
    if lines:
        print(f"\n  These two lines are the result. They are also saved to")
        print(f"  {RESULT} next to this project, so you can copy them from there.")
    if a.demo:
        print("\n  (generated data — these numbers describe a fixture, "
              "not the market)")

    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
    else:
        print(f"\n  open: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
