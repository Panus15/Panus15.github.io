#!/usr/bin/env python3
"""aifilter: fact-check the text of an AI answer.

    python3 cli.py check answer.md                 # report to stdout
    python3 cli.py check answer.md -o report.md
    cat answer.txt | python3 cli.py check -        # read stdin
    python3 cli.py check answer.md --json
    python3 cli.py check answer.md --offline       # no network; lookups come back UNVERIFIED
    python3 cli.py check answer.md --run-code      # ALSO execute Python blocks (see warning)
    python3 cli.py check answer.md --ai            # AI claim extraction, official-source and page checks

--ai SENDS DATA TO THIRD PARTIES: the answer text goes to the Anthropic API, and
search queries derived from it go to the Brave Search API; fetched pages are sent
to Anthropic for judging. Off by default. Needs `pip install anthropic`, working
Anthropic credentials, and BRAVE_API_KEY for the official-source search. Every AI
verdict is checked mechanically (quotes must appear verbatim in the source) before
it can change a result.

--run-code executes code taken from the answer on THIS machine. The sandbox limits
time, memory, files and network, but it is not a security boundary against hostile
code. Off by default; only use it on answers you would be willing to run yourself.

Exit code: 0 no problems found / nothing checkable, 1 needs review, 2 do not use,
3 usage or input error. Gate a script on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import pipeline, report
from engine.net import OfflineFetcher
from engine.score import EXIT_CODE, verdict

MAX_INPUT_BYTES = 2_000_000


def _build_ai(args, env=None):
    """Return (AIConfig, None) or (None, error message). Imports AI modules lazily so the
    stdlib-only core never needs them."""
    env = os.environ if env is None else env
    from engine.llm import LLMError, make_llm
    from engine.pipeline import AIConfig
    from engine.search import BraveSearch
    if args.max_llm_calls < 1:
        return None, "--max-llm-calls must be at least 1"
    try:
        llm = make_llm(args.model, max_calls=args.max_llm_calls)
    except LLMError as e:
        return None, str(e)
    search = BraveSearch() if env.get("BRAVE_API_KEY") else None
    print("aifilter: --ai is on. The answer text is being sent to the Anthropic API"
          + (", search queries to Brave Search," if search else "; no BRAVE_API_KEY, so official-source search is skipped,")
          + " and fetched pages to Anthropic for judging.", file=sys.stderr)
    return AIConfig(llm=llm, search=search), None


def main(argv=None, stdin=None, stdout=None) -> int:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    p = argparse.ArgumentParser(prog="aifilter", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check an answer")
    c.add_argument("file", help="path to the answer text, or - for stdin")
    c.add_argument("-o", "--output", help="write the report here instead of stdout")
    c.add_argument("--json", action="store_true", help="machine-readable output")
    c.add_argument("--offline", action="store_true", help="never touch the network")
    c.add_argument("--run-code", action="store_true", help="execute Python code blocks (read the warning)")
    c.add_argument("--timeout", type=float, default=10.0, help="per-request network timeout, seconds")
    c.add_argument("--ai", action="store_true", help="enable AI-assisted checks (sends text to third parties; read the notice)")
    c.add_argument("--model", default=None, help="Claude model for --ai (default: claude-opus-5-5)")
    c.add_argument("--max-llm-calls", type=int, default=40, help="hard cap on LLM calls per run (default 40)")
    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        return 3 if e.code else 0

    try:
        if args.file == "-":
            raw = stdin.buffer.read(MAX_INPUT_BYTES + 1) if hasattr(stdin, "buffer") else stdin.read().encode()
        else:
            with open(args.file, "rb") as fh:
                raw = fh.read(MAX_INPUT_BYTES + 1)
    except OSError as e:
        print(f"aifilter: cannot read {args.file}: {e.strerror}", file=sys.stderr)
        return 3
    if len(raw) > MAX_INPUT_BYTES:
        print(f"aifilter: input larger than {MAX_INPUT_BYTES} bytes; refusing", file=sys.stderr)
        return 3
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print("aifilter: input is not valid UTF-8", file=sys.stderr)
        return 3

    ai = None
    if args.ai:
        if args.offline:
            print("aifilter: --ai cannot be combined with --offline", file=sys.stderr)
            return 3
        ai, err = _build_ai(args)
        if err:
            print(f"aifilter: {err}", file=sys.stderr)
            return 3
    findings = pipeline.run(text, OfflineFetcher() if args.offline else None,
                            run_code=args.run_code, timeout=args.timeout, ai=ai)
    out = (json.dumps(report.to_dict(findings), ensure_ascii=False, indent=2)
           if args.json else report.render(findings))
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(out + "\n")
        except OSError as e:
            print(f"aifilter: cannot write {args.output}: {e.strerror}", file=sys.stderr)
            return 3
    else:
        stdout.write(out + "\n")
    return EXIT_CODE[verdict(findings)]


if __name__ == "__main__":
    sys.exit(main())
