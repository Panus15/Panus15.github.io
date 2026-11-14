#!/usr/bin/env python3
"""aifilter: fact-check the text of an AI answer.

    python3 cli.py check answer.md                 # report to stdout
    python3 cli.py check answer.md -o report.md
    cat answer.txt | python3 cli.py check -        # read stdin
    python3 cli.py check answer.md --json
    python3 cli.py check answer.md --offline       # no network; lookups come back UNVERIFIED
    python3 cli.py check answer.md --run-code      # ALSO execute Python blocks (see warning)

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

    findings = pipeline.run(text, OfflineFetcher() if args.offline else None,
                            run_code=args.run_code, timeout=args.timeout)
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
