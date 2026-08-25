"""The daily job — one command, so the clock that expires actually runs.

    python3 -m tools.daily run                  # do today's work
    python3 -m tools.daily status               # how far along the clocks are
    python3 -m tools.daily install              # print the scheduler line for this OS

The largest risk to this project is not a modelling error. It is that nobody runs
the daily job. `engine/crowding_backtest.py` is finished, tested and
mutation-verified, and it needs roughly eighteen months of dated fund books that
can only be captured one day at a time — a fund publishes today's option book and
overwrites it, so a day missed is gone at any price. A study that cannot start
until someone remembers a command every morning does not start.

So this collapses the routine to one invocation, keeps going when a step fails,
and reports the clocks in the only units that matter: how many days are banked and
what date the study becomes possible. Progress you can see is progress that keeps
happening.

WHAT IT DELIBERATELY DOES NOT DO. It does not trade, place orders, or size
anything. It captures data and records forecasts. Everything downstream of that is
a decision a human makes after reading a report.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: crowding_backtest needs >= 30 paired dates at a 21-trading-day spacing.
PAIRS_NEEDED = 30
DTE_SPACING = 21
TRADING_DAYS_NEEDED = PAIRS_NEEDED * DTE_SPACING       # ~630 trading days


def _today() -> str:
    return datetime.date.today().isoformat()


def _step(name: str, fn) -> dict:
    """Run one step, catching everything: a failing step must not stop the rest."""
    try:
        detail = fn()
        return {"step": name, "ok": True, "detail": detail or ""}
    except SystemExit as e:                      # argparse/main() returning non-zero
        return {"step": name, "ok": int(getattr(e, "code", 1) or 0) == 0,
                "detail": f"exit {getattr(e, 'code', 1)}"}
    except Exception as e:                       # noqa: BLE001
        return {"step": name, "ok": False, "detail": f"{type(e).__name__}: {e}"}


def run(cfg: dict) -> list:
    """Today's capture. Returns one status dict per step."""
    out = []

    if cfg.get("holdings"):
        def _holdings():
            from tools.archive_holdings import fetch_all
            src = (json.load(open(cfg["sources"])) if cfg.get("sources") else None)
            rows = fetch_all(cfg["holdings"], sources=src)
            got = sum(r["status"] in ("archived", "already have it") for r in rows)
            bad = [f"{r['fund']}: {r['status']}" for r in rows
                   if r["status"] not in ("archived", "already have it")]
            return (f"{got}/{len(rows)} books"
                    + (f"  |  {'; '.join(bad)}" if bad else ""))
        out.append(_step("fund holdings", _holdings))

    if cfg.get("ledger"):
        def _ledger():
            argv = ["record", "--ledger", cfg["ledger"], "--source", cfg["source"],
                    "--dte", str(cfg["dte"])]
            if cfg["source"] == "deribit":
                argv += ["--currency", cfg.get("currency", "BTC")]
            else:
                argv += ["--symbol", cfg.get("symbol", "SPX")]
            if cfg.get("events"):
                argv += ["--events-json", cfg["events"]]
            from tools.paper_trade import main as pt
            rc = pt(argv)
            if rc:
                raise RuntimeError(f"paper_trade record returned {rc}")
            return f"recorded into {os.path.basename(cfg['ledger'])}"
        out.append(_step("paper ledger", _ledger))

    if cfg.get("prices"):
        def _prices():
            from tools.fetch_prices import main as fp
            rc = fp(["--out", cfg["prices"], "--pause", "0.3"])
            if rc:
                raise RuntimeError(f"fetch_prices returned {rc}")
            return f"refreshed {os.path.basename(cfg['prices'])}"
        out.append(_step("price basket", _prices))

    return out


def status(cfg: dict) -> str:
    """Where the two clocks stand, in days and in dates."""
    lines = []

    if cfg.get("holdings"):
        from tools.archive_holdings import coverage
        cov = coverage(cfg["holdings"])
        if not cov:
            lines.append(
                "  fund holdings   NOT STARTED\n"
                "                  crowding_backtest cannot run at all until this\n"
                "                  does, and the days in between are unrecoverable.")
        else:
            best = max(cov.values(), key=lambda c: c["days"])
            have = best["days"]
            left = max(0, TRADING_DAYS_NEEDED - have)
            eta = (datetime.date.today()
                   + datetime.timedelta(days=int(left * 7 / 5))).isoformat()
            pct = min(100, int(100 * have / TRADING_DAYS_NEEDED))
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            lines.append(f"  fund holdings   [{bar}] {pct}%   {have}/"
                         f"{TRADING_DAYS_NEEDED} trading days")
            lines.append(f"                  first {best['first']}  "
                         f"latest {best['last']}  missing {best['missing']}")
            lines.append(f"                  the crowding study becomes possible "
                         f"around {eta}"
                         if left else
                         "                  ENOUGH DATA — run the crowding study now")

    if cfg.get("ledger") and os.path.exists(cfg["ledger"]):
        n = settled = corrupt = 0
        with open(cfg["ledger"]) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    corrupt += 1        # a line that will not parse is not an entry
                    continue
                n += 1
                settled += entry.get("status") == "settled"
        need = 30
        lines.append(f"  paper ledger    {n} entries, {settled} settled "
                     f"(need {need} settled for a verdict)"
                     + (f"  !! {corrupt} unreadable line(s)" if corrupt else ""))
        if settled < need:
            lines.append(f"                  the first score appears {cfg['dte']} "
                         f"records after the first entry — that lag is the honest "
                         f"cost of an out-of-sample test")
    elif cfg.get("ledger"):
        lines.append("  paper ledger    NOT STARTED — forward-test time only "
                     "accrues once it does")

    if cfg.get("prices"):
        p = cfg["prices"]
        if os.path.exists(p):
            with open(p) as fh:
                rows = sum(1 for _ in fh) - 1
            lines.append(f"  price basket    {rows} rows in "
                         f"{os.path.basename(p)}  (rotation study is ready to run)")
        else:
            lines.append("  price basket    not fetched yet — "
                         "`python3 -m tools.fetch_prices --out sectors.csv`")

    return "\n".join(lines) or "  nothing configured; see --help"


#: cfg key -> CLI flag, and when it must be emitted. Derived from ONE table so a
#: new flag cannot be added to the job and forgotten here. The old code listed
#: five flags by hand and appended two more, which silently dropped --symbol:
#: a job configured for SPY scheduled itself with no --symbol and fell back to
#: SPX at record time, forever, without a word.
_EMIT = {
    "holdings": lambda c: bool(c.get("holdings")),
    "ledger": lambda c: bool(c.get("ledger")),
    "prices": lambda c: bool(c.get("prices")),
    "events": lambda c: bool(c.get("events")),
    "sources": lambda c: bool(c.get("sources")),
    "source": lambda c: True,
    "dte": lambda c: True,
    "currency": lambda c: c.get("source") == "deribit",
    "symbol": lambda c: c.get("source") == "tradier",
}


def _q(v) -> str:
    """Quote a value for a shell/batch line. Paths on Windows contain spaces."""
    v = str(v)
    return v if v and not any(ch in v for ch in ' \t"&|<>^()') else '"' + v + '"'


def job_command(cfg: dict, *, python: str | None = None) -> str:
    """The full `daily run` command line, with every configured flag present."""
    py = python or sys.executable or "python3"
    parts = [_q(py), "-m", "tools.daily", "run"]
    for key, want in _EMIT.items():
        if want(cfg) and cfg.get(key) not in (None, ""):
            parts += [f"--{key.replace('_', '-')}", _q(cfg[key])]
    return " ".join(parts)


def wrapper_path() -> str:
    sysname = platform.system()
    return os.path.join(ROOT, "daily-job.bat" if sysname == "Windows"
                        else "daily-job.sh")


def write_wrapper(cfg: dict, *, python: str | None = None,
                  path: str | None = None) -> str:
    """Write a one-line runnable script and return its path.

    The scheduler gets a PATH, never a compound command. schtasks takes its
    payload in /tr as one double-quoted string with no way to escape an inner
    quote, so the previous template — `/tr "cmd /c cd <root> && <python> -m ..."`
    — could not survive a space anywhere in the interpreter path or the repo
    path. C:\Program Files\Python313\python.exe split at the space and the task
    failed every single day while schtasks still reported it registered, which
    is the worst available outcome: a clock the operator believes is running.
    """
    path = path or wrapper_path()
    cmd = job_command(cfg, python=python)
    log = _q(os.path.join(ROOT, "daily.log"))
    if path.endswith(".bat"):
        body = ("@echo off\r\n"
                "REM generated by `python -m tools.daily install` — safe to regenerate\r\n"
                f"cd /d {_q(ROOT)}\r\n"
                f"{cmd} >> {log} 2>&1\r\n")
        with open(path, "w", newline="") as fh:
            fh.write(body)
    else:
        body = ("#!/bin/sh\n"
                "# generated by `python3 -m tools.daily install` — safe to regenerate\n"
                f"cd {_q(ROOT)} || exit 1\n"
                f"exec {cmd} >> {log} 2>&1\n")
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    return path


TASK_NAME = "options-alpha-daily"


def scheduler_command(path: str, *, hour: int = 18) -> str:
    """The one line that registers the wrapper with this machine's scheduler."""
    if platform.system() == "Windows":
        # always quoted, not only when it contains a space: schtasks accepts a
        # quoted path unconditionally, and the day someone clones into
        # "C:\Users\First Last\..." is not the day to discover this
        return (f'schtasks /create /f /tn "{TASK_NAME}" /sc daily '
                f'/st {hour:02d}:00 /tr "{path}"')
    return (f"(crontab -l 2>/dev/null | grep -v {_q(path)}; "
            f"echo '0 {hour} * * 1-5 {path}') | crontab -")


def install_line(cfg: dict, *, hour: int = 18) -> str:
    """What to run, and what it will do — written but NOT executed by default."""
    path = wrapper_path()
    sysname = platform.system()
    where = ("an Administrator PowerShell" if sysname == "Windows"
             else "a terminal")
    return (f"wrote {path}\n"
            f"  it runs: {job_command(cfg)}\n"
            f"  logging to: {os.path.join(ROOT, 'daily.log')}\n\n"
            f"To schedule it, run this in {where}:\n\n"
            f"  {scheduler_command(path, hour=hour)}\n\n"
            f"Or let this do it for you:  "
            f"python -m tools.daily install --apply\n")


def _cfg(a) -> dict:
    return {"holdings": a.holdings, "ledger": a.ledger, "prices": a.prices,
            "events": a.events, "sources": a.sources, "source": a.source,
            "dte": a.dte, "currency": a.currency, "symbol": a.symbol}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "status", "install"])
    ap.add_argument("--holdings", default="holdings",
                    help="archive dir for fund books ('' to skip)")
    ap.add_argument("--ledger", default="", help="paper-trade ledger ('' to skip)")
    ap.add_argument("--prices", default="", help="price basket CSV ('' to skip)")
    ap.add_argument("--events", default="")
    ap.add_argument("--sources", default="", help="JSON of {FUND: url}")
    ap.add_argument("--source", default="deribit",
                    choices=["deribit", "tradier", "replay"])
    ap.add_argument("--dte", type=int, default=30)
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--symbol", default="SPX")
    ap.add_argument("--hour", type=int, default=18)
    ap.add_argument("--apply", action="store_true",
                    help="install: actually register the scheduled task")
    a = ap.parse_args(argv)
    cfg = _cfg(a)

    if a.cmd == "install":
        path = write_wrapper(cfg)
        print(install_line(cfg, hour=a.hour))
        if not a.apply:
            # Printing by default is deliberate. Registering a scheduled task is
            # a side effect on the machine, and on Windows it needs elevation
            # the operator may not have granted this shell. Writing the wrapper
            # is not a side effect worth asking about; scheduling it is.
            print("Then check it took effect with:  python -m tools.daily status")
            return 0
        cmd = scheduler_command(path, hour=a.hour)
        print(f"--apply: running\n  {cmd}\n")
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            print(f"\n  the scheduler refused (exit {rc}). On Windows this is "
                  f"almost always\n  a shell without Administrator rights — "
                  f"reopen it as Administrator\n  and run the line above.")
            return 1
        verify = (f'schtasks /query /tn "{TASK_NAME}"'
                  if platform.system() == "Windows" else "crontab -l")
        print(f"  registered. Verify with:  {verify}")
        return 0

    if a.cmd == "status":
        print(f"Clocks as of {_today()}\n")
        print(status(cfg))
        return 0

    print(f"daily run {_today()}")
    rows = run(cfg)
    if not rows:
        print("  nothing configured — pass --holdings and/or --ledger")
        return 1
    for r in rows:
        print(f"  {'ok ' if r['ok'] else '!! '}{r['step']:16}{r['detail']}")
    print()
    print(status(cfg))
    failed = [r for r in rows if not r["ok"]]
    if failed:
        print(f"\n  {len(failed)} step(s) failed. The others still ran — a broken "
              f"vendor must not cost you the whole day's capture.")
    return 1 if len(failed) == len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
