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


def install_line(cfg: dict, *, hour: int = 18) -> str:
    """The scheduler entry for this machine, ready to paste."""
    py = sys.executable or "python3"
    cmd = f"cd {ROOT} && {py} -m tools.daily run"
    for flag in ("holdings", "ledger", "prices", "events", "sources"):
        if cfg.get(flag):
            cmd += f" --{flag} {cfg[flag]}"
    cmd += f" --source {cfg['source']} --dte {cfg['dte']}"

    sysname = platform.system()
    if sysname == "Windows":
        return ("Windows — run in an Administrator PowerShell:\n\n"
                f'  schtasks /create /tn "options-alpha-daily" /sc daily '
                f'/st {hour:02d}:00 /tr "cmd /c {cmd}"\n')
    if sysname == "Darwin":
        return ("macOS — cron still works and is the least fuss:\n\n"
                f"  (crontab -l 2>/dev/null; echo '0 {hour} * * 1-5 {cmd} "
                f">> {os.path.join(ROOT, 'daily.log')} 2>&1') | crontab -\n\n"
                "  Note: a laptop asleep at that hour simply misses the day. If\n"
                "  that is likely, prefer a machine that stays awake.\n")
    return ("Linux — weekdays after the US close:\n\n"
            f"  (crontab -l 2>/dev/null; echo '0 {hour} * * 1-5 {cmd} "
            f">> {os.path.join(ROOT, 'daily.log')} 2>&1') | crontab -\n")


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
    a = ap.parse_args(argv)
    cfg = _cfg(a)

    if a.cmd == "install":
        print(install_line(cfg, hour=a.hour))
        print("Then check it took effect with:  python3 -m tools.daily status")
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
