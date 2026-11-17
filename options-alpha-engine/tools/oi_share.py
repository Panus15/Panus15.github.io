"""Answer the crowding premise in one afternoon instead of in eighteen months.

    python3 -m tools.oi_share --holdings holdings/QQQI/2026-09-11.csv \\
                              --underlying QQQ --source tradier --symbol QQQ

    python3 -m tools.oi_share --holdings book.csv --underlying QQQ \\
                              --chain-json chain.json          # offline replay

PREREGISTRATION.md §2.2 needs ~18 months of daily fund books that cannot be bought
retroactively. Before spending that, this checks the premise's NECESSARY condition:
that the fund holds a meaningful share of the open interest at its own strikes. The
decision rule and its bands are written down in models/oi_share.py and were fixed
before any data was seen.

It is a screen, not the study. Passing does not show an effect exists.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fund_flow import FundBook                       # noqa: E402
from models.oi_share import share_of_open_interest          # noqa: E402


def asof_from_path(path: str) -> str:
    """`tools/archive_holdings.py` writes `outdir/FUND/YYYY-MM-DD.csv`, so the date
    is the archiver's own convention rather than a guess. Returns "" if the name
    does not carry one — an empty as-of is reported, never silently replaced with
    today, because a book dated a week ago read as today shifts every line's days
    to expiry by seven and matches nothing."""
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--holdings", required=True,
                    help="one archived fund book (.csv or .json)")
    ap.add_argument("--underlying", required=True,
                    help="the ticker the fund writes options on, e.g. QQQ")
    ap.add_argument("--fund", default="", help="label for the report")
    ap.add_argument("--source", default="replay",
                    choices=["tradier", "replay"],
                    help="where the CHAIN comes from; tradier carries open interest")
    ap.add_argument("--symbol", default="", help="chain symbol (default --underlying)")
    ap.add_argument("--chain-json", dest="chain_json", default="",
                    help="replay: a --dump'ed chain")
    ap.add_argument("--max-expirations", dest="max_expirations", type=int, default=12,
                    help="how many expiries to pull; the fund's book can be spread "
                         "across several, and a short window looks like a mismatch")
    ap.add_argument("--asof", default="",
                    help="the book's as-of date (default: taken from the archived "
                         "filename, which tools/archive_holdings.py names by date)")
    a = ap.parse_args(argv)

    symbol = a.symbol or a.underlying
    asof = a.asof or asof_from_path(a.holdings)
    book = FundBook.from_file(a.holdings, fund=a.fund or os.path.basename(a.holdings),
                              asof=asof)
    # Said out loud because the match is on DAYS to expiry: an as-of a week wrong
    # shifts every line by seven days and matches nothing, which reads as "the fund
    # holds no listed options" rather than as "the date was wrong".
    if not getattr(book, "asof", ""):
        print("  !! this book carries no as-of date and none was supplied, so days "
              "to expiry\n     cannot be computed and nothing will match. Pass "
              "--asof YYYY-MM-DD.\n")
    else:
        print(f"  book as of {book.asof}"
              + ("" if a.asof else "  (from the filename)"))

    if a.source == "tradier":
        from engine.tradier import TradierAdapter
        ad = TradierAdapter(max_expirations=a.max_expirations)
        chain = ad.option_chain(symbol)
    else:
        if not a.chain_json:
            print("replay needs --chain-json (dump one with "
                  "`python3 -m tools.run_live tradier --symbol QQQ --dump c.json`)")
            return 1
        from engine.adapters import JsonFileAdapter
        chain = JsonFileAdapter(chain_json=a.chain_json).option_chain(symbol)

    rep = share_of_open_interest(book, chain, a.underlying)
    print(rep.summary())

    if not rep.known_rows:
        # Two different failures used to print the same sentence. "No strike
        # matched" and "strikes matched but carried no open interest" have
        # different causes and different fixes, and blaming the chain for a date
        # mismatch sends the operator to re-dump a chain that was fine.
        if not rep.rows:
            # the verdict above already names this case; add only what the model
            # cannot know, which is which funds are known to overwrite off-exchange
            print("\n  JEPI and JEPQ, for two, run their overwriting through OTC\n"
                  "  equity-linked notes and hold no listed options at all, so for\n"
                  "  those there is nothing to find a share OF.")
            return 1
        print("\n  The chain carried no open interest. In this codebase OI 0 means\n"
              "  UNKNOWN, not zero, so nothing is concluded. Tradier supplies it;\n"
              "  a replayed chain carries it only if it was dumped with it — which,\n"
              "  until engine/deribit.save_chain_json was fixed, it never was.")
        return 1
    if rep.unmatched and rep.unmatched >= len(rep.rows):
        print(f"\n  NOTE: {rep.unmatched} fund line(s) matched no listed quote, more\n"
              f"  than matched. That is itself a finding — JEPI and JEPQ, for one,\n"
              f"  run their overwriting through OTC equity-linked notes and hold no\n"
              f"  listed options at all, so there is nothing to find a share OF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
