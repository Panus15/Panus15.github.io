"""Tests for the per-symbol export merger (tools/merge_csv.py).

The failure this module exists to prevent is silent and fatal: lining twelve
exports up by ROW NUMBER instead of by DATE pairs one sector's Tuesday with
another's Wednesday, and every relative-strength number computed afterwards is
wrong in a way nothing downstream can detect. So the join is tested against files
with deliberately mismatched holidays and listing dates.

Run: python3 tests/test_merge_csv.py
"""

import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.merge_csv import (merge, read_series, symbol_from_filename,
                             write_wide, _norm_date)


def _write(d, name, rows, header=("time", "open", "high", "low", "close", "Volume")):
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


def test_symbols_come_out_of_the_filenames():
    cases = {
        "AMEX_XLK, 1D.csv": "XLK",
        "BATS_SPY, 1D.csv": "SPY",
        "NASDAQ_QQQ, 1D.csv": "QQQ",
        "XLRE.csv": "XLRE",
        "spy_daily.csv": "SPY",
        "/some/path/AMEX_XLU, 1D.csv": "XLU",
    }
    for name, want in cases.items():
        assert symbol_from_filename(name) == want, (name, symbol_from_filename(name))


def test_dates_are_normalised_from_every_common_format():
    assert _norm_date("2024-01-02T00:00:00-05:00") == "2024-01-02"
    assert _norm_date("2024-01-02") == "2024-01-02"
    assert _norm_date("01/02/2024") == "2024-01-02"
    assert _norm_date("1704153600") == "2024-01-02"      # unix seconds
    assert _norm_date("1704153600000") == "2024-01-02"   # unix millis
    assert _norm_date("") is None
    assert _norm_date("not a date") is None


def test_rows_are_joined_on_the_DATE_not_the_row_number():
    """The bug that would poison every downstream number, invisibly."""
    d = tempfile.mkdtemp()
    try:
        # SPY trades on the 3rd; XLK does not (a deliberate hole)
        _write(d, "BATS_SPY, 1D.csv", [
            ["2024-01-02T00:00:00-05:00", 1, 1, 1, 100.0, 1],
            ["2024-01-03T00:00:00-05:00", 1, 1, 1, 101.0, 1],
            ["2024-01-04T00:00:00-05:00", 1, 1, 1, 102.0, 1]])
        _write(d, "AMEX_XLK, 1D.csv", [
            ["2024-01-02T00:00:00-05:00", 1, 1, 1, 200.0, 1],
            ["2024-01-04T00:00:00-05:00", 1, 1, 1, 202.0, 1]])
        dates, cols, _ = merge([os.path.join(d, f) for f in os.listdir(d)])

        assert dates == ["2024-01-02", "2024-01-04"], dates
        # the 4th must pair 102 with 202 — a row-number merge would pair 101/202
        assert cols["SPY"] == [100.0, 102.0], cols["SPY"]
        assert cols["XLK"] == [200.0, 202.0], cols["XLK"]
    finally:
        shutil.rmtree(d)


def test_a_late_listing_binds_the_whole_join_and_says_so():
    """XLRE only lists in 2015; the report must name what truncated the sample."""
    d = tempfile.mkdtemp()
    try:
        _write(d, "SPY.csv", [[f"2024-01-{i:02d}", 1, 1, 1, 100 + i, 1]
                              for i in range(1, 21)])
        _write(d, "XLRE.csv", [[f"2024-01-{i:02d}", 1, 1, 1, 50 + i, 1]
                               for i in range(15, 21)])
        dates, cols, report = merge([os.path.join(d, f) for f in os.listdir(d)])
        assert len(dates) == 6, dates
        assert "XLRE" in report and "dropped by the join" in report
        assert "20" in report and "6 rows" in report
    finally:
        shutil.rmtree(d)


def test_no_overlap_is_an_error_not_an_empty_file():
    d = tempfile.mkdtemp()
    try:
        _write(d, "SPY.csv", [["2024-01-02", 1, 1, 1, 100.0, 1]])
        _write(d, "XLK.csv", [["2025-06-02", 1, 1, 1, 200.0, 1]])
        try:
            merge([os.path.join(d, f) for f in os.listdir(d)])
        except ValueError as e:
            assert "every file" in str(e) or "timeframe" in str(e)
            return
        raise AssertionError("disjoint files should raise, not write an empty CSV")
    finally:
        shutil.rmtree(d)


def test_a_required_symbol_that_is_absent_is_refused():
    d = tempfile.mkdtemp()
    try:
        _write(d, "XLK.csv", [["2024-01-02", 1, 1, 1, 200.0, 1]])
        try:
            merge([os.path.join(d, "XLK.csv")], require=["SPY"])
        except ValueError as e:
            assert "SPY" in str(e)
            return
        raise AssertionError("a missing benchmark must be refused up front")
    finally:
        shutil.rmtree(d)


def test_alternative_column_names_and_dirty_numbers():
    d = tempfile.mkdtemp()
    try:
        p = _write(d, "SPY.csv",
                   [["01/02/2024", "$1,234.50"], ["01/03/2024", "1,240.00"]],
                   header=("Date", "Close/Last"))
        s = read_series(p)
        assert s == {"2024-01-02": 1234.50, "2024-01-03": 1240.00}, s
    finally:
        shutil.rmtree(d)


def test_a_file_without_a_close_column_is_refused():
    d = tempfile.mkdtemp()
    try:
        p = _write(d, "SPY.csv", [["2024-01-02", 1]], header=("time", "volume"))
        try:
            read_series(p)
        except ValueError as e:
            assert "close" in str(e).lower()
            return
        raise AssertionError("a file with no close should raise")
    finally:
        shutil.rmtree(d)


def test_two_files_for_the_same_symbol_are_refused():
    d = tempfile.mkdtemp()
    try:
        a = _write(d, "AMEX_XLK, 1D.csv", [["2024-01-02", 1, 1, 1, 1.0, 1]])
        os.makedirs(os.path.join(d, "sub"))
        b = _write(os.path.join(d, "sub"), "XLK.csv", [["2024-01-02", 1, 1, 1, 2.0, 1]])
        try:
            merge([a, b])
        except ValueError as e:
            assert "XLK" in str(e)
            return
        raise AssertionError("a duplicate symbol should be refused, not silently won")
    finally:
        shutil.rmtree(d)


def test_the_output_is_exactly_what_the_rotation_tools_read():
    """End to end: merged file -> rotation_dashboard's own loader -> a chart."""
    from tools.rotation_dashboard import build_payload, load_csv
    from models.rotation import SECTOR_ETFS

    d = tempfile.mkdtemp()
    try:
        import math
        import random
        rng = random.Random(3)
        n = 400
        for i, sym in enumerate(list(SECTOR_ETFS) + ["SPY"]):
            px, rows = 100.0, []
            for k in range(n):
                px *= math.exp(rng.gauss(0.0003, 0.010))
                day = 1 + k
                rows.append([f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}",
                             px, px, px, round(px, 4), 1])
            _write(d, f"AMEX_{sym}, 1D.csv", rows)

        out = os.path.join(d, "sectors.csv")
        dates, cols, _ = merge([os.path.join(d, f) for f in os.listdir(d)
                                if f.endswith(".csv")], require=["SPY"])
        write_wide(out, dates, cols)

        series, bench, read_dates = load_csv(out)
        assert len(series) == len(SECTOR_ETFS) and len(bench) == len(dates)
        assert read_dates[0] == dates[0]
        payload = build_payload(series, bench, read_dates, window=63, mom_lag=5,
                                tail=12, horizon=21, run_test=False)
        assert len(payload["points"]) == len(SECTOR_ETFS)
        assert all(p["quadrant"] for p in payload["points"])
    finally:
        shutil.rmtree(d)


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
