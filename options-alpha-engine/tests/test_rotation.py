"""Tests for sector rotation (models/rotation.py, engine/rotation_backtest.py).

Two jobs. First the RRG math, checked against constructed cases where the right
answer is known by hand — a sector that tracks the benchmark exactly must land on
the origin, one that steadily outruns it must sit strong, and the quadrant labels
must fall where the axes say.

Then the part that matters: the harness must SEPARATE a world where sector
strength persists from one where it is fresh noise every day. A rotation study
that reports a signal in the noise world is worse than useless, because the chart
looks equally organised in both.

Run: python3 tests/test_rotation.py
"""

import math
import os
import re
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import rotation_backtest as rbt
from engine.rotation_backtest import (QuadrantPanel, RotationBacktestResult,
                                      preregistered_verdict, quadrant_panel,
                                      run_rotation_backtest)
from models.rotation import (IMPROVING, LAGGING, LEADING, SECTOR_ETFS, WEAKENING,
                             leaderboard, quadrant, quadrant_history,
                             relative_strength, rotation_map, rrg_series,
                             summary_table, transitions)

SYMS = list(SECTOR_ETFS)


def test_trimming_the_history_changes_nothing():
    """rotation_map trims to the rolling lookback for speed — it must be EXACT.

    Every statistic here is rolling, so bars older than the lookback cannot affect
    the answer. That turns an O(history) recompute per rebalance into a constant
    one, which is what makes the backtest runnable at all. If it were merely
    approximate, the speedup would be buying a different (wrong) answer.
    """
    px, bench = _world(n=900)
    full = rotation_map({s: p[:800] for s, p in px.items()}, bench[:800])
    trimmed = rotation_map({s: p[400:800] for s, p in px.items()}, bench[400:800])
    assert len(full) == len(trimmed) > 0
    for a, b in zip(sorted(full, key=lambda z: z.symbol),
                    sorted(trimmed, key=lambda z: z.symbol)):
        assert a.symbol == b.symbol
        assert a.rs_ratio == b.rs_ratio, (a.symbol, a.rs_ratio, b.rs_ratio)
        assert a.rs_momentum == b.rs_momentum
        assert a.quadrant == b.quadrant


def _world(n=1300, phi=0.0, seed=5, noise=0.006, alpha_sd=0.0010):
    """Benchmark + 11 sectors whose relative alpha is an AR(1).

    The innovation sd is set so the STATIONARY sd of alpha is ``alpha_sd`` whatever
    phi is — phi changes how long an edge persists, not how big it is. phi=0 means
    a sector's edge is re-drawn every day and no chart can predict it; phi=0.98
    means strength lasts about a month and momentum genuinely exists.
    """
    rng = random.Random(seed)
    eps = alpha_sd * math.sqrt(max(1e-9, 1.0 - phi * phi))
    bench = [100.0]
    for _ in range(n):
        bench.append(bench[-1] * math.exp(rng.gauss(0.0003, 0.009)))
    alpha = {s: rng.gauss(0, alpha_sd) for s in SYMS}
    px = {s: [100.0] for s in SYMS}
    for i in range(n):
        for s in SYMS:
            alpha[s] = phi * alpha[s] + rng.gauss(0, eps)
            px[s].append(px[s][-1] * math.exp(
                math.log(bench[i + 1] / bench[i]) + alpha[s] + rng.gauss(0, noise)))
    return px, bench


# --------------------------------------------------------------------------
# 1. the RRG math
# --------------------------------------------------------------------------

def test_quadrant_labels_follow_the_axes():
    assert quadrant(101.0, 101.0) == LEADING
    assert quadrant(101.0, 99.0) == WEAKENING
    assert quadrant(99.0, 99.0) == LAGGING
    assert quadrant(99.0, 101.0) == IMPROVING
    # the axes cross at exactly 100 and the boundary belongs to the strong side
    assert quadrant(100.0, 100.0) == LEADING
    assert quadrant(99.99, 100.0) == IMPROVING


def test_a_sector_that_tracks_the_benchmark_sits_at_the_origin():
    """Zero dispersion has no z-score; it must map to 100, not to an infinity."""
    bench = [100.0 * (1.01 ** i) for i in range(200)]
    same = list(bench)
    rs = relative_strength(same, bench)
    assert all(abs(x - 100.0) < 1e-9 for x in rs)
    ratio, mom = rrg_series(same, bench, window=63)
    assert ratio[-1] == 100.0
    assert mom[-1] == 100.0
    assert quadrant(ratio[-1], mom[-1]) == LEADING       # on the boundary


def test_steady_outperformance_reads_strong():
    bench = [100.0 * (1.0002 ** i) for i in range(400)]
    strong = [100.0 * (1.0012 ** i) for i in range(400)]
    weak = [100.0 * (0.9992 ** i) for i in range(400)]
    r_s, _ = rrg_series(strong, bench, window=63)
    r_w, _ = rrg_series(weak, bench, window=63)
    assert r_s[-1] > 100.0 > r_w[-1], (r_s[-1], r_w[-1])


def test_no_point_is_emitted_before_the_windows_fill():
    bench = [100.0 + i for i in range(300)]
    px = [100.0 + 1.5 * i for i in range(300)]
    ratio, mom = rrg_series(px, bench, window=63, mom_lag=5)
    assert ratio[:62] == [None] * 62, "RS-Ratio must not exist before its window"
    assert ratio[62] is not None
    assert mom[:62] == [None] * 62
    # too little history at all -> nothing, rather than a point built on nothing
    assert rrg_series(px[:20], bench[:20], window=63) == ([], [])


def test_rotation_map_drops_symbols_without_enough_history():
    px, bench = _world(n=300)
    px["SHORTY"] = [100.0] * 10
    pts = rotation_map(px, bench)
    assert "SHORTY" not in {p.symbol for p in pts}
    assert len(pts) == len(SYMS)
    assert all(p.quadrant in (LEADING, WEAKENING, LAGGING, IMPROVING) for p in pts)
    # ranked strongest-first, and the tail is real history not a placeholder
    assert all(pts[i].rs_ratio >= pts[i + 1].rs_ratio for i in range(len(pts) - 1))
    assert all(len(p.tail) > 1 for p in pts)
    assert "z-scores" in summary_table(pts)


def test_leaderboard_axes_are_selectable_so_the_two_axis_claim_is_testable():
    px, bench = _world(n=400)
    pts = rotation_map(px, bench)
    by_rs = [p.symbol for p in leaderboard(pts, rank_by="rs")]
    by_mom = [p.symbol for p in leaderboard(pts, rank_by="momentum")]
    both = [p.symbol for p in leaderboard(pts, rank_by="both")]
    assert by_rs[0] == max(pts, key=lambda p: p.rs_ratio).symbol
    assert by_mom[0] == max(pts, key=lambda p: p.rs_momentum).symbol
    assert by_rs != by_mom, "the two axes must not be the same measurement"
    assert len(both) == len(pts)


def test_quadrant_history_and_clockwise_transitions():
    px, bench = _world(n=600)
    sym = SYMS[0]
    hist = quadrant_history(px[sym], bench, bars=200)
    assert hist and all(q in (LEADING, WEAKENING, LAGGING, IMPROVING)
                        for _, q in hist)
    assert [i for i, _ in hist] == sorted(i for i, _ in hist)

    # a hand-built clockwise loop is all clockwise; reversing it is all not
    loop = [(0, IMPROVING), (1, LEADING), (2, WEAKENING), (3, LAGGING), (4, IMPROVING)]
    assert all(cw for _, _, _, cw in transitions(loop))
    assert not any(cw for _, _, _, cw in transitions(list(reversed(loop))))
    # repeats produce no transition at all
    assert transitions([(0, LEADING), (1, LEADING), (2, LEADING)]) == []


# --------------------------------------------------------------------------
# 2. does it predict anything? — the harness must tell the two worlds apart
# --------------------------------------------------------------------------

def test_the_panel_finds_nothing_when_there_is_nothing():
    px, bench = _world(phi=0.0)
    panel = quadrant_panel(px, bench)
    assert panel.n_dates > 50
    assert min(panel.stats[q]["n"] for q in panel.stats) > 100
    assert "NO SIGNIFICANT EFFECT" in panel.verdict(), panel.summary()


def test_the_panel_finds_the_effect_when_strength_really_persists():
    px, bench = _world(phi=0.98)
    panel = quadrant_panel(px, bench)
    assert "LEADING BEATS LAGGING" in panel.verdict(), panel.summary()
    assert panel.stats[LEADING]["mean"] > 0 > panel.stats[LAGGING]["mean"]


def test_the_long_short_book_separates_the_two_worlds():
    """Separation is judged on the permutation p, not on the sign of the total.

    The sign was always a proxy: in the null world the book's total is whatever
    costs and one sample happen to make it, so the test passed or failed on the
    cost assumption rather than on whether the harness can tell signal from
    noise. The p-value asks the question directly.

    Judged on MEDIANS over several worlds, not on one of each. A calibrated test
    fires on about one null world in ten by design, so a single null world
    landing under the threshold is the test working, not failing — asserting on
    one would make this flaky for the very reason the harness is correct.
    """
    def med(phi):
        # nine worlds, so the median is an actual observation and one unlucky
        # draw cannot decide the test; five was not enough and gave 0.099
        ps = sorted(run_rotation_backtest(*_world(n=1300, phi=phi, seed=s),
                                          placebo_draws=100).permutation_p()
                    for s in range(9))
        return ps[len(ps) // 2]

    real = run_rotation_backtest(*_world(phi=0.98))
    assert real.n_rebalances > 50
    assert med(0.98) <= 0.10 < med(0.0), (med(0.98), med(0.0))
    assert "worth a longer sample" in real.verdict(), real.verdict()


def test_costs_are_charged_every_rebalance():
    px, bench = _world(phi=0.98)
    free = run_rotation_backtest(px, bench, cost_bps=0.0)
    paid = run_rotation_backtest(px, bench, cost_bps=25.0)
    assert paid.metrics.total_return < free.metrics.total_return
    drag = free.metrics.total_return - paid.metrics.total_return
    assert drag > 0.10, f"25bp a leg over {paid.n_rebalances} rebalances should bite: {drag:.1%}"
    assert paid.cost_per_rebalance > free.cost_per_rebalance


def test_the_map_is_built_strictly_from_the_past():
    """The forward return is scored on data after t; the MAP must never see it."""
    px, bench = _world(n=900, phi=0.98)
    seen = {"calls": 0}
    real_map = rbt.rotation_map

    def spy(prices_by_symbol, benchmark, **kw):
        seen["calls"] += 1
        n = len(benchmark)
        for s, p in prices_by_symbol.items():
            assert len(p) == n, f"{s}: {len(p)} bars vs benchmark {n}"
            assert p == px[s][:n], f"{s} was handed prices it should not have"
        assert benchmark == bench[:n]
        return real_map(prices_by_symbol, benchmark, **kw)

    rbt.rotation_map = spy
    try:
        res = run_rotation_backtest(px, bench)
    finally:
        rbt.rotation_map = real_map
    assert seen["calls"] == res.n_rebalances > 20


def test_the_control_fires_at_its_nominal_rate_on_null_worlds():
    """The control must be CALIBRATED, not merely present.

    This is the defect that made the rewrite necessary. Comparing to ONE shuffled
    book, the placebo's own total swung from -16.6% to +7.9% purely on its seed,
    and on a fixture world built with no signal at all it declared an edge in 6
    of 20 seeds. A control that fires on a third of null worlds cannot support
    the sentence "the book beat the placebo".

    A permutation p over a distribution of shuffles should fire at its nominal
    rate: about 1 world in 10 at p <= 0.10. Ten worlds is a small check, so the
    bound is generous — it is here to catch a control that is broken, not to
    measure it to two decimals.
    """
    fired = 0
    for seed in range(10):
        r = run_rotation_backtest(*_world(n=1300, phi=0.0, seed=seed), cost_bps=0.0)
        if r.permutation_p() <= 0.10 and r.metrics.total_return > 0:
            fired += 1
    assert fired <= 3, (f"{fired}/10 null worlds declared an edge at p<=0.10; "
                        f"the control is not calibrated")


def test_the_permutation_p_is_invariant_to_the_fee():
    """The sharpest check that the control is MATCHED.

    Costs are identical for the book and every shuffle, so they cancel in the
    comparison and the p-value must not move when the fee changes. If a shuffle
    paid its own turnover instead, the ranked book — which is stickier, 66% churn
    against 77% — would get cheaper than its own control as the fee rose, and the
    p-value would drift down for a reason that is not prediction. Measured that
    way, the null world's median p fell to 0.079.
    """
    w = _world(n=1300, phi=0.0, seed=3)
    ps = [run_rotation_backtest(*w, cost_bps=bps, placebo_draws=200).permutation_p()
          for bps in (0.0, 10.0, 50.0)]
    assert len(set(ps)) == 1, f"p moved with the fee: {ps} — the control is unmatched"


def test_costs_are_charged_on_the_legs_that_actually_change():
    """Billing a full round trip when nothing moved is a fee for a trade nobody
    placed. What is charged must equal the fee times the measured churn."""
    w = _world(n=1300, phi=0.9, seed=2)
    free = run_rotation_backtest(*w, cost_bps=0.0)
    paid = run_rotation_backtest(*w, cost_bps=10.0)
    assert 0.0 < paid.mean_turnover < 1.0, paid.mean_turnover
    charged = free.metrics.total_return - paid.metrics.total_return
    expect = paid.cost_per_rebalance * paid.mean_turnover * paid.n_rebalances
    assert abs(charged - expect) < 1e-9, (charged, expect)
    flat = paid.cost_per_rebalance * paid.n_rebalances
    assert charged < flat * 0.95, "turnover made no difference to the bill"


def test_no_number_of_shuffles_can_prove_impossibility():
    """p must never be 0. A finite null cannot rule anything out, and a printed
    0.0% would read as certainty the resampling cannot deliver."""
    r = RotationBacktestResult(metrics=_Curve(9.99), n_rebalances=91,
                               placebo=_Curve(0.0), placebo_totals=[-1.0] * 200)
    p = r.permutation_p()
    assert p == 1.0 / 201.0, p
    assert p > 0.0


def test_a_shuffle_that_ties_counts_against_the_finding():
    """`>=`, not `>`. A shuffle that merely MATCHED the book is evidence that
    chance reaches this result, so it belongs in the numerator. Counting only
    strict betters is the convention that flatters, and with a discrete return
    grid — or a shuffle that happens to pick the same names — ties are not rare
    enough to wave away."""
    tied = RotationBacktestResult(metrics=_Curve(0.25), n_rebalances=91,
                                  placebo=_Curve(0.0),
                                  placebo_totals=[0.25] * 30 + [-1.0] * 170)
    assert tied.permutation_p() == 31.0 / 201.0, tied.permutation_p()
    assert tied.permutation_p() > 0.10, "30 ties were not counted against it"
    assert "INDISTINGUISHABLE" in tied.verdict(), tied.verdict()


def test_the_verdict_refuses_a_book_sitting_in_its_null():
    """A positive total that a fifth of shuffles match is not a finding, and the
    sentence the user reads has to say so."""
    mid = RotationBacktestResult(metrics=_Curve(0.40), n_rebalances=91,
                                 placebo=_Curve(0.01),
                                 placebo_totals=[0.9] * 40 + [-0.9] * 160)
    v = mid.verdict()
    assert "INDISTINGUISHABLE FROM CHANCE" in v, v
    assert "worth a longer sample" not in v, v


def test_the_control_still_finds_a_real_signal():
    """Calibration is worthless if it is bought by never firing at all."""
    found = sum(run_rotation_backtest(*_world(n=1300, phi=0.98, seed=s),
                                      cost_bps=0.0).permutation_p() <= 0.10
                for s in range(5))
    assert found >= 4, f"only {found}/5 signal worlds were detected"


def test_one_shuffle_is_reported_as_one_shuffle():
    """The degenerate path must not quietly speak as if it had a null."""
    class _M:
        def __init__(self, tr):
            self.total_return, self.sharpe, self.max_drawdown = tr, 1.0, -0.1

        def summary(self):
            return ""

    one = RotationBacktestResult(metrics=_M(0.40), n_rebalances=60, placebo=_M(0.02))
    v = one.verdict()
    assert one.permutation_p() is None
    assert "ONE shuffle" in v, v
    assert "% of" not in v, f"it quoted a null distribution it does not have: {v}"


def test_a_result_the_placebo_matches_is_rejected():
    """If shuffling the labels earns the same, the harness measured something else."""
    class _M:
        def __init__(self, tr):
            self.total_return, self.sharpe, self.max_drawdown = tr, 1.0, -0.1

        def summary(self):
            return ""

    good = RotationBacktestResult(metrics=_M(0.40), n_rebalances=60,
                                  placebo=_M(0.02),
                                  placebo_totals=[0.02] * 100)
    assert "worth a longer sample" in good.verdict(), good.verdict()
    fake = RotationBacktestResult(metrics=_M(0.40), n_rebalances=60,
                                  placebo=_M(0.35))
    assert "REJECTED BY THE PLACEBO" in fake.verdict(), fake.verdict()
    thin = RotationBacktestResult(metrics=_M(0.40), n_rebalances=5, placebo=_M(0.0))
    assert "NOT ENOUGH DATA" in thin.verdict()


def test_the_momentum_axis_carries_no_edge_of_its_own():
    """The RRG's selling point is its SECOND axis. Measured, it does not pay.

    Checked across independent worlds rather than one, because a single sample
    said the opposite of the next one: on n=2500 ranking by strength alone beat
    the diagonal by 12 points of return, and on n=1300 the diagonal won. Neither
    was a finding. Over 15 worlds the picture that survives is narrower and worth
    stating exactly:

        rs alone        Sharpe +9.52  [+6.34, +12.79]
        both axes       Sharpe +7.60  [+3.86, +12.19]
        momentum alone  Sharpe -1.17  [-3.24,  +1.22]   <- spans zero
        rs minus both   Sharpe +1.93  [-0.37,  +3.98]   <- spans zero

    So: the momentum axis alone has NO detectable edge, and adding it to relative
    strength does not clearly help (strength alone won 13 of 15 worlds, but the
    interval on the difference includes zero, so that is suggestive, not shown).
    The fixture's alpha is AR(1) — it persists but never accelerates — and an
    accelerating world is exactly where a momentum axis should earn its place.
    """
    sharpes = {"both": [], "rs": [], "momentum": []}
    for seed in (5, 6, 7):
        px, bench = _world(phi=0.98, seed=seed)
        for mode in sharpes:
            sharpes[mode].append(
                run_rotation_backtest(px, bench, rank_by=mode).metrics.sharpe)

    mean = {k: sum(v) / len(v) for k, v in sharpes.items()}
    assert mean["rs"] > 2.0, mean          # relative strength does work here
    assert mean["both"] > 2.0, mean
    assert mean["momentum"] < mean["rs"] / 2, (
        f"the momentum axis should not rival relative strength: {mean}")
    assert min(sharpes["momentum"]) < 2.0, (
        "momentum alone must not look like a reliable edge on an AR(1) fixture")



# --------------------------------------------------------------------------
# The pre-registered criterion, which must not bend around a result
# --------------------------------------------------------------------------

def _panel(lead_mean, lag_mean, sd=0.05, n=250, n_lead=None, n_lag=None):
    """A panel with a chosen Leading-Lagging spread and dispersion."""
    st = {q: {"n": n, "mean": 0.0, "sd": sd, "t": 0.0}
          for q in ("Leading", "Weakening", "Lagging", "Improving")}
    st["Leading"].update(mean=lead_mean, n=n_lead or n)
    st["Lagging"].update(mean=lag_mean, n=n_lag or n)
    return QuadrantPanel(stats=st, n_dates=n // 11)


# The first real-data run, as reported by the dashboard: 91 rebalance dates,
# Leading n=228 mean +0.04%, Lagging n=274 mean -0.21%, spread +0.25%, t=+0.71.
# sd is solved from those so the fixture reproduces the published t rather than
# an invented one.
def _real_panel():
    return _panel(0.0004, -0.0021, sd=0.0393, n_lead=228, n_lag=274)


class _Curve:
    def __init__(self, total):
        self.total_return = total
        self.sharpe = 0.0


def _book(real, fake, n=91, p=0.01, draws=200):
    """A book whose permutation p can be dialled by stacking the null totals."""
    beat = max(0, int(round(p * (1 + draws))) - 1)
    totals = [real + 1.0] * beat + [real - 1.0] * (draws - beat)
    return RotationBacktestResult(metrics=_Curve(real), n_rebalances=n,
                                  placebo=_Curve(fake), placebo_totals=totals)


def test_the_real_result_is_recorded_as_a_failure():
    """The first real-data run, verbatim: t=+0.71 and a book that lost 27%
    while the shuffled placebo lost 18.4%. Both criteria fail. The book losing
    LESS badly than a coin flip is not a pass, and this is the exact shape of
    result that invites the phrase "beat the placebo" to be reinterpreted."""
    v = preregistered_verdict(_real_panel(), _book(-0.270, -0.184))
    assert not v["passed"] and "ANSWERED NO" in v["headline"]
    assert not v["panel_passed"] and not v["book_passed"], v


def test_a_losing_book_never_passes_however_bad_the_placebo():
    for fake in (-0.50, -0.99, 0.0):
        v = preregistered_verdict(_panel(0.02, -0.02), _book(-0.05, fake))
        assert not v["book_passed"], f"a losing book passed against placebo {fake}"


def test_both_halves_are_required():
    strong_panel, weak_panel = _panel(0.02, -0.02), _panel(0.0001, 0.0)
    good_book, bad_book = _book(0.40, 0.05), _book(0.40, 0.30)
    assert preregistered_verdict(strong_panel, good_book)["passed"]
    assert not preregistered_verdict(weak_panel, good_book)["passed"]
    assert not preregistered_verdict(strong_panel, bad_book)["passed"]


def test_a_book_that_sits_inside_its_null_never_passes():
    """Amendment 6.1. Beating one shuffle by 8x means nothing if a fifth of the
    shuffles do the same — which is the case a single-draw placebo cannot see."""
    strong = _panel(0.02, -0.02)
    assert preregistered_verdict(strong, _book(0.40, 0.05, p=0.01))["passed"]
    v = preregistered_verdict(strong, _book(0.40, 0.05, p=0.20))
    assert not v["book_passed"], v["why_book"]
    assert "permutation p=19" in v["why_book"], v["why_book"]


def test_a_book_with_no_null_distribution_cannot_pass():
    from engine.rotation_backtest import RotationBacktestResult as R
    naked = R(metrics=_Curve(0.40), n_rebalances=91, placebo=_Curve(0.05))
    v = preregistered_verdict(_panel(0.02, -0.02), naked)
    assert not v["book_passed"] and "chance is unmeasured" in v["why_book"]


def test_the_amendment_can_only_make_passing_harder():
    """A stricter rule added after a result is safe; a looser one is not.

    Every book that fails the registered 2x rule must still fail with the
    permutation requirement bolted on, whatever its p. If any combination could
    pass under the amendment but not under the original, the amendment would be
    a rescue rather than a tightening.
    """
    panel = _panel(0.02, -0.02)
    for real, fake in ((0.40, 0.30), (-0.05, -0.50), (0.10, 0.09), (0.0, -0.9)):
        for p in (0.001, 0.05, 0.10, 0.50):
            b = _book(real, fake, p=p)
            registered = real > 0 and fake < real / 2.0
            got = preregistered_verdict(panel, b)["book_passed"]
            assert got <= registered, (
                f"real={real} fake={fake} p={p}: passed under the amendment "
                f"but not under the registered rule")


def test_an_inverted_panel_is_not_a_pass():
    """|t| >= 2 the WRONG way round means the chart is backwards, not working."""
    v = preregistered_verdict(_panel(-0.02, 0.02), _book(0.40, 0.05))
    assert not v["panel_passed"], v["why_panel"]


def test_a_thin_panel_refuses_rather_than_guesses():
    thin = _panel(0.02, -0.02, n=5)
    v = preregistered_verdict(thin, _book(0.40, 0.05))
    assert not v["panel_passed"] and "too few" in v["why_panel"]


def test_the_thresholds_are_the_registered_ones_and_cannot_drift():
    """The whole point of a pre-registration is that this constant does not move.

    Relaxing |t| from 2 to 1 is the cheapest way to turn a failure into a
    finding, it looks like a reasonable judgement call in a diff, and no test
    about panels or books would notice. So the constants are checked against the
    document itself, and a panel that clears 1 but not 2 must still fail.
    """
    doc = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "PREREGISTRATION.md"), encoding="utf-8").read()
    # parsed, not string-formatted: 2.0 vs "2" is a formatting difference, and a
    # test that fails on formatting gets weakened rather than fixed
    registered = re.findall(r"\|\s*`t_thresh`\s*\|\s*([0-9.]+)\s*\|", doc)
    assert registered, "PREREGISTRATION.md has no t_thresh row"
    assert float(registered[0]) == rbt.PREREG_T_THRESH, \
        f"code has |t| >= {rbt.PREREG_T_THRESH}, §2.1 registered {registered[0]}"
    assert "more than 2×" in doc and rbt.PREREG_PLACEBO_RATIO == 2.0

    # spread solved to land at t ~ 1.4: significant at the 10-20% level, and
    # therefore exactly the result that invites the threshold to be reconsidered
    mid = _panel(0.0050, 0.0, sd=0.0393, n_lead=228, n_lag=274)
    _, t = mid.spread_t()
    assert 1.0 < abs(t) < 2.0, f"fixture t={t:+.2f} does not sit between 1 and 2"
    assert not preregistered_verdict(mid, _book(0.40, 0.05))["panel_passed"], \
        f"a panel at t={t:+.2f} passed a criterion registered at |t| >= 2"


def test_the_reasons_carry_the_numbers_that_decided_it():
    v = preregistered_verdict(_real_panel(), _book(-0.270, -0.184))
    assert "t=+0.7" in v["why_panel"], v["why_panel"]
    assert "-27.0%" in v["why_book"] and "-18.4%" in v["why_book"], v["why_book"]


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
