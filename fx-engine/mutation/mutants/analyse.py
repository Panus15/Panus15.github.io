"""Ways the command line could mislead: running on broken data,
accepting unmeasured costs, advising from a sample too small to measure."""

TARGET = "tools/analyse.py"
TESTS = "tests/test_analyse.py"

MUTANTS = [
 ("unusable data not stopped",   "    if not series.usable:", "    if False:"),
 ("stops on usable data too",    "    if not series.usable:", "    if True:"),
 ("stop returns success",        "        return 2", "        return 0"),
 ("costs always confirmed",      "                  costs_confirmed=costs_measured, evidence=ev,", "                  costs_confirmed=True, evidence=ev,"),
 ("costs never confirmed",       "                  costs_confirmed=costs_measured, evidence=ev,", "                  costs_confirmed=False, evidence=ev,"),
 ("bar size ignored",            "        res = run(series.bars, fn(series.bars), costs, bars_per_night=bpn)", "        res = run(series.bars, fn(series.bars), costs)"),
 ("bar size fallback wrong",     "        bpn = 1.0", "        bpn = 24.0"),
 ("short history not flagged",   "    if series.n < 200:", "    if False:"),
 ("pattern filter ignored",      "        for p_ in (patterns or list(DETECTORS)):", "        for p_ in list(DETECTORS):"),
 ("search width hardcoded",      "    n_hyp = sum(pc for _, pc, _ in chosen.values())", "    n_hyp = 1"),
 ("declared counts ignored",     "    n_hyp = sum(pc for _, pc, _ in chosen.values())", "    n_hyp = len(chosen) * 4"),
 ("rule evidence dropped",       "        chosen[r.name] = (r.detect, r.parameter_count, r.evidence)", "        chosen[r.name] = (r.detect, r.parameter_count, None)"),
 ("rule never loaded",           "    for rp in (rule_paths or []):", "    for rp in []:"),
 ("closing message always same", "    if passed:", "    if False:"),
 ("closing message inverted",    "    if passed:", "    if not passed:"),
 ("side not passed through",     "    series = load_csv(path, pair, side=side, time_format=time_format)", "    series = load_csv(path, pair, side='mid', time_format=time_format)"),
 ("advice on a 1-trade sample",  "                  if TOO_FEW_TRADES not in v.codes\n                  and (NEGATIVE_NET_EXPECTANCY in v.codes", "                  if (NEGATIVE_NET_EXPECTANCY in v.codes"),
 ("levers on any refusal",       "    if not passed and arithmetic:", "    if not passed and refused:"),
 ("advice shown despite a pass", "    if not passed and arithmetic:", "    if arithmetic:"),
 ("data-constraint note dropped","    elif not passed and refused:", "    elif False:"),
 ("rate uses the raw count",     "        days = max(len(series.bars) / bpn, 1.0)", "        days = 365.0"),
 ("holdout ignored",             "    if holdout is not None:", "    if False:"),
 ("holdout fraction ignored",    "        out(holdout_evaluate(series.bars, costs, frac=holdout,", "        out(holdout_evaluate(series.bars, costs, frac=0.9,"),
 ("report never written",        "    if report_path:", "    if False:"),
 ("report ignores the costs",    "            fh.write(render_report(series, results, judged, costs, bpn, n_hyp))", "            fh.write(render_report(series, results, judged, CostModel(pair=pair), bpn, n_hyp))"),
 ("report gets no results",      "            fh.write(render_report(series, results, judged, costs, bpn, n_hyp))", "            fh.write(render_report(series, {k: BacktestResult() for k in judged}, judged, costs, bpn, n_hyp))"),
 ("errors not caught",           "    except (OSError, ValueError) as e:", "    except (KeyboardInterrupt,) as e:"),
 ("error exit code is success",  "        return 1\n\n\nif __name__", "        return 0\n\n\nif __name__"),
]
