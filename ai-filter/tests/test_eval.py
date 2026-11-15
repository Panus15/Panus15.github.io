"""Tests for the evaluation tooling itself (a grader that grades wrong is worse than none).

Run: python3 tests/test_eval.py
"""
import json, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

import run_eval, new_case


def test_truth_grading_table():
    g = run_eval.classify_truth
    assert [g("false", s) for s in ("contradicted", "unverified", "verified", "absent")] == ["CAUGHT", "FLAGGED", "MISS", "MISS"]
    assert [g("true", s) for s in ("contradicted", "unverified", "verified", "absent")] == ["FALSE ALARM", "NOISE", "OK", "OK"]


def test_new_case_builds_a_runnable_corpus_and_the_runner_grades_it():
    with tempfile.TemporaryDirectory() as d:
        ans = os.path.join(d, "a.md"); corpus = os.path.join(d, "mine.json")
        open(ans, "w", encoding="utf-8").write("Friday, 15 March 2021 was warm. 2 + 2 = 4 is basic.")
        answers = iter(["Friday, 15 March 2021", "f", "2 + 2 = 4", "t", ""])
        assert new_case.main([ans, "--corpus", corpus, "--id", "t1"], input_fn=lambda p: next(answers)) == 0
        case = json.load(open(corpus, encoding="utf-8"))["cases"][0]
        assert case["expect"] == [{"find": "Friday, 15 March 2021", "truth": "false"}, {"find": "2 + 2 = 4", "truth": "true"}]
        r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "run_eval.py"), "--corpus", corpus, "--offline", "-v"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr          # wrong weekday CAUGHT, true sum OK
        assert "จับเป็น ❌ 1" in r.stdout


def test_a_missed_error_fails_the_run():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "m.json")
        json.dump({"cases": [{"id": "x", "needs_network": False, "text": "Sydney is the capital of Australia.",
                              "expect": [{"find": "Sydney is the capital", "truth": "false"}]}]}, open(corpus, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "run_eval.py"), "--corpus", corpus, "--offline"],
                           capture_output=True, text=True)
        assert r.returncode == 1 and "MISS" in r.stdout


def test_new_case_rejects_unknown_snippets_and_empty_input():
    with tempfile.TemporaryDirectory() as d:
        ans = os.path.join(d, "a.md"); open(ans, "w", encoding="utf-8").write("hello world")
        answers = iter(["not in the text", ""])
        assert new_case.main([ans, "--corpus", os.path.join(d, "c.json"), "--id", "e"], input_fn=lambda p: next(answers)) == 1
        assert not os.path.exists(os.path.join(d, "c.json"))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback; fails += 1; print("FAIL", name, repr(e)); traceback.print_exc()
    sys.exit(1 if fails else 0)
