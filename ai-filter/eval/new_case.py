#!/usr/bin/env python3
"""Turn one AI answer into an evaluation case by answering questions. No JSON editing.

    python3 eval/new_case.py answer.md --corpus mine.json --id my-1

For each statement you care about you give (1) a short snippet that appears in the answer and
(2) whether that statement is TRUE or FALSE, as YOU verified it. You do not need to know what the
filter will say: run_eval.py grades it (CAUGHT / FLAGGED / MISS / NOISE / FALSE ALARM).
"""
import argparse, json, os, sys


def ask(prompt, input_fn=input):
    try:
        return input_fn(prompt).strip()
    except EOFError:
        return ""


def build_case(case_id, text, items):
    needs_net = any(k in text.lower() for k in ("http", "doi", "pip install", "npm "))
    return {"id": case_id, "needs_network": needs_net, "text": text,
            "expect": [{"find": f, "truth": t} for f, t in items]}


def main(argv=None, input_fn=input):
    ap = argparse.ArgumentParser()
    ap.add_argument("answer")
    ap.add_argument("--corpus", default="mine.json")
    ap.add_argument("--id", required=True)
    a = ap.parse_args(argv)
    text = open(a.answer, encoding="utf-8").read()
    print(text, "\n" + "-" * 60)
    items = []
    while True:
        snippet = ask("\nพิมพ์ข้อความสั้นๆ ที่อยู่ในคำตอบข้างบน เพื่อระบุข้อที่จะเฉลย (เว้นว่าง = เสร็จ): ", input_fn)
        if not snippet:
            break
        n = text.count(snippet)
        if n == 0:
            print("  ไม่พบข้อความนี้ในคำตอบ ลองพิมพ์ใหม่ (ต้องตรงทุกตัวอักษร)")
            continue
        if n > 1:
            print(f"  พบ {n} ที่ จะใช้ทุกที่ที่ตรง ถ้าไม่ต้องการให้พิมพ์ให้ยาวขึ้น")
        t = ""
        while t not in ("t", "f"):
            t = ask("  ข้อนี้ จริง(t) หรือ ผิด(f) ตามที่คุณตรวจเองแล้ว? [t/f]: ", input_fn).lower()
            if t == "":
                print("  ข้ามข้อนี้")
                break
        if t in ("t", "f"):
            items.append((snippet, "true" if t == "t" else "false"))
    if not items:
        print("ไม่มีข้อที่เฉลย ไม่เขียนอะไร")
        return 1
    corpus = {"note": "mine", "cases": []}
    if os.path.exists(a.corpus):
        corpus = json.load(open(a.corpus, encoding="utf-8"))
    corpus["cases"] = [c for c in corpus["cases"] if c["id"] != a.id] + [build_case(a.id, text, items)]
    json.dump(corpus, open(a.corpus, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    runner = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_eval.py"))
    print(f"\nบันทึกแล้ว: {a.corpus} ({len(corpus['cases'])} เคส) รันด้วย: python3 {runner} --corpus {a.corpus} -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
