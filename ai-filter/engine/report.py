"""Findings -> Markdown (for people) or JSON-ready dicts (for scripts).

Everything that originates outside this program -- the claim text, and any
evidence string, which can contain a paper title or a page title returned by a
remote API -- is treated as hostile when it is written into Markdown: it goes
inside a code span sized to out-fence any backticks it contains, control
characters are removed, and it is length-capped. Otherwise a crafted answer (or a
crafted remote title) could inject links, images or headings into a report that
a person then trusts.
"""

from __future__ import annotations

import re
from typing import Sequence

from engine.models import Finding, Status
from engine.score import Verdict, counts, risk, verdict

ICON = {Status.CONTRADICTED: "❌", Status.UNVERIFIED: "⚠️", Status.VERIFIED: "✅"}
ORDER = (Status.CONTRADICTED, Status.UNVERIFIED, Status.VERIFIED)
HEADLINE = {
    Verdict.DO_NOT_USE: "❌ ห้ามใช้ก่อนแก้ — พบข้อที่ขัดกับหลักฐาน",
    Verdict.REVIEW: "⚠️ ต้องตรวจเพิ่ม — มีบางข้อที่ยืนยันไม่ได้",
    Verdict.NO_CHECKABLE_CLAIMS: "ℹ️ ไม่พบสิ่งที่ตัวกรองนี้ตรวจได้ (ไม่ได้แปลว่าถูกต้อง)",
    Verdict.NO_PROBLEMS_FOUND: "✅ ไม่พบปัญหาในข้อที่ตรวจ (ไม่ได้แปลว่าถูกต้องทั้งหมด)",
}
RISK_TH = {"high": "สูง", "medium": "กลาง", "low": "ต่ำ", "none": "ไม่พบสัญญาณ"}
MAX_SHOWN = 200

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")


def _clean(s: str, limit: int = MAX_SHOWN) -> str:
    s = _CTRL.sub("", s.replace("\r", " ").replace("\n", " ").replace("\t", " "))
    s = re.sub(r" {2,}", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def code_span(s: str) -> str:
    s = _clean(s)
    longest = max((len(m) for m in re.findall(r"`+", s)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def to_dict(findings: Sequence[Finding]) -> dict:
    return {
        "verdict": verdict(findings).value,
        "counts": {s.value: n for s, n in counts(findings).items()},
        "risk": {"score": risk(findings)["score"], "level": risk(findings)["level"]},
        "findings": [{
            "claim_id": f.claim.id, "kind": f.claim.kind.value, "text": f.claim.text,
            "start": f.claim.start, "end": f.claim.end, "status": f.status.value,
            "module": f.module, "reason": f.reason, "evidence": list(f.evidence),
        } for f in findings],
    }


def render(findings: Sequence[Finding]) -> str:
    v, c = verdict(findings), counts(findings)
    r = risk(findings)
    out = ["# รายงานตรวจคำตอบ AI", "", f"**{HEADLINE[v]}**", "",
           f"ตรวจ {len(findings)} ข้อ: ❌ {c[Status.CONTRADICTED]} · "
           f"⚠️ {c[Status.UNVERIFIED]} · ✅ {c[Status.VERIFIED]}",
           f"ระดับความเสี่ยง: **{RISK_TH[r['level']]}** (คะแนน {r['score']}; น้ำหนักเป็นค่าประมาณ ยังไม่ได้ปรับจากข้อมูลจริง)", ""]
    if r["top"]:
        out += ["จุดที่ควรดูก่อน:"] + [f"1. {code_span(f.claim.text) if f.claim.text else '(ทั้งคำตอบ)'} — {_clean(f.reason, 160)}"
                                     for f in r["top"]] + [""]
    for status in ORDER:
        group = [f for f in findings if f.status is status]
        if not group:
            continue
        out += [f"## {ICON[status]} {status.value} ({len(group)})", ""]
        for f in group:
            out.append(f"- {code_span(f.claim.text)} — {_clean(f.reason, 300)} _[{f.module}]_")
            for e in f.evidence[:3]:
                out.append(f"  - {code_span(e)}")
        out.append("")
    out += ["---", "ตัวกรองนี้ยืนยันได้แค่ว่า \"มีหลักฐานหรือไม่\" — ลิงก์ที่เปิดได้ไม่ได้พิสูจน์ว่า"
            "หน้านั้นพูดตามที่ AI อ้าง และข้อที่ ✅ ก็ยังควรอ่านต้นฉบับก่อนนำไปใช้จริง", ""]
    return "\n".join(out)
