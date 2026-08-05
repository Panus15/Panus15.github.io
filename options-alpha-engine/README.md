# Options Alpha Engine — MVP Core

> Multi-Modal Probability & Volatility Estimation Engine
> ระบบล่าส่วนต่างราคาออปชัน (mispricing) ด้วยสถิติ + AI

โค้ดชุดนี้คือ **ก้าวแรกที่จับต้องได้และรันได้จริง** ของ engine ตาม roadmap 4 ขั้น
รันได้ทันทีด้วย Python stdlib ล้วน ๆ (ไม่ต้อง `pip install` อะไรเลย):

```bash
cd options-alpha-engine
python3 demo.py            # เดินครบ loop: data -> vol forecast -> scan -> sizing -> backtest
python3 tests/test_engine.py   # 7 correctness tests (put-call parity, IV round-trip, ฯลฯ)
```

**อยากเห็นผลจากราคาจริงเลย — คำสั่งเดียว:**

```bash
python3 -m tools.quickstart        # Windows: ดับเบิลคลิก START.bat
python3 -m tools.quickstart --demo # ไม่มีเน็ต / โดนบล็อก: ใช้ข้อมูลจำลอง
```

โหลดราคา → วาดชาร์ต → เปิดในเบราว์เซอร์ → เขียนผล 2 บรรทัดลง `RESULT.txt`
ทุกขั้นรันจากโฟลเดอร์ของตัวเอง ไม่ว่า shell จะอยู่ที่ไหนก็ได้ และถ้าขั้นไหนพัง
มันจะหยุดตรงนั้นพร้อมบอกสาเหตุ แทนที่จะไปตายที่ขั้นถัดไป

---


> **Start here:** [`PIPELINE.md`](PIPELINE.md) — the whole system on one
> page: the flow, every measured number, and how to run it.
> [`PREREGISTRATION.md`](PREREGISTRATION.md) locks the parameters and the
> decision rules *before* any real data is seen.

## TL;DR — คำตอบ 2 ข้อที่ถามมา

### 1) เริ่มจากอะไรก่อน: ท่อ Greeks/IV หรือ NLP ข่าว?

**เริ่มที่ Greeks / IV / Vol Surface ก่อน — ไม่ใช่ NLP.** เหตุผล:

- **หัวใจของโปรดักต์คือการหา mispricing** (ถูก/แพงกว่าความจริง) ซึ่ง *ต้องมี pricing engine ก่อน*
  ถ้ายังไม่มีตัวตีราคายุติธรรม NLP sentiment ก็ไม่มีที่ให้เสียบ — มันเป็น *feature* ไม่ใช่ *core*
- **ฝั่งตัวเลขตรวจสอบได้ทันที** (deterministic): คำนวณ IV/Greeks แล้วเช็คกับ put-call parity, ค่าที่รู้ผลอยู่แล้วได้เลย
  → progress เร็ว, พิสูจน์ถูก/ผิดได้ทุกวัน. ตรงข้ามกับ NLP ที่ signal มั่วและ validate ยาก
- **NLP sentiment → alpha คือของยากและ noisy มาก** FinBERT score กับผลตอบแทนมี correlation อ่อนและไม่เสถียร
  โดนตลาด arbitrage หมดแล้ว. เก็บไว้เป็น **Phase 2** ที่มาเสริม signal หลัก ไม่ใช่ตัวเปิด MVP

> สรุป: **Structured data + pricing/vol มาก่อน. NLP ตามทีหลัง.** โค้ดใน repo นี้คือ Phase 1 ที่ว่า

### 2) ทำยังไงให้ "มันออกมาเวิค"?

ความจริงที่ต้องยอมรับ: **การเอาชนะตลาดออปชันยากมาก และ 90% ของโปรเจกต์แบบนี้ตายที่ backtest ที่โกหกตัวเอง**
ทางที่ทำให้เวิคจริง เรียงตามลำดับลดความเสี่ยงเร็วที่สุด:

1. **หด scope ให้โหด** — เลือก underlying เดียวที่ liquid (เช่น SPY หรือ mega-cap ไม่กี่ตัว) + edge เดียว
   อย่าเพิ่งไปทำนาย density (S_T) ของทุกหุ้น. MVP = **mispricing scanner บนออปชัน liquid** พอ
2. **โฟกัส "vol premium" ไม่ใช่ "ทิศทาง"** — edge ที่ทนที่สุดคือ **variance risk premium**
   (implied vol มักแพงกว่า realized vol). ทำนาย realized vol ให้แม่นแล้วขาย IV แพง/ซื้อ IV ถูก + delta-hedge
   → ยากน้อยกว่าและ robust กว่าการทำนายว่าหุ้นจะขึ้น/ลง (MDN density มักไม่ชนะ lognormal ธรรมดา)
3. **Backtest ต้องซื่อสัตย์** — จุดที่โปรเจกต์ตาย:
   - bid/ask spread กว้าง (ห้ามเทรดที่ mid), fill/liquidity assumption, look-ahead ใน IV surface,
     survivorship bias, commission, assignment/early-exercise risk
   - `engine/backtest.py` **บังคับให้ใส่ต้นทุน** (spread + commission + slippage) เป็น default
4. **Data คือคอขวดและค่าใช้จ่ายจริง** — ข้อมูล options history ที่ดี (chain เต็ม + Greeks + timestamp)
   ราคาแพง (ORATS, OptionMetrics/IvyDB, CBOE DataShop, Polygon.io). ข้อมูลฟรีไม่พอสำหรับ backtest จริงจัง.
   ตั้งงบส่วนนี้ไว้ตั้งแต่แรก
5. **เริ่มจาก baseline ที่ง่ายและ interpretable** — HAR-RV / EWMA สำหรับ vol, กฎง่าย ๆ สำหรับ execution
   **ต้องชนะ baseline ก่อน** ค่อยเติม TFT / MDN / PPO. ของหรู = 10% สุดท้ายและเสี่ยงสุด (RL execution overfit ง่ายมาก)
6. **Risk management คือ Phase 0 ไม่ใช่ Phase 3** — fractional Kelly + hard cap ต่อไม้ (ดู `engine/sizing.py`)
   ต้องมีตั้งแต่ backtest วันแรก. Risk of ruin ฆ่าแม้แต่กลยุทธ์ที่มี edge จริง
7. **Paper trade ก่อนเสมอ** — ต่อ IBKR/sandbox วัด latency + ค่าธรรมเนียมแฝง ก่อนใส่เงินจริง

> **meta-answer:** สร้างสิ่งที่เล็กที่สุดที่ให้ **equity curve out-of-sample ที่รวมต้นทุนแล้ว** บน 1 กลยุทธ์
> ถ้าเส้นนั้นมี edge หลังหักต้นทุน → คุณมี startup. ถ้าไม่มี → TFT/RL/NLP ก็ช่วยไม่ได้
> **พิสูจน์ว่า edge มีอยู่จริงก่อน ค่อยสร้าง engine หรู**

---

## โค้ดนี้แมปกับ roadmap 4 ขั้นของคุณยังไง

| Roadmap ขั้น | โมดูลใน repo | สถานะ |
|---|---|---|
| **1. Data Pipeline & Features** — Greeks/IV/IV-surface | `engine/pricing.py`, `engine/iv.py`, `engine/data.py` | ✅ รันได้ (synthetic adapter; เสียบ vendor จริงผ่าน `MarketDataAdapter`) |
| **2. Core ML** — Vol model, density, pricing evaluator | `engine/volforecast.py` (HAR-RV/EWMA baseline), `engine/signal.py` (evaluator) | ✅ baseline; TFT/MDN = งานถัดไป |
| **3. RL Execution & Risk** — sizing, exit | `engine/sizing.py` (fractional Kelly + cap) | 🟡 sizing พร้อม; PPO agent = งานถัดไป |
| **4. Validation & Proof** — backtest, Sortino/MaxDD | `engine/backtest.py` | ✅ engine + metrics พร้อม; ต่อ data 10 ปี = งานถัดไป |

**NLP (FinBERT sentiment)** = จงใจยังไม่ทำใน Phase 1 ตามเหตุผลข้อ 1 ด้านบน — เสียบเป็น feature เพิ่มใน Phase 2

---

## สถาปัตยกรรม

```
option data (vendor)                prices (vendor)
        │                                   │
        ▼                                   ▼
   data.py ─ OptionChain            volforecast.py ─ RV forecast (HAR-RV)
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
                   signal.py  ── scan: market IV vs forecast vol,
                        │          edge NET of spread -> RICH / CHEAP
                        ▼
                   sizing.py  ── fractional Kelly + 2% risk cap
                        ▼
                  backtest.py ── cost-aware equity curve
                                 Sharpe / Sortino / MaxDD / hit-rate
```

Data ทั้งหมดพูดภาษาเดียว: `OptionQuote` / `OptionChain`
ไปเทรดจริง = เขียน `MarketDataAdapter` ตัวใหม่ (Polygon/ORATS/IBKR) ให้คืน object เดิม — downstream ไม่ต้องแก้อะไร

---

## Roadmap การ build (เรียงตามลดความเสี่ยง)

- **สัปดาห์ 1–2:** ต่อ data จริง 1 underlying → IV/Greeks/vol surface เก็บลง **TimescaleDB** *(repo นี้ = ตัวอย่างที่รันได้แล้ว)*
- **สัปดาห์ 3–4:** signal variance-risk-premium ให้แน่น + validate IV surface ไม่มี look-ahead
- **สัปดาห์ 5–6:** backtest 10 ปีจริง รวมต้นทุนจริง → เอา **honest Sharpe/Sortino/MaxDD** ออกมาให้ได้
- **หลังจากนั้น:** ถ้า baseline มี edge → เติม **TFT** (vol forecast), **MDN** (density), **PPO** (exit timing)
- **ก่อนเงินจริง:** **paper trade** บน IBKR sim วัด latency + ค่าธรรมเนียม

---

## หมายเหตุ (สำคัญ)

- `SyntheticAdapter` เป็น **fixture** สำหรับพิสูจน์ว่า pipeline เดินครบ — **ไม่ใช่ market simulator**
  ตัวเลข backtest ที่ออกมาไม่มีความหมายจนกว่าจะเปลี่ยนเป็นข้อมูลจริง
- core engine เป็น **stdlib ล้วน** เพื่อให้รันได้ทุกที่. Production ควรสลับเป็น `numpy`/`pandas`/`scipy`
  (แทน OLS/normal-CDF ที่เขียนเอง) เพื่อความเร็วและ vectorization
- ทุกไม้ **หักต้นทุนก่อนตัดสินใจ** — edge ที่หายไปหลังข้าม spread ไม่ใช่ edge

## License

Prototype scaffold for internal R&D.
