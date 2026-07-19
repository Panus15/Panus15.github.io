# Core ML Technique — Spec & Rationale

> เอกสารนี้บันทึก "เทคนิค ML หลัก" ของ engine + เหตุผล + จุดที่ต้องระวัง
> (ผ่านการ stress-test แบบ adversarial ด้วยหลายเอเจนต์ก่อนเขียนโค้ด)

## 1. เทคนิคหลัก — Probabilistic (Distributional) Forecasting

ทำนาย **การแจกแจงความน่าจะเป็นเต็มใบของราคาปลายทาง `S_T`** ไม่ใช่ทิศทาง ไม่ใช่ค่าเดียว
รูปธรรม: **Temporal encoder (TFT) → Mixture Density Network (MDN) head** ที่ปล่อยออกมาเป็น
**mixture ของ log-normal** (จับ skew + fat tail ได้)

- Output type กลาง = `density.MixtureLogNormal` — **ทั้ง baseline และ MDN ในอนาคตปล่อยชนิดเดียวกัน**
  ⇒ deep net เป็น drop-in แทน `forecast()` โดยไม่ต้องแก้ downstream เลย
- MDN Gaussian-mixture head = softmax weights, linear means, `softplus(+floor)` sigmas → map ตรงกับ `LogNormalComponent(weight, mu, sigma)`
- **ทำไมไม่ใช่ "ทายทิศทาง":** direction ทำนายยากและ arbitrage หมด; ส่วน **variance/skew** มี premium ที่ยั่งยืน
- **ทำไมไม่ใช่ RL เป็นแกน:** PPO/RL คือชั้น *execution* (จังหวะเข้า-ออก) ไม่ใช่ตัวหา edge — มาทีหลังสุด

## 2. หัวใจแนวคิด — Measure P vs Q

| | P (physical) | Q (risk-neutral) |
|---|---|---|
| คืออะไร | สิ่งที่ **จะเกิดจริง** | สิ่งที่ **ตลาดตั้งราคา** |
| ได้จาก | price history → `baseline.py` | option chain → `rnd.py` |
| ค่าเฉลี่ย | forward `S·e^{(r-q)T}` (ตั้งใจ neutral) | forward เดียวกัน (parity) |

**Edge = ช่องว่าง P − Q**
- **Variance Risk Premium** `VRP = E^Q[var] − E^P[var]` (IV มักแพงกว่า RV → sell vol) — Carr-Wu 2009, Bollerslev-Tauchen-Zhou 2009
- **Skew Risk Premium** `SRP = skew_Q − skew_P`

⚠️ **นี่คือ "premium" ไม่ใช่ "arbitrage ฟรี"** — เป็นค่าตอบแทนการรับความเสี่ยง crash → ต้อง size + gate เสมอ

## 3. โมดูล Phase-1 (build แล้ว รันได้ + มี oracle test)

| ไฟล์ | ทำอะไร | test |
|---|---|---|
| `density.py` | `MixtureLogNormal` output type (pdf/cdf/quantile/moments/pricing) | ตี option = Black-Scholes เป๊ะ (6) |
| `baseline.py` | HAR-RV vol → mixture calm+crash (**skew data-driven**) | leverage+horizon+vol recover |
| `rnd.py` | Q-moments จากกระดาน: model-free (VIX), **BKM**, Breeden-Litzenberger | flat chain → skew≈0 (5) |
| `edge.py` | เทียบ P vs Q ต่อ expiry → VRP/SRP + verdict | matched→FAIR, rich→RICH, z-score, regime, cost (11) |
| `objective.py` | **promotion gate** (NLL/CRPS/pinball + left-tail) พิสูจน์ MDN ชนะ baseline OOS | NLL properness (4) |

**รวม 24 tests ผ่านหมด** (เดิม engine 7 + density 6 + rnd/edge 11 + objective 4 — ที่นี่นับ models เท่านั้น 26)

## 4. สิ่งที่แก้ตาม adversarial review (สำคัญมาก — ป้องกันกับดักคลาสสิก)

1. **Baseline skew เคยเป็นค่าคงที่ (−1.19 ทุก input) → strawman.** แก้แล้ว: skew ขึ้นกับ vol (leverage), หด
   ตาม horizon (~1/√T), seed จาก realized 3rd moment
2. **BKM annualization bug:** ใช้ `Var_Q = e^{rT}·V − μ²` แล้วรายงาน `vol = √(Var_Q/T)`
   (ไม่ใช่ `√(V/T)` ที่ bias variance ต่ำ) — ให้เทียบ P/Q แบบ like-for-like
3. **BKM ต้อง spot-anchored** (weights `ln(K/S)`); forward ใช้จาก put-call parity เฉพาะฝั่ง VIX/BL
4. **VRP level = แค่เก็บ premium ไม่ใช่ alpha** → เทรด **z-score (deviation)** เทียบประวัติ ไม่ใช่ระดับ
5. **HAR ตามหลัง crash** → **regime gate** ปิดสัญญาณ sell-vol เมื่อ realized vol เร่งตัว
6. **SRP = diagnostic-only** จนกว่าจะมี physical skew ที่ดีจริง (อย่าลงเงินบน skew อย่างเดียว)
7. **ต้นทุนกิน edge** → คิด **round-trip spread + commission + hedge** แปลงเป็น vol points ผ่าน vega
8. **Look-ahead** → เพิ่ม `OptionChain.asof` (P ต้อง truncate ≤ asof)

## 5. สถานะ roadmap (verifier ชี้ว่าต้องมีก่อน go-live)

**✅ ทำแล้ว:**
- **Portfolio risk layer** (`engine/portfolio.py`) — correlation-aware short-vega cap (N short = N× ไม่ใช่ √N) + **CVaR sizing** (ไม่ใช่ Kelly binary) + drawdown kill-switch + defined-risk spread
- **Delta-hedged walk-forward backtest** (`engine/hedged_backtest.py`) — **non-overlapping** trades (Sharpe ไม่เฟ้อ) + crash-aware (calm Sharpe 2.0 → crash 0.1) + regime gate/kill-switch ทำงานจริง
- **MDN pathology guards** — sigma floor + weight-entropy reg (ทั้ง `mdn.py` และ `neural.py`)
- **Data adapters** (`engine/adapters.py`) — Polygon/ORATS live + CSV/JSON offline (asof กัน look-ahead)
- **Trained encoder** (`models/neural.py`) — MLP encoder เทรน end-to-end (numpy) = ก้าวสู่ TFT/GRU

**🟠 ยังขาด:**
- **Vega loss ใน backtest:** ตอนนี้ hold IV คงที่/trade → จับ gamma loss แต่ยังไม่จับ vega loss ตอน IV spike (ต่อ vol-of-vol path)
- **Survivorship-free data:** point-in-time universe เก็บ delisted names (CRSP + OptionMetrics/IvyDB)
- **American exercise + dividends:** single-name เป็น American; BKM/VIX สมมติ European → de-Americanize หรือใช้ **index/ETF (SPX/SPY)** ก่อน
- **Wing-truncation guard ปรับตาม regime:** wings หายตอนเครียด → ต้องเข้มขึ้นเมื่อ vol สูง
- **Newey-West SE** สำหรับ overlapping backtest (ตอนนี้เลี่ยงด้วย non-overlapping)

## 6. Production target (หลัง baseline ผ่าน gate)

```
features (price, RV, IV-surface, [sentiment])
    → trained encoder → MDN head → MixtureLogNormal   # แทน baseline.forecast()
```
- `models/mdn.py` = fixed random-feature encoder + trained head (stdlib)
- `models/neural.py` = **trained MLP encoder end-to-end** (numpy) — ก้าวถัดไปคือสลับ MLP เป็น **TFT/GRU** โดย type ที่ปล่อยออก (MixtureLogNormal) ไม่เปลี่ยน

Promotion rule: โมเดลใหม่ต้องชนะ HAR baseline **ทั้ง** aggregate NLL **และ** left-tail pinball (แยก) บน OOS
ที่ครอบ crash regime อย่างน้อย 1 ครั้ง + ผ่าน net-of-cost hedged backtest — ถึงจะขึ้น production
(บน synthetic data ตอนนี้ baseline ยังชนะ — gate ทำงานถูก)

### ข้อมูลต้องเตรียม (งบ)
IVDB/OptionMetrics (point-in-time, ~5 หลัก/ปี) หรือ ORATS (~$1-2k+/เดือน) · Polygon ถูกแต่ history ตื้น ·
HAR price history ฟรี (stooq/Yahoo) · 30d horizon ให้ ~12 obs ไม่ทับซ้อน/ปี/ticker → ต้องหลายปี×หลาย ticker
