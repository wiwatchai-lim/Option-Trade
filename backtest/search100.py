"""ค้นหาหลายร้อยวิธีจากสูตรในไฟล์ แบบป้องกันการหลอกตัวเอง

ข้อมูล 3 ส่วน (กำหนดก่อนเริ่ม ห้ามเปลี่ยน):
  DEV_TRAIN  ก่อน 2025-12-01         ใช้สร้าง/เทรน AI
  DEV_VAL    2025-12-01..2026-03-01  ใช้จัดอันดับวิธี
  HOLDOUT    2026-03-01 เป็นต้นไป     ล็อกไว้ เปิดตรวจครั้งเดียวกับวิธีที่ชนะ
เกณฑ์สำเร็จ: HOLDOUT win rate >= 55%, ไม้ >= 200, ขอบล่าง 95% > 52.6%
"""
import itertools
import math
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import DATA, TECHS, load
from pick3 import NUM, build
from filter_search import enrich

sys.stdout.reconfigure(encoding="utf-8")
T_VAL, T_HOLD = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01")
MIN_VAL, MIN_HOLD, GOAL, BREAKEVEN = 150, 200, 0.55, 0.526


def ci(w, n):
    p = w / n
    return p - 1.96 * math.sqrt(p * (1 - p) / n), p + 1.96 * math.sqrt(p * (1 - p) / n)


frames = {p.stem.replace("_M5", ""): enrich(load(p)) for p in sorted(DATA.glob("*_M5.csv"))}

# ---------- 1) วิธีจากกฎ: ตีความทุกแบบ x ตาม/สวน x ช่วงเวลา x เทรนด์
parts = []
for vi, (name, bear, var, fn, _) in enumerate(TECHS):
    for pair, df in frames.items():
        s = fn(df).fillna(False) & df.valid
        d = df[s]
        r = np.sign((d.n_open - d.n_close) if bear else (d.n_close - d.n_open))
        thai_h = (d.time.dt.hour + 5) % 24
        aligned = np.where(bear, ~d.h1_up, d.h1_up)
        parts.append(pd.DataFrame({"vi": vi, "time": d.time, "res": r, "thai_h": thai_h, "aligned": aligned}))
T = pd.concat(parts)
T = T[T.res != 0]

HOURS = {"ทุกเวลา": lambda h: h >= 0, "ไม่เอา04-07น.": lambda h: ~((h >= 4) & (h < 7)),
         "เฉพาะ14-23น.": lambda h: (h >= 14) & (h < 23), "เฉพาะ07-14น.": lambda h: (h >= 7) & (h < 14)}
TRENDS = {"ไม่ดูเทรนด์": None, "ตามกราฟใหญ่": True, "สวนกราฟใหญ่": False}

methods = []  # (ชื่อ, Series ผล +1/-1 พร้อม time)
for vi, g in T.groupby("vi"):
    name, bear, var, *_ = TECHS[vi]
    for (hn, hf), (tn, tv), flip in itertools.product(HOURS.items(), TRENDS.items(), (False, True)):
        m = hf(g.thai_h)
        if tv is not None:
            m &= g.aligned == tv
        sub = g[m]
        res = -sub.res if flip else sub.res
        label = f"{'สวนสัญญาณ ' if flip else ''}{name} | {var} | {hn} | {tn}"
        methods.append((label, pd.DataFrame({"time": sub.time.values, "res": res.values})))

# ---------- 2) วิธีจาก AI (เทรนบน DEV_TRAIN เท่านั้น)
t = build(frames)
X = NUM + ["tech_id", "pair_id", "is_put"]
tr = t[t.time < T_VAL]
models = {
    "AI ต้นไม้ (GBM)": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                       min_samples_leaf=200, random_state=0),
    "AI ป่าไม้ (Random Forest)": RandomForestClassifier(n_estimators=300, min_samples_leaf=100, n_jobs=-1,
                                                        random_state=0),
    "AI เส้นตรง (Logistic)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}
for mname, m in models.items():
    m.fit(tr[X], tr.win)
    s_tr, s_all = m.predict_proba(tr[X])[:, 1], m.predict_proba(t[X])[:, 1]
    for top in (1, 2, 5, 10):
        hi = np.quantile(s_tr, 1 - top / 100)
        lo = np.quantile(s_tr, top / 100)
        sel = t[s_all >= hi]
        methods.append((f"{mname} คะแนนบนสุด {top}%", pd.DataFrame({"time": sel.time.values, "res": sel.res.values})))
        sel = t[s_all <= lo]  # คะแนนต่ำสุด -> เข้าทางตรงข้าม
        methods.append((f"{mname} คะแนนล่างสุด {top}% แล้วสวนทาง",
                        pd.DataFrame({"time": sel.time.values, "res": -sel.res.values})))

# ---------- 3) จัดอันดับด้วย DEV_VAL
rows = []
for label, d in methods:
    tr_ = d[d.time < T_VAL]; va = d[(d.time >= T_VAL) & (d.time < T_HOLD)]
    if len(va) < MIN_VAL:
        continue
    rows.append({"วิธี": label, "ไม้_สร้าง": len(tr_), "WR_สร้าง": (tr_.res > 0).mean() if len(tr_) else np.nan,
                 "ไม้_จัดอันดับ": len(va), "WR_จัดอันดับ": (va.res > 0).mean()})
R = pd.DataFrame(rows).sort_values("WR_จัดอันดับ", ascending=False).reset_index(drop=True)
print(f"ลองทั้งหมด {len(methods)} วิธี | มีไม้พอให้จัดอันดับ {len(R)} วิธี")
print(f"วิธีที่ได้ ≥55% ในช่วงจัดอันดับ: {(R.WR_จัดอันดับ >= GOAL).sum()} วิธี\n")

pd.set_option("display.width", 300, "display.max_colwidth", 110)
show = R.head(10).copy()
for c in ("WR_สร้าง", "WR_จัดอันดับ"):
    show[c] = show[c].map("{:.1%}".format)
print("10 อันดับแรก (ยังไม่เปิดข้อมูลที่ล็อกไว้):")
print(show.to_string())

# ---------- 4) เปิดข้อมูลที่ล็อกไว้ ครั้งเดียว
lookup = dict(methods)
print("\n=========== เปิดข้อมูลที่ล็อกไว้ (มี.ค.-พ.ย. 2026) ===========")
for rank, row in R.head(10).iterrows():
    d = lookup[row["วิธี"]]
    h = d[d.time >= T_HOLD]
    w, n = int((h.res > 0).sum()), len(h)
    lo, hi = ci(w, n)
    ok = n >= MIN_HOLD and w / n >= GOAL and lo > BREAKEVEN
    print(f"#{rank + 1:2d} จัดอันดับ {row['WR_จัดอันดับ']:.1%} -> ข้อมูลล็อก {w / n:.1%} ({n} ไม้, ช่วง {lo:.1%}-{hi:.1%})"
          f"  {'✅ ผ่าน' if ok else '❌ ไม่ผ่าน'}  | {row['วิธี']}")
best = R.iloc[0]
d = lookup[best["วิธี"]]; h = d[d.time >= T_HOLD]
w, n = int((h.res > 0).sum()), len(h)
lo, _ = ci(w, n)
print(f"\nคำตัดสินตามกติกา (ใช้อันดับ 1 เท่านั้น): "
      f"{'สำเร็จ' if n >= MIN_HOLD and w / n >= GOAL and lo > BREAKEVEN else 'ไม่สำเร็จ'} "
      f"— {w / n:.1%} จาก {n} ไม้")
R.to_csv("search100_results.csv", index=False, encoding="utf-8-sig")
