"""เลือกเทรดแค่วันละ 3 ไม้: ให้ AI ให้คะแนนสัญญาณ แล้วเข้าเฉพาะไม้คะแนนสูง เทียบกับไม่เลือก/สุ่ม

เรียนจากข้อมูลก่อน 2026-05-01 (train) แล้ววัดผลกับ 2026-05-01 เป็นต้นไป (test) ที่ AI ไม่เคยเห็น
ในวันจริงเราไม่รู้ล่วงหน้าว่าไม้ไหนดีที่สุดของวัน จึงใช้กติกา: เข้า 3 ไม้แรกของวันที่คะแนนผ่านเกณฑ์
"""
import math
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import DATA, TECHS, load
from filter_search import enrich

sys.stdout.reconfigure(encoding="utf-8")
SPLIT = pd.Timestamp("2026-05-01")
PER_DAY, STAKE, PAYOUT = 3, 10, 0.90
NUM = ["hour", "dow", "vol_pct", "body_atr", "prev_body_atr", "size_ratio", "upper_atr", "lower_atr",
       "dist_sma5", "slope_sma5", "dist_sma240", "dist_hi20", "dist_lo20", "confluence"]


def build(frames):
    out = []
    for pi, (pair, df) in enumerate(frames.items()):
        a = df.atr
        f = pd.DataFrame({
            "hour": df.time.dt.hour, "dow": df.time.dt.dayofweek,
            "vol_pct": df.atr.rolling(288 * 5, min_periods=288).rank(pct=True),
            "body_atr": df.body / a, "prev_body_atr": df.body.shift(1) / a,
            "size_ratio": np.minimum(df.body, df.body.shift(1)) / np.maximum(df.body, df.body.shift(1)).replace(0, np.nan),
            "upper_atr": df.upper / a, "lower_atr": df.lower / a,
            "dist_sma5": (df.close - df.sma5) / a, "slope_sma5": (df.sma5 - df.sma5.shift(3)) / a,
            "dist_sma240": (df.close - df.sma240) / a,
            "dist_hi20": (df.high.shift(1).rolling(20).max() - df.close) / a,
            "dist_lo20": (df.close - df.low.shift(1).rolling(20).min()) / a})
        sigs = [(i, name, bear, fn(df).fillna(False) & df.valid)
                for i, (name, bear, _, fn, base) in enumerate([t for t in TECHS if t[4]])]
        n_put = sum(s.astype(int) for *_, b, s in sigs if b)
        n_call = sum(s.astype(int) for *_, b, s in sigs if not b)
        for ti, name, bear, s in sigs:
            d = df[s]
            x = f[s].copy()
            sign = -1 if bear else 1  # หมุนทิศให้ฟีเจอร์ PUT/CALL เทียบกันได้
            for c in ("dist_sma5", "slope_sma5", "dist_sma240"):
                x[c] *= sign
            if bear:
                x["dist_hi20"], x["dist_lo20"] = f.dist_lo20[s], f.dist_hi20[s]
            x["confluence"] = (n_put if bear else n_call)[s]
            x["tech_id"], x["pair_id"], x["is_put"] = ti, pi, int(bear)
            x["time"], x["tech"], x["pair"] = d.time, name, pair
            x["res"] = np.sign((d.n_open - d.n_close) if bear else (d.n_close - d.n_open))
            out.append(x)
    t = pd.concat(out, ignore_index=True)
    t = t[t.res != 0].dropna(subset=NUM).sort_values("time").reset_index(drop=True)
    t["date"] = t.time.dt.date
    t["win"] = (t.res > 0).astype(int)
    return t


def pick_first(t, score=None, thr=None, seed=None):
    d = t if score is None else t[score >= thr]
    if seed is not None:
        d = d.sample(frac=1, random_state=seed).sort_values("date", kind="stable")
    return d.groupby("date", sort=False).head(PER_DAY)


def show(label, d, days):
    n, w = len(d), int(d.win.sum())
    p = w / n if n else float("nan")
    se = 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")
    pnl = w * STAKE * PAYOUT - (n - w) * STAKE
    print(f"{label:44s} ไม้ {n:5d} ({n / days:.1f}/วัน)  Win rate {p:6.1%} (±{se:.1%})  กำไร/ขาดทุนรวม ${pnl:+,.0f}")


def main():
    frames = {p.stem.replace("_M5", ""): enrich(load(p)) for p in sorted(DATA.glob("*_M5.csv"))}
    t = build(frames)
    tr, te = t[t.time < SPLIT], t[t.time >= SPLIT]
    days_tr, days_te = tr.date.nunique(), te.date.nunique()
    X = NUM + ["tech_id", "pair_id", "is_put"]
    print(f"สัญญาณ train {len(tr):,} ({days_tr} วัน) | test {len(te):,} ({days_te} วัน)\n")

    models = {
        "AI แบบต้นไม้ (Gradient Boosting)": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
            categorical_features=[X.index("tech_id"), X.index("pair_id")], random_state=0),
        "AI แบบเส้นตรง (Logistic)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    for name, m in models.items():
        m.fit(tr[X], tr.win)
        s_tr, s_te = m.predict_proba(tr[X])[:, 1], m.predict_proba(te[X])[:, 1]
        print(f"=== {name} ===")
        for top in (0.5, 1, 2, 5):  # เกณฑ์ = คะแนนสูงสุด top% ของ train
            thr = np.quantile(s_tr, 1 - top / 100)
            show(f"  เกณฑ์คะแนนบนสุด {top}% (train)", pick_first(tr, s_tr, thr), days_tr)
            show(f"  เกณฑ์คะแนนบนสุด {top}% (test ที่ไม่เคยเห็น)", pick_first(te, s_te, thr), days_te)
        # ถ้ารู้ล่วงหน้าว่าไม้ไหนคะแนนสูงสุดของวัน (ทำจริงไม่ได้ แต่ดูเพดาน)
        top3 = te.assign(s=s_te).sort_values("s", ascending=False).groupby("date").head(PER_DAY)
        show("  เพดาน: 3 ไม้คะแนนสูงสุดของวัน (test)", top3, days_te)
        print()

    print("=== ไม่ใช้ AI (test) ===")
    show("  เข้า 3 สัญญาณแรกของวัน", pick_first(te), days_te)
    for tech in dict.fromkeys(te.tech):
        show(f"  3 สัญญาณแรกของวัน เฉพาะ {tech}", pick_first(te[te.tech == tech]), days_te)
    wr = [pick_first(te, seed=s).win.mean() for s in range(200)]
    print(f"  สุ่ม 3 ไม้/วัน 200 รอบ: Win rate เฉลี่ย {np.mean(wr):.1%}  (ต่ำสุด {min(wr):.1%} – สูงสุด {max(wr):.1%})")


if __name__ == "__main__":
    main()
