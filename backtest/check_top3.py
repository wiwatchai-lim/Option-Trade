"""เช็คว่า 3 สภาพตลาดที่เกิน 53% (กราฟ 5 นาที) ชนะสม่ำเสมอทุกช่วงเวลาหรือไม่"""
import sys

import numpy as np
import pandas as pd

from backtest import DATA, TECHS, load

sys.stdout.reconfigure(encoding="utf-8")

# เหมือนใน regimes.py
def regime(df):
    sma = df.close.rolling(240).mean()
    slope = (sma - sma.shift(12)) / df.atr
    q = slope.rolling(288 * 20, min_periods=288 * 5).rank(pct=True)
    h = df.time.dt.hour  # เซิร์ฟเวอร์ UTC+2 (ไทย = +5 ชม.)
    return pd.DataFrame({
        "time": df.time,
        "strong_up": q > .8, "strong_dn": q <= .2, "side": (q > .4) & (q <= .6),
        "asia": (h >= 2) & (h < 9), "london": (h >= 9) & (h < 14), "overlap": (h >= 14) & (h < 18)})


SETUPS = {
    "A: เขียวกลืนกินแดง + ขาขึ้นแรง + เอเชีย 07-14น.": ("2", lambda c: c.strong_up & c.asia),
    "B: เขียวกลืนกินแดง + ไซด์เวย์ + 19-23น.": ("2", lambda c: c.side & c.overlap),
    "C: แดงกลืนกินเขียว + ขาขึ้นแรง + ลอนดอน 14-19น.": ("1", lambda c: c.strong_up & c.london),
}

rows = []
for p in sorted(DATA.glob("*_M5.csv")):
    df = load(p)
    ctx = regime(df)
    for label, (tid, cond) in SETUPS.items():
        name, bear, _, fn, _ = next(t for t in TECHS if t[4] and t[0].startswith(tid))
        s = fn(df).fillna(False) & df.valid & cond(ctx).fillna(False)
        d = df[s]
        r = np.sign((d.n_open - d.n_close) if bear else (d.n_close - d.n_open))
        rows.append(pd.DataFrame({"setup": label, "time": d.time, "pair": p.stem.split("_")[0], "res": r}))
t = pd.concat(rows)
t = t[t.res != 0]
t["win"] = t.res > 0
t["ไตรมาส"] = t.time.dt.to_period("Q").astype(str)
t["ครึ่ง"] = np.where(t.time < pd.Timestamp("2026-03-01"), "ครึ่งแรก", "ครึ่งหลัง")

pd.set_option("display.width", 250)
for label, g in t.groupby("setup"):
    print(f"\n===== {label} =====")
    print(f"ทั้งหมด {len(g)} ไม้, Win rate {g.win.mean():.1%}  (~{len(g) / g.time.dt.date.nunique():.1f} ไม้ต่อวันที่เกิด)")
    for col in ("ครึ่ง", "ไตรมาส", "pair"):
        s = g.groupby(col).win.agg(["size", "mean"])
        print(f"  ตาม{col}: " + " | ".join(f"{k} {v['mean']:.0%} ({int(v['size'])})" for k, v in s.iterrows()))
