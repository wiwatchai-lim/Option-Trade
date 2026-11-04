"""ภาพรวมทุก Timeframe แล้ววิเคราะห์กราฟ 5 นาทีแยกตามสภาพตลาด (ทุกแบบหมดอายุ 1 แท่ง, เข้าตอนเปิดแท่ง)"""
import sys

import numpy as np
import pandas as pd

import backtest
from backtest import DATA, TECHS, load

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
TFS = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}
SERVER_TO_THAI = 5  # เซิร์ฟเวอร์ UTC+2, ไทย UTC+7
BASE = [t for t in TECHS if t[4]]


def load_tf(tf):
    backtest.BAR = pd.Timedelta(minutes=TFS[tf])
    return {p.stem.split("_")[0]: load(p) for p in sorted(DATA.glob(f"*_{tf}.csv"))}


def trades(frames, extra=None):
    out = []
    for pair, df in frames.items():
        ctx = extra(df) if extra else pd.DataFrame(index=df.index)
        for name, bear, _, fn, _ in BASE:
            s = fn(df).fillna(False) & df.valid
            d = df[s]
            r = np.sign((d.n_open - d.n_close) if bear else (d.n_close - d.n_open))
            x = ctx[s].copy()
            x["tech"], x["bear"], x["res"] = name, bear, r
            out.append(x)
    t = pd.concat(out)
    return t[t.res != 0]


def wr(r):
    return pd.Series({"ไม้": len(r), "Win rate": (r > 0).mean()})


def fmt(df):
    df = df.copy()
    df["ไม้"] = df["ไม้"].astype(int)
    df["Win rate"] = df["Win rate"].map("{:.1%}".format)
    return df


# ------------------------------------------------------------------ ส่วน A: ทุก Timeframe
print("################ A. ภาพรวมทุก Timeframe ################")
rows = []
for tf in TFS:
    frames = load_tf(tf)
    t = trades(frames)
    row = {"TF": tf}
    for name in dict.fromkeys(t.tech):
        row[name.split(" (")[0]] = f"{(t[t.tech == name].res > 0).mean():.1%}"
    row["รวม 6 เทคนิค"] = f"{(t.res > 0).mean():.1%}"
    row["ไม้ทั้งหมด"] = f"{len(t):,}"
    # ตลาดเอง: แท่งถัดไปสีเดียวกับแท่งนี้บ่อยแค่ไหน (50% = เดาไม่ได้)
    same = tot = 0
    for df in frames.values():
        d = df[df.valid & (df.body > 0)]
        nxt = np.sign(d.n_close - d.n_open)
        cur = np.sign(d.close - d.open)
        same += int((nxt == cur).sum()); tot += int((nxt != 0).sum())
    row["แท่งถัดไปสีเดิม"] = f"{same / tot:.1%}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ ส่วน B: M5 แยกสภาพตลาด
print("\n################ B. กราฟ 5 นาที แยกตามสภาพตลาด ################")
frames = load_tf("M5")


def regime(df):
    sma = df.close.rolling(240).mean()  # ~ SMA20 กราฟ 1 ชม.
    slope = (sma - sma.shift(12)) / df.atr
    q = slope.rolling(288 * 20, min_periods=288 * 5).rank(pct=True)
    trend = pd.cut(q, [0, .2, .4, .6, .8, 1],
                   labels=["1 ขาลงแรง", "2 ขาลงอ่อน", "3 ไซด์เวย์", "4 ขาขึ้นอ่อน", "5 ขาขึ้นแรง"])
    vq = df.atr.rolling(288 * 5, min_periods=288).rank(pct=True)
    vol = pd.cut(vq, [0, 1 / 3, 2 / 3, 1], labels=["1 ต่ำ", "2 กลาง", "3 สูง"])
    h = df.time.dt.hour  # เวลาเซิร์ฟเวอร์ UTC+2
    sess = np.select([(h >= 2) & (h < 9), (h >= 9) & (h < 14), (h >= 14) & (h < 18), (h >= 18) & (h < 23)],
                     ["1 เอเชีย (07-14น.)", "2 ลอนดอน (14-19น.)", "3 ลอนดอน+นิวยอร์ก (19-23น.)",
                      "4 นิวยอร์ก (23-04น.)"], "5 เปลี่ยนวัน/เงียบ (04-07น.)")
    return pd.DataFrame({"trend": trend.astype(str), "vol": vol.astype(str), "session": sess,
                         "thai_hour": (h + SERVER_TO_THAI) % 24}, index=df.index)


t = trades(frames, regime)
t = t[(t.trend != "nan") & (t.vol != "nan")]
print(f"ไม้ทั้งหมดที่ระบุสภาพตลาดได้: {len(t):,}\n")

for col, title in (("trend", "เทรนด์ (ดูจากกราฟ 1 ชม.)"), ("vol", "ความผันผวน"), ("session", "ช่วงตลาด (เวลาไทย)")):
    print(f"--- {title}: รวม 6 เทคนิค ---")
    print(fmt(t.groupby(col).res.apply(wr).unstack()).to_string())
    m = t.assign(w=(t.res > 0)).pivot_table(index="tech", columns=col, values="w", aggfunc="mean")
    print(f"\n--- {title}: แยกเทคนิค (Win rate) ---")
    print(m.map("{:.1%}".format).to_string())
    print()

# ตามเทรนด์ vs สวนเทรนด์
up = t.trend.isin(["4 ขาขึ้นอ่อน", "5 ขาขึ้นแรง"])
dn = t.trend.isin(["1 ขาลงแรง", "2 ขาลงอ่อน"])
t["ทิศ"] = np.select([(~t.bear & up) | (t.bear & dn), (~t.bear & dn) | (t.bear & up)],
                     ["ตามเทรนด์", "สวนเทรนด์"], "ไซด์เวย์")
print("--- เข้าตามเทรนด์ vs สวนเทรนด์ ---")
m = t.assign(w=(t.res > 0)).pivot_table(index="tech", columns="ทิศ", values="w", aggfunc="mean")
print(m.map("{:.1%}".format).to_string())

# สภาพตลาดที่ดีที่สุด 10 อันดับ (เทคนิค x เทรนด์ x ช่วงเวลา, ≥300 ไม้)
print("\n--- 10 สภาพตลาดที่ดีที่สุด (เทคนิค + เทรนด์ + ช่วงตลาด, อย่างน้อย 300 ไม้) ---")
g = t.groupby(["tech", "trend", "session"]).res.apply(wr).unstack()
g = g[g["ไม้"] >= 300].sort_values("Win rate", ascending=False)
print(fmt(g.head(10)).to_string())
print(f"(จาก {len(g)} สภาพตลาดที่มีไม้พอ: เกิน 53% {int((g['Win rate'] >= .53).sum())} แบบ)")
