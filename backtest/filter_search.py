"""ค้นหาตัวกรองที่ทำให้ Win rate สูงสุด แล้วตรวจกับข้อมูลช่วงหลังที่ไม่ได้ใช้ค้นหา

ข้อมูลช่วงค้นหา (train): ก่อน 2026-05-01  |  ข้อมูลตรวจ (test): ตั้งแต่ 2026-05-01
มิติที่กรอง: เทคนิค, คู่เงิน, ช่วงเวลา (ทีละ 3 ชม.), วันในสัปดาห์, ความผันผวน, ทิศกราฟใหญ่, สัญญาณซ้อน
ลองทุกชุด 1-3 มิติ ทุกช่องที่มีไม้ใน train ≥ 300

การทดลองหลอก (placebo): สุ่มแท่งและทิศ CALL/PUT ด้วยจำนวนไม้เท่ากัน แล้วค้นหาแบบเดียวกัน
"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtest import DATA, TECHS, load

sys.stdout.reconfigure(encoding="utf-8")
SPLIT = pd.Timestamp("2026-05-01")
MIN_TRAIN, MIN_TEST, BREAKEVEN = 300, 100, 0.53
DIMS = ["tech", "pair", "hour3", "dow", "vol", "big_trend", "confluence"]


def enrich(df):
    df["hour3"] = (df.time.dt.hour // 3 * 3).map(lambda h: f"{h:02d}-{h + 3:02d}น.")
    df["dow"] = df.time.dt.day_name().str[:3]
    pct = df.atr.rolling(288 * 5, min_periods=288).rank(pct=True)
    df["vol"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=["ผันผวนต่ำ", "ผันผวนกลาง", "ผันผวนสูง"])
    df["sma240"] = df.close.rolling(240).mean()  # ~ SMA20 ของกราฟ 1 ชม.
    df["h1_up"] = df.close > df.sma240
    return df


def build_trades(frames):
    out = []
    for pair, df in frames.items():
        sigs = {}
        for name, bear, _, fn, base in TECHS:
            if base:
                sigs[(name, bear)] = fn(df).fillna(False) & df.valid
        n_put = sum(s.astype(int) for (n, b), s in sigs.items() if b)
        n_call = sum(s.astype(int) for (n, b), s in sigs.items() if not b)
        for (name, bear), s in sigs.items():
            d = df[s]
            diff = (d.n_open - d.n_close) if bear else (d.n_close - d.n_open)
            same = (n_put if bear else n_call)[s]
            aligned = (~d.h1_up) if bear else d.h1_up
            out.append(pd.DataFrame({
                "time": d.time, "tech": name, "pair": pair, "hour3": d.hour3, "dow": d.dow,
                "vol": d.vol.astype(str), "big_trend": np.where(aligned, "ตามกราฟใหญ่", "สวนกราฟใหญ่"),
                "confluence": np.where(same >= 2, "สัญญาณซ้อน≥2", "สัญญาณเดียว"), "res": np.sign(diff)}))
    t = pd.concat(out, ignore_index=True)
    return t[(t.res != 0) & (t.vol != "nan")]


def placebo(frames, n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    per = n // len(frames)
    for pair, df in frames.items():
        d = df[df.valid & df.atr.notna()].sample(per, random_state=int(rng.integers(1e9)))
        bear = rng.random(len(d)) < 0.5
        diff = np.where(bear, d.n_open - d.n_close, d.n_close - d.n_open)
        aligned = np.where(bear, ~d.h1_up, d.h1_up)
        out.append(pd.DataFrame({
            "time": d.time, "tech": rng.choice([x[0] for x in TECHS if x[4]], len(d)), "pair": pair,
            "hour3": d.hour3, "dow": d.dow, "vol": d.vol.astype(str),
            "big_trend": np.where(aligned, "ตามกราฟใหญ่", "สวนกราฟใหญ่"),
            "confluence": rng.choice(["สัญญาณซ้อน≥2", "สัญญาณเดียว"], len(d), p=[.2, .8]), "res": np.sign(diff)}))
    t = pd.concat(out, ignore_index=True)
    return t[(t.res != 0) & (t.vol != "nan")]


def search(t):
    tr, te = t[t.time < SPLIT], t[t.time >= SPLIT]
    rows = []
    for k in (1, 2, 3):
        for dims in itertools.combinations(DIMS, k):
            dims = list(dims)
            a = tr.groupby(dims).res.agg(n="size", w=lambda r: (r > 0).sum())
            b = te.groupby(dims).res.agg(n="size", w=lambda r: (r > 0).sum())
            a = a[a.n >= MIN_TRAIN]
            j = a.join(b, rsuffix="_test", how="left").fillna(0)
            for key, r in j.iterrows():
                key = key if isinstance(key, tuple) else (key,)
                rows.append({"filter": " + ".join(f"{v}" for v in key), "n_train": int(r.n),
                             "wr_train": r.w / r.n, "n_test": int(r.n_test),
                             "wr_test": r.w_test / r.n_test if r.n_test else np.nan})
    return pd.DataFrame(rows).sort_values("wr_train", ascending=False)


def report(label, t):
    s = search(t)
    ok = s[s.n_test >= MIN_TEST]
    print(f"\n######## {label} ########")
    print(f"ไม้ทั้งหมด {len(t):,} | ชุดตัวกรองที่ลอง {len(s):,}")
    print(f"ชุดที่ train ≥ 53%: {(s.wr_train >= BREAKEVEN).sum()}  | "
          f"ชุดที่ทั้ง train และ test ≥ 53%: {((ok.wr_train >= BREAKEVEN) & (ok.wr_test >= BREAKEVEN)).sum()}")
    top = ok.head(15).copy()
    for c in ("wr_train", "wr_test"):
        top[c] = top[c].map("{:.1%}".format)
    print("\n15 อันดับแรกตาม train -> ผลจริงใน test:")
    print(top.to_string(index=False))
    t20 = ok.head(20)
    print(f"\nเฉลี่ย 20 อันดับแรก: train {t20.wr_train.mean():.1%}  ->  test {t20.wr_test.mean():.1%}")
    return s


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    frames = {p.stem.replace("_M5", ""): enrich(load(p)) for p in sorted(DATA.glob("*_M5.csv"))}
    t = build_trades(frames)
    s = report("เทคนิคจากไฟล์", t)
    s.to_csv("filter_results.csv", index=False, encoding="utf-8-sig")
    report("การทดลองหลอก: กดสุ่ม", placebo(frames, len(t)))
