"""ทดสอบย้อนหลัง 6 เทคนิคจากไฟล์ "เทคนิค สคลิปนำเทรด.docx" แบบ Binary หมดอายุ 1 แท่ง (5 นาที)

แต่ละเทคนิคถูกตีความหลายแบบ (variant) เพราะคำในไฟล์ เช่น "ขนาดพอๆกัน", "ไส้ยาว", "แนวรับ"
ไม่ได้ระบุเป็นตัวเลข ผลลัพธ์จึงรายงานทั้งแบบที่ใกล้ไฟล์ที่สุด (baseline) และช่วงของทุกแบบ

ผลแพ้ชนะ: CALL ชนะเมื่อราคาปิดแท่งถัดไป > ราคาเข้า, PUT ชนะเมื่อ < ราคาเข้า, เท่ากัน = เสมอ (คืนเงิน)
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
BAR = pd.Timedelta(minutes=5)
WARMUP = 30  # ต้องมีแท่งต่อเนื่องย้อนหลังอย่างน้อยเท่านี้ (สำหรับ SMA/แนวรับแนวต้าน)


def load(path: Path) -> pd.DataFrame:
    return prepare(pd.read_csv(path, parse_dates=["time"]))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ SMA/ATR/ขนาดแท่ง ฯลฯ (ใช้ร่วมกันทั้งทดสอบย้อนหลังและโปรแกรมแจ้งเตือน)"""
    df = df.reset_index(drop=True)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    df["sma5"] = df.close.rolling(5).mean()
    df["sma20"] = df.close.rolling(20).mean()
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    df["atr"] = pd.Series(tr).rolling(14).mean()
    df["body"] = abs(c - o)
    df["upper"] = h - np.maximum(o, c)
    df["lower"] = np.minimum(o, c) - l
    df["green"] = c > o
    df["red"] = c < o
    # ความต่อเนื่อง: นับจำนวนแท่งต่อเนื่องก่อนหน้า (ไม่มีช่องว่างเวลา)
    contig = df.time.diff().eq(BAR)
    run = contig.groupby((~contig).cumsum()).cumsum()
    df["run"] = run
    # แท่งถัดไป (แท่งที่เทรด)
    for k in ("open", "high", "low", "close"):
        df[f"n_{k}"] = df[k].shift(-1)
    df["next_ok"] = contig.shift(-1, fill_value=False)
    df["valid"] = (df.run >= WARMUP) & df.next_ok
    return df


def near_resistance(df, i_hi, n, tol):
    """ยอดของแท่งสัญญาณอยู่ใกล้/เหนือจุดสูงสุดของ n แท่งก่อนหน้า (แนวแท่งเทียนเดิมซ้ายมือ)"""
    prior_max = df.high.shift(2).rolling(n).max()
    return i_hi >= prior_max - tol * df.atr


def near_support(df, i_lo, n, tol):
    prior_min = df.low.shift(2).rolling(n).min()
    return i_lo <= prior_min + tol * df.atr


def trend(df, kind, up):
    slope = df.sma5 - df.sma5.shift(3)
    s_up, s_dn = slope > 0, slope < 0
    m_up, m_dn = df.sma5 > df.sma20, df.sma5 < df.sma20
    if kind == "slope":
        return s_up if up else s_dn
    if kind == "sma20":
        return m_up if up else m_dn
    return (s_up & m_up) if up else (s_dn & m_dn)  # both


# ---------------------------------------------------------------- สัญญาณแต่ละเทคนิค
def sig_engulf(df, bear, eq, loc):
    prev_ok = df.green.shift(1) if bear else df.red.shift(1)
    cur_ok = df.red if bear else df.green
    b1, b2 = df.body.shift(1), df.body
    size_ok = (np.minimum(b1, b2) / np.maximum(b1, b2).replace(0, np.nan)) >= eq
    if bear:  # เนื้อแดงทับ SMA5 ลงมา
        cross = (df.open > df.sma5) & (df.close < df.sma5)
    else:
        cross = (df.open < df.sma5) & (df.close > df.sma5)
    s = prev_ok.astype(bool) & cur_ok & size_ok & cross
    return s & location(df, bear, loc, pair=True)


def sig_wick(df, bear, wick, loc):
    if bear:  # แดงดาวตก: ไส้บนยาว เนื้อทับ SMA5 ลงมา
        s = df.red & (df.upper >= wick * df.body) & (df.open > df.sma5) & (df.close < df.sma5)
    else:  # เขียวบั้งไฟ: ไส้ล่างยาว เนื้อทับ SMA5 ขึ้นไป หรืออยู่บนเส้น
        s = df.green & (df.lower >= wick * df.body) & (df.close > df.sma5) & (df.low <= df.sma5)
    s = s & (df.body > 0)
    return s & location(df, bear, loc, pair=False)


def sig_pullback(df, bear, tkind):
    if bear:  # เขียวใต้เส้นในเทรนลง -> PUT
        s = df.green & (df.close < df.sma5) & trend(df, tkind, up=False)
    else:  # แดงบนเส้นในเทรนขึ้น -> CALL
        s = df.red & (df.close > df.sma5) & trend(df, tkind, up=True)
    return s


def location(df, bear, loc, pair):
    if loc is None:
        return pd.Series(True, index=df.index)
    n, tol = loc
    if bear:
        hi = np.maximum(df.high, df.high.shift(1)) if pair else df.high
        return near_resistance(df, hi, n, tol)
    lo = np.minimum(df.low, df.low.shift(1)) if pair else df.low
    return near_support(df, lo, n, tol)


# ---------------------------------------------------------------- ผลการเทรด
def outcome(df, sig, bear, entry):
    d = df[sig.fillna(False) & df.valid]
    if entry == "open":  # เข้าตอนเปิดแท่งถัดไป
        price = d.n_open
        taken = pd.Series(True, index=d.index)
    else:  # รอราคาย่อกลับมาหา SMA5 (ตามคำแนะนำในรูป)
        if bear:
            price = np.maximum(d.n_open, d.sma5)
            taken = d.n_high >= price
        else:
            price = np.minimum(d.n_open, d.sma5)
            taken = d.n_low <= price
    d, price = d[taken], price[taken]
    diff = (price - d.n_close) if bear else (d.n_close - price)
    return int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())


LOCS = {"ไม่ใช้": None, "แนว20แท่ง±0.5ATR": (20, 0.5), "แนว12แท่ง±0.25ATR": (12, 0.25), "แนว40แท่ง±1ATR": (40, 1.0)}
ENTRIES = {"open": "เปิดแท่งถัดไป", "sma": "รอย่อหา SMA5"}

TECHS = []  # (ชื่อเทคนิค, ทิศ, รายละเอียด variant, ฟังก์ชันสัญญาณ, baseline?)
for bear, name in ((True, "1 แดงกลืนกินเขียว (PUT)"), (False, "2 เขียวกลืนกินแดง (CALL)")):
    for eq, (ln, lv) in itertools.product((0.5, 0.6, 0.75), LOCS.items()):
        TECHS.append((name, bear, f"ขนาด≥{eq:.0%} | {ln}", lambda df, b=bear, e=eq, l=lv: sig_engulf(df, b, e, l),
                      eq == 0.6 and ln == "แนว20แท่ง±0.5ATR"))
for bear, name in ((True, "3 แดงดาวตก (PUT)"), (False, "4 เขียวบั้งไฟ (CALL)")):
    for w, (ln, lv) in itertools.product((1.0, 1.5, 2.0), LOCS.items()):
        TECHS.append((name, bear, f"ไส้≥{w}xเนื้อ | {ln}", lambda df, b=bear, w=w, l=lv: sig_wick(df, b, w, l),
                      w == 1.5 and ln == "แนว20แท่ง±0.5ATR"))
for bear, name in ((True, "5 เขียวใต้เส้น (PUT)"), (False, "6 แดงบนเส้น (CALL)")):
    for tk, tl in (("slope", "เทรน=SMA5ชัน"), ("sma20", "เทรน=SMA5เทียบSMA20"), ("both", "เทรน=ทั้งสองแบบ")):
        TECHS.append((name, bear, tl, lambda df, b=bear, t=tk: sig_pullback(df, b, t), tk == "both"))


def wilson(w, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    frames = {p.stem.replace("_M5", ""): load(p) for p in sorted(DATA.glob("*_M5.csv"))}
    rows = []
    for (name, bear, var, fn, base), entry in itertools.product(TECHS, ENTRIES):
        for half in ("all", "first", "second"):
            W = L = D = 0
            for pair, df in frames.items():
                mid = len(df) // 2
                sub_sig = fn(df)
                if half == "first":
                    sub_sig = sub_sig & (df.index < mid)
                elif half == "second":
                    sub_sig = sub_sig & (df.index >= mid)
                w, l, d = outcome(df, sub_sig, bear, entry)
                W, L, D = W + w, L + l, D + d
            rows.append(dict(tech=name, variant=var, entry=ENTRIES[entry], baseline=base, half=half,
                             wins=W, losses=L, draws=D))
    res = pd.DataFrame(rows)
    res["trades"] = res.wins + res.losses
    res["winrate"] = res.wins / res.trades
    res[["ci_lo", "ci_hi"]] = res.apply(lambda r: pd.Series(wilson(r.wins, r.trades)), axis=1)
    res.to_csv(Path(__file__).parent / "results_all.csv", index=False, encoding="utf-8-sig")

    # ฐานเปรียบเทียบ: กด CALL/PUT สุ่มทุกแท่ง
    W = L = 0
    for df in frames.values():
        d = df[df.valid]
        W += int((d.n_close > d.n_open).sum()); L += int((d.n_close < d.n_open).sum())
    print(f"ฐานเปรียบเทียบ (กด CALL ทุกแท่ง): winrate {W / (W + L):.2%} จาก {W + L:,} แท่ง\n")

    allr = res[res.half == "all"]
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 500)
    print("=== baseline (ตีความใกล้ไฟล์ที่สุด) ===")
    b = allr[allr.baseline]
    print(b[["tech", "variant", "entry", "trades", "draws", "winrate", "ci_lo", "ci_hi"]].to_string(index=False))

    print("\n=== ช่วงผลของทุกการตีความ (เฉพาะแบบที่มีเทรด ≥ 200 ครั้ง) ===")
    g = allr[allr.trades >= 200].groupby(["tech", "entry"])
    print(g.agg(variants=("winrate", "size"), min_wr=("winrate", "min"), median_wr=("winrate", "median"),
                max_wr=("winrate", "max"), trades_median=("trades", "median")).to_string())

    print("\n=== แบบที่ดีที่สุดของแต่ละเทคนิค (ครึ่งแรก) แล้วเช็คกับครึ่งหลัง ===")
    first = res[(res.half == "first") & (res.trades >= 100)]
    second = res[res.half == "second"].set_index(["tech", "variant", "entry"])
    for tech, grp in first.groupby("tech"):
        best = grp.sort_values("winrate").iloc[-1]
        s = second.loc[(tech, best.variant, best.entry)]
        print(f"{tech:28s} | {best.variant} | {best.entry}: ครึ่งแรก {best.winrate:.2%} ({best.trades}) "
              f"-> ครึ่งหลัง {s.winrate:.2%} ({s.trades})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
