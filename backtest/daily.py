"""Win rate รายวัน ของ 6 เทคนิค (แบบตีความใกล้ไฟล์ที่สุด, เข้าตอนเปิดแท่งถัดไป)
สมมติลงไม้ละ $10, ชนะได้ 90%, แพ้เสียทั้งหมด, เสมอคืนเงิน (วันอิงเวลาเซิร์ฟเวอร์ MT5)
"""
import sys

import numpy as np
import pandas as pd

from backtest import DATA, TECHS, load

sys.stdout.reconfigure(encoding="utf-8")
STAKE, PAYOUT = 10, 0.90


def trades(df, sig, bear, tech, pair):
    d = df[sig.fillna(False) & df.valid]
    diff = (d.n_open - d.n_close) if bear else (d.n_close - d.n_open)
    res = np.sign(diff)
    return pd.DataFrame({"time": d.time, "date": d.time.dt.date, "tech": tech, "pair": pair, "res": res})


frames = {p.stem.replace("_M5", ""): load(p) for p in sorted(DATA.glob("*_M5.csv"))}
all_t = pd.concat([trades(df, fn(df), bear, name, pair)
                   for name, bear, _, fn, base in TECHS if base
                   for pair, df in frames.items()])
all_t["pnl"] = np.select([all_t.res > 0, all_t.res < 0], [STAKE * PAYOUT, -STAKE], 0.0)


def daily(t):
    g = t.groupby("date")
    d = pd.DataFrame({"trades": g.res.apply(lambda r: (r != 0).sum()),
                      "wins": g.res.apply(lambda r: (r > 0).sum()), "pnl": g.pnl.sum()})
    d = d[d.trades > 0]
    d["wr"] = d.wins / d.trades
    return d


def summary(name, t):
    d = daily(t)
    print(f"{name:26s} {len(d):5d} {d.trades.mean():8.1f} {d.wr.mean():8.1%} {d.wr.quantile(.1):7.1%} "
          f"{d.wr.quantile(.9):7.1%} {(d.wr >= 0.53).mean():8.1%} {(d.pnl > 0).mean():8.1%} {d.pnl.mean():+9.2f}")


print(f"{'เทคนิค':26s} {'วัน':>5} {'ไม้/วัน':>8} {'WRเฉลี่ย':>8} {'วันแย่10%':>7} {'วันดี10%':>7} "
      f"{'วัน≥53%':>8} {'วันกำไร':>8} {'$/วัน':>9}")
for name in dict.fromkeys(all_t.tech):
    summary(name, all_t[all_t.tech == name])
summary("รวมทุกเทคนิค", all_t)
excl = all_t[~all_t.tech.str.startswith(("5", "6"))]
summary("รวมเทคนิค 1-4 (ไม้น้อยกว่า)", excl)

print("\n=== 14 วันล่าสุด (รวมทุกเทคนิค, ทุกคู่เงิน) ===")
d = daily(all_t).tail(14)
d["wr"] = d.wr.map("{:.1%}".format)
d["pnl"] = d.pnl.map("${:+,.0f}".format)
print(d.to_string())

print("\n=== 14 วันล่าสุด (เทคนิค 1-4 เท่านั้น) ===")
d = daily(excl).tail(14)
d["wr"] = d.wr.map("{:.1%}".format)
d["pnl"] = d.pnl.map("${:+,.0f}".format)
print(d.to_string())
