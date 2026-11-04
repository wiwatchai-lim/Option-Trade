"""พิสูจน์: ค้นหาชุดเงื่อนไขที่ชนะ 100% ในข้อมูลช่วงแรก แล้วดูผลจริงในช่วงหลัง"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtest import DATA, load
from filter_search import DIMS, SPLIT, build_trades, enrich, placebo

sys.stdout.reconfigure(encoding="utf-8")


def perfect(t, min_n=8):
    tr, te = t[t.time < SPLIT], t[t.time >= SPLIT]
    rows = []
    for k in (2, 3, 4):
        for dims in itertools.combinations(DIMS, k):
            dims = list(dims)
            a = tr.groupby(dims).res.agg(n="size", w=lambda r: (r > 0).sum())
            a = a[(a.n >= min_n) & (a.w == a.n)]
            if a.empty:
                continue
            b = te.groupby(dims).res.agg(n="size", w=lambda r: (r > 0).sum())
            for key, r in a.join(b, rsuffix="_t", how="left").fillna(0).iterrows():
                rows.append({"เงื่อนไข": " + ".join(map(str, key)), "ช่วงแรก": f"{int(r.w)}/{int(r.n)} = 100%",
                             "n_t": int(r.n_t), "w_t": int(r.w_t)})
    return pd.DataFrame(rows)


frames = {p.stem.replace("_M5", ""): enrich(load(p)) for p in sorted(DATA.glob("*_M5.csv"))}
for label, t in (("สูตรจากไฟล์", build_trades(frames)), ("กดสุ่ม (โยนเหรียญ)", None)):
    if t is None:
        t = placebo(frames, 127_725)
    p = perfect(t)
    print(f"\n##### {label}: เจอชุดที่ชนะ 100% ในช่วงแรก {len(p)} ชุด #####")
    if p.empty:
        continue
    tested = p[p.n_t > 0]
    if len(tested):
        wr = tested.w_t.sum() / tested.n_t.sum()
        print(f"เอาไปใช้ช่วงหลัง: ชนะ {tested.w_t.sum()} จาก {tested.n_t.sum()} ไม้ = {wr:.1%}")
        tested = tested.assign(ช่วงหลัง=lambda d: d.w_t.astype(str) + "/" + d.n_t.astype(str) + " = "
                               + (d.w_t / d.n_t).map("{:.0%}".format))
        print(tested.sort_values("n_t", ascending=False).head(10)[["เงื่อนไข", "ช่วงแรก", "ช่วงหลัง"]]
              .to_string(index=False))
