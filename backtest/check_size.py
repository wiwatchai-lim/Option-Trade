"""ทดสอบเทคนิค 1-2 (กลืนกิน) เมื่อกำหนด "ขนาดพอๆ กัน" ตั้งแต่ 75% ขึ้นไป"""
import itertools
import sys

from backtest import DATA, LOCS, load, outcome, sig_engulf, wilson

sys.stdout.reconfigure(encoding="utf-8")
frames = [load(p) for p in sorted(DATA.glob("*_M5.csv"))]
ENTRY = {"open": "เปิดแท่ง", "sma": "ย่อหาSMA"}
for bear, name in ((True, "1 แดงกลืนกินเขียว PUT"), (False, "2 เขียวกลืนกินแดง CALL")):
    print(f"\n== {name} ==")
    print(f"{'ขนาด':>6}  {'แนวรับ/ต้าน':<20} {'เข้า':<9} {'ไม้':>6} {'Winrate':>8}  ช่วง95%")
    for eq in (0.75, 0.8, 0.85, 0.9):
        for (ln, lv), entry in itertools.product(LOCS.items(), ENTRY):
            W = L = 0
            for df in frames:
                w, l, _ = outcome(df, sig_engulf(df, bear, eq, lv), bear, entry)
                W, L = W + w, L + l
            lo, hi = wilson(W, W + L)
            print(f">={eq:.0%}  {ln:<20} {ENTRY[entry]:<9} {W + L:6d} {W / (W + L):8.2%}  {lo:.1%}-{hi:.1%}")
