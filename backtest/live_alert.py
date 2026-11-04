"""แจ้งเตือนสัญญาณ 6 เทคนิคจากไฟล์ "เทคนิค สคลิปนำเทรด.docx" แบบเรียลไทม์ (กราฟ 5 นาที, ข้อมูลจาก MT5 ในเครื่อง)

วิธีใช้:  python live_alert.py            (ทุกเทคนิค)
         python live_alert.py 1 2 4      (เลือกเฉพาะเทคนิคหมายเลข 1, 2, 4)
กด Ctrl+C เพื่อหยุด

ทุกสัญญาณและผลแพ้/ชนะ (ราคาปิดแท่งถัดไปเทียบราคาเปิด) ถูกบันทึกใน live_log.csv
หมายเหตุ: ราคา MT5 อาจต่างจากเว็บ Binary เล็กน้อย และไม่รองรับคู่ OTC
"""
import csv
import ctypes
import sys
import threading
import time
import winsound
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtest import TECHS, prepare

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY", "USDJPY", "NZDUSD", "USDCHF", "EURGBP"]
LOG = Path(__file__).parent / "live_log.csv"
HISTORY_BARS = 6000  # พอสำหรับคำนวณเทรนด์ใหญ่ย้อนหลัง 20 วัน
POLL_SEC = 2
# Win rate จากการทดสอบย้อนหลัง ~16 เดือน (จุดคุ้มทุน ~52-53%)
BACKTEST_WR = {"1": 48.5, "2": 49.2, "3": 47.4, "4": 49.1, "5": 48.8, "6": 50.0}
THAI = timezone(timedelta(hours=7))

BASE = [t for t in TECHS if t[4]]


def server_offset_hours():
    tick = mt5.symbol_info_tick("EURUSD")
    return round((tick.time - time.time()) / 3600)


def to_thai(server_ts: pd.Timestamp, offset: int) -> datetime:
    return (server_ts.to_pydatetime() - timedelta(hours=offset)).replace(tzinfo=timezone.utc).astimezone(THAI)


def setup_a(df: pd.DataFrame, i: int, thai_hour: int) -> bool:
    """แบบ A: เขียวกลืนกินแดง + กราฟใหญ่ขาขึ้นแรง (SMA240 ชันติด 20% บน) + 07-14 น. เวลาไทย"""
    sma = df.close.rolling(240).mean()
    slope = (sma - sma.shift(12)) / df.atr
    q = slope.rolling(288 * 20, min_periods=288 * 5).rank(pct=True)
    return 7 <= thai_hour < 14 and bool(q.iloc[i] > 0.8)


def popup(text: str):
    threading.Thread(target=lambda: ctypes.windll.user32.MessageBoxW(0, text, "สัญญาณเทรด", 0x40000 | 0x40),
                     daemon=True).start()


def log_row(row: dict):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def fetch(pair):
    r = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M5, 0, HISTORY_BARS)
    if r is None or len(r) < 300:
        return None
    df = pd.DataFrame(r)[["time", "open", "high", "low", "close"]]
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return prepare(df)


def main():
    wanted = set(sys.argv[1:]) or set(BACKTEST_WR)
    techs = [t for t in BASE if t[0][0] in wanted]
    if not mt5.initialize():
        sys.exit(f"เชื่อม MT5 ไม่ได้: {mt5.last_error()} (เปิดโปรแกรม MetaTrader 5 ก่อน)")
    for p in PAIRS:
        mt5.symbol_select(p, True)
    offset = server_offset_hours()
    print(f"เริ่มเฝ้ากราฟ 5 นาที {len(PAIRS)} คู่เงิน | เทคนิค: {', '.join(t[0] for t in techs)}")
    print("Win rate ย้อนหลังของทุกเทคนิคอยู่ที่ 47-50% (ต่ำกว่าจุดคุ้มทุน 52-53%) ใช้เพื่อทดลอง/บัญชี Demo")
    print("กด Ctrl+C เพื่อหยุด\n")

    last_bar = {}  # pair -> เวลาแท่งที่ปิดล่าสุดที่เช็คแล้ว
    pending = []   # สัญญาณที่รอผล
    while True:
        try:
            for pair in PAIRS:
                head = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M5, 0, 2)  # เช็คเร็วว่ามีแท่งใหม่ปิดไหม
                if head is None or len(head) < 2:
                    continue
                if last_bar.get(pair) == pd.Timestamp(head[0]["time"], unit="s"):
                    continue
                df = fetch(pair)
                if df is None:
                    continue
                i = len(df) - 2  # แท่งที่ปิดแล้วล่าสุด (แท่งสุดท้ายยังวิ่งอยู่)
                bar_t = df.time.iloc[i]
                if last_bar.get(pair) == bar_t:
                    continue
                first_run = pair not in last_bar
                last_bar[pair] = bar_t

                # อัปเดตผลของสัญญาณที่แท่งเทรดเพิ่งปิด
                for s in [s for s in pending if s["pair"] == pair and s["trade_bar"] == bar_t]:
                    o, c = df.open.iloc[i], df.close.iloc[i]
                    diff = (o - c) if s["dir"] == "PUT" else (c - o)
                    res = "ชนะ" if diff > 0 else "แพ้" if diff < 0 else "เสมอ"
                    print(f"   ผล {s['pair']} {s['tech']} ({s['thai_time']}): {res}")
                    log_row({**{k: s[k] for k in ("thai_time", "pair", "tech", "dir", "setup_a", "danger_time")},
                             "entry": o, "close": c, "result": res})
                    pending.remove(s)
                if first_run:
                    continue

                thai = to_thai(bar_t + timedelta(minutes=5), offset)  # เวลาที่ต้องเข้า = ต้นแท่งถัดไป
                for name, bear, _, fn, _ in techs:
                    if not bool(fn(df).fillna(False).iloc[i]):
                        continue
                    d = "PUT" if bear else "CALL"
                    star = name.startswith("2") and setup_a(df, i, thai.hour)
                    danger = 4 <= thai.hour < 7
                    msg = (f"{'★ ' if star else ''}{pair}  {d}  เข้าตอน {thai:%H:%M} น. (หมดอายุ 5 นาที)\n"
                           f"เทคนิค: {name}\n"
                           f"Win rate ย้อนหลัง: {54.3 if star else BACKTEST_WR[name[0]]}%"
                           f"{'  (ตรงแบบ A)' if star else ''}"
                           f"{chr(10) + '⚠ ช่วง 04-07 น. ผลแย่ที่สุด ไม่แนะนำให้เข้า' if danger else ''}")
                    print(f"[{datetime.now(THAI):%H:%M:%S}] " + msg.replace("\n", " | "))
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    popup(msg)
                    pending.append({"pair": pair, "tech": name, "dir": d, "setup_a": star, "danger_time": danger,
                                    "thai_time": f"{thai:%Y-%m-%d %H:%M}",
                                    "trade_bar": bar_t + timedelta(minutes=5)})
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("\nหยุดแล้ว")
            break
        except Exception as e:  # MT5 หลุดชั่วคราว ฯลฯ
            print(f"ผิดพลาด: {e} — ลองใหม่ใน 10 วินาที")
            time.sleep(10)
            mt5.initialize()
    mt5.shutdown()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
