"""ดึงกราฟ 5 นาทีย้อนหลังจาก MetaTrader 5 แล้วบันทึกเป็น CSV ใน backtest/data"""
import sys
import time
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY", "USDJPY", "NZDUSD", "USDCHF", "EURGBP"]
BARS = 99_000  # MT5 จำกัดที่ maxbars (100,000)
OUT = Path(__file__).parent / "data"


TF = sys.argv[1] if len(sys.argv) > 1 else "M5"  # เช่น M1, M5, M15, M30, H1


def fetch(symbol: str):
    for attempt in range(6):
        if not mt5.initialize():
            time.sleep(3)
            continue
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{TF}"), 0, BARS)
        if rates is not None and len(rates) > 0:
            return rates
        print(f"  {symbol}: retry {attempt + 1} ({mt5.last_error()})")
        mt5.shutdown()
        time.sleep(5)
    return None


def main():
    OUT.mkdir(exist_ok=True)
    for s in PAIRS:
        rates = fetch(s)
        if rates is None:
            print(f"{s}: FAILED")
            continue
        df = pd.DataFrame(rates)[["time", "open", "high", "low", "close"]]
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.to_csv(OUT / f"{s}_{TF}.csv", index=False)
        print(f"{s}: {len(df)} bars  {df.time.min()} -> {df.time.max()}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
