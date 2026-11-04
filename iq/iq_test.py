"""ทดสอบเชื่อม IQ Option (บัญชี Demo/PRACTICE) แล้วดึงแท่งเทียน 5 นาทีของคู่เงินจริงและ OTC

ใส่อีเมล/รหัสผ่านใน iq_login.txt (ไฟล์อยู่ในเครื่องนี้เท่านั้น) แล้วรัน:
    ..\\iqenv\\Scripts\\python iq_test.py
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "iqoptionapi"))
from iqoptionapi.stable_api import IQ_Option  # noqa: E402

THAI = timezone(timedelta(hours=7))


def read_login():
    cfg = dict(line.split("=", 1) for line in (Path(__file__).parent / "iq_login.txt").read_text(
        encoding="utf-8").splitlines() if "=" in line)
    email, pw = cfg.get("email", "").strip(), cfg.get("password", "").strip()
    if not email or not pw:
        sys.exit("ยังไม่ได้ใส่อีเมล/รหัสผ่านใน iq_login.txt")
    return email, pw


def main():
    email, pw = read_login()
    iq = IQ_Option(email, pw, active_account_type="PRACTICE")
    ok, reason = iq.connect()
    if not ok and reason == "2FA":
        code = input("IQ Option ส่งรหัสยืนยัน (2FA) ไปที่มือถือ/อีเมล ใส่รหัส: ").strip()
        ok, reason = iq.connect_2fa(code)
    if not ok:
        sys.exit(f"เชื่อมไม่สำเร็จ: {reason}")
    iq.change_balance("PRACTICE")
    print("เชื่อมสำเร็จ (บัญชี Demo) ยอดเงิน Demo:", iq.get_balance())

    for asset in ("EURUSD", "EURUSD-OTC", "USDJPY-OTC"):
        candles = iq.get_candles(asset, 300, 6, time.time())
        if not candles:
            print(f"\n{asset}: ไม่มีข้อมูล (ตลาดปิด หรือชื่อไม่ตรง)")
            continue
        print(f"\n{asset} (แท่ง 5 นาทีล่าสุด, เวลาไทย):")
        for c in candles:
            t = datetime.fromtimestamp(c["from"], THAI)
            print(f"  {t:%H:%M}  O {c['open']:.6f}  H {c['max']:.6f}  L {c['min']:.6f}  C {c['close']:.6f}")
    iq.api.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
