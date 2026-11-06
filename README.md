# Wiwat_Dev — ทดสอบเทคนิคเทรด Binary (แท่งเทียน + SMA5)

> **Private repo** — มีเอกสารเทคนิคส่วนตัว ห้ามเปลี่ยนเป็น Public

## สรุปผล
ทดสอบ 6 เทคนิคจาก `เทคนิค สคลิปนำเทรด.docx` กับกราฟ 9 คู่เงินจริง (ข้อมูล MT5 ~16 เดือน, หมดอายุ 1 แท่ง)

- ทุกเทคนิค ทุก Timeframe (M1–H1) ทุกสภาพตลาด: Win rate ~46–51% ต่ำกว่าจุดคุ้มทุน Binary (~52–53%)
- ค้นหา 1,320 วิธี (ตีความกฎ/สวนสัญญาณ/ตัวกรอง/AI) แล้วตรวจกับข้อมูลที่ล็อกไว้ 8 เดือน: ไม่มีวิธีไหนผ่าน
- ช่วง 04–07 น. เวลาไทย ผลแย่ที่สุด (~46%)

## โครงสร้าง
| ไฟล์ | หน้าที่ |
|---|---|
| `backtest/fetch_data.py [M5]` | ดึงกราฟจาก MT5 → `backtest/data/` |
| `backtest/backtest.py` | กฎ 6 เทคนิค + ทดสอบหลายการตีความ |
| `backtest/check_size.py`, `daily.py`, `filter_search.py`, `pick3.py`, `find_100.py`, `regimes.py`, `check_top3.py`, `search100.py` | การวิเคราะห์แต่ละรอบ |
| `backtest/live_alert.py`, `เปิดแจ้งเตือน.bat` | แจ้งเตือนสัญญาณเรียลไทม์จาก MT5 + จดผลลง `live_log.csv` |
| `backtest/dashboard.py`, `dashboard.html`, `เปิดหน้าเว็บสัญญาณ.bat` | หน้าเว็บดูสัญญาณเรียลไทม์ http://127.0.0.1:8765 — กราฟแท่ง 5 นาที, ป๊อปอัปสัญญาณ, สถิติ/กำไรจำลอง, ตั้งค่าเทคนิค · `เปิดหน้าเว็บสัญญาณ (มือถือ).bat` (หรือ `--lan`) ให้มือถือใน Wi-Fi เดียวกันเข้าดูได้ |
| `iq/iq_test.py` | ทดสอบดึงแท่งเทียนจาก IQ Option (บัญชี Demo) |

## ติดตั้งใหม่
```powershell
pip install MetaTrader5 pandas numpy scikit-learn
python backtest\fetch_data.py M5        # ต้องเปิด MetaTrader 5 ไว้

# ส่วน IQ Option
python -m venv iqenv
iqenv\Scripts\python -m pip install requests certifi "websocket-client==0.56" pandas numpy
git clone https://github.com/iqoptionapi/iqoptionapi.git vendor\iqoptionapi
git -C vendor\iqoptionapi apply ..\..\iq\iqoptionapi_security.patch   # เปิดการตรวจใบรับรอง HTTPS
# ใส่อีเมล/รหัสผ่านใน iq\iq_login.txt (ไฟล์นี้ถูก .gitignore ไว้)
```
