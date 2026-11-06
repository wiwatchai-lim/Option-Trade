"""หน้าเว็บดูสัญญาณเทรดเรียลไทม์ http://127.0.0.1:8765

วิธีใช้:  python dashboard.py            (ทุกเทคนิค)
         python dashboard.py 1 2 4      (เลือกเฉพาะเทคนิคหมายเลข 1, 2, 4)
         python dashboard.py --lan      (เปิดให้มือถือใน Wi-Fi เดียวกันเข้าดูได้)
ต้องเปิดโปรแกรม MetaTrader 5 ไว้ก่อน กด Ctrl+C เพื่อหยุด
"""
import csv
import json
import socket
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import live_alert
from live_alert import LOG, STATE, STATE_LOCK, THAI

PORT = 8765
PAGE = Path(__file__).parent / "dashboard.html"
_cache = {"key": None, "rows": []}


def history():
    """ผลจริงทั้งหมดที่เคยจดไว้ใน live_log.csv (อ่านใหม่เฉพาะเมื่อไฟล์เปลี่ยน)"""
    if not LOG.exists():
        return []
    s = LOG.stat()
    key = (s.st_mtime_ns, s.st_size)
    if _cache["key"] != key:
        with LOG.open(encoding="utf-8-sig", newline="") as f:
            _cache["rows"] = list(csv.DictReader(f))
        _cache["key"] = key
    return _cache["rows"]


def summarize(rows):
    """สรุปผลจริงทั้งหมดเป็นก้อนเล็ก ๆ ให้หน้าเว็บ (ไม่ต้องส่งทุกแถวทุก 2 วินาที)"""
    code = {"ชนะ": "W", "แพ้": "L", "เสมอ": "D"}
    stats, hours, pairs, days = {}, {}, {}, {}
    seq = []
    for r in rows:
        res = r.get("result", "")
        k = stats.setdefault(r["tech"], {"ชนะ": 0, "แพ้": 0, "เสมอ": 0})
        k[res] = k.get(res, 0) + 1
        idx = {"ชนะ": 0, "แพ้": 1, "เสมอ": 2}.get(res)
        if idx is None:
            continue
        hours.setdefault(r["thai_time"][11:13], [0, 0, 0])[idx] += 1
        pairs.setdefault(r["pair"], [0, 0, 0])[idx] += 1
        days.setdefault(r["thai_time"][:10], [0, 0, 0])[idx] += 1
        seq.append(code[res] + r["tech"][:1])  # เช่น "W5" = ชนะ เทคนิค 5
    today = datetime.now(THAI).strftime("%Y-%m-%d")
    return {"stats": stats, "hours": hours, "pair_results": pairs, "days": days, "seq": seq,
            "today_rows": [r for r in rows if r["thai_time"].startswith(today)]}


def state_json() -> bytes:
    with STATE_LOCK:
        snap = json.loads(json.dumps(STATE, default=str))
    rows = history()
    snap.update(summarize(rows))
    snap["history"] = rows[-200:][::-1]
    snap["history_total"] = len(rows)
    snap["server_time"] = datetime.now(THAI).strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps(snap, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        headers = {}
        if self.path in ("/", "/index.html"):
            body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
        elif self.path.startswith("/state"):
            body, ctype = state_json(), "application/json; charset=utf-8"
        elif self.path.startswith("/live_log.csv"):
            body = LOG.read_bytes() if LOG.exists() else "﻿ไม่มีข้อมูล\n".encode("utf-8")
            ctype = "text/csv; charset=utf-8"
            headers["Content-Disposition"] = f'attachment; filename="live_log_{datetime.now(THAI):%Y%m%d}.csv"'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # หน้าเว็บนี้อ่านอย่างเดียว
        self.send_error(405)

    def log_message(self, *a):
        pass


def run_watcher(wanted):
    try:
        live_alert.main(wanted, web=True)
    except SystemExit as e:  # เช่น เชื่อม MT5 ไม่ได้
        with STATE_LOCK:
            STATE["error"] = str(e)
        print(e)
    except Exception as e:
        with STATE_LOCK:
            STATE["error"] = f"โปรแกรมเฝ้ากราฟหยุดทำงาน: {e}"
        print(STATE["error"])


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # ไม่ได้ส่งข้อมูลจริง แค่ให้ระบบเลือกการ์ดเครือข่าย
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    args = sys.argv[1:]
    lan = "--lan" in args
    wanted = [a for a in args if a != "--lan"]
    threading.Thread(target=run_watcher, args=(wanted,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0" if lan else "127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"หน้าเว็บสัญญาณ: {url}  (กด Ctrl+C เพื่อหยุด)")
    if lan:
        print(f"เปิดบนมือถือ (Wi-Fi เดียวกัน): http://{lan_ip()}:{PORT}")
        print("  ถ้าเข้าไม่ได้ ให้กด 'อนุญาต' ที่หน้าต่าง Windows Firewall")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nหยุดแล้ว")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
