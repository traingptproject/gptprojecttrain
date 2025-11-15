#!/usr/bin/env python3
import hashlib
from pathlib import Path
import sys

SRC = Path('training_data_1.1m_final.jsonl')
OUT = Path('dataset_read_all_report.txt')

if not SRC.exists():
    print(f"ERROR: {SRC} not found", file=sys.stderr)
    sys.exit(1)

line_count = 0
byte_count = 0
sha_all = hashlib.sha256()
first_line = None
last_line = None
first_sha = None
last_sha = None

with SRC.open('rb') as f:  # อ่านแบบไบต์ ครบทุกอักขระ ไม่เว้นบรรทัด
    for raw in f:
        byte_count += len(raw)
        sha_all.update(raw)
        try:
            s = raw.decode('utf-8', errors='replace')
        except Exception:
            s = raw.decode('utf-8', errors='replace')
        if first_line is None:
            first_line = s
            first_sha = hashlib.sha256(raw).hexdigest()
        last_line = s
        last_sha = hashlib.sha256(raw).hexdigest()
        line_count += 1
        if line_count % 100000 == 0:
            print(f"[READ] lines={line_count} bytes={byte_count}")

with OUT.open('w', encoding='utf-8') as r:
    r.write("=== DATASET FULL READ CONFIRMATION ===\n")
    r.write(f"File: {SRC}\n")
    r.write(f"Total lines: {line_count}\n")
    r.write(f"Total bytes: {byte_count}\n")
    r.write(f"SHA256(all-bytes): {sha_all.hexdigest()}\n")
    r.write(f"First line SHA256: {first_sha}\n")
    r.write(f"Last line SHA256: {last_sha}\n")
    r.write("--- First line (preview) ---\n")
    r.write((first_line or '').strip()[:2000] + "\n")
    r.write("--- Last line (preview) ---\n")
    r.write((last_line or '').strip()[:2000] + "\n")
    r.write("=== END ===\n")

print(f"[DONE] Read all lines successfully. Report -> {OUT}")

