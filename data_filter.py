#!/usr/bin/env python3
import json
from pathlib import Path

SRC = Path('training_data_1.1m_final.jsonl')
DST = Path('training_data_unfiltered.jsonl')

KEEP_COUNT = 0
DROP_COUNT = 0

with SRC.open('r', encoding='utf-8') as fin, DST.open('w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            KEEP_COUNT += 1
        except Exception as e:
            # เก็บเฉพาะข้อมูลที่ parse JSON ได้
            DROP_COUNT += 1
            continue

print(f"Unfiltered dataset written to {DST}")
print(f"Kept: {KEEP_COUNT}  Dropped: {DROP_COUNT} (invalid JSON only)")
