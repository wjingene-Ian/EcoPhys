#!/usr/bin/env python3
"""Verify the included data against the SHA-256 manifest. No third-party dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root / 'data_manifest.json').read_text())
errors = []
for record in records:
    path = root / record['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        errors.append(record['path'])
if errors:
    raise SystemExit('数据校验失败：\n' + '\n'.join(errors))
print(f'数据校验通过：{len(records)}个文件，全部SHA-256一致。')
