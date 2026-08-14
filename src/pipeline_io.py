"""Utility helpers for JSON/NDJSON IO and checkpoint persistence."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def save_json_atomic(path: str, payload: Any) -> None:
    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path) or ".", encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def append_ndjson(path: str, record: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=True))
        outfile.write("\n")


def iter_ndjson(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def ndjson_file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def collect_values_from_ndjson(path: str, key: str) -> set[str]:
    values: set[str] = set()
    for row in iter_ndjson(path):
        value = row.get(key)
        if value:
            values.add(value)
    return values


def collect_match_ids_from_matches_ndjson(path: str) -> set[str]:
    match_ids: set[str] = set()
    for row in iter_ndjson(path):
        metadata = row.get("metadata", {})
        match_id = metadata.get("matchId")
        if match_id:
            match_ids.add(match_id)
    return match_ids


def append_many_ndjson(path: str, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for record in records:
        append_ndjson(path, record)
        count += 1
    return count
