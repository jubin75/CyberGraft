"""Filesystem cache for neuPrint query results."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class QueryCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def query_hash(query: dict) -> str:
        canonical = json.dumps(query, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[list]:
        path = self.cache_dir / f"{key}.json"
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return entry["payload"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, payload, *, source_url: str, dataset: str, query: dict) -> None:
        entry = {
            "source_url": source_url,
            "dataset": dataset,
            "query_hash": key,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        path = self.cache_dir / f"{key}.json"
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
