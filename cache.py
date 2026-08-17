"""
cache.py — 跨进程持久化缓存。

面试现场你会跑十几轮 pipeline。没有缓存的话，每轮都要重新调
embedding + LLM，时间和 quota 全烧在重复调用上。

设计要点：
- sqlite 落盘，进程重启后仍然命中（改 prompt 才失效）
- key = sha256(namespace + payload)，payload 里带上 model / prompt_version，
  所以改了 prompt 会自动 miss，不会拿到脏结果
- 记录 hit/miss 统计，面试时可以直接报"缓存命中率 87%"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _stable_key(namespace: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{namespace}::{blob}".encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4)}


class DiskCache:
    def __init__(self, path: str = "cache.sqlite3", enabled: bool = True):
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "  k TEXT PRIMARY KEY,"
            "  ns TEXT,"
            "  v TEXT,"
            "  created_at REAL DEFAULT (strftime('%s','now'))"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON kv(ns)")
        self._conn.commit()

    def get(self, namespace: str, payload: Any) -> Optional[Any]:
        if not self.enabled:
            return None
        k = _stable_key(namespace, payload)
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return json.loads(row[0])

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        k = _stable_key(namespace, payload)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, ns, v) VALUES (?, ?, ?)",
                (k, namespace, json.dumps(value, ensure_ascii=False, default=str)),
            )
            self._conn.commit()

    async def aget_or_set(
        self,
        namespace: str,
        payload: Any,
        producer: Callable[[], "asyncio.Future"],
    ) -> Any:
        """异步 get-or-compute。producer 是一个无参 async callable。"""
        cached = self.get(namespace, payload)
        if cached is not None:
            return cached
        value = await producer()
        self.set(namespace, payload, value)
        return value

    def purge(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace:
                cur = self._conn.execute("DELETE FROM kv WHERE ns = ?", (namespace,))
            else:
                cur = self._conn.execute("DELETE FROM kv")
            self._conn.commit()
            return cur.rowcount
