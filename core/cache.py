"""In-memory LRU cache for full QueryPipeline.ask() responses, keyed on the
normalized query plus every flag/filter that affects the answer - so
"What is BVLOS?" with hybrid=True and jurisdiction=india caches separately
from the same question with different settings. Process-lifetime only (no
persistence across restarts) - see docs/superpowers/specs/2026-07-08-advanced-rag-upgrade-design.md
for why that's sufficient here.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

DEFAULT_MAX_SIZE = 128


def normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def make_cache_key(query: str, **flags: Any) -> tuple:
    normalized = normalize_query(query)
    return (normalized, tuple(sorted(flags.items(), key=lambda kv: kv[0])))


class QueryCache:
    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        self.max_size = max_size
        self._store: OrderedDict[tuple, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Any | None:
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: tuple, value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


if __name__ == "__main__":
    cache = QueryCache(max_size=2)
    k1 = make_cache_key("What is BVLOS?", use_hybrid=True)
    k2 = make_cache_key("what is bvlos?  ", use_hybrid=True)  # normalizes to same key as k1
    k3 = make_cache_key("What is BVLOS?", use_hybrid=False)  # different flags -> different key

    assert k1 == k2
    assert k1 != k3

    assert cache.get(k1) is None
    cache.set(k1, "answer A")
    assert cache.get(k2) == "answer A"
    cache.set(k3, "answer B")
    print(cache.stats())
    print("OK")
