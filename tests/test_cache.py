from core.cache import QueryCache, make_cache_key, normalize_query


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  What   is BVLOS? ") == "what is bvlos?"


def test_cache_key_ignores_case_whitespace_and_flag_order():
    k1 = make_cache_key("What is BVLOS?", use_hybrid=True, final_k=5)
    k2 = make_cache_key("what is bvlos?  ", final_k=5, use_hybrid=True)
    assert k1 == k2


def test_cache_key_differs_when_flags_differ():
    assert make_cache_key("q", use_hybrid=True) != make_cache_key("q", use_hybrid=False)


def test_cache_get_set_and_stats():
    cache = QueryCache(max_size=4)
    key = make_cache_key("q")

    assert cache.get(key) is None
    cache.set(key, "answer")

    assert cache.get(key) == "answer"
    assert cache.stats() == {"size": 1, "hits": 1, "misses": 1, "hit_rate": 0.5}


def test_cache_evicts_least_recently_used_entry():
    cache = QueryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # "a" is now most recently used

    cache.set("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_cache_clear_resets_entries_and_counters():
    cache = QueryCache()
    cache.set("a", 1)
    cache.get("a")

    cache.clear()

    assert cache.stats() == {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
