from cybergraft.data_ingest.cache import QueryCache


def test_cache_round_trip(tmp_path):
    cache = QueryCache(tmp_path / "cache")
    query = {"op": "resolve_neurons", "criteria": {"type": ["GNG232"], "status": "Traced"}}
    key = cache.query_hash(query)
    payload = [{"bodyId": 31018, "type": "GNG232"}]
    cache.put(key, payload, source_url="https://example.com", dataset="male-cns:v1.0", query=query)
    assert cache.get(key) == payload


def test_cache_missing_key_returns_none(tmp_path):
    cache = QueryCache(tmp_path / "cache")
    assert cache.get("does-not-exist") is None


def test_query_hash_is_deterministic_and_order_insensitive():
    first = QueryCache.query_hash({"b": 1, "a": [1, 2, 3]})
    second = QueryCache.query_hash({"a": [1, 2, 3], "b": 1})
    assert first == second
    assert first == QueryCache.query_hash({"a": [1, 2, 3], "b": 1})
