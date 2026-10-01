from cybergraft.data_ingest.cache import QueryCache
from cybergraft.data_ingest.malecns_queries import MaleCNSQueries

from fakes import feeding_backend


def test_resolve_neurons_by_type_and_cache(tmp_path):
    backend = feeding_backend()
    cache = QueryCache(tmp_path / "cache")
    queries = MaleCNSQueries(
        backend, cache, dataset="male-cns:v1.0", source_url="https://neuprint.janelia.org/"
    )
    rows = queries.resolve_neurons_by_type(["GNG232", "MN9"])
    assert rows
    assert all({"bodyId", "type"} <= set(row) for row in rows)
    assert {row["type"] for row in rows} <= {"GNG232", "MN9"}
    calls_after_first = backend.calls
    rows_again = queries.resolve_neurons_by_type(["GNG232", "MN9"])
    assert rows_again == rows
    assert backend.calls == calls_after_first


def test_query_vpn_candidates(tmp_path):
    backend = feeding_backend()
    cache = QueryCache(tmp_path / "cache")
    queries = MaleCNSQueries(
        backend, cache, dataset="male-cns:v1.0", source_url="https://neuprint.janelia.org/"
    )
    result = queries.query_vpn_candidates()
    assert result["candidate_type"] == "visual_projection"
    assert result["count"] >= 0
    assert queries.query_log
