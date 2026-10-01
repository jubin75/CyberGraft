from cybergraft.data_ingest.cache import QueryCache
from cybergraft.data_ingest.fly_pathway import FlyFeedingPathway
from cybergraft.data_ingest.malecns_queries import MaleCNSQueries

from fakes import feeding_backend


def _config():
    return {
        "pathway": {"stages": ["GNG232", "DNg67", "DNge080", "MN9"], "readout": "MN9"},
        "data": {"max_input_nodes": 10, "min_edge_weight": 1},
    }


def test_resolve_pathway(tmp_path):
    backend = feeding_backend()
    cache = QueryCache(tmp_path / "cache")
    queries = MaleCNSQueries(
        backend, cache, dataset="male-cns:v1.0", source_url="https://neuprint.janelia.org/"
    )
    pathway = FlyFeedingPathway(queries, _config()).resolve()

    groups = pathway["groups"]
    assert groups["input"]
    assert groups["gng232"]
    assert groups["dn"]
    assert groups["mn9"]
    assert groups["output"] == groups["mn9"]

    assert pathway["input_source"] == "gustatory_grn"
    assert pathway["dataset"] == "male-cns:v1.0"

    in_set = set(groups["input"])
    gng = set(groups["gng232"])
    dn = set(groups["dn"])
    mn = set(groups["mn9"])
    edges = pathway["edges"]
    assert any(e["pre"] in in_set and e["post"] in gng for e in edges)
    assert any(e["pre"] in gng and e["post"] in dn for e in edges)
    assert any(e["pre"] in dn and e["post"] in mn for e in edges)

    assert pathway["query_hashes"]

    node_ids = {n["bodyId"] for n in pathway["nodes"]}
    for edge in edges:
        assert edge["pre"] in node_ids
        assert edge["post"] in node_ids
    for group in groups.values():
        assert set(group) <= node_ids

    input_nodes = [n for n in pathway["nodes"] if n["bodyId"] in in_set]
    assert input_nodes
    assert all(n["class"] == "gustatory" for n in input_nodes)
    assert 900004 not in in_set
