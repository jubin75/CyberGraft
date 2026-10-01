from cybergraft.experiment.bypass_selection import select_bypass_candidates

from fakes import synthetic_zebrafish_graph


def test_select_bypass_candidates():
    graph = synthetic_zebrafish_graph()
    result = select_bypass_candidates(
        graph, output_group="output", lesion_group="lesion", max_candidates=10, seed=0
    )
    # node 7 (extra integrator) projects to output and is not lesioned/readout.
    assert result["candidate_ids"] == [7]
    assert 3 not in result["candidate_ids"]
    assert 4 not in result["candidate_ids"]
    assert 5 not in result["candidate_ids"]
    assert 6 not in result["candidate_ids"]
    assert set(result["excluded_lesioned"]) == {3, 4}
    assert set(result["excluded_readout"]) == {5, 6}
    for key in ("candidate_ids", "selection_score", "excluded_lesioned", "excluded_readout", "seed"):
        assert key in result


def test_select_respects_max_candidates():
    graph = synthetic_zebrafish_graph()
    result = select_bypass_candidates(
        graph, output_group="output", lesion_group="lesion", max_candidates=1, seed=0
    )
    assert len(result["candidate_ids"]) <= 1
