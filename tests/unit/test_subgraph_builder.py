import numpy as np

from cybergraft.data_ingest.subgraph_builder import SubgraphBuilder


def _pathway():
    return {
        "dataset": "male-cns:v1.0",
        "input_source": "upstream_gng232",
        "nodes": [
            {"bodyId": 900001, "type": "sugar_grn", "instance": "", "superclass": "cb_sensory",
             "subsystem": "taste_input", "input_class": "stimulus", "output_class": "interneuron"},
            {"bodyId": 31018, "type": "GNG232", "instance": "GNG232_L", "superclass": "gnathal_ganglion",
             "subsystem": "gng232", "input_class": "interneuron", "output_class": "interneuron"},
            {"bodyId": 22758, "type": "DNg67", "instance": "DNg67_L", "superclass": "descending_neuron",
             "subsystem": "descending_neuron", "input_class": "interneuron", "output_class": "interneuron"},
            {"bodyId": 10331, "type": "MN9", "instance": "MN9_L", "superclass": "motor_neuron",
             "subsystem": "motor", "input_class": "interneuron", "output_class": "readout"},
        ],
        "edges": [
            {"pre": 900001, "post": 31018, "weight": 5},
            {"pre": 31018, "post": 22758, "weight": 4},
            {"pre": 22758, "post": 10331, "weight": 4},
        ],
        "groups": {
            "input": [900001],
            "gng232": [31018],
            "dn": [22758],
            "mn9": [10331],
            "output": [10331],
        },
        "query_hashes": ["abc"],
    }


def test_build_subgraph():
    graph, provenance = SubgraphBuilder(max_nodes=500, synaptic_delay_ms=1.8).build(_pathway())
    assert graph.node_count == 4
    assert [n.id for n in graph.nodes] == [0, 1, 2, 3]
    assert graph.weights.dtype == np.float32
    assert graph.edge_count == 3

    assert graph.weights[1, 0] == 5.0
    assert graph.weights[2, 1] == 4.0
    assert graph.weights[3, 2] == 4.0

    assert graph.groups["input"] == [0]
    assert graph.groups["output"] == [3]

    for key in ("dataset", "input_source", "stage_counts", "query_hashes"):
        assert key in provenance
    assert provenance["stage_counts"] == {"GNG232": 1, "DNg67": 1, "DNge080": 0, "MN9": 1}
