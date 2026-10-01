"""Fake in-memory neuPrint backend for M1 tests (never touches the network)."""


class FakeBackend:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections
        self.calls = 0

    @property
    def available(self):
        return True

    def status(self):
        return {
            "installed": True,
            "available": True,
            "status": "passed",
            "server": "fake",
            "dataset": "male-cns:v1.0",
        }

    def query_neurons(self, criteria):
        self.calls += 1
        return [dict(n) for n in self.neurons if _match_neuron(n, criteria)]

    def query_connections(self, sources, targets, min_weight=1):
        self.calls += 1
        out = []
        for c in self.connections:
            if c["weight"] < min_weight:
                continue
            if sources is not None and c["bodyId_pre"] not in sources:
                continue
            if targets is not None and c["bodyId_post"] not in targets:
                continue
            out.append({"bodyId_pre": c["bodyId_pre"], "bodyId_post": c["bodyId_post"], "weight": c["weight"]})
        return out


def _match_neuron(n, criteria):
    for key, value in criteria.items():
        neuron_value = n.get(key)
        if key in ("bodyId", "type", "superclass"):
            if isinstance(value, list):
                if neuron_value not in value:
                    return False
            elif neuron_value != value:
                return False
        else:
            if neuron_value != value:
                return False
    return True


def feeding_backend():
    neurons = [
        {"bodyId": 31018, "type": "GNG232", "instance": "GNG232_L", "superclass": "cb_intrinsic", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 89638, "type": "GNG232", "instance": "GNG232_R", "superclass": "cb_intrinsic", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 22758, "type": "DNg67", "instance": "DNg67_L", "superclass": "descending_neuron", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 516215, "type": "DNg67", "instance": "DNg67_R", "superclass": "descending_neuron", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 12364, "type": "DNge080", "instance": "DNge080_L", "superclass": "descending_neuron", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 12752, "type": "DNge080", "instance": "DNge080_R", "superclass": "descending_neuron", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 10331, "type": "MN9", "instance": "MN9_L", "superclass": "cb_motor", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 16949, "type": "MN9", "instance": "MN9_R", "superclass": "cb_motor", "class": "", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 900001, "type": "grn_a", "instance": "", "superclass": "cb_sensory", "class": "gustatory", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 900002, "type": "grn_b", "instance": "", "superclass": "cb_sensory", "class": "gustatory", "predictedNt": "acetylcholine", "status": "Traced"},
        {"bodyId": 900003, "type": "grn_c", "instance": "", "superclass": "cb_sensory", "class": "gustatory", "predictedNt": "gaba", "status": "Traced"},
        {"bodyId": 900004, "type": "mechano_grn", "instance": "", "superclass": "cb_sensory", "class": "mechanosensory", "predictedNt": "acetylcholine", "status": "Traced"},
    ]
    gustatory = [900001, 900002, 900003]
    mechano = [900004]
    gng232 = [31018, 89638]
    dng67 = [22758, 516215]
    dnge080 = [12364, 12752]
    dn = dng67 + dnge080
    mn9 = [10331, 16949]

    connections = []
    for s in gustatory:
        for g in gng232:
            connections.append({"bodyId_pre": s, "bodyId_post": g, "weight": 5})
    for s in mechano:
        for g in gng232:
            connections.append({"bodyId_pre": s, "bodyId_post": g, "weight": 10})
    for g in gng232:
        for d in dng67:
            connections.append({"bodyId_pre": g, "bodyId_post": d, "weight": 4})
        for d in dnge080:
            connections.append({"bodyId_pre": g, "bodyId_post": d, "weight": 4})
    for d in dn:
        for m in mn9:
            connections.append({"bodyId_pre": d, "bodyId_post": m, "weight": 4})

    return FakeBackend(neurons, connections)


def synthetic_zebrafish_graph():
    """Small deterministic chain: 3 input (_DOs_) -> 2 integrator (_Int_) -> 2 motor (ABD_m), plus 1 extra integrator."""
    from cybergraft.graph.edge_schema import Edge
    from cybergraft.graph.node_schema import Node
    from cybergraft.graph.sparse_graph import SparseGraph

    labels = ["_DOs_", "_DOs_", "_DOs_", "_Int_", "_Int_", "ABD_m", "ABD_m", "_Int_"]
    nodes = [
        Node(
            id=index,
            species="zebrafish",
            cell_type=labels[index],
            subsystem=labels[index],
            input_class="stimulus" if labels[index] == "_DOs_" else "interneuron",
            output_class="readout" if labels[index] == "ABD_m" else "interneuron",
        )
        for index in range(8)
    ]
    edges = []
    for source in (0, 1, 2):
        for target in (3, 4):
            edges.append(Edge(source=source, target=target, weight=1.0, provenance="synthetic", sign="excitatory", sign_source="model_assumption", delay_ms=1.8))
    for source in (3, 4):
        for target in (5, 6):
            edges.append(Edge(source=source, target=target, weight=1.0, provenance="synthetic", sign="excitatory", sign_source="model_assumption", delay_ms=1.8))
    edges.append(Edge(source=0, target=7, weight=1.0, provenance="synthetic", sign="excitatory", sign_source="model_assumption", delay_ms=1.8))
    for target in (5, 6):
        edges.append(Edge(source=7, target=target, weight=1.0, provenance="synthetic", sign="excitatory", sign_source="model_assumption", delay_ms=1.8))

    groups = {"input": [0, 1, 2], "lesion": [3, 4], "output": [5, 6]}
    return SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=8)
