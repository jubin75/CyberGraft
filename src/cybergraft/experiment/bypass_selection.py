"""Selection of bypass candidate nodes that project onto the readout."""

import numpy as np


def select_bypass_candidates(
    graph,
    *,
    output_group: str = "output",
    lesion_group: str = "lesion",
    max_candidates: int = 100,
    seed: int = 0,
) -> dict:
    output_ids = set(graph.groups[output_group])
    lesion_ids = set(graph.groups[lesion_group])
    output_list = list(output_ids)
    drive_onto_output = np.asarray(graph.weights[output_list, :].sum(axis=0)).ravel()
    scored = []
    for node in range(graph.node_count):
        if node in output_ids or node in lesion_ids:
            continue
        score = float(drive_onto_output[node])
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:max_candidates]
    return {
        "candidate_ids": [node for _, node in selected],
        "selection_score": [score for score, _ in selected],
        "excluded_lesioned": sorted(lesion_ids),
        "excluded_readout": sorted(output_ids),
        "seed": seed,
    }
