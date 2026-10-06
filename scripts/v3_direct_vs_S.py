#!/usr/bin/env python
"""V3 direct_vs_S investigation (caveat 2).

Site-level S_i shows direct_weight -> relay (rho=0.543) but per-node S_i (single
neuron) shows ~0 (rho=0.032). This isolates the cause by recomputing per-node
S_i on an EXCITATORY-ONLY graph (inhibitory_fraction=0) with the SAME slow tau,
so the only difference vs the E/I per-node run is the E/I balance.

If excitatory-only per-node S_i recovers direct_vs_S > 0, the E/I balance is what
kills single-neuron relay; if it stays ~0, single-neuron relay is intrinsically
degenerate (a population-level property) regardless of E/I.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.analysis import attractor_graft_relationship as ar
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import (
    _build_stimulus,
    _run_fixed_drive,
    generate_informative_source,
)
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.utils.config import load_config


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    data = config["data"]
    seed = int(config["experiment"]["seed"])

    adapter = ZebrafishAdapter(
        mat_path=project_root / data["mat_file"], max_nodes=int(data["max_nodes"]),
        synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
    )
    # excitatory-only: inhibitory_fraction = 0, same slow tau
    graph, _ = adapter.build_full_subgraph(
        readout_type=data["readout_type"], lesion_type=data["lesion_type"],
        input_type=data["input_type"], max_nodes=int(data["max_nodes"]),
        inhibitory_fraction=0.0, inhibitory_gain=1.0,
    )
    parameters = LIFParameters.from_config(config["simulation"])
    dt_ms = parameters.dt_ms
    duration_ms = float(config["simulation"]["duration_ms"])
    n_steps = int(round(duration_ms / dt_ms))

    output_ids = [n.id for n in graph.nodes if n.cell_type == data["readout_type"]]
    input_ids = [n.id for n in graph.nodes if n.cell_type == data["input_type"]]
    lesion_ids = [n.id for n in graph.nodes if n.cell_type == data["lesion_type"]]
    drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
    ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
    n_lesion = int(np.ceil(float(data["lesion_fraction"]) * len(lesion_ids)))
    lesion_target = ranked[:n_lesion]
    lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
    lesion_set = set(lesion_target)
    output_set = set(output_ids)
    graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

    candidate_nodes = sorted(
        n.id for n in graph.nodes if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > 0
    )
    print(f"[ds] excitatory-only graph nodes={graph.node_count} candidate_nodes={len(candidate_nodes)}")

    # readout graft (single neuron) — mirror per-node S_i
    stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
    act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
    baseline_rate = output_rate(act0.spikes, output_ids)
    act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
    lesioned_rate = output_rate(act_lesion.spikes, output_ids)
    reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
    headroom = 1.0 - float(reward_baseline)
    missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)
    baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
    command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
    source = generate_informative_source(
        n_steps, int(data["n_source"]), 0.0, command_rate, dt_ms,
        float(config["graft"]["source_onset_ms"]), float(config["graft"]["source_duration_ms"]), seed,
    )
    w_offline, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
    drive = source @ w_offline
    mapping_gain = float(config["mapping"]["gain"])

    S_i = {}
    for nid in candidate_nodes:
        spikes = _run_fixed_drive(lesioned_graph, parameters, [nid], drive, mapping_gain, duration_ms, stimulus)
        rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
        S_i[nid] = float((rec - reward_baseline) / headroom) if headroom > 0 else float("nan")

    records = [
        {"direct_readout_weight": float(drive_onto[nid]), "readout_substitutability": S_i[nid]}
        for nid in candidate_nodes
    ]
    corr = ar.correlate(records, "direct_readout_weight", "readout_substitutability")

    svals = np.array([v for v in S_i.values() if not np.isnan(v)], dtype=float)
    print(f"\n[ds] excitatory-only per-node S_i: n={len(svals)} mean={svals.mean():+.4f} std={svals.std():.4f} "
          f"max={svals.max():+.4f} nonzero={int((np.abs(svals) > 1e-9).sum())}")
    print(f"[ds] direct_vs_S (excitatory-only) rho={corr['rho']:+.3f} p={corr['p_perm']:.4f} n={corr['n']}")
    print(f"[ds]   (compare E/I per-node: rho=+0.032;  E/I site-level: rho=+0.543)")

    out = {"n": len(records), "S_i_mean": float(svals.mean()), "S_i_std": float(svals.std()),
           "S_i_max": float(svals.max()), "direct_vs_S_rho": corr["rho"], "p": corr["p_perm"]}
    out_dir = project_root / config["experiment"]["output_dir"] / (
        f"v3-ds-exc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "direct_vs_S_exc.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ds] wrote {out_dir}/direct_vs_S_exc.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
