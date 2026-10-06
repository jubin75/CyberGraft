#!/usr/bin/env python
"""V3 zebrafish per-node analysis (statistical power over hundreds of neurons).

Site-level results established the dissociation: surviving_Int is the unique
computational graft site (G_comp=0.637), feedforward-direct sites are the relay
sites. This script lifts the analysis to PER-NODE resolution (oracle Problem 3):

  * ``A_i``        — held-state attractor support (fractional memory shift)
  * ``S_i``        — readout graft into a SINGLE neuron (relay, reuse V2 oracle)
  * ``C_i^supp``   — silence neuron i during the surviving_Int computational
                     graft; ``C_i^supp = G_comp(quiet) - G_comp(silence i)`` is a
                     causal per-node computational-support score.

All three share one node set: surviving, non-output, non-lesioned neurons with
``direct_readout_weight > 0`` (the meaningful graft candidates).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.analysis.grafting import _select_graft_site, _site_features
from cybergraft.attractor_analysis import scoring
from cybergraft.attractor_analysis import structural
from cybergraft.attractor_analysis.dynamics import Injection, run_dynamics
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import (
    _build_stimulus,
    _run_fixed_drive,
    generate_informative_source,
)
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.graft_analysis import computational_graft as cg
from cybergraft.graft_analysis import role_index as ri
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.utils.config import load_config
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _natural_afferent(graph, input_ids) -> np.ndarray:
    return np.asarray(graph.drive_weights[:, input_ids].sum(axis=1)).ravel().astype(np.float32)


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    experiment_id = f"v3-zfish-pn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    print(f"[pn] result_dir = {result_dir}")

    data = config["data"]
    mat = project_root / data["mat_file"]
    seed = int(config["experiment"]["seed"])
    rng = np.random.default_rng(seed)

    adapter = ZebrafishAdapter(
        mat_path=mat, max_nodes=int(data["max_nodes"]),
        synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
    )
    ei = config.get("ei", {})
    graph, _ = adapter.build_full_subgraph(
        readout_type=data["readout_type"], lesion_type=data["lesion_type"],
        input_type=data["input_type"], max_nodes=int(data["max_nodes"]),
        inhibitory_fraction=float(ei.get("inhibitory_fraction", 0.0)),
        inhibitory_gain=float(ei.get("inhibitory_gain", 1.0)),
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

    # candidate nodes = surviving direct-projectors
    candidate_nodes = [
        n.id for n in graph.nodes
        if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > 0
    ]
    candidate_nodes = sorted(candidate_nodes)
    print(f"[pn] graph={graph.node_count} lesion={len(lesion_target)} candidate_nodes={len(candidate_nodes)}")

    struct = structural.structural_scores(graph)
    phi = _natural_afferent(graph, input_ids)

    # surviving_Int site (the computational site) for C_i^supp reference
    n_target = int(config["graft"]["n_target"])
    int_site = _select_graft_site(graph, output_ids, lesion_target, "surviving_Int", n_target, drive_onto, rng)

    T_on_ms = float(config["graft"].get("step_hold_on_ms", 500.0))
    step_hold_dur_ms = float(config["graft"].get("step_hold_duration_ms", 3000.0))
    amplitude = float(config["velocity"]["amplitude"])
    rate_window = int(config["readout"]["rate_window"])
    test_frac = float(config["readout"]["test_frac"])
    gain_grid = [float(g) for g in config["graft"]["gain_grid"]]

    v = cg.build_velocity_probe("step_hold", dt_ms, int(round(step_hold_dur_ms / dt_ms)), step_hold_on_ms=T_on_ms)
    p_star = cg.position_target(v, dt_ms)

    # ---- per-node S_i (readout graft into a single neuron) ------------------
    print("[pn] per-node S_i (readout graft)...")
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

    S_i_node = {}
    for nid in candidate_nodes:
        spikes = _run_fixed_drive(lesioned_graph, parameters, [nid], drive, mapping_gain, duration_ms, stimulus)
        rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
        S_i_node[nid] = float((rec - reward_baseline) / headroom) if headroom > 0 else float("nan")

    # ---- per-node A_i (held-state fractional memory shift) ------------------
    print("[pn] per-node A_i (held-state perturbation)...")
    T_hold_ms = float(config["attractor"].get("hold_ms", 2000.0))
    p_dur_ms = T_on_ms + T_hold_ms
    p_n_steps = int(round(p_dur_ms / dt_ms))
    t_p = np.arange(p_n_steps, dtype=np.float64) * dt_ms
    v_step = np.zeros(p_n_steps, dtype=np.float32)
    v_step[t_p < T_on_ms] = np.float32(1.0)
    inj = Injection(
        target_ids=np.asarray(input_ids, dtype=np.int64),
        weights=np.ones(len(input_ids), dtype=np.float32),
        signal=np.float32(amplitude) * v_step, gain=1.0,
    )
    base = run_dynamics(lesioned_graph, parameters, duration_ms=p_dur_ms, inject=inj, record_voltage=False)
    R0 = output_rate(base.spikes, output_ids, win=12).astype(np.float64)
    s_start = int(round((T_on_ms + 100.0) / dt_ms))
    s_end = int(round((T_on_ms + 200.0) / dt_ms))
    t_star = slice(p_n_steps - 100, p_n_steps)
    R0_star = float(np.mean(R0[t_star]))

    A_persist = {}
    for nid in candidate_nodes:
        pert = run_dynamics(
            lesioned_graph, parameters, duration_ms=p_dur_ms, inject=inj,
            clamp=(int(nid), s_start, s_end), record_voltage=False,
        )
        Ri = output_rate(pert.spikes, output_ids, win=12).astype(np.float64)
        Ri_star = float(np.mean(Ri[t_star]))
        A_persist[int(nid)] = abs(float((Ri_star - R0_star) / max(R0_star, 1e-6)))

    # ---- per-node C_i^supp (silence node during surviving_Int graft) --------
    print("[pn] per-node C_i^supp (silencing dG)...")
    w_int = phi[np.asarray(int_site, dtype=np.int64)]
    int_assay = cg.run_computational_graft_assay(
        lesioned_graph, parameters, site_ids=int_site, output_ids=output_ids, velocity=v,
        name="step_hold", amplitude=amplitude, gain_grid=gain_grid,
        rate_window=rate_window, test_frac=test_frac, duration_ms=step_hold_dur_ms,
        weights=w_int,
    )
    best_gain = float(int_assay["best_gain"])
    G_quiet = float(int_assay["computational_graftability"])
    signal_int = np.float32(best_gain) * np.float32(amplitude) * v.astype(np.float32)
    inj_int = Injection(target_ids=np.asarray(int_site, dtype=np.int64), weights=w_int, signal=signal_int, gain=1.0)

    C_supp = {}
    n_fit = int(round(step_hold_dur_ms / dt_ms))
    for nid in candidate_nodes:
        pert = run_dynamics(
            lesioned_graph, parameters, duration_ms=step_hold_dur_ms, inject=inj_int,
            clamp=(int(nid), 0, n_fit), record_voltage=False,
        )
        out = np.convolve(pert.spikes[:, output_ids].mean(axis=1).astype(float), np.ones(rate_window), mode="same")
        r2_silenced = cg.static_readout_r2(out, p_star, slice(0, n_fit), slice(0, n_fit))
        C_supp[int(nid)] = float(G_quiet - r2_silenced)  # >0 => node carries computation

    # ---- records + analyses ------------------------------------------------
    scored = scoring.attractor_support_score({}, A_persist)
    node_Ai = {nid: d["attractor_support_score"] for nid, d in scored["node_scores"].items()}
    feat_by_node = {nid: _site_features(graph, output_ids, [nid]) for nid in candidate_nodes}

    records = []
    for nid in candidate_nodes:
        f = feat_by_node[nid]
        ct = graph.nodes[nid].cell_type
        records.append({
            "node_id": int(nid),
            "population": ct,
            "structural_role": "recurrent" if ct == "_Int_" else ("feedforward" if ct in ("untyped", "gold_top19") else "upstream"),
            "direct_readout_weight": f["direct_weight"],
            "recurrence": f["recurrence"],
            "degree": float(f["out_degree"] + f["in_degree"]),
            "attractor_support_score": node_Ai.get(nid),
            "readout_substitutability": S_i_node.get(nid),
            "computational_support": C_supp.get(nid),
            "role_index": None,
        })

    from cybergraft.analysis import attractor_graft_relationship as ar
    from cybergraft.analysis import multivariate_models as mm

    key_corrs = {
        "A_vs_C_supp": ar.correlate(records, "attractor_support_score", "computational_support"),
        "direct_vs_S": ar.correlate(records, "direct_readout_weight", "readout_substitutability"),
        "A_vs_S": ar.correlate(records, "attractor_support_score", "readout_substitutability"),
        "A_vs_recurrence": ar.correlate(records, "attractor_support_score", "recurrence"),
    }

    summary = {
        "experiment_id": experiment_id,
        "n_nodes": len(records),
        "rho_R": struct["spectral_radius"]["rho_R"],
        "surviving_Int_G_comp_quiet": G_quiet,
        "best_gain": best_gain,
        "records": records,
        "correlations": {k: {kk: vv for kk, vv in v.items() if kk not in ("x", "y")} for k, v in key_corrs.items()},
        "config_sha256": config_hash(config),
    }
    _write_json(result_dir / "summary.json", summary)
    _write_json(result_dir / "records.json", records)

    manifest = build_manifest(
        experiment_id=experiment_id, config=config, node_count=graph.node_count,
        edge_count=graph.edge_count, project_root=project_root,
        mlx_smoke={"available": False}, status="passed", milestone="V3",
        data_provenance={
            "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
            "source_urls": {"zebrafish": "https://seunglab.org/zebrafish/"},
            "query_hashes": [], "data_status": {"available": True},
            "license": "CC BY-NC-ND 4.0 (Vishwanathan 2024)",
            "description": "V3 zebrafish per-node function-specific graftability (A_i, S_i, C_i^supp).",
        },
    )
    write_manifest(result_dir / "manifest.json", manifest)

    print("\n===== V3 ZEBRAFISH PER-NODE (n=%d) =====" % len(records))
    print(f"surviving_Int G_comp(quiet) = {G_quiet:.3f} (best_gain={best_gain})")
    print("key correlations:")
    for k, v in key_corrs.items():
        print(f"  {k}: rho={v['rho']:.3f} p={v['p_perm']:.4f} n={v['n']}")

    # per-population summary of computational_support (causal)
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in records:
        by_type[r["population"]].append(r["computational_support"] if r["computational_support"] is not None else np.nan)
    print("per-population mean C_i^supp (causal computational support):")
    for t in sorted(by_type, key=lambda t: -np.nanmean(by_type[t])):
        vals = [x for x in by_type[t] if not np.isnan(x)]
        print(f"  {t:<14} n={len(vals):4d} mean={np.nanmean(vals):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
