#!/usr/bin/env python
"""V3 zebrafish experiment runner (corrected, post positive-control).

Applies the three oracle fixes to the first-run artifacts:

  1. Computational graft ``W`` = per-neuron NATURAL-AFFERENT pattern
     (``φ_j = net signed vestibular (_DOs_) input onto neuron j``), times a swept
     scalar gain — a differential static-linear map, not uniform common-mode.
  2. Velocity probe = STEP-AND-HOLD (velocity step, then u=0 hold), with the raw
     velocity source (NO DC pedestal), so the surviving circuit's own dynamics —
     not membrane low-pass — must produce the position.
  3. Attractor support ``A_persist`` = fractional memory shift under a HELD state
     (velocity step -> hold, silence node i briefly, measure the readout shift at
     fixed t*), instead of "deviation from a decaying free-run baseline".

``S_i`` (readout graft) is unchanged from V2: source = intact output command,
injected uniformly (relay). Only the computational branch's source semantics and
injection pattern differ (PRD §28).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.analysis.grafting import SITES, _select_graft_site, _site_features
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
    """``φ_j`` = net signed vestibular (_DOs_) input onto each node (shape N)."""
    return np.asarray(graph.drive_weights[:, input_ids].sum(axis=1)).ravel().astype(np.float32)


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    experiment_id = f"v3-zfish-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    print(f"[v3] result_dir = {result_dir}")

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
    print(f"[v3] graph nodes={graph.node_count} edges={graph.edge_count}")

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
    surviving = [n.id for n in graph.nodes if n.id not in output_set and n.id not in lesion_set]
    print(f"[v3] lesion {len(lesion_target)} _Int_; surviving={len(surviving)}")
    graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

    struct = structural.structural_scores(graph)
    phi = _natural_afferent(graph, input_ids)
    print(f"[v3] rho_R={struct['spectral_radius']['rho_R']:.2f} "
          f"phi(natural afferent) range=[{phi.min():.2f},{phi.max():.2f}] nonzero={int(np.count_nonzero(phi))}")

    n_target = int(config["graft"]["n_target"])
    site_ids_map = {}
    site_feats = {}
    for site in SITES:
        ids = _select_graft_site(graph, output_ids, lesion_target, site, n_target, drive_onto, rng)
        if ids:
            site_ids_map[site] = ids
            site_feats[site] = _site_features(graph, output_ids, ids)

    # ---- readout graft S_i (source = intact output command, uniform relay) ----
    print("[v3] readout graft (S_i)...")
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
    for site, ids in site_ids_map.items():
        spikes = _run_fixed_drive(lesioned_graph, parameters, ids, drive, mapping_gain, duration_ms, stimulus)
        rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
        S_i[site] = float((rec - reward_baseline) / headroom) if headroom > 0 else float("nan")
    print(f"[v3] S_i: { {k: round(v, 3) for k, v in S_i.items()} }")

    # ---- computational graft G_comp (step-and-hold, natural-afferent W) ------
    print("[v3] computational graft (G_comp, step-hold + natural-afferent W)...")
    T_on_ms = float(config["graft"].get("step_hold_on_ms", 500.0))
    step_hold_dur_ms = float(config["graft"].get("step_hold_duration_ms", 3000.0))
    v = cg.build_velocity_probe("step_hold", dt_ms, int(round(step_hold_dur_ms / dt_ms)), step_hold_on_ms=T_on_ms)
    gain_grid = [float(g) for g in config["graft"]["gain_grid"]]
    G_comp = {}
    for site, ids in site_ids_map.items():
        w_site = phi[np.asarray(ids, dtype=np.int64)]
        assay = cg.run_computational_graft_assay(
            lesioned_graph, parameters, site_ids=ids, output_ids=output_ids, velocity=v,
            name="step_hold", amplitude=float(config["velocity"]["amplitude"]), gain_grid=gain_grid,
            rate_window=int(config["readout"]["rate_window"]),
            test_frac=float(config["readout"]["test_frac"]), duration_ms=step_hold_dur_ms,
            weights=w_site,
        )
        G_comp[site] = float(assay["computational_graftability"])
    print(f"[v3] G_comp: { {k: round(v, 3) for k, v in G_comp.items()} }")

    # ---- attractor support A_persist (held-state fractional memory shift) ----
    print("[v3] held-state perturbation (A_persist)...")
    T_hold_ms = float(config["attractor"].get("hold_ms", 2000.0))
    p_dur_ms = T_on_ms + T_hold_ms
    p_n_steps = int(round(p_dur_ms / dt_ms))
    t_p = np.arange(p_n_steps, dtype=np.float64) * dt_ms
    v_step = np.zeros(p_n_steps, dtype=np.float32)
    v_step[t_p < T_on_ms] = np.float32(1.0)
    inj = Injection(
        target_ids=np.asarray(input_ids, dtype=np.int64),
        weights=np.ones(len(input_ids), dtype=np.float32),
        signal=np.float32(config["velocity"]["amplitude"]) * v_step, gain=1.0,
    )
    base = run_dynamics(lesioned_graph, parameters, duration_ms=p_dur_ms, inject=inj, record_voltage=False)
    R0 = output_rate(base.spikes, output_ids, win=12).astype(np.float64)
    s_start = int(round((T_on_ms + 100.0) / dt_ms))
    s_end = int(round((T_on_ms + 200.0) / dt_ms))
    t_star = slice(p_n_steps - 100, p_n_steps)
    R0_star = float(np.mean(R0[t_star]))

    candidate_nodes = sorted({nid for ids in site_ids_map.values() for nid in ids})
    A_persist = {}
    for nid in candidate_nodes:
        pert = run_dynamics(
            lesioned_graph, parameters, duration_ms=p_dur_ms, inject=inj,
            clamp=(int(nid), s_start, s_end), record_voltage=False,
        )
        Ri = output_rate(pert.spikes, output_ids, win=12).astype(np.float64)
        Ri_star = float(np.mean(Ri[t_star]))
        shift = (Ri_star - R0_star) / max(R0_star, 1e-6)
        A_persist[int(nid)] = abs(float(shift))
    scored = scoring.attractor_support_score({}, A_persist)
    node_Ai = {nid: d["attractor_support_score"] for nid, d in scored["node_scores"].items()}
    print(f"[v3] A_persist n={len(A_persist)} range=[{min(A_persist.values()):.3f},{max(A_persist.values()):.3f}]")

    # ---- assemble site-level records ----------------------------------------
    records = []
    for site in SITES:
        if site not in site_ids_map:
            continue
        ids = site_ids_map[site]
        feats = site_feats[site]
        site_A = [node_Ai.get(nid) for nid in ids]
        site_A = [a for a in site_A if a is not None]
        A_mean = float(np.mean(site_A)) if site_A else None
        records.append({
            "population": site,
            "structural_role": "recurrent" if site == "surviving_Int" else "feedforward" if site in ("untyped_direct", "gold_top19") else "upstream",
            "direct_readout_weight": feats["direct_weight"],
            "recurrence": feats["recurrence"],
            "degree": float(feats["out_degree"] + feats["in_degree"]),
            "attractor_support_score": A_mean,
            "readout_substitutability": S_i.get(site),
            "computational_graftability": G_comp.get(site),
            "role_index": None,
        })

    G_list = [r["computational_graftability"] for r in records]
    S_list = [r["readout_substitutability"] for r in records]
    D = ri.role_index(G_list, S_list)
    for r, d in zip(records, D):
        r["role_index"] = None if np.isnan(d) else float(d)

    from cybergraft.analysis import attractor_graft_relationship as ar
    from cybergraft.analysis import multivariate_models as mm

    key_corrs = {
        "A_vs_G_comp": ar.correlate(records, "attractor_support_score", "computational_graftability"),
        "direct_vs_S": ar.correlate(records, "direct_readout_weight", "readout_substitutability"),
        "A_vs_S": ar.correlate(records, "attractor_support_score", "readout_substitutability"),
        "A_vs_recurrence": ar.correlate(records, "attractor_support_score", "recurrence"),
    }

    summary = {
        "experiment_id": experiment_id,
        "node_count": graph.node_count, "n_surviving": len(surviving), "n_lesioned": len(lesion_target),
        "rho_R": struct["spectral_radius"]["rho_R"],
        "records": records,
        "per_node_A_persist": A_persist,
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
            "description": "V3 zebrafish function-specific graftability (corrected W + step-hold + held-state A_persist).",
        },
    )
    write_manifest(result_dir / "manifest.json", manifest)

    print("\n===== V3 ZEBRAFISH SUMMARY (corrected) =====")
    for r in records:
        print(f"  {r['population']:<16} S_i={ (round(r['readout_substitutability'],3) if r['readout_substitutability'] is not None else None)!s:<8} "
              f"G_comp={ (round(r['computational_graftability'],3) if r['computational_graftability'] is not None else None)!s:<8} "
              f"A_i={ (round(r['attractor_support_score'],3) if r['attractor_support_score'] is not None else None)!s:<8} "
              f"direct_w={r['direct_readout_weight']:.2f} role={ (round(r['role_index'],3) if r['role_index'] is not None else None)!s}")
    print("\nkey correlations:")
    for k, v in key_corrs.items():
        print(f"  {k}: rho={v['rho']:.3f} p={v['p_perm']:.3f} n={v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
