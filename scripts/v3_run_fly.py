#!/usr/bin/env python
"""V3 Drosophila EPG/PEG ring attractor-validation runner (PRD §25).

Does the EPG/PEG head-direction ring show attractor-like dynamics — a circular
(low-dimensional ring) manifold?

The previous protocol free-ran from a single ``.final_voltage`` snapshot, which
is a reset-state (voltage = v_reset) with synaptic current zeroed, so everything
collapsed to rest (0 spikes, trivial convergence, ``A_i = 0``). That was an assay
artifact, not a finding.

A ring attractor is revealed by observing the bump ROTATING DURING drive, not by
free-run. This revision therefore:

  1. Builds the hemibrain compass graph (PFL3 readout, EPG/PEG lesion, PEN input).
  2. Drives the PEN input with a CONSTANT angular-velocity step for the full
     drive window ``T_drive`` (``step_signal = 1`` throughout), recording the
     EPG/PEG voltage trajectory during the drive.
  3. Builds ``X = (T_drive_steps, N_epgpeg)`` directly from the drive trajectory.
  4. PCAs ``X`` and measures ring geometry; also sweeps 3 amplitudes (0.5x / 1x /
     2x the reference) for per-amplitude circularity and a concatenated PCA.
  5. Only runs a short per-node silencing check if the drive produces *sustained*
     EPG/PEG activity (spikes in the second half of the window); otherwise
     reports ``mean_Ai_by_type = null``.
  6. Reports the raw EPG/PEG spike count during drive.

Head-direction note (PRD §10.4): there is NO reliable head-direction target, so
we report manifold geometry only and DO NOT compute head-direction alignment.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.attractor_analysis import scoring
from cybergraft.attractor_analysis.dynamics import Injection, run_dynamics
from cybergraft.attractor_analysis.manifold import circular_metrics, pca_manifold
from cybergraft.attractor_analysis.perturbation import (
    a_persist_from_metrics,
    perturbation_metrics,
)
from cybergraft.data_ingest.hemibrain_compass import build_compass_graph
from cybergraft.sim.lif_reference import LIFParameters
from cybergraft.utils.config import load_config
from cybergraft.utils.manifest import config_hash


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


def _run_drive(
    graph,
    parameters,
    *,
    amplitude,
    duration_ms,
    input_ids,
    clamp=None,
):
    """Drive the input with a constant angular-velocity step and record voltage.

    Returns a ``DynamicsResult`` with ``record_voltage=True`` so the EPG/PEG
    voltage trajectory (the rotating bump) is available.
    """
    n_steps = int(round(duration_ms / parameters.dt_ms))
    step_signal = np.ones(n_steps, dtype=np.float32)
    inject = Injection(
        target_ids=np.asarray(input_ids, dtype=np.int64),
        weights=np.ones(len(input_ids), dtype=np.float32),
        signal=np.float32(amplitude) * step_signal,
        gain=1.0,
    )
    return run_dynamics(
        graph,
        parameters,
        duration_ms=duration_ms,
        inject=inject,
        clamp=clamp,
        record_voltage=True,
    )


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    experiment_id = f"v3-fly-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    print(f"[v3-fly] result_dir = {result_dir}")

    # 1. Compass graph.
    graph, provenance = build_compass_graph(config, project_root, full=True)
    print(f"[v3-fly] graph nodes={graph.node_count} edges={graph.edge_count}")

    # 2. LIF parameters.
    parameters = LIFParameters.from_config(config["simulation"])
    dt_ms = parameters.dt_ms
    T_drive_ms = float(config["simulation"]["duration_ms"])

    # 3. Population ids.
    input_ids = list(graph.groups["input"])
    epg_peg = sorted(n.id for n in graph.nodes if n.cell_type in ("EPG", "EPGt", "PEG"))
    epg_peg_idx = np.asarray(epg_peg, dtype=np.int64)
    print(f"[v3-fly] input={len(input_ids)} epg_peg={len(epg_peg)} T_drive={T_drive_ms:.0f}ms")

    # 4. Drive at 3 amplitudes and measure ring geometry during the drive.
    reference_amplitude = float(config.get("velocity", {}).get("amplitude", 300.0))
    amplitudes = [0.5 * reference_amplitude, reference_amplitude, 2.0 * reference_amplitude]

    drive_results = {}
    per_amplitude = {}
    trajectories_by_amp = []
    for amp in amplitudes:
        res = _run_drive(
            graph, parameters, amplitude=amp, duration_ms=T_drive_ms, input_ids=input_ids
        )
        drive_results[amp] = res
        x_amp = np.asarray(res.voltage[:, epg_peg_idx], dtype=np.float64)
        trajectories_by_amp.append(x_amp)
        m = pca_manifold(x_amp, n_components=3)
        cm = circular_metrics(m["z1"], m["z2"])
        per_amplitude[round(amp, 4)] = {
            "amplitude": amp,
            "epg_peg_spikes": int(res.spikes[:, epg_peg_idx].sum()),
            "explained_variance_ratio": np.asarray(m["explained_variance_ratio"]).tolist(),
            "radial_variance": cm["radial_variance"],
            "angular_coverage": cm["angular_coverage"],
            "circularity_error": cm["circularity_error"],
            "mean_radius": cm["mean_radius"],
        }

    # Concatenated PCA across amplitudes (one shared manifold).
    X_concat = np.vstack(trajectories_by_amp)
    manifold = pca_manifold(X_concat, n_components=3)
    circ = circular_metrics(manifold["z1"], manifold["z2"])

    # 5. Raw EPG/PEG spike count during the reference-amplitude drive, plus a
    #    "sustained" check (activity in the second half of the window).
    ref_res = drive_results[reference_amplitude]
    n_drive_steps = int(round(T_drive_ms / dt_ms))
    epg_peg_spikes = int(ref_res.spikes[:, epg_peg_idx].sum())
    epg_peg_spikes_late = int(ref_res.spikes[n_drive_steps // 2:, epg_peg_idx].sum())
    sustained = epg_peg_spikes_late > 0

    # 6. Short per-node silencing check, only if the drive is sustained.
    mean_ai_by_type = None
    if sustained:
        s_start = int(round(float(config["attractor"]["silence_start_ms"]) / dt_ms))
        s_end = int(round(
            (float(config["attractor"]["silence_start_ms"]) + float(config["attractor"]["silence_duration_ms"])) / dt_ms
        ))
        baseline = ref_res  # driven baseline (has voltage)
        nodes_metrics = {}
        for nid in epg_peg:
            pert = _run_drive(
                graph,
                parameters,
                amplitude=reference_amplitude,
                duration_ms=T_drive_ms,
                input_ids=input_ids,
                clamp=(int(nid), s_start, s_end),
            )
            nodes_metrics[int(nid)] = perturbation_metrics(
                baseline, pert, silence_end_step=s_end
            )
        a_persist = a_persist_from_metrics(nodes_metrics)
        scored = scoring.attractor_support_score({}, a_persist)

        by_type: dict = {}
        for nid in epg_peg:
            t = graph.nodes[nid].cell_type
            ai = scored["node_scores"][int(nid)]["attractor_support_score"]
            by_type.setdefault(t, []).append(ai)
        mean_ai_by_type = {
            t: (float(np.nanmean(v)) if any(x is not None for x in v) else None)
            for t, v in sorted(by_type.items())
        }
        print(f"[v3-fly] sustained drive -> ran silencing check over {len(epg_peg)} EPG/PEG nodes")
    else:
        print("[v3-fly] drive NOT sustained -> mean_Ai_by_type = null")

    # Ring-emergence heuristic: near-constant radius (small circularity error)
    # and broad angular sweep.
    ring_emerges = bool(circ["circularity_error"] < 0.5 and circ["angular_coverage"] > 0.6)

    # 7. Assemble and write summary.
    summary = {
        "experiment_id": experiment_id,
        "milestone": "V3",
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "n_epg_peg": int(len(epg_peg)),
        "T_drive_ms": T_drive_ms,
        "reference_amplitude": reference_amplitude,
        "amplitudes": amplitudes,
        "explained_variance_ratio": np.asarray(manifold["explained_variance_ratio"], dtype=np.float64).tolist(),
        "circularity": {
            "radial_variance": circ["radial_variance"],
            "angular_coverage": circ["angular_coverage"],
            "circularity_error": circ["circularity_error"],
            "mean_radius": circ["mean_radius"],
        },
        "per_amplitude": per_amplitude,
        "epg_peg_drive_spikes": epg_peg_spikes,
        "epg_peg_drive_spikes_late": epg_peg_spikes_late,
        "sustained_drive": sustained,
        "ring_emerges": ring_emerges,
        "mean_Ai_by_type": mean_ai_by_type,
        "config_sha256": config_hash(config),
    }
    _write_json(result_dir / "summary.json", summary)

    print("\n===== V3 FLY RING-ATTRACTOR VALIDATION (drive, not free-run) =====")
    print(f"explained_variance_ratio = {[round(v, 4) for v in summary['explained_variance_ratio']]}")
    print(f"circularity: radial_var={circ['radial_variance']:.4f} "
          f"angular_cov={circ['angular_coverage']:.4f} "
          f"circ_error={circ['circularity_error']:.4f} mean_radius={circ['mean_radius']:.4f}")
    for amp_key, pa in per_amplitude.items():
        print(f"  amp={pa['amplitude']:>7.1f}: spikes={pa['epg_peg_spikes']:>6} "
              f"angular_cov={pa['angular_coverage']:.4f} circ_error={pa['circularity_error']:.4f}")
    print(f"EPG/PEG drive spikes (ref amp {reference_amplitude:.0f}): "
          f"total={epg_peg_spikes} late={epg_peg_spikes_late} sustained={sustained}")
    print(f"ring_emerges = {ring_emerges}")
    print(f"mean_Ai_by_type = {mean_ai_by_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_fly.yaml"))
