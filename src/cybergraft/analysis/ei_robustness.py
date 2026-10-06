"""Milestone V2.4 — E/I parameter robustness of the computational-headroom ranking.

The reviewer's objection: the "opposite structural prerequisites" result — that the
velocity->position integration lives in the integrator remnant ``surviving_Int``
rather than in the feedforward direct-projectors (``untyped_direct``, ``gold_top19``)
— rests on hand-tuned E/I + slow-dynamics parameters (f=0.5, g=1.0, tau_syn=50ms,
tau_m=100ms).

This experiment perturbs ONE parameter at a time around that base point. For every
perturbed configuration it:

  1. measures the intact circuit's integrator signature — the sinusoidal phase lag of
     the ``ABD_m`` output behind the injected velocity (reusing the exact machinery
     ``scripts/ei_sweep.py`` uses, i.e. :func:`cybergraft.analysis.integrator_check._phase_lag_deg`
     and ``_reconstruct_velocity``), and
  2. re-runs the per-site computational site scan
     (:func:`cybergraft.analysis.computational_graft.run_computational_site_scan`) to get
     the per-site ``computational_headroom`` for all 6 graft sites.

A configuration "integrates" if its phase lag falls inside the integrator window
``30 deg < lag < 150 deg`` (the window used by ``integrator_check._verdict`` /
``ei_sweep._verdict``). The inversion holds if ``surviving_Int`` is the top
POSITIVE-headroom site AND the feedforward direct-projectors stay at ~0 headroom.
The result is ROBUST if the inversion holds at the base point and in >= 80% of the
integrating configurations.

This module is self-contained and reuses existing code paths; it modifies none.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cybergraft.analysis.computational_graft import run_computational_site_scan
from cybergraft.analysis.grafting import SITES
from cybergraft.analysis.integrator_check import (
    _phase_lag_deg,
    _reconstruct_velocity,
)
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_training import output_rate
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import RhythmicStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"

# --- Integrator-signature measurement constants (mirror scripts/ei_sweep.py) ---
# These reproduce the reference base-point phase lag of 88.2 deg that the
# original result was anchored on, so the perturbation grid is comparable.
READOUT_TYPE = "ABD_m"
INPUT_TYPE = "_DOs_"
LESION_TYPE = "_Int_"

SIG_DT_MS = 0.5
SIG_DURATION_MS = 1000.0
SIG_N_STEPS = int(round(SIG_DURATION_MS / SIG_DT_MS))

SIG_BASE_SIM = {
    "dt_ms": 0.5,
    "v_rest": -52.0,
    "v_reset": -52.0,
    "v_th": -45.0,
    "refractory_ms": 2.2,
    "synaptic_delay_ms": 1.8,
    "synaptic_gain": 20.0,
}

SIG_SINE_CFG = {
    "type": "rhythmic",
    "target_group": "input",
    "amplitude": 300.0,
    "frequency_hz": 5.0,
    "start_ms": 100.0,
    "duration_ms": 600.0,
}

# Thresholds (overridable via the config's ``ei_robustness`` block).
DEFAULT_PHASE_LAG_WINDOW_DEG = (30.0, 150.0)
DEFAULT_TOP_SITE = "surviving_Int"
DEFAULT_DIRECT_SITES = ("untyped_direct", "gold_top19")
DEFAULT_DIRECT_HEADROOM_MAX = 0.05

DEFAULT_BASE = {
    "inhibitory_fraction": 0.5,
    "inhibitory_gain": 1.0,
    "tau_syn_ms": 50.0,
    "tau_m_ms": 100.0,
}
DEFAULT_GRID = {
    "tau_m_ms": [20.0, 50.0, 100.0, 200.0],
    "tau_syn_ms": [10.0, 25.0, 50.0, 100.0],
    "inhibitory_fraction": [0.3, 0.4, 0.5, 0.6, 0.7],
}


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_grid(rb_cfg: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """Deduplicated 1D perturbation grid of ``(f, g, tau_syn_ms, tau_m_ms)``.

    Varies one parameter at a time around the base point; the other two stay at
    base. The base point is shared by all three axes, so the unique set has
    4 + (4-1) + (5-1) = 11 configurations (not 13).
    """
    base = dict(DEFAULT_BASE)
    base.update(rb_cfg.get("base", {}))
    grid = dict(DEFAULT_GRID)
    grid.update(rb_cfg.get("grid", {}))

    b_f = float(base["inhibitory_fraction"])
    b_g = float(base["inhibitory_gain"])
    b_s = float(base["tau_syn_ms"])
    b_m = float(base["tau_m_ms"])

    seen: set = set()
    configs: List[Tuple[float, float, float, float]] = []

    def add(f: float, g: float, s: float, m: float) -> None:
        key = (round(float(f), 6), round(float(g), 6), round(float(s), 6), round(float(m), 6))
        if key not in seen:
            seen.add(key)
            configs.append((float(f), float(g), float(s), float(m)))

    for tau_m in grid["tau_m_ms"]:
        add(b_f, b_g, b_s, tau_m)
    for tau_syn in grid["tau_syn_ms"]:
        add(b_f, b_g, tau_syn, b_m)
    for f in grid["inhibitory_fraction"]:
        add(f, b_g, b_s, b_m)

    return configs


def _measure_phase_lag(
    mat_path: Path,
    max_nodes: int,
    synaptic_delay_ms: float,
    f: float,
    g: float,
    tau_syn: float,
    tau_m: float,
    graph_cache: Dict[Tuple[float, float], Tuple[Any, Dict[str, list], List[int]]],
) -> float:
    """Sinusoidal phase lag (deg) of intact ``ABD_m`` output behind injected velocity.

    Reuses ``_phase_lag_deg`` / ``_reconstruct_velocity`` from
    :mod:`cybergraft.analysis.integrator_check` with the exact stimulus/simulation
    constants of ``scripts/ei_sweep.py``. Graphs are cached per ``(f, g)`` since
    ``tau_syn`` / ``tau_m`` do not change the wiring.
    """
    key = (round(float(f), 6), round(float(g), 6))
    if key not in graph_cache:
        adapter = ZebrafishAdapter(
            mat_path=mat_path, max_nodes=max_nodes, synaptic_delay_ms=synaptic_delay_ms
        )
        graph, _ = adapter.build_full_subgraph(
            readout_type=READOUT_TYPE,
            lesion_type=LESION_TYPE,
            input_type=INPUT_TYPE,
            max_nodes=max_nodes,
            inhibitory_fraction=float(f),
            inhibitory_gain=float(g),
        )
        output_ids = [n.id for n in graph.nodes if n.cell_type == READOUT_TYPE]
        input_ids = [n.id for n in graph.nodes if n.cell_type == INPUT_TYPE]
        groups = {"output": output_ids, "input": input_ids}
        graph_cache[key] = (graph, groups, output_ids)
    else:
        graph, groups, output_ids = graph_cache[key]

    sine_stim = RhythmicStimulus.from_config(SIG_SINE_CFG, groups)
    velocity = _reconstruct_velocity(sine_stim, SIG_DT_MS, SIG_N_STEPS)

    sim_dict = dict(SIG_BASE_SIM)
    sim_dict["tau_syn_ms"] = float(tau_syn)
    sim_dict["tau_m_ms"] = float(tau_m)
    parameters = LIFParameters.from_config(sim_dict)

    sine_spikes = ReferenceLIFSimulator(graph, parameters).run(SIG_DURATION_MS, sine_stim).spikes
    sine_rate = output_rate(sine_spikes, output_ids)
    phase_lag = _phase_lag_deg(
        velocity,
        sine_rate,
        SIG_DT_MS,
        float(SIG_SINE_CFG["frequency_hz"]),
        float(SIG_SINE_CFG["start_ms"]),
        float(SIG_SINE_CFG["duration_ms"]),
    )
    return float(phase_lag)


def _inversion_holds(
    headroom: Dict[str, float],
    top_site: str,
    direct_sites: Tuple[str, ...],
    direct_max: float,
) -> bool:
    """``surviving_Int`` is the top POSITIVE-headroom site AND the feedforward
    direct-projectors have headroom <= ``direct_max`` (i.e. ~0)."""
    si = float(headroom.get(top_site, float("nan")))
    others = {s: float(headroom.get(s, float("-inf"))) for s in SITES if s != top_site}
    top = (not np.isnan(si)) and si > 0.0 and all(si >= v for v in others.values())

    direct_ok = all(
        (not np.isnan(float(headroom.get(s, float("nan")))))
        and float(headroom.get(s, float("nan"))) <= direct_max
        for s in direct_sites
    )
    return bool(top and direct_ok)


def run_ei_robustness(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Run the 1D E/I perturbation grid and test the inversion's robustness."""
    experiment_id = f"v2-ei-robustness-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(result_dir / "run.log", verbose=verbose)
    guard = MemoryGuard(
        warning_gb=float(config["runtime"]["memory_warning_gb"]),
        abort_gb=float(config["runtime"]["memory_abort_gb"]),
    )
    mlx_smoke = run_mlx_smoke_test()

    data = config["data"]
    mat = Path(mat_path) if mat_path else (project_root / data["mat_file"])
    n_seeds = int(config["reproducibility"].get("n_seeds", n_seeds))

    rb_cfg = config.get("ei_robustness", {})
    grid = _build_grid(rb_cfg)
    phase_min, phase_max = (float(x) for x in rb_cfg.get("phase_lag_window_deg", list(DEFAULT_PHASE_LAG_WINDOW_DEG)))
    top_site = str(rb_cfg.get("top_site", DEFAULT_TOP_SITE))
    direct_sites = tuple(str(s) for s in rb_cfg.get("direct_sites", list(DEFAULT_DIRECT_SITES)))
    direct_max = float(rb_cfg.get("direct_headroom_max", DEFAULT_DIRECT_HEADROOM_MAX))

    try:
        max_nodes = int(data["max_nodes"])
        synaptic_delay_ms = float(config["simulation"]["synaptic_delay_ms"])
        graph_cache: Dict[Tuple[float, float], Tuple[Any, Dict[str, list], List[int]]] = {}

        rows: List[Dict[str, Any]] = []
        last_node_count = 0
        last_edge_count = 0
        for (f, g, tau_syn, tau_m) in grid:
            phase_lag = _measure_phase_lag(
                mat, max_nodes, synaptic_delay_ms, f, g, tau_syn, tau_m, graph_cache
            )
            integrates = (not np.isnan(phase_lag)) and (phase_min < phase_lag < phase_max)

            # Per-site computational headroom via the existing site-scan code path.
            cfg = deepcopy(config)
            cfg["ei"] = {"inhibitory_fraction": f, "inhibitory_gain": g}
            cfg["simulation"]["tau_syn_ms"] = tau_syn
            cfg["simulation"]["tau_m_ms"] = tau_m
            _, site_metrics = run_computational_site_scan(
                cfg, project_root, n_seeds=n_seeds, verbose=verbose, mat_path=mat
            )
            last_node_count = int(site_metrics.get("node_count", last_node_count))
            last_edge_count = int(site_metrics.get("edge_count", last_edge_count))
            headroom = {row["site"]: float(row["computational_headroom"]) for row in site_metrics["sites"]}

            inversion_holds = _inversion_holds(headroom, top_site, direct_sites, direct_max)

            rows.append(
                {
                    "f": f,
                    "g": g,
                    "tau_syn_ms": tau_syn,
                    "tau_m_ms": tau_m,
                    "phase_lag_deg": phase_lag,
                    "integrates": bool(integrates),
                    "headroom_per_site": headroom,
                    "inversion_holds": inversion_holds,
                }
            )
            logger.info(
                "f=%.1f g=%.1f tau_syn=%g tau_m=%g phase=%.1f integrates=%s inversion_holds=%s",
                f, g, tau_syn, tau_m, phase_lag, integrates, inversion_holds,
            )
            guard.check(f"after config f={f} tau_syn={tau_syn} tau_m={tau_m}", logger)

        n_integrating = int(sum(1 for r in rows if r["integrates"]))
        n_inversion_holds = int(sum(1 for r in rows if r["integrates"] and r["inversion_holds"]))
        robust_fraction = (n_inversion_holds / n_integrating) if n_integrating > 0 else float("nan")

        base_row = next(
            r
            for r in rows
            if r["f"] == DEFAULT_BASE["inhibitory_fraction"]
            and r["g"] == DEFAULT_BASE["inhibitory_gain"]
            and r["tau_syn_ms"] == DEFAULT_BASE["tau_syn_ms"]
            and r["tau_m_ms"] == DEFAULT_BASE["tau_m_ms"]
        )
        base_inversion = bool(base_row["inversion_holds"])
        passes = base_inversion and (not np.isnan(robust_fraction)) and robust_fraction >= 0.8
        verdict = "ROBUST" if passes else "NOT robust"

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "V2.4",
            "backend": "reference",
            "node_count": last_node_count,
            "edge_count": last_edge_count,
            "n_seeds": n_seeds,
            "n_configs": len(rows),
            "phase_lag_window_deg": [phase_min, phase_max],
            "top_site": top_site,
            "direct_sites": list(direct_sites),
            "direct_headroom_max": direct_max,
            "note": (
                "integrates = phase_lag_deg within the window (30, 150) deg, the threshold used by "
                "integrator_check._verdict / ei_sweep._verdict. NB: _phase_lag_deg (cross-correlation) "
                "can alias/wrap the lag for a periodic 5 Hz signal, so a few configs whose lag is "
                "equivalent to ~+90 deg (e.g. -268.2 deg == +91.8 deg) are classified non-integrating "
                "by the raw window. This lowers the denominator and hence robust_fraction."
            ),
            "configs": rows,
            "n_integrating": n_integrating,
            "n_inversion_holds": n_inversion_holds,
            "robust_fraction": robust_fraction,
            "base_inversion_holds": base_inversion,
            "pass": bool(passes),
            "verdict": verdict,
        }
        _write_metrics(result_dir / "metrics.json", metrics)

        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=last_node_count,
            edge_count=last_edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="V2.4",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Milestone V2.4 E/I robustness: 1D perturbation grid around "
                    "(f=0.5, g=1.0, tau_syn=50ms, tau_m=100ms); does surviving_Int stay the "
                    "top positive-headroom site while feedforward direct-projectors stay at ~0?"
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info(
            "integrating=%d inversion_holds=%d robust_fraction=%.3f base_inversion=%s verdict=%s",
            n_integrating, n_inversion_holds, robust_fraction, base_inversion, verdict,
        )
        return result_dir, metrics

    except Exception as exc:
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=0,
            edge_count=0,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="error",
            milestone="V2.4",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("E/I robustness sweep failed")
        raise
