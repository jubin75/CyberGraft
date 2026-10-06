"""Step-response discriminator: does a site ramp (integrate) or saturate (relay)?

A single-frequency sine probe cannot distinguish a recurrent integrator from an
RC low-pass filter: above the corner frequency a low-pass is phase-indistinguishable
from an integrator, confounding both ``run_computational_site_scan`` and
``recompute_graft``. The step response resolves this ambiguity:

  * a true integrator RAMPS without bound under a velocity step;
  * a low-pass filter JUMPS then SATURATES.

We feed a velocity step through the natural input (``_DOs_``) of the lesioned
circuit and, for each graft site, fit the site's summed (box-smoothed) population
rate over the step window:

  * ``ramp_slope_a`` (Hz/ms): linear slope of the population rate during the step;
    large positive => integrator, ~0 => low-pass relay.
  * ``saturation_ratio_sr``: second-half slope vs first-half slope;
    ``sr ~ 1`` => linear ramp (integrator), ``sr -> 0`` (or negative) => saturating
    (low-pass).

A recurrence ablation (removing all ``_Int_``->``_Int_`` edges among the surviving
integrators) causally tests whether the ramp is carried by recurrence: if so,
ablating recurrence collapses ``ramp_slope_a``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import _run_fixed_drive
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard
from cybergraft.graph.sparse_graph import SparseGraph

from cybergraft.analysis.computational_graft import _build_probe
from cybergraft.analysis.grafting import SITES, _select_graft_site, _site_features

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _site_rate(spikes: np.ndarray, site_ids: list, rate_window: int, n_steps: int) -> np.ndarray:
    """Sum of per-neuron box-smoothed rates across a site's neurons (``n_steps``,)."""
    R_site = np.zeros(n_steps, dtype=np.float64)
    for node_id in site_ids:
        R_site += np.convolve(
            spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
        )
    return R_site


def _step_response_metrics(
    R_site: np.ndarray, t_start_idx: int, t_end_idx: int, dt_ms: float
) -> Tuple[float, float]:
    """Ramp slope (Hz/ms) and saturation ratio over the step window.

    ``ramp_slope_a`` is the linear slope of the population rate during the step.
    ``saturation_ratio_sr`` compares the second-half slope to the first-half
    slope: ~1 => linear ramp (integrator), ->0 or negative => saturating (low-pass).
    """
    t_step = np.arange(t_start_idx, t_end_idx, dtype=np.float64) * dt_ms
    a = float(np.polyfit(t_step, R_site[t_start_idx:t_end_idx], 1)[0])
    m = (t_start_idx + t_end_idx) // 2
    eps = 1e-9
    sr = float((R_site[t_end_idx - 1] - R_site[m]) / (R_site[m] - R_site[t_start_idx] + eps))
    return a, sr


def run_step_response_discriminator(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 1,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Step-response discriminator: does a site ramp (integrate) or saturate (relay)?

    Lesions the top integrator neurons (``_Int_`` ranked by total weight onto the
    readout), feeds a velocity STEP through the natural input (``_DOs_``), and -- for
    each graft site -- measures the step response of the site's summed population
    rate (ramp slope + saturation ratio). A recurrence ablation on the surviving
    integrator causally tests whether the ramp is carried by recurrence.
    """
    experiment_id = f"p12-step-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

    try:
        adapter = ZebrafishAdapter(
            mat_path=mat,
            max_nodes=int(data["max_nodes"]),
            synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
        )
        ei = config.get("ei", {})
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
            inhibitory_fraction=float(ei.get("inhibitory_fraction", 0.0)),
            inhibitory_gain=float(ei.get("inhibitory_gain", 1.0)),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        dt_ms = parameters.dt_ms
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / dt_ms))

        output_ids = [n.id for n in graph.nodes if n.cell_type == data["readout_type"]]
        input_ids = [n.id for n in graph.nodes if n.cell_type == data["input_type"]]
        lesion_ids = [n.id for n in graph.nodes if n.cell_type == data["lesion_type"]]

        # Lesion the top integrator neurons ranked by total weight onto readout.
        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
        guard.check("after lesion", logger)

        # Set groups BEFORE selecting sites: _select_graft_site (gold_top19) reads
        # graph.groups["output"] / ["lesion"].
        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

        rng = np.random.default_rng(int(config["experiment"]["seed"]))
        n_target = int(data.get("graft_n_target", 19))

        site_ids_map: Dict[str, list] = {}
        site_feats_map: Dict[str, Dict[str, Any]] = {}
        for site in SITES:
            ids = _select_graft_site(graph, output_ids, lesion_target, site, n_target, drive_onto, rng)
            if not ids:
                continue
            site_ids_map[site] = ids
            site_feats_map[site] = _site_features(graph, output_ids, ids)

        velocity_cfg = config["velocity"]
        amplitude = float(velocity_cfg["amplitude"])
        rate_window = int(config["readout"]["rate_window"])

        # Step velocity probe (0/1, non-negative -- no DC offset needed).
        v = _build_probe("step", velocity_cfg, dt_ms, n_steps)
        drive = amplitude * v

        t_start_idx = int(round(float(velocity_cfg["step_start_ms"]) / dt_ms))
        t_end_idx = int(round((float(velocity_cfg["step_start_ms"]) + float(velocity_cfg["step_duration_ms"])) / dt_ms))

        # Drive the step through the NATURAL input (_DOs_).
        spikes = _run_fixed_drive(
            lesioned_graph, parameters, input_ids, drive, 1.0, duration_ms, stimulus=None
        )
        guard.check("after step drive", logger)

        # Per-site step response over the step window.
        rows: list = []
        for site, ids in site_ids_map.items():
            R_site = _site_rate(spikes, ids, rate_window, n_steps)
            a, sr = _step_response_metrics(R_site, t_start_idx, t_end_idx, dt_ms)
            rows.append(
                {
                    "site": site,
                    "ramp_slope_a": a,
                    "saturation_ratio_sr": sr,
                    "n": len(ids),
                    **site_feats_map[site],
                }
            )

        # Recurrence ablation (only for the integrator): remove all _Int_->_Int_
        # edges among the FULL surviving _Int_ population, then re-run the step and
        # recompute the surviving_Int site's ramp/saturation.
        int_site = site_ids_map.get("surviving_Int")
        if not int_site:
            int_site = _select_graft_site(graph, output_ids, lesion_target, "surviving_Int", n_target, drive_onto, rng)
        if not int_site:
            raise RuntimeError("surviving_Int site is empty; cannot run recurrence ablation")

        lesion_set = set(lesion_target)
        surviving_int = [
            n.id for n in graph.nodes if n.cell_type == "_Int_" and n.id not in lesion_set
        ]
        s = set(surviving_int)
        kept = [e for e in lesioned_graph.edges if not (e.source in s and e.target in s)]
        ablated_graph = SparseGraph.from_edges(
            lesioned_graph.nodes, kept,
            groups=lesioned_graph.groups, max_nodes=lesioned_graph.node_count,
        )
        guard.check("after ablation graph build", logger)

        spikes_ablated = _run_fixed_drive(
            ablated_graph, parameters, input_ids, drive, 1.0, duration_ms, stimulus=None
        )
        guard.check("after ablated step drive", logger)

        R_int = _site_rate(spikes, int_site, rate_window, n_steps)
        intact_a, intact_sr = _step_response_metrics(R_int, t_start_idx, t_end_idx, dt_ms)
        R_int_ablated = _site_rate(spikes_ablated, int_site, rate_window, n_steps)
        ablated_a, ablated_sr = _step_response_metrics(R_int_ablated, t_start_idx, t_end_idx, dt_ms)

        n_removed = int(len(lesioned_graph.edges) - len(kept))
        ablation = {
            "surviving_Int_intact_a": intact_a,
            "surviving_Int_ablated_a": ablated_a,
            "surviving_Int_intact_sr": intact_sr,
            "surviving_Int_ablated_sr": ablated_sr,
            "delta_a": ablated_a - intact_a,
            "n_surviving_Int": len(surviving_int),
            "n_recurrent_edges_removed": n_removed,
        }

        def _sort_key(row: Dict[str, Any]) -> float:
            h = row["ramp_slope_a"]
            return 0.0 if (isinstance(h, float) and np.isnan(h)) else float(h)

        ranked_rows = sorted(rows, key=lambda r: -_sort_key(r))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "n_target": n_target,
            "step_start_ms": float(velocity_cfg["step_start_ms"]),
            "step_duration_ms": float(velocity_cfg["step_duration_ms"]),
            "sites": ranked_rows,
            "ablation": ablation,
            "interpretation": (
                "ramp_slope_a>0 sustained => integrator; a~0 saturating => low-pass relay"
            ),
        }
        _write_metrics(result_dir / "metrics.json", metrics)

        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M6",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Step-response discriminator: does a site ramp (integrate) or saturate (relay) under a velocity step?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("step response by site (ranked by ramp_slope_a):")
        for r in ranked_rows:
            logger.info(
                "site %-14s ramp_slope_a=%.6f saturation_ratio_sr=%.4f n=%d",
                r["site"], r["ramp_slope_a"], r["saturation_ratio_sr"], r["n"],
            )
        logger.info(
            "ablation surviving_Int: intact_a=%.6f ablated_a=%.6f delta_a=%+.6f (recurrent edges removed=%d)",
            intact_a, ablated_a, ablation["delta_a"], n_removed,
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
            milestone="M6",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("step response discriminator failed")
        raise
