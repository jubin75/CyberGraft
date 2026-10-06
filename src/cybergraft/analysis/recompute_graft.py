"""Velocity-only recompute graft: does a surviving site integrate velocity -> position?

This replaces the tautological "readout graft" (which built its source from the
intact output command, fit ``W = lstsq(source, missing)``, and injected ``source
@ W`` into a graft neuron whose only job was to relay current onward). In that
construction ``substitutability ~ direct synaptic weight onto readout`` was true
by definition.

Here we graft a VELOCITY source (the circuit's INPUT, not its output) into each
surviving site and ask whether the readout can reconstruct POSITION (the integral
of velocity, the circuit's OUTPUT). This is non-tautological: a site can only
reconstruct position if its own dynamics (slow intrinsic/synaptic time constants
under E/I) actually perform the integration.

Expected signature:
  * feedforward-direct sites relay velocity (high ``r2_velocity``, low
    ``r2_position``);
  * the surviving integrator (with slow dynamics) reconstructs position (high
    ``r2_position``) -- IF the E/I + slow-tau circuit actually integrates.
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

from cybergraft.analysis.computational_graft import _build_probe, _r2
from cybergraft.analysis.grafting import SITES, _select_graft_site, _site_features

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_recompute_graft(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 1,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Graft a VELOCITY source into each surviving site and ask whether the readout
    reconstructs POSITION (non-tautological recompute graft).

    Lesions the top integrator neurons (``_Int_`` ranked by total weight onto the
    readout), then -- for each of the 6 graft sites -- injects the velocity signal
    directly into that site's neurons and fits a memoryless readout of the READOUT
    neurons' smoothed rates against (a) position (the integral of velocity) and
    (b) velocity itself. Two controls: direct injection into the readout (a LIF
    readout relays velocity but cannot integrate) and a time-shuffled velocity
    drive (no temporal structure, so position reconstruction ~ 0).
    """
    experiment_id = f"p11-recompute-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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
        test_frac = float(config["readout"]["test_frac"])

        # Sine velocity probe only. DC offset keeps the relay from half-wave
        # rectifying the bipolar velocity signal (mirror run_computational_graft).
        v = _build_probe("sine", velocity_cfg, dt_ms, n_steps)
        drive = amplitude * (0.5 + 0.5 * v)

        # The position target: the integral of velocity.
        p_star = np.cumsum(v) * dt_ms / 1000.0

        # Temporal held-out: fit on the first (1 - test_frac), eval on the tail.
        n_train = int(n_steps * (1.0 - test_frac))
        fit, evl = slice(0, n_train), slice(n_train, n_steps)

        def _readout_reconstruction(spikes: np.ndarray) -> Dict[str, float]:
            """Static (memoryless) readout of the READOUT neurons' smoothed rates.

            Fits an intercept-inclusive least-squares readout of the readout
            neurons' box-smoothed rates against BOTH the position target (integral
            of the ORIGINAL velocity) and the velocity target itself, evaluating
            each on the held-out tail. A memoryless readout can reconstruct the
            position only if the injected drive was already integrated upstream.
            """
            R = np.zeros((n_steps, len(output_ids)), dtype=np.float64)
            for j, node_id in enumerate(output_ids):
                R[:, j] = np.convolve(
                    spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
                )

            X = np.hstack([R, np.ones((n_steps, 1), dtype=np.float64)])

            W_p = np.linalg.lstsq(X[fit], p_star[fit], rcond=None)[0]
            r2_position = _r2(p_star[evl], X[evl] @ W_p)

            W_v = np.linalg.lstsq(X[fit], v[fit], rcond=None)[0]
            r2_velocity = _r2(v[evl], X[evl] @ W_v)

            return {"r2_position": float(r2_position), "r2_velocity": float(r2_velocity)}

        # Per-site recompute graft: inject VELOCITY into the site, reconstruct
        # position/velocity from the readout's smoothed rates.
        site_results: Dict[str, Dict[str, float]] = {}
        for site, ids in site_ids_map.items():
            spikes = _run_fixed_drive(
                lesioned_graph, parameters, ids, drive, 1.0, duration_ms, stimulus=None
            )
            site_results[site] = _readout_reconstruction(spikes)
            guard.check(f"after site {site}", logger)

        # Control (a): inject velocity directly into the READOUT. A LIF readout
        # relays velocity (high r2_velocity) but cannot integrate -> position
        # (low r2_position).
        spikes_direct = _run_fixed_drive(
            lesioned_graph, parameters, output_ids, drive, 1.0, duration_ms, stimulus=None
        )
        direct_injection = _readout_reconstruction(spikes_direct)
        guard.check("after direct injection", logger)

        # Control (b): time-shuffled velocity (same values, no temporal structure)
        # injected into a representative site. Fitting against the ORIGINAL p_star
        # should give r2_position ~ 0.
        v_noise = v.copy()
        rng.shuffle(v_noise)
        drive_noise = amplitude * (0.5 + 0.5 * v_noise)

        if "surviving_Int" in site_ids_map:
            noise_site = "surviving_Int"
        else:
            noise_site = next(iter(site_ids_map))
        spikes_noise = _run_fixed_drive(
            lesioned_graph, parameters, site_ids_map[noise_site], drive_noise, 1.0, duration_ms, stimulus=None
        )
        noise_control = {
            "site": noise_site,
            "r2_position": _readout_reconstruction(spikes_noise)["r2_position"],
        }
        guard.check("after noise control", logger)

        rows: list = []
        for site, res in site_results.items():
            rows.append(
                {
                    "site": site,
                    "r2_position": res["r2_position"],
                    "r2_velocity": res["r2_velocity"],
                    "n": len(site_ids_map[site]),
                    **site_feats_map[site],
                }
            )

        def _sort_key(row: Dict[str, Any]) -> float:
            h = row["r2_position"]
            return 0.0 if (isinstance(h, float) and np.isnan(h)) else float(h)

        ranked_rows = sorted(rows, key=lambda r: -_sort_key(r))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "n_target": n_target,
            "sites": ranked_rows,
            "direct_injection": {
                "r2_position": direct_injection["r2_position"],
                "r2_velocity": direct_injection["r2_velocity"],
            },
            "noise_control": noise_control,
            "interpretation": (
                "feedforward-direct sites relay velocity (high r2_velocity, low "
                "r2_position) while the surviving integrator (with slow dynamics) "
                "reconstructs position (high r2_position) -- IF the E/I + slow-tau "
                "circuit actually integrates"
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
                "description": "Velocity-only recompute graft: does a surviving site integrate velocity -> position (non-tautological)?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("recompute graft by site (ranked by r2_position):")
        for r in ranked_rows:
            logger.info(
                "site %-14s r2_position=%.4f r2_velocity=%.4f n=%d",
                r["site"], r["r2_position"], r["r2_velocity"], r["n"],
            )
        logger.info(
            "direct injection (readout): r2_position=%.4f r2_velocity=%.4f",
            direct_injection["r2_position"], direct_injection["r2_velocity"],
        )
        logger.info(
            "noise control (%s): r2_position=%.4f",
            noise_control["site"], noise_control["r2_position"],
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
        logger.exception("recompute graft failed")
        raise
