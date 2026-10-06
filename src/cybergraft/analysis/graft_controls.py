"""Graft circularity controls: direct-injection baseline + noise-source control.

The readout-substitutability experiment (``grafting.py``) builds its graft source
from the INTACT output command (``command_rate = baseline_rate / baseline_max``),
fits ``W = lstsq(source, missing)``, and injects ``source @ W`` into a graft site.
Because the source already encodes the output, substitutability is near-tautological
(≈ direct synaptic weight). Two controls bound this:

  * **Direct-injection baseline**: inject the SAME command drive directly into the
    READOUT (``ABD_m``) instead of a graft site. If this already reaches ~1.0,
    grafting through surviving neurons adds nothing.
  * **Noise-source control**: inject a source with NO output information (the
    ``generate_informative_source`` ``lambda_=1.0`` flat-Poisson mode) into each
    graft site. If substitutability collapses to ~0, the result depends on the
    source encoding the output, not just on the site's structure.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import (
    _build_stimulus,
    _run_fixed_drive,
    generate_informative_source,
)
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from cybergraft.analysis.grafting import SITES, _select_graft_site, _site_features, _resolve_ids

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_graft_controls(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 5,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Two controls bounding the readout-substitutability experiment.

    Mirrors :func:`run_graft_site_sweep` for the setup (lesion, site selection,
    source construction), then -- per seed -- computes three quantities:

      * per-site command substitutability (the normal graft),
      * a direct-injection baseline (command drive injected into the readout),
      * a per-site noise-source control (flat-Poisson source, ``lambda_=1.0``).
    """
    experiment_id = f"p6-controls-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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
    base_seed = int(config["experiment"]["seed"])

    try:
        adapter = ZebrafishAdapter(
            mat_path=mat,
            max_nodes=int(data["max_nodes"]),
            synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
        )
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        duration_ms = float(config["simulation"]["duration_ms"])
        dt_ms = parameters.dt_ms
        n_steps = int(round(duration_ms / dt_ms))

        output_ids, lesion_ids = _resolve_ids(graph, data["readout_type"], data["lesion_type"])
        input_ids = [n.id for n in graph.nodes if n.cell_type == data["input_type"]]
        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
        guard.check("after lesion", logger)

        n_source = int(data["n_source"])
        n_target = int(data.get("graft_n_target", 19))
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        rng = np.random.default_rng(base_seed)
        site_ids: Dict[str, list] = {}
        site_feats: Dict[str, Dict[str, Any]] = {}
        for site in SITES:
            ids = _select_graft_site(graph, output_ids, lesion_target, site, n_target, drive_onto, rng)
            if not ids:
                continue
            site_ids[site] = ids
            site_feats[site] = _site_features(graph, output_ids, ids)

        per_site_gain: Dict[str, list] = {s: [] for s in site_ids}
        per_site_noise_gain: Dict[str, list] = {s: [] for s in site_ids}
        per_seed_headroom: list = []
        per_seed_direct_gain: list = []

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            per_seed_headroom.append(1.0 - float(reward_baseline))
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            source_f = source.astype(np.float64)
            w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
            drive = source @ w_offline

            # Per-site command substitutability (the normal graft).
            for site, ids in site_ids.items():
                spikes = _run_fixed_drive(lesioned_graph, parameters, ids, drive, mapping_gain, duration_ms, stimulus)
                rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                per_site_gain[site].append(float(rec - reward_baseline))

            # Direct-injection control: inject the command drive into the READOUT.
            spikes_d = _run_fixed_drive(lesioned_graph, parameters, output_ids, drive, mapping_gain, duration_ms, stimulus)
            rec_d = rate_similarity(output_rate(spikes_d, output_ids), baseline_rate)
            per_seed_direct_gain.append(float(rec_d - reward_baseline))

            # Noise-source control: flat-Poisson source (lambda_=1.0), no output info.
            source_noise = generate_informative_source(
                n_steps, n_source, 1.0, command_rate, dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            w_n, _, _, _ = np.linalg.lstsq(source_noise.astype(np.float64), missing, rcond=None)
            drive_noise = source_noise @ w_n
            for site, ids in site_ids.items():
                spikes_n = _run_fixed_drive(lesioned_graph, parameters, ids, drive_noise, mapping_gain, duration_ms, stimulus)
                rec_n = rate_similarity(output_rate(spikes_n, output_ids), baseline_rate)
                per_site_noise_gain[site].append(float(rec_n - reward_baseline))

            guard.check(f"after seed {i}", logger)

        headroom = float(np.mean(per_seed_headroom))

        command_sites: list = []
        for site in site_ids:
            gains = np.asarray(per_site_gain[site], dtype=np.float64)
            gain = float(gains.mean())
            substitutability = gain / headroom if headroom > 0 else float("nan")
            command_sites.append(
                {
                    "site": site,
                    "substitutability": substitutability,
                    "gain": gain,
                    **site_feats[site],
                }
            )

        direct_gain = float(np.mean(per_seed_direct_gain))
        direct_injection = {
            "substitutability": direct_gain / headroom if headroom > 0 else float("nan"),
            "gain": direct_gain,
        }

        noise_sites: list = []
        for site in site_ids:
            gains = np.asarray(per_site_noise_gain[site], dtype=np.float64)
            gain = float(gains.mean())
            noise_sites.append(
                {
                    "site": site,
                    "substitutability": gain / headroom if headroom > 0 else float("nan"),
                    "gain": gain,
                }
            )

        def _sort_key(row: Dict[str, Any]) -> float:
            h = row["substitutability"]
            return 0.0 if (isinstance(h, float) and np.isnan(h)) else float(h)

        command_sites = sorted(command_sites, key=lambda r: -_sort_key(r))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "headroom": headroom,
            "command_sites": command_sites,
            "direct_injection": direct_injection,
            "noise_sites": noise_sites,
            "interpretation": (
                "direct_injection ~ ceiling (source already encodes output => inject "
                "into readout directly); noise_source ~ 0 (substitutability requires "
                "the source to carry the output command)."
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
                "description": "Graft circularity controls: direct-injection baseline + noise-source control.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("command substitutability by site (ranked):")
        for r in command_sites:
            logger.info(
                "site %-14s substitutability=%.4f gain=%.4f direct_weight=%.4f recurrence=%.4f",
                r["site"], r["substitutability"], r["gain"], r["direct_weight"], r["recurrence"],
            )
        logger.info(
            "direct injection control: substitutability=%.4f gain=%.4f",
            direct_injection["substitutability"], direct_injection["gain"],
        )
        logger.info("noise-source control by site:")
        for r in noise_sites:
            logger.info(
                "site %-14s substitutability=%.4f gain=%.4f",
                r["site"], r["substitutability"], r["gain"],
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
        logger.exception("graft controls failed")
        raise
