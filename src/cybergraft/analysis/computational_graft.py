"""Computational substitutability: can the surviving circuit still integrate?

Lesions the top integrator neurons (``_Int_`` ranked by total weight onto the
readout), feeds ONLY the velocity signal into the surviving circuit's input
(``_DOs_``), and asks three closed-form questions about a memoryless (static)
readout of surviving-neuron activity:

  * ``r2_velocity`` (relay capacity): can a static readout reconstruct the
    velocity itself? This measures whether the surviving circuit merely relays
    the injected velocity signal.
  * ``r2_position`` (computational headroom): can a static readout reconstruct
    the POSITION (integral of velocity)? A memoryless readout can only do this
    if the surviving circuit already performs the integration.
  * ``r2_position_integrated`` (relocation ceiling): how well can the relayed
    velocity, when passed through a leaky-integrator readout, reconstruct the
    position?

``relocation = r2_position_integrated - r2_position`` quantifies how much
integration a dynamic readout must supply because the surviving circuit does
not. The expected signature is ``relay_capacity ~ 1``, ``computational_headroom
~ 0``, ``relocation`` large: the surviving circuit relays velocity but does NOT
integrate it, so a dynamic readout recovers the integrator's computation.

All readouts are fit with an intercept column. The sine probe is evaluated
with temporal held-out (fit on the first ``1 - test_frac`` of the signal,
evaluated on the unseen tail); the step probe (piecewise-constant velocity) is
evaluated in-sample.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import _run_fixed_drive
from cybergraft.experiment.bypass_training import output_rate
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard
from cybergraft.graph.sparse_graph import SparseGraph

from cybergraft.analysis.integrator_check import _phase_lag_deg
from cybergraft.analysis.grafting import SITES, _select_graft_site, _site_features

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; NaN when the target has zero variance."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _leaky_integrator(x: np.ndarray, alpha: float) -> np.ndarray:
    """First-order leaky integrator y[t] = alpha*y[t-1] + x[t] (bounded memory)."""
    y = np.empty_like(x, dtype=np.float64)
    acc = 0.0
    for t in range(x.shape[0]):
        acc = alpha * acc + x[t]
        y[t] = acc
    return y


def _build_probe(name: str, cfg: Dict[str, Any], dt_ms: float, n_steps: int) -> np.ndarray:
    """Unit-amplitude velocity probe (before scaling to injected amplitude)."""
    t = np.arange(n_steps, dtype=np.float64) * dt_ms
    v = np.zeros(n_steps, dtype=np.float64)
    if name == "step":
        start = float(cfg["step_start_ms"])
        dur = float(cfg["step_duration_ms"])
        v[(t >= start) & (t < start + dur)] = 1.0
    elif name == "sine":
        freq = float(cfg["sine_frequency_hz"])
        start = float(cfg["sine_start_ms"])
        dur = float(cfg["sine_duration_ms"])
        mask = (t >= start) & (t < start + dur)
        v[mask] = np.sin(2.0 * np.pi * freq * (t[mask] - start) / 1000.0)
    else:
        raise ValueError(f"unknown probe: {name}")
    return v


def _static_position_headroom(
    R: np.ndarray,
    site_ids: list,
    p_star: np.ndarray,
    fit: slice,
    evl: slice,
) -> float:
    """Static (memoryless) readout of ``site_ids`` smoothed rates -> position.

    Fits an intercept-inclusive least-squares readout on the ``fit`` slices and
    evaluates R^2 on the ``evl`` slices. A memoryless readout can reconstruct the
    position only if the site's population already performs the integration.
    """
    X = np.hstack([R[:, site_ids], np.ones((R.shape[0], 1), dtype=np.float64)])
    W = np.linalg.lstsq(X[fit], p_star[fit], rcond=None)[0]
    p_hat = X[evl] @ W
    return _r2(p_star[evl], p_hat)


def _site_computational_headroom(
    graph,
    parameters: LIFParameters,
    input_ids: list,
    site_ids: list,
    velocity_cfg: Dict[str, Any],
    readout_cfg: Dict[str, Any],
    duration_ms: float,
    n_steps: int,
) -> float:
    """Run the sine-velocity probe on ``graph`` and return the static-readout
    position-reconstruction R^2 (temporal held-out) for ``site_ids``.

    Mirrors the drive + smoothing + headroom logic in
    :func:`run_computational_site_scan`, extracted for the recurrence ablation.
    """
    dt_ms = parameters.dt_ms
    amplitude = float(velocity_cfg["amplitude"])
    rate_window = int(readout_cfg["rate_window"])
    test_frac = float(readout_cfg["test_frac"])

    v = _build_probe("sine", velocity_cfg, dt_ms, n_steps)
    drive = amplitude * (0.5 + 0.5 * v)

    spikes = _run_fixed_drive(
        graph, parameters, input_ids, drive, 1.0, duration_ms, stimulus=None
    )

    R = np.zeros((n_steps, graph.node_count), dtype=np.float64)
    for node_id in range(graph.node_count):
        R[:, node_id] = np.convolve(
            spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
        )

    p_star = np.cumsum(v) * dt_ms / 1000.0
    n_train = int(n_steps * (1.0 - test_frac))
    fit, evl = slice(0, n_train), slice(n_train, n_steps)
    return _static_position_headroom(R, site_ids, p_star, fit, evl)


def run_computational_graft(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 1,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = f"p10-compgraft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        lesion_set = set(lesion_target)
        output_set = set(output_ids)
        surviving = [
            n.id for n in graph.nodes if n.id not in output_set and n.id not in lesion_set
        ]

        velocity_cfg = config["velocity"]
        amplitude = float(velocity_cfg["amplitude"])
        rate_window = int(config["readout"]["rate_window"])
        tau_ms = float(config["readout"]["tau_ms"])
        test_frac = float(config["readout"]["test_frac"])
        alpha = float(np.exp(-dt_ms / tau_ms))

        probes: Dict[str, Dict[str, Any]] = {}
        for name in ("step", "sine"):
            v = _build_probe(name, velocity_cfg, dt_ms, n_steps)
            # Inject non-negative current: a DC offset for the sine keeps the
            # relay from half-wave rectifying the bipolar velocity signal.
            drive = amplitude * (0.5 + 0.5 * v) if name == "sine" else amplitude * v

            spikes = _run_fixed_drive(
                lesioned_graph, parameters, input_ids, drive, 1.0, duration_ms, stimulus=None
            )
            guard.check(f"after drive {name}", logger)

            # Per-neuron box-smoothed rate (no cross-neuron summation).
            R = np.zeros((n_steps, len(surviving)), dtype=np.float64)
            for j, node_id in enumerate(surviving):
                R[:, j] = np.convolve(
                    spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
                )

            p_star = np.cumsum(v) * dt_ms / 1000.0

            # Design matrix with an intercept column: absorbs the firing-rate
            # DC baseline so the relayed velocity is zero-mean and integration
            # does not drift linearly.
            X = np.hstack([R, np.ones((n_steps, 1), dtype=np.float64)])

            if name == "sine":
                # Temporal held-out: fit on the first (1 - test_frac) of the
                # signal, evaluate on the unseen tail.
                n_train = int(n_steps * (1.0 - test_frac))
                fit, evl = slice(0, n_train), slice(n_train, n_steps)
            else:
                # Step velocity is piecewise constant, so a temporal split
                # yields zero-variance targets; evaluate in-sample instead.
                fit = evl = slice(0, n_steps)

            # Relay capacity: static readout of velocity.
            W_v = np.linalg.lstsq(X[fit], v[fit], rcond=None)[0]
            v_hat = X @ W_v
            r2_velocity = _r2(v[evl], v_hat[evl])

            # Static computational headroom: static readout of position.
            W_p = np.linalg.lstsq(X[fit], p_star[fit], rcond=None)[0]
            p_hat = X @ W_p
            r2_position = _r2(p_star[evl], p_hat[evl])

            # Dynamic/integrator readout: leaky-integrate the relayed velocity
            # (bounded memory; pure cumsum would drift under residual bias).
            p_int = _leaky_integrator(v_hat, alpha) * dt_ms / 1000.0
            r2_position_integrated = _r2(p_star[evl], p_int[evl])

            relocation = r2_position_integrated - r2_position

            entry: Dict[str, Any] = {
                "r2_velocity": r2_velocity,
                "r2_position": r2_position,
                "r2_position_integrated": r2_position_integrated,
                "relocation": relocation,
            }
            if name == "sine":
                entry["phase_lag_integrated_deg"] = _phase_lag_deg(
                    v,
                    p_int,
                    dt_ms,
                    float(velocity_cfg["sine_frequency_hz"]),
                    float(velocity_cfg["sine_start_ms"]),
                    float(velocity_cfg["sine_duration_ms"]),
                )
            probes[name] = entry

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_surviving": len(surviving),
            "probes": probes,
            "headline": {
                "computational_headroom": probes["sine"]["r2_position"],
                "relay_capacity": probes["sine"]["r2_velocity"],
                "relocation": probes["sine"]["relocation"],
                "phase_lag_integrated_deg": probes["sine"]["phase_lag_integrated_deg"],
            },
            "interpretation": (
                "relay_capacity≈1 + computational_headroom≈0 => surviving circuit relays "
                "velocity but does NOT integrate it; relocation = integration a dynamic "
                "readout can supply"
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
                "description": "Computational substitutability: lesion integrator, feed velocity, static vs dynamic readout of surviving activity.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        for name in ("step", "sine"):
            logger.info(
                "probe %-4s r2_velocity=%.4f r2_position=%.4f r2_position_integrated=%.4f relocation=%.4f",
                name,
                probes[name]["r2_velocity"],
                probes[name]["r2_position"],
                probes[name]["r2_position_integrated"],
                probes[name]["relocation"],
            )
        logger.info(
            "headline: relay_capacity=%.4f computational_headroom=%.4f relocation=%.4f",
            metrics["headline"]["relay_capacity"],
            metrics["headline"]["computational_headroom"],
            metrics["headline"]["relocation"],
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
        logger.exception("computational graft failed")
        raise


def run_computational_site_scan(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 1,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Per-site COMPUTATIONAL headroom: which surviving structure carries the
    velocity->position integration?

    Lesions the top integrator neurons (``_Int_`` ranked by total weight onto the
    readout), feeds ONLY the velocity signal into the surviving circuit's input
    (``_DOs_``), runs the whole surviving circuit once, then -- for each of the 6
    graft sites -- asks whether a STATIC readout of that site's neurons can
    reconstruct the POSITION (integral of velocity). A memoryless readout can only
    do this if that site's population already performs the integration.

    Contrast with the readout-substitutability ranking (feedforward-direct sites
    best, recurrent ``surviving_Int`` worst): if ``surviving_Int`` now ranks
    HIGHEST here, the recurrent integrator remnant carries the integration even
    though it is the worst graft recipient for a dynamic readout.
    """
    experiment_id = f"p10-compsite-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        # gold_top19 selection reads graph.groups["output"]/["lesion"].
        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

        rng = np.random.default_rng(int(config["experiment"]["seed"]))
        n_target = 19

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

        spikes = _run_fixed_drive(
            lesioned_graph, parameters, input_ids, drive, 1.0, duration_ms, stimulus=None
        )
        guard.check("after drive", logger)

        # Per-neuron box-smoothed rate, indexed by node id (so R[:, site_ids]
        # directly selects a site's surviving neurons).
        R = np.zeros((n_steps, graph.node_count), dtype=np.float64)
        for node_id in range(graph.node_count):
            R[:, node_id] = np.convolve(
                spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
            )

        p_star = np.cumsum(v) * dt_ms / 1000.0

        # Temporal held-out: fit on the first (1 - test_frac), eval on the tail.
        n_train = int(n_steps * (1.0 - test_frac))
        fit, evl = slice(0, n_train), slice(n_train, n_steps)

        rows: list = []
        for site in site_ids_map:
            site_ids = site_ids_map[site]
            headroom = _static_position_headroom(R, site_ids, p_star, fit, evl)
            rows.append(
                {
                    "site": site,
                    "computational_headroom": float(headroom),
                    "n": len(site_ids),
                    **site_feats_map[site],
                }
            )

        def _sort_key(row: Dict[str, Any]) -> float:
            h = row["computational_headroom"]
            return 0.0 if (isinstance(h, float) and np.isnan(h)) else float(h)

        ranked_rows = sorted(rows, key=lambda r: -_sort_key(r))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "n_sites": len(rows),
            "n_target": n_target,
            "sites": ranked_rows,
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
                "description": "Per-site computational headroom: which surviving structure carries the velocity->position integration?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("computational headroom by site (ranked):")
        for r in ranked_rows:
            logger.info(
                "site %-14s headroom=%.4f direct_weight=%.4f recurrence=%.4f",
                r["site"],
                r["computational_headroom"],
                r["direct_weight"],
                r["recurrence"],
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
        logger.exception("computational site scan failed")
        raise


def run_computational_ablation(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Causal test: does the surviving integrator's recurrence carry the computation?

    Lesions the top integrator neurons (as in :func:`run_computational_site_scan`),
    then measures the ``surviving_Int`` site's computational headroom (static
    readout -> position, temporal held-out) BEFORE and AFTER removing the recurrent
    (``_Int_`` -> ``_Int_``) synapses among the surviving integrators. If recurrence
    causally carries the velocity->position integration, the headroom should drop
    after ablation.
    """
    experiment_id = f"p10-compablate-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

        rng = np.random.default_rng(int(config["experiment"]["seed"]))
        n_target = 19
        int_site = _select_graft_site(
            graph, output_ids, lesion_target, "surviving_Int", n_target, drive_onto, rng
        )
        if not int_site:
            raise RuntimeError("surviving_Int site is empty; cannot run ablation")

        velocity_cfg = config["velocity"]
        readout_cfg = config["readout"]

        baseline = _site_computational_headroom(
            lesioned_graph, parameters, input_ids, int_site,
            velocity_cfg, readout_cfg, duration_ms, n_steps,
        )
        guard.check("after baseline drive", logger)

        # Ablate: drop recurrent edges among the FULL surviving integrator
        # population (mirror run_graft_role_intervention, which filters on the
        # whole surviving ``_Int_`` set rather than the top-n site). The site is
        # only the readout target; the recurrence that could carry the
        # integration spans the whole surviving population.
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

        ablated = _site_computational_headroom(
            ablated_graph, parameters, input_ids, int_site,
            velocity_cfg, readout_cfg, duration_ms, n_steps,
        )
        guard.check("after ablated drive", logger)

        n_removed = int(len(lesioned_graph.edges) - len(kept))
        delta = float(ablated - baseline)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_target": n_target,
            "n_surviving_Int": len(surviving_int),
            "n_recurrent_edges_removed": n_removed,
            "surviving_Int_baseline_headroom": float(baseline),
            "surviving_Int_ablated_headroom": float(ablated),
            "delta": delta,
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
                "description": "Computational recurrence ablation: does the surviving integrator's recurrence causally carry the velocity->position integration?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info(
            "surviving_Int headroom: baseline=%.4f ablated=%.4f delta=%+.4f (recurrent edges removed=%d)",
            baseline, ablated, delta, n_removed,
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
        logger.exception("computational ablation failed")
        raise
