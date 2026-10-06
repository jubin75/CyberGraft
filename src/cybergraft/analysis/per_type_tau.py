"""Per-cell-type slow-tau diagnostic: does slow ``_Int_`` membrane tau ramp?

The step-response discriminator (``step_response.py``) showed that with UNIFORM
slow tau (tau_m = 100 ms everywhere) NO site ramps -- the "integrator" is a
low-pass. Here we test the "memory without feedback" hypothesis (Goldman 2009)
directly: give the ``_Int_`` population its OWN, slower membrane time constant (a
per-cell-type tau the connectome does NOT supply) and check whether the
``surviving_Int`` site now shows a SUSTAINED RAMP (true/approximate integrator)
instead of saturating, while a feedforward site (``untyped_direct``) does not.

If the ramp emerges in ``surviving_Int`` but NOT ``untyped_direct`` as
``tau_m_int`` grows, slow membrane dynamics confer integration (memory without
feedback). If ``untyped_direct`` ramps too, the effect is tau-generic rather than
integrator-specific.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from cybergraft.analysis.computational_graft import _build_probe
from cybergraft.analysis.grafting import _select_graft_site
from cybergraft.analysis.step_response import _site_rate, _step_response_metrics

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_fixed_drive_hetero(
    graph,
    parameters: LIFParameters,
    target_ids: list,
    drive: np.ndarray,
    mapping_gain: float,
    duration_ms: float,
    tau_m_ms: np.ndarray,
    tau_syn_ms: np.ndarray,
    stimulus=None,
) -> np.ndarray:
    """Standalone LIF pass with PER-NEURON membrane/synaptic time constants.

    Identical to ``cybergraft.experiment.bypass_helpers._run_fixed_drive`` except
    that ``tau_m_ms`` and ``tau_syn_ms`` are per-neuron arrays (shape
    ``(node_count,)``, or scalars broadcastable to it), so the Euler leak terms
    ``alpha_m`` and ``alpha_syn`` become per-neuron arrays applied element-wise.
    Everything else (spike history, refractory, injected current, target
    injection, stimulus) is unchanged.
    """
    p = parameters
    step_count = int(round(duration_ms / p.dt_ms))
    node_count = graph.node_count
    delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
    spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
    spikes_out = np.zeros((step_count, node_count), dtype=np.bool_)
    voltage = np.full(node_count, p.v_rest, dtype=np.float32)
    synaptic_current = np.zeros(node_count, dtype=np.float32)
    refractory = np.zeros(node_count, dtype=np.int32)

    tau_m_arr = np.broadcast_to(np.asarray(tau_m_ms, dtype=np.float64), (node_count,))
    tau_syn_arr = np.broadcast_to(np.asarray(tau_syn_ms, dtype=np.float64), (node_count,))
    alpha_m = (p.dt_ms / tau_m_arr).astype(np.float32)
    alpha_syn = np.exp(-p.dt_ms / tau_syn_arr).astype(np.float32)

    refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))
    target_arr = np.asarray(target_ids, dtype=np.int64)

    for step in range(step_count):
        delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
        synaptic_current *= alpha_syn
        synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
        injected_current = np.zeros(node_count, dtype=np.float32)
        injected_current[target_arr] += np.float32(drive[step]) * np.float32(mapping_gain)
        if stimulus is not None:
            injected_current += stimulus.current_at(step * p.dt_ms, node_count)
        active = refractory == 0
        voltage[active] += alpha_m[active] * (
            (p.v_rest - voltage[active]) + synaptic_current[active] + injected_current[active]
        )
        voltage[~active] = np.float32(p.v_reset)
        refractory[~active] -= 1
        spikes = active & (voltage >= p.v_th)
        voltage[spikes] = np.float32(p.v_reset)
        refractory[spikes] = refractory_steps
        spikes_out[step] = spikes
        spike_history[step % spike_history.shape[0]] = spikes

    return spikes_out


def run_per_type_tau_step(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Per-cell-type slow-tau diagnostic: does slow ``_Int_`` membrane tau ramp?

    Lesions the top integrator neurons, feeds a velocity STEP through the natural
    input (``_DOs_``), and -- sweeping the ``_Int_``-specific membrane time constant
    -- measures whether the ``surviving_Int`` site develops a sustained ramp
    (integrator) while the feedforward ``untyped_direct`` site remains a low-pass
    relay.
    """
    experiment_id = f"p13-tau-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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
        node_count = graph.node_count

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
        n_target = int(data.get("graft_n_target", 19))

        int_site = _select_graft_site(graph, output_ids, lesion_target, "surviving_Int", n_target, drive_onto, rng)
        untyped_site = _select_graft_site(graph, output_ids, lesion_target, "untyped_direct", n_target, drive_onto, rng)
        if not int_site:
            raise RuntimeError("surviving_Int site is empty; cannot run per-type tau sweep")

        velocity_cfg = config["velocity"]
        amplitude = float(velocity_cfg["amplitude"])
        rate_window = int(config["readout"]["rate_window"])

        # Step velocity probe (0/1, non-negative -- no DC offset needed).
        v = _build_probe("step", velocity_cfg, dt_ms, n_steps)
        drive = amplitude * v

        t_start_idx = int(round(float(velocity_cfg["step_start_ms"]) / dt_ms))
        t_end_idx = int(round((float(velocity_cfg["step_start_ms"]) + float(velocity_cfg["step_duration_ms"])) / dt_ms))

        default_tau_m = float(config["simulation"]["tau_m_ms"])
        default_tau_syn = float(config["simulation"]["tau_syn_ms"])
        tau_sweep = [float(x) for x in config["tau"]["sweep_ms"]]

        sweep_rows: list = []
        for tau_int in tau_sweep:
            # Give every _Int_ neuron (including lesioned, which are silent anyway)
            # the swept membrane tau; everything else keeps the default.
            tau_m_ms = np.full(node_count, default_tau_m, dtype=np.float64)
            for node in graph.nodes:
                if node.cell_type == "_Int_":
                    tau_m_ms[node.id] = tau_int
            tau_syn_ms = np.full(node_count, default_tau_syn, dtype=np.float64)

            spikes = _run_fixed_drive_hetero(
                lesioned_graph, parameters, input_ids, drive, 1.0, duration_ms,
                tau_m_ms, tau_syn_ms, stimulus=None,
            )
            guard.check(f"after tau {tau_int}", logger)

            R_int = _site_rate(spikes, int_site, rate_window, n_steps)
            int_a, int_sr = _step_response_metrics(R_int, t_start_idx, t_end_idx, dt_ms)

            R_un = _site_rate(spikes, untyped_site, rate_window, n_steps)
            un_a, un_sr = _step_response_metrics(R_un, t_start_idx, t_end_idx, dt_ms)

            sweep_rows.append(
                {
                    "tau_m_int": tau_int,
                    "surviving_Int_a": int_a,
                    "surviving_Int_sr": int_sr,
                    "untyped_direct_a": un_a,
                    "untyped_direct_sr": un_sr,
                }
            )

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_target": n_target,
            "default_tau_m_ms": default_tau_m,
            "default_tau_syn_ms": default_tau_syn,
            "tau_sweep": sweep_rows,
            "interpretation": (
                "a sustained ramp (sr->1) emerging in surviving_Int but NOT "
                "untyped_direct as tau_m_int grows => per-type slow membrane "
                "dynamics confer integration (memory without feedback); if "
                "untyped_direct ramps too, the effect is tau-generic not "
                "integrator-specific."
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
                "description": "Per-cell-type slow-tau diagnostic: does slow _Int_ membrane tau produce a sustained ramp (integrator)?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("per-type tau step sweep (surviving_Int vs untyped_direct):")
        for row in sweep_rows:
            logger.info(
                "tau_m_int=%7.1f  surviving_Int a=%.6f sr=%.4f | untyped_direct a=%.6f sr=%.4f",
                row["tau_m_int"],
                row["surviving_Int_a"], row["surviving_Int_sr"],
                row["untyped_direct_a"], row["untyped_direct_sr"],
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
        logger.exception("per-type tau step failed")
        raise
