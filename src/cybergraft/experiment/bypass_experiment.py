"""M3 cross-species mapping + STDP bypass experiment orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.sim.activity import ActivityRecord
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import DirectCurrentStimulus, PoissonStimulus, RhythmicStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .bypass_selection import select_bypass_candidates
from .bypass_training import BypassResult, output_rate, rate_similarity, run_bypass_simulation, run_rstdp_training, timing_similarity
from .lesion import Lesion
from .mapping_interface import MappingInterface
from .motor_primitive import EyeMovementPrimitive, MotorPrimitive

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024); cite Vishwanathan 2017 EM / 2024 analysis"


def _experiment_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['experiment']['name']}-{timestamp}-{config_hash(config)[:8]}"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_stimulus(config: dict, groups: dict, seed: int, dt_ms: float):
    stimulus_type = config.get("type")
    if stimulus_type == "poisson":
        return PoissonStimulus.from_config(config, groups, seed, dt_ms=dt_ms)
    if stimulus_type == "direct_current":
        return DirectCurrentStimulus.from_config(config, groups)
    if stimulus_type == "rhythmic":
        return RhythmicStimulus.from_config(config, groups)
    raise ValueError(f"unsupported stimulus type: {stimulus_type}")


def _build_source_signal(n_steps, n_source, rate_hz, seed, dt_ms, onset_ms, duration_ms):
    """Synthetic Poisson source with a rectangular burst envelope aligned to onset/duration."""
    rng = np.random.default_rng(seed)
    onset_step = int(round(onset_ms / dt_ms))
    burst_steps = int(round(duration_ms / dt_ms))
    end_step = min(n_steps, onset_step + burst_steps)
    probability = min(1.0, rate_hz * dt_ms / 1000.0)
    signal = np.zeros((n_steps, n_source), dtype=bool)
    for step in range(onset_step, end_step):
        signal[step, :] = rng.random(n_source) < probability
    return signal


def _first_spike_latency_ms(spikes, output_ids, onset_ms, dt_ms):
    onset_step = int(round(onset_ms / dt_ms))
    for step in range(onset_step, spikes.shape[0]):
        if spikes[step, output_ids].any():
            return step * dt_ms - onset_ms
    return None


def _decode(primitive, spikes, dt_ms):
    activity = ActivityRecord(
        spikes=spikes,
        mean_membrane_potential=np.zeros(spikes.shape[0], dtype=np.float32),
        dt_ms=dt_ms,
    )
    return primitive.decode(activity)


def run_bypass_experiment(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        if data.get("use_full_connectome", False):
            graph, provenance = adapter.build_full_subgraph(
                readout_type=data["readout_type"],
                lesion_type=data["lesion_type"],
                input_type=data["input_type"],
                max_nodes=int(data["max_nodes"]),
            )
        else:
            graph, provenance = adapter.build_subgraph(
                include_types=list(data["include_types"]),
                readout_type=data["readout_type"],
                input_type=data["input_type"],
                lesion_type=data["lesion_type"],
            )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))

        # Lesion a ranked subset of the integrator population.
        lesion_fraction = float(data["lesion_fraction"])
        output_ids = graph.groups["output"]
        lesion_ids = graph.groups["lesion"]
        drive = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked_lesion = sorted(lesion_ids, key=lambda n: (-drive[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target_ids = ranked_lesion[:n_lesion]
        lesioned_graph, lesion_record = Lesion(method="node_silence").apply(graph, lesion_target_ids)

        # Bypass candidate selection on the lesioned graph.
        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=int(config["experiment"]["seed"]),
        )
        target_node_ids = sel["candidate_ids"]
        n_target = len(target_node_ids)
        if n_target == 0:
            raise ValueError("no bypass candidates")

        n_source = int(data["n_source"])
        source_rate_hz = float(data["source_rate_hz"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        base_seed = int(config["experiment"]["seed"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        source_signal = _build_source_signal(
            n_steps, n_source, source_rate_hz, base_seed, parameters.dt_ms,
            source_onset_ms, source_duration_ms,
        )
        # Matched random control: same mean rate and spike count, different rng seed
        # (matches first-order statistics, not the temporal burst correlation).
        random_signal = _build_source_signal(
            n_steps, n_source, source_rate_hz, base_seed + 1_000_000, parameters.dt_ms,
            source_onset_ms, source_duration_ms,
        )

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        stdp_lr = float(map_cfg["stdp_lr"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        # Amplitude-sensitive reward baseline: intact vs lesioned output rate.
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act_base = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act_base.spikes, output_ids)
        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        reward_baseline = rate_similarity(output_rate(act_lesion.spikes, output_ids), baseline_rate)

        group_names = ["g0", "g1", "g2", "g3", "g4", "g5"]
        output_spikes = {name: [] for name in group_names}
        recoveries = {name: [] for name in group_names}
        recovery_rates: list[float] = []
        recovery_rate_lesion: list[float] = []
        bypass_gains: list[float] = []
        bypass_gains_g3: list[float] = []
        latencies: list[Optional[float]] = []
        activity_g0_0 = None
        activity_g1_0 = None
        activity_g2_0 = None
        final_mapping = None
        final_w0 = None

        for rep in range(n_seeds):
            seed = base_seed + rep
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)

            # G0 intact.
            act_g0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            base = primitive.decode(act_g0)
            output_spikes["g0"].append(int(act_g0.spikes[:, output_ids].sum()))

            # G1 lesioned (no bypass).
            act_g1 = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            post_lesion = primitive.decode(act_g1)
            output_spikes["g1"].append(int(act_g1.spikes[:, output_ids].sum()))

            recovery_lesion = MotorPrimitive.similarity(base, post_lesion)
            recoveries["g1"].append(recovery_lesion)
            recovery_rate_lesion.append(
                rate_similarity(output_rate(act_g1.spikes, output_ids), baseline_rate)
            )

            # G2 bypass (R-STDP training).
            mapping = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            w0 = mapping.W_map.copy()
            res_spikes, mapping = run_rstdp_training(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping,
                target_node_ids=target_node_ids,
                source_signal=source_signal,
                mapping_gain=mapping_gain,
                rstdp_lr=rstdp_lr,
                duration_ms=duration_ms,
                baseline_rate=baseline_rate,
                reward_baseline=reward_baseline,
                n_episodes=n_episodes,
                stimulus=stimulus,
                tau_elig_ms=tau_elig_ms,
            )
            post_bypass = _decode(primitive, res_spikes, parameters.dt_ms)
            output_spikes["g2"].append(int(res_spikes[:, output_ids].sum()))
            recovery = MotorPrimitive.similarity(base, post_bypass)
            recoveries["g2"].append(recovery)
            recovery_rates.append(rate_similarity(output_rate(res_spikes, output_ids), baseline_rate))
            bypass_gains.append(recovery - recovery_lesion)
            latencies.append(_first_spike_latency_ms(res_spikes, output_ids, source_onset_ms, parameters.dt_ms))
            if rep == 0:
                activity_g2_0 = BypassResult(spikes=res_spikes, output_spikes=int(res_spikes[:, output_ids].sum()), mapping=mapping)

            # G3 matched-random bypass (train).
            mapping_g3 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            res_g3 = run_bypass_simulation(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping_g3,
                target_node_ids=target_node_ids,
                source_signal=random_signal,
                mapping_gain=mapping_gain,
                stdp_lr=stdp_lr,
                duration_ms=duration_ms,
                train=True,
                stimulus=stimulus,
            )
            post_g3 = _decode(primitive, res_g3.spikes, parameters.dt_ms)
            output_spikes["g3"].append(int(res_g3.output_spikes))
            recovery_g3 = MotorPrimitive.similarity(base, post_g3)
            recoveries["g3"].append(recovery_g3)
            bypass_gains_g3.append(recovery_g3 - recovery_lesion)

            # G4 no-training bypass.
            mapping_g4 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            res_g4 = run_bypass_simulation(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping_g4,
                target_node_ids=target_node_ids,
                source_signal=source_signal,
                mapping_gain=mapping_gain,
                stdp_lr=stdp_lr,
                duration_ms=duration_ms,
                train=False,
                stimulus=stimulus,
            )
            post_g4 = _decode(primitive, res_g4.spikes, parameters.dt_ms)
            output_spikes["g4"].append(int(res_g4.output_spikes))
            recoveries["g4"].append(MotorPrimitive.similarity(base, post_g4))

            # G5 no-input bypass (train on silence).
            mapping_g5 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            res_g5 = run_bypass_simulation(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping_g5,
                target_node_ids=target_node_ids,
                source_signal=np.zeros((n_steps, n_source), dtype=bool),
                mapping_gain=mapping_gain,
                stdp_lr=stdp_lr,
                duration_ms=duration_ms,
                train=True,
                stimulus=stimulus,
            )
            post_g5 = _decode(primitive, res_g5.spikes, parameters.dt_ms)
            output_spikes["g5"].append(int(res_g5.output_spikes))
            recoveries["g5"].append(MotorPrimitive.similarity(base, post_g5))

            if rep == 0:
                activity_g0_0 = act_g0
                activity_g1_0 = act_g1
            final_mapping = mapping
            final_w0 = w0
            guard.check(f"after simulation rep {rep}", logger)

        def _mean(values):
            return float(np.mean(values)) if values else 0.0

        def _std(values):
            return float(np.std(values)) if values else 0.0

        assert final_mapping is not None and final_w0 is not None
        recovery_mean = _mean(recoveries["g2"])
        recovery_lesion_mean = _mean(recoveries["g1"])
        bypass_gain_mean = _mean(bypass_gains)
        bypass_gain_std = _std(bypass_gains)
        recovery_rate_mean = _mean(recovery_rates)
        recovery_rate_lesion_mean = _mean(recovery_rate_lesion)
        bypass_gain_rate_mean = recovery_rate_mean - recovery_rate_lesion_mean
        lesion_severity_mean = 1.0 - recovery_lesion_mean
        latencies_filtered = [lat for lat in latencies if lat is not None]

        W_final = final_mapping.W_map
        W_delta = W_final - final_w0
        weight_reorganization = {
            "l1": float(np.abs(W_delta).sum()),
            "l2": float(np.linalg.norm(W_delta)),
            "sparsity": float(np.mean(W_final <= 1e-6)),
            "n_changed": int(np.count_nonzero(np.abs(W_delta) > 1e-6)),
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "primitive": primitive.name,
            "n_source": n_source,
            "n_target": n_target,
            "n_seeds": n_seeds,
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": lesion_record["target_count"],
            "bypass_candidate_count": n_target,
            "intact_output_spikes": _mean(output_spikes["g0"]),
            "lesioned_output_spikes": _mean(output_spikes["g1"]),
            "bypass_output_spikes": _mean(output_spikes["g2"]),
            "g3_output_spikes": _mean(output_spikes["g3"]),
            "g4_output_spikes": _mean(output_spikes["g4"]),
            "g5_output_spikes": _mean(output_spikes["g5"]),
            "recovery_mean": recovery_mean,
            "recovery_lesion_mean": recovery_lesion_mean,
            "bypass_gain_mean": bypass_gain_mean,
            "bypass_gain_std": bypass_gain_std,
            "bypass_gain_per_seed": [float(g) for g in bypass_gains],
            "recovery_rate_mean": recovery_rate_mean,
            "recovery_rate_lesion_mean": recovery_rate_lesion_mean,
            "bypass_gain_rate_mean": bypass_gain_rate_mean,
            "rstdp_n_episodes": n_episodes,
            "rstdp_lr": rstdp_lr,
            "recovery_g3_mean": _mean(recoveries["g3"]),
            "recovery_g4_mean": _mean(recoveries["g4"]),
            "recovery_g5_mean": _mean(recoveries["g5"]),
            "bypass_gain_g3_mean": _mean(bypass_gains_g3),
            "lesion_severity_mean": lesion_severity_mean,
            "latency_ms": _mean(latencies_filtered) if latencies_filtered else None,
            "weight_reorganization": weight_reorganization,
            "dataset": provenance["dataset"],
            "source_url": provenance["source_url"],
            "cell_type_counts": provenance["cell_type_counts"],
            "source_signal_note": "synthetic Poisson burst (drosophila_source=synthetic_poisson); correlated with stimulus onset",
            "acceptance": {
                "min_bypass_gain": float(config["acceptance"]["min_bypass_gain"]),
                "min_recovery": float(config["acceptance"]["min_recovery"]),
            },
        }
        metrics["passed"] = bool(
            bypass_gain_rate_mean > metrics["acceptance"]["min_bypass_gain"]
            and recovery_rate_mean > metrics["acceptance"]["min_recovery"]
        )

        assert activity_g0_0 is not None and activity_g1_0 is not None and activity_g2_0 is not None
        activity_g0_0.save(result_dir / "activity_g0_intact.npz")
        activity_g1_0.save(result_dir / "activity_g1_lesioned.npz")
        np.savez_compressed(
            result_dir / "activity_g2_bypass.npz",
            spikes=activity_g2_0.spikes,
            dt_ms=np.float32(parameters.dt_ms),
        )
        final_mapping.save(result_dir / "mapping.npz")
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Cross-species mapping: synthetic Drosophila-like Poisson source mapped onto zebrafish hindbrain readout via STDP. Source signal is synthetic, not real connectome activity.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished M3 experiment: passed=%s, bypass_gain_rate_mean=%.4f, bypass_gain_cosine=%.4f, recovery_rate_mean=%.4f, g0=%.1f, g1=%.1f, g2=%.1f",
            metrics["passed"],
            bypass_gain_rate_mean,
            bypass_gain_mean,
            recovery_rate_mean,
            _mean(output_spikes["g0"]),
            _mean(output_spikes["g1"]),
            _mean(output_spikes["g2"]),
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 experiment failed")
        raise


def generate_informative_source(
    n_steps: int,
    n_source: int,
    lambda_: float,
    command_rate: np.ndarray,
    dt_ms: float,
    onset_ms: float,
    duration_ms: float,
    seed: int,
) -> np.ndarray:
    """Source whose correlation with the target is controlled by lambda_ in [0,1].

    command_rate is the normalized intact output rate (the "command"). lambda_=0
    fires exactly where the target fires; lambda_=1 fires at a flat Poisson rate
    (the mean command rate, i.e. no shared information).
    """
    rng = np.random.default_rng(seed)
    onset_step = int(round(onset_ms / dt_ms))
    end_step = min(n_steps, onset_step + int(round(duration_ms / dt_ms)))
    flat = float(np.mean(command_rate[onset_step:end_step])) if end_step > onset_step else 0.0
    signal = np.zeros((n_steps, n_source), dtype=bool)
    for step in range(onset_step, end_step):
        probability = (1.0 - lambda_) * float(command_rate[step]) + lambda_ * flat
        probability = max(0.0, min(1.0, probability))
        signal[step, :] = rng.random(n_source) < probability
    return signal


def _save_sweep_plot(path: Path, sweep_results: list, m1_point: Optional[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [s["lambda"] for s in sweep_results]
    gains = [s["recovery_gain_mean"] for s in sweep_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, gains, marker="o", color="tab:blue", label="synthetic source")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    if m1_point and m1_point.get("status") == "available":
        ax.scatter(
            [1.0], [m1_point["gain"]], marker="*", s=220, color="tab:red",
            label="M1 activity", zorder=5,
        )
    ax.set_xlabel("lambda (source informativeness)")
    ax.set_ylabel("recovery gain")
    ax.set_title("Bypass recovery gain vs source informativeness")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_informativeness_sweep(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_subgraph(
            include_types=list(data["include_types"]),
            readout_type=data["readout_type"],
            input_type=data["input_type"],
            lesion_type=data["lesion_type"],
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))

        lesion_fraction = float(data["lesion_fraction"])
        output_ids = graph.groups["output"]
        lesion_ids = graph.groups["lesion"]
        drive = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked_lesion = sorted(lesion_ids, key=lambda n: (-drive[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target_ids = ranked_lesion[:n_lesion]
        lesioned_graph, lesion_record = Lesion(method="node_silence").apply(graph, lesion_target_ids)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=int(config["experiment"]["seed"]),
        )
        target_node_ids = sel["candidate_ids"]
        n_target = len(target_node_ids)
        if n_target == 0:
            raise ValueError("no bypass candidates")

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        base_seed = int(config["experiment"]["seed"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act_base = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act_base.spikes, output_ids)
        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        reward_baseline = rate_similarity(output_rate(act_lesion.spikes, output_ids), baseline_rate)
        lesioned_output_spikes = int(act_lesion.spikes[:, output_ids].sum())

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        lambdas = list(config["sweep"]["lambdas"])
        sweep_results = []
        activity_lam0 = None
        activity_lam1 = None

        for lam in lambdas:
            gains = []
            output_spike_list = []
            for rep in range(n_seeds):
                seed = base_seed + rep
                source = generate_informative_source(
                    n_steps, n_source, float(lam), command_rate, parameters.dt_ms,
                    source_onset_ms, source_duration_ms, seed,
                )
                mapping = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
                res_spikes, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=mapping,
                    target_node_ids=target_node_ids,
                    source_signal=source,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus0,
                    tau_elig_ms=tau_elig_ms,
                )
                gain = rate_similarity(output_rate(res_spikes, output_ids), baseline_rate) - reward_baseline
                gains.append(gain)
                output_spike_list.append(int(res_spikes[:, output_ids].sum()))
                if rep == 0:
                    if lam == lambdas[0]:
                        activity_lam0 = res_spikes
                    if lam == lambdas[-1]:
                        activity_lam1 = res_spikes
            sweep_results.append(
                {
                    "lambda": float(lam),
                    "recovery_gain_mean": float(np.mean(gains)),
                    "recovery_gain_std": float(np.std(gains)),
                    "output_spikes_mean": float(np.mean(output_spike_list)),
                }
            )
            guard.check(f"after sweep lambda={lam}", logger)

        boundary_demonstrated = bool(
            sweep_results[0]["recovery_gain_mean"] > sweep_results[-1]["recovery_gain_mean"]
        )

        # Real M1 activity data point (optional).
        m1_point = None
        m1_activity_path = config["sweep"].get("m1_activity_path")
        if m1_activity_path:
            m1_path = Path(m1_activity_path)
            if not m1_path.is_absolute():
                m1_path = project_root / m1_path
            try:
                if m1_path.exists():
                    m1_spikes = np.load(m1_path)["spikes"].astype(bool)
                    n_source_m1 = int(m1_spikes.shape[1])
                    m1_source = np.zeros((n_steps, n_source_m1), dtype=bool)
                    m1_len = min(n_steps, m1_spikes.shape[0])
                    m1_source[:m1_len, :] = m1_spikes[:m1_len, :]
                    m1_mapping = MappingInterface(
                        n_source=n_source_m1, n_target=n_target, seed=base_seed, **mapping_kwargs
                    )
                    m1_res, _ = run_rstdp_training(
                        graph=lesioned_graph,
                        parameters=parameters,
                        mapping=m1_mapping,
                        target_node_ids=target_node_ids,
                        source_signal=m1_source,
                        mapping_gain=mapping_gain,
                        rstdp_lr=rstdp_lr,
                        duration_ms=duration_ms,
                        baseline_rate=baseline_rate,
                        reward_baseline=reward_baseline,
                        n_episodes=n_episodes,
                        stimulus=stimulus0,
                        tau_elig_ms=tau_elig_ms,
                    )
                    m1_recovery = rate_similarity(output_rate(m1_res, output_ids), baseline_rate)
                    m1_point = {
                        "status": "available",
                        "n_source": n_source_m1,
                        "recovery": m1_recovery,
                        "gain": m1_recovery - reward_baseline,
                        "path": str(m1_path),
                    }
                else:
                    m1_point = {"status": "unavailable", "reason": "file not found", "path": str(m1_path)}
            except Exception as exc:
                m1_point = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "path": str(m1_path)}

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": lesion_record["target_count"],
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "lesioned_output_spikes": lesioned_output_spikes,
            "reward_baseline": reward_baseline,
            "sweep": sweep_results,
            "boundary_demonstrated": boundary_demonstrated,
            "m1": m1_point,
            "rstdp_n_episodes": n_episodes,
            "rstdp_lr": rstdp_lr,
            "source_signal_note": "synthetic informative source (lambda controls shared information with target) + optional real M1 activity",
        }
        metrics["passed"] = boundary_demonstrated

        if activity_lam0 is not None:
            np.savez_compressed(
                result_dir / "activity_lam0.npz", spikes=activity_lam0, dt_ms=np.float32(parameters.dt_ms)
            )
        if activity_lam1 is not None:
            np.savez_compressed(
                result_dir / "activity_lam1.npz", spikes=activity_lam1, dt_ms=np.float32(parameters.dt_ms)
            )
        _save_sweep_plot(result_dir / "sweep_curve.png", sweep_results, m1_point)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Informativeness sweep: recovery gain vs source-target shared information (lambda), plus a real-M1-activity source data point.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished M3 sweep: passed=%s, boundary=%s, sweep=%s",
            metrics["passed"],
            boundary_demonstrated,
            [(s["lambda"], round(s["recovery_gain_mean"], 4)) for s in sweep_results],
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 sweep failed")
        raise


def _save_residual_plot(path: Path, grid: list, lambdas: list, fit_lam0: dict, fit_lam1: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in lambdas:
        xs = [g["residual_gap"] for g in grid if g["lambda_"] == lam]
        ys = [g["bypass_recovery_mean"] for g in grid if g["lambda_"] == lam]
        ax.plot(xs, ys, marker="o", label=f"lambda={lam}")
    gaps = sorted({g["residual_gap"] for g in grid})
    if gaps:
        xline = np.linspace(min(gaps), max(gaps), 50)
        ax.plot(
            xline, fit_lam0["slope"] * xline + fit_lam0["intercept"],
            "--", color="tab:blue", alpha=0.6, label="fit lambda=0",
        )
        ax.plot(
            xline, fit_lam1["slope"] * xline + fit_lam1["intercept"],
            "--", color="tab:red", alpha=0.6, label="fit lambda=1",
        )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("residual gap")
    ax.set_ylabel("bypass recovery")
    ax.set_title("Residual-filling decomposition")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_residual_filling(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        G0 = int(act0.spikes[:, output_ids].sum())
        baseline_rate = output_rate(act0.spikes, output_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        lesion_fractions = list(config["sweep"]["lesion_fractions"])
        lambdas = list(config["sweep"]["lambdas"])

        drive = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        grid = []

        for lf in lesion_fractions:
            lf = float(lf)
            ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive[n], n))
            n_lesion = int(np.ceil(lf * len(graph.groups["lesion"])))
            lesion_target = ranked[:n_lesion]
            lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

            act1 = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
            G1 = int(act1.spikes[:, output_ids].sum())
            residual_gap = max(0.0, (G0 - G1) / G0) if G0 > 0 else 0.0
            reward_baseline = rate_similarity(output_rate(act1.spikes, output_ids), baseline_rate)

            sel = select_bypass_candidates(
                lesioned_graph,
                output_group="output",
                lesion_group="lesion",
                max_candidates=int(data["max_bypass_candidates"]),
                seed=base_seed,
            )
            target_ids = sel["candidate_ids"]
            if not target_ids:
                continue

            for lam in lambdas:
                lam = float(lam)
                recoveries = []
                g2s = []
                for rep in range(n_seeds):
                    seed = base_seed + rep
                    source = generate_informative_source(
                        n_steps, n_source, lam, command_rate, parameters.dt_ms,
                        source_onset_ms, source_duration_ms, seed,
                    )
                    mapping = MappingInterface(
                        n_source=n_source, n_target=len(target_ids), seed=seed, **mapping_kwargs
                    )
                    res_spikes, _ = run_rstdp_training(
                        graph=lesioned_graph,
                        parameters=parameters,
                        mapping=mapping,
                        target_node_ids=target_ids,
                        source_signal=source,
                        mapping_gain=mapping_gain,
                        rstdp_lr=rstdp_lr,
                        duration_ms=duration_ms,
                        baseline_rate=baseline_rate,
                        reward_baseline=reward_baseline,
                        n_episodes=n_episodes,
                        stimulus=stimulus0,
                        tau_elig_ms=tau_elig_ms,
                    )
                    G2 = int(res_spikes[:, output_ids].sum())
                    rec = rate_similarity(output_rate(res_spikes, output_ids), baseline_rate) - reward_baseline
                    recoveries.append(float(rec))
                    g2s.append(G2)
                grid.append(
                    {
                        "lesion_fraction": lf,
                        "residual_gap": residual_gap,
                        "lambda_": lam,
                        "g1_mean": float(G1),
                        "g2_mean": float(np.mean(g2s)),
                        "bypass_recovery_mean": float(np.mean(recoveries)),
                        "bypass_recovery_std": float(np.std(recoveries)),
                    }
                )
            guard.check(f"after lesion_fraction={lf}", logger)

        def _fit(lam):
            points = [g for g in grid if g["lambda_"] == lam]
            xs = [g["residual_gap"] for g in points]
            ys = [g["bypass_recovery_mean"] for g in points]
            if len(xs) < 2:
                return {"slope": 0.0, "intercept": 0.0}
            slope, intercept = np.polyfit(xs, ys, 1)
            return {"slope": float(slope), "intercept": float(intercept)}

        fit_lam0 = _fit(lambdas[0])
        fit_lam1 = _fit(lambdas[-1])

        slope_diff = float(fit_lam0["slope"] - fit_lam1["slope"])
        acceptance = {
            "min_slope_diff": float(config["acceptance"].get("min_slope_diff", 0.0)),
        }
        passed = bool(slope_diff > acceptance["min_slope_diff"])

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "G0": G0,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "grid": grid,
            "fit_lam0": fit_lam0,
            "fit_lam1": fit_lam1,
            "slope_diff": slope_diff,
            "acceptance": acceptance,
            "source_signal_note": "synthetic informative source; residual-filling decomposition",
        }
        metrics["passed"] = passed

        _save_residual_plot(result_dir / "residual_filling.png", grid, lambdas, fit_lam0, fit_lam1)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Residual-filling decomposition: 2D sweep over lesion fraction x source informativeness, testing bypass_recovery ~ alpha * residual_gap * informativeness.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished residual-filling: passed=%s, fit_lam0.slope=%.4f, fit_lam1.slope=%.4f",
            metrics["passed"],
            fit_lam0["slope"],
            fit_lam1["slope"],
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 residual-filling failed")
        raise


def _run_fixed_drive(graph, parameters, target_ids, drive, mapping_gain, duration_ms, stimulus=None):
    """Standalone LIF pass injecting a scalar per-step drive into target nodes (no STDP)."""
    p = parameters
    step_count = int(round(duration_ms / p.dt_ms))
    node_count = graph.node_count
    delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
    spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
    spikes_out = np.zeros((step_count, node_count), dtype=np.bool_)
    voltage = np.full(node_count, p.v_rest, dtype=np.float32)
    synaptic_current = np.zeros(node_count, dtype=np.float32)
    refractory = np.zeros(node_count, dtype=np.int32)
    alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
    alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
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
        voltage[active] += alpha_m * (
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


def _save_optimal_plot(path: Path, optimal_results: list, lesioned_recovery: float, timing_lesioned: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [r["lambda_"] for r in optimal_results]
    rate_gains = [r["optimal_gain"] for r in optimal_results]
    timing_gains = [r["timing_gain"] for r in optimal_results]
    residual_norms = [r["residual_norm"] for r in optimal_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, rate_gains, marker="o", color="tab:blue", label="rate_gain")
    ax1.plot(lambdas, timing_gains, marker="s", color="tab:orange", label="timing_gain")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("lambda (source informativeness)")
    ax1.set_ylabel("gain (recovery - lesioned)")
    ax1.set_title("Optimal mapping: rate vs timing gain")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, residual_norms, marker="^", color="tab:red", linestyle=":", label="residual_norm")
    ax2.set_ylabel("residual_norm")
    ax2.legend(loc="upper right")

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_optimal_mapping(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        lesioned_recovery = rate_similarity(lesioned_rate, baseline_rate)
        timing_lesioned = timing_similarity(lesioned_rate, baseline_rate)
        missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])

        map_cfg = config["mapping"]
        mapping_gain = float(map_cfg["gain"])

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        lambdas = list(config["sweep"]["lambdas"])
        optimal_results = []

        for lam in lambdas:
            lam = float(lam)
            source = generate_informative_source(
                n_steps, n_source, lam, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, base_seed,
            )
            w_opt, _, _, _ = np.linalg.lstsq(
                source.astype(np.float64), missing.astype(np.float64), rcond=None
            )
            drive = source @ w_opt
            spikes = _run_fixed_drive(
                lesioned_graph, parameters, target_ids, drive, mapping_gain, duration_ms, stimulus0
            )
            optimal_rate = output_rate(spikes, output_ids)
            optimal_recovery = rate_similarity(optimal_rate, baseline_rate)
            optimal_gain = optimal_recovery - lesioned_recovery
            timing_recovery = timing_similarity(optimal_rate, baseline_rate)
            timing_gain = timing_recovery - timing_lesioned
            residual = source.astype(np.float64) @ w_opt - missing
            residual_norm = float(np.linalg.norm(residual) / (np.linalg.norm(missing) or 1.0))
            optimal_results.append(
                {
                    "lambda_": lam,
                    "lesioned_recovery": lesioned_recovery,
                    "optimal_recovery": optimal_recovery,
                    "optimal_gain": optimal_gain,
                    "timing_recovery": timing_recovery,
                    "timing_gain": timing_gain,
                    "residual_norm": residual_norm,
                }
            )
            guard.check(f"after lambda={lam}", logger)

        passed = bool(optimal_results[0]["optimal_gain"] > optimal_results[-1]["optimal_gain"])

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": len(target_ids),
            "n_source": n_source,
            "lesioned_recovery": lesioned_recovery,
            "rate_lesioned": lesioned_recovery,
            "timing_lesioned": timing_lesioned,
            "optimal": optimal_results,
            "mapping_gain": mapping_gain,
            "source_signal_note": "least-squares optimal mapping (upper bound) from source to missing output drive",
        }
        metrics["passed"] = passed

        _save_optimal_plot(result_dir / "optimal_mapping.png", optimal_results, lesioned_recovery, timing_lesioned)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Optimal-mapping upper bound: least-squares source->drive mapping vs STDP recovery, to test whether STDP is the bottleneck or the informativeness effect is structural.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished optimal-mapping: passed=%s, optimal_gain lam0=%.4f lam1=%.4f, lesioned_recovery=%.4f",
            metrics["passed"],
            optimal_results[0]["optimal_gain"],
            optimal_results[-1]["optimal_gain"],
            lesioned_recovery,
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 optimal-mapping failed")
        raise


def _save_learning_plot(path: Path, rules: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rules.keys())
    gains = [rules[name]["gain"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, gains, color="tab:blue")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("learning rule")
    ax.set_ylabel("rate gain")
    ax.set_title("Learning rule comparison (rate gain)")
    for bar, gain in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{gain:.4f}", ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_learning_rule_comparison(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        lesioned_recovery = rate_similarity(lesioned_rate, baseline_rate)
        missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))
        delta_lr = float(map_cfg.get("delta_lr", 0.01))
        delta_epochs = int(map_cfg.get("delta_epochs", 500))

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        source = generate_informative_source(
            n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
            source_onset_ms, source_duration_ms, base_seed,
        )

        def _recovery_from_drive(drive):
            spikes = _run_fixed_drive(
                lesioned_graph, parameters, target_ids, drive, mapping_gain, duration_ms, stimulus0
            )
            return rate_similarity(output_rate(spikes, output_ids), baseline_rate) - lesioned_recovery

        rules: Dict[str, Any] = {}

        # rstdp_clipped (weight_max = 1.0).
        mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=1.0, seed=base_seed, **mapping_kwargs)
        res_spikes, _ = run_rstdp_training(
            graph=lesioned_graph,
            parameters=parameters,
            mapping=mapping,
            target_node_ids=target_ids,
            source_signal=source,
            mapping_gain=mapping_gain,
            rstdp_lr=rstdp_lr,
            duration_ms=duration_ms,
            baseline_rate=baseline_rate,
            reward_baseline=lesioned_recovery,
            n_episodes=n_episodes,
            stimulus=stimulus0,
            tau_elig_ms=tau_elig_ms,
        )
        gain = rate_similarity(output_rate(res_spikes, output_ids), baseline_rate) - lesioned_recovery
        rules["rstdp_clipped"] = {"gain": gain, "recovery": lesioned_recovery + gain}
        guard.check("after rstdp_clipped", logger)

        # rstdp_unclipped (weight_max = 100.0).
        mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=100.0, seed=base_seed, **mapping_kwargs)
        res_spikes, _ = run_rstdp_training(
            graph=lesioned_graph,
            parameters=parameters,
            mapping=mapping,
            target_node_ids=target_ids,
            source_signal=source,
            mapping_gain=mapping_gain,
            rstdp_lr=rstdp_lr,
            duration_ms=duration_ms,
            baseline_rate=baseline_rate,
            reward_baseline=lesioned_recovery,
            n_episodes=n_episodes,
            stimulus=stimulus0,
            tau_elig_ms=tau_elig_ms,
        )
        gain = rate_similarity(output_rate(res_spikes, output_ids), baseline_rate) - lesioned_recovery
        rules["rstdp_unclipped"] = {"gain": gain, "recovery": lesioned_recovery + gain}
        guard.check("after rstdp_unclipped", logger)

        source_f = source.astype(np.float64)

        # supervised_clipped (delta rule, w clipped to [0,1]).
        w = np.zeros(n_source, dtype=np.float64)
        for _ in range(delta_epochs):
            error = missing - source_f @ w
            grad = source_f.T @ error / n_steps
            w = w + delta_lr * grad
            np.clip(w, 0.0, 1.0, out=w)
        gain = _recovery_from_drive(source @ w)
        rules["supervised_clipped"] = {"gain": gain, "recovery": lesioned_recovery + gain}
        guard.check("after supervised_clipped", logger)

        # supervised_unclipped (delta rule, no clipping).
        w = np.zeros(n_source, dtype=np.float64)
        for _ in range(delta_epochs):
            error = missing - source_f @ w
            grad = source_f.T @ error / n_steps
            w = w + delta_lr * grad
        gain = _recovery_from_drive(source @ w)
        rules["supervised_unclipped"] = {"gain": gain, "recovery": lesioned_recovery + gain}
        guard.check("after supervised_unclipped", logger)

        # oracle (exact least squares).
        w_opt, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
        gain = _recovery_from_drive(source @ w_opt)
        rules["oracle"] = {"gain": gain, "recovery": lesioned_recovery + gain}
        guard.check("after oracle", logger)

        clip_effect = float(rules["rstdp_unclipped"]["gain"] - rules["rstdp_clipped"]["gain"])
        error_signal_effect = float(rules["supervised_clipped"]["gain"] - rules["rstdp_clipped"]["gain"])

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "lesioned_recovery": lesioned_recovery,
            "rules": rules,
            "clip_effect": clip_effect,
            "error_signal_effect": error_signal_effect,
            "source_signal_note": "informative (lambda=0) source; decomposition of STDP->oracle gap into weight clipping vs error signal",
        }
        metrics["passed"] = True

        _save_learning_plot(result_dir / "learning_rule.png", rules)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Learning-rule comparison: decompose the STDP->oracle recovery gap into weight clipping (weight_max) and error signal (R-STDP reward vs supervised delta rule).",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished learning-rule comparison: clip_effect=%.4f, error_signal_effect=%.4f",
            clip_effect,
            error_signal_effect,
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 learning-rule comparison failed")
        raise


def summary_stats(values) -> dict:
    """Mean/std/Cohen's d/95% CI for a list of per-seed measurements."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        cohen_d = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
        return {"mean": mean, "std": 0.0, "cohen_d": cohen_d, "ci_lo": mean, "ci_hi": mean, "n": n}
    cohen_d = mean / std
    se = std / np.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "cohen_d": cohen_d,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "n": n,
    }


def _save_statistical_plot(path: Path, bypass_stats: dict, info_stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def _panel(ax, stats, title):
        mean = stats["mean"]
        ci_lo = stats["ci_lo"]
        ci_hi = stats["ci_hi"]
        per_seed = stats["per_seed"]
        ax.bar(
            [0], [mean], yerr=[[mean - ci_lo], [ci_hi - mean]],
            capsize=8, color="tab:blue", width=0.4, alpha=0.6,
        )
        jitter = np.linspace(-0.1, 0.1, len(per_seed))
        ax.scatter(jitter, per_seed, color="tab:red", alpha=0.6, zorder=3)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks([0])
        ax.set_xticklabels([title])
        ax.set_ylabel("value")
        ax.set_title(title)
        ax.set_xlim(-0.6, 0.6)

    _panel(axes[0], bypass_stats, "bypass_gain")
    _panel(axes[1], info_stats, "informativeness_effect")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_statistical_validation(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        reward_baseline = rate_similarity(lesioned_rate, baseline_rate)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        bypass_gains: list[float] = []
        informativeness_effects: list[float] = []

        for rep in range(n_seeds):
            seed = base_seed + rep
            source0 = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            mapping0 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            res0, _ = run_rstdp_training(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping0,
                target_node_ids=target_ids,
                source_signal=source0,
                mapping_gain=mapping_gain,
                rstdp_lr=rstdp_lr,
                duration_ms=duration_ms,
                baseline_rate=baseline_rate,
                reward_baseline=reward_baseline,
                n_episodes=n_episodes,
                stimulus=stimulus0,
                tau_elig_ms=tau_elig_ms,
            )
            bypass_gain = rate_similarity(output_rate(res0, output_ids), baseline_rate) - reward_baseline

            source1 = generate_informative_source(
                n_steps, n_source, 1.0, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            mapping1 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
            res1, _ = run_rstdp_training(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping1,
                target_node_ids=target_ids,
                source_signal=source1,
                mapping_gain=mapping_gain,
                rstdp_lr=rstdp_lr,
                duration_ms=duration_ms,
                baseline_rate=baseline_rate,
                reward_baseline=reward_baseline,
                n_episodes=n_episodes,
                stimulus=stimulus0,
                tau_elig_ms=tau_elig_ms,
            )
            gain_lam1 = rate_similarity(output_rate(res1, output_ids), baseline_rate) - reward_baseline

            bypass_gains.append(float(bypass_gain))
            informativeness_effects.append(float(bypass_gain - gain_lam1))
            guard.check(f"after seed {rep}", logger)

        bypass_stats = summary_stats(bypass_gains)
        bypass_stats["significant"] = bool(bypass_stats["ci_lo"] > 0.0)
        bypass_stats["per_seed"] = [float(g) for g in bypass_gains]

        info_stats = summary_stats(informativeness_effects)
        info_stats["significant"] = bool(info_stats["ci_lo"] > 0.0)
        info_stats["per_seed"] = [float(g) for g in informativeness_effects]

        passed = bool(bypass_stats["significant"] and info_stats["significant"])

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M4",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "reward_baseline": reward_baseline,
            "bypass_gain": bypass_stats,
            "informativeness_effect": info_stats,
            "acceptance": {
                "min_bypass_gain": float(config["acceptance"]["min_bypass_gain"]),
                "min_informativeness": float(config["acceptance"]["min_informativeness"]),
            },
            "source_signal_note": "statistical validation: informative (lambda=0) bypass gain + informativeness effect (lambda=0 - lambda=1) over seeds",
        }
        metrics["passed"] = passed

        _save_statistical_plot(result_dir / "statistical_validation.png", bypass_stats, info_stats)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M4",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M4 statistical validation: effect size (Cohen's d) + 95% CI + significance for the bypass gain and the informativeness effect, over many seeds.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished statistical validation: passed=%s, bypass_gain d=%.3f CI=[%.4f, %.4f], informativeness d=%.3f CI=[%.4f, %.4f]",
            metrics["passed"],
            bypass_stats["cohen_d"],
            bypass_stats["ci_lo"],
            bypass_stats["ci_hi"],
            info_stats["cohen_d"],
            info_stats["ci_lo"],
            info_stats["ci_hi"],
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
            milestone="M4",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M4 statistical validation failed")
        raise


def _run_vector_drive(graph, parameters, target_ids, drive, mapping_gain, duration_ms, stimulus=None):
    """Standalone LIF pass injecting a per-candidate drive vector per step (no STDP).

    drive has shape (n_target, n_steps): drive[i, step] is injected into target_ids[i].
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
    alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
    alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
    refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))
    target_arr = np.asarray(target_ids, dtype=np.int64)

    for step in range(step_count):
        delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
        synaptic_current *= alpha_syn
        synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
        injected_current = np.zeros(node_count, dtype=np.float32)
        injected_current[target_arr] += drive[:, step].astype(np.float32) * np.float32(mapping_gain)
        if stimulus is not None:
            injected_current += stimulus.current_at(step * p.dt_ms, node_count)
        active = refractory == 0
        voltage[active] += alpha_m * (
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


def _save_directional_plot(path: Path, lesioned_recovery: float, scalar_gain: float, directional_gain: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Documented R-STDP clipped reference gain (from the M3 findings), for visual comparison only.
    stdp_ref = 0.1515
    labels = ["lesioned_recovery", "STDP-ref", "scalar_gain", "directional_gain"]
    values = [lesioned_recovery, stdp_ref, scalar_gain, directional_gain]
    colors = ["tab:gray", "tab:red", "tab:blue", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("value")
    ax.set_title("Directional oracle vs scalar oracle vs STDP")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_directional_oracle(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        lesioned_recovery = rate_similarity(lesioned_rate, baseline_rate)
        missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])

        map_cfg = config["mapping"]
        mapping_gain = float(map_cfg["gain"])

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        source = generate_informative_source(
            n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
            source_onset_ms, source_duration_ms, base_seed,
        )

        # Scalar oracle (directionless), for direct comparison.
        w_scalar, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
        scalar_drive = source @ w_scalar
        scalar_spikes = _run_fixed_drive(
            lesioned_graph, parameters, target_ids, scalar_drive, mapping_gain, duration_ms, stimulus0
        )
        scalar_gain = rate_similarity(output_rate(scalar_spikes, output_ids), baseline_rate) - lesioned_recovery

        # Per-candidate synaptic weight onto the readout.
        syn_weight = np.asarray(
            lesioned_graph.weights[output_ids, :][:, target_ids].sum(axis=0)
        ).ravel().astype(np.float64)
        if syn_weight.sum() == 0.0:
            syn_weight = np.full(n_target, 1.0 / n_target, dtype=np.float64)
        weight_fraction = syn_weight / syn_weight.sum()

        # Distribute the missing output-drive across candidates and fit per-candidate least squares.
        W_map = np.zeros((n_target, n_source), dtype=np.float64)
        source_f = source.astype(np.float64)
        for i in range(n_target):
            missing_i = missing * weight_fraction[i]
            W_map[i, :], _, _, _ = np.linalg.lstsq(source_f, missing_i, rcond=None)

        directional_drive = W_map @ source_f.T  # (n_target, n_steps)
        directional_spikes = _run_vector_drive(
            lesioned_graph, parameters, target_ids, directional_drive, mapping_gain, duration_ms, stimulus0
        )
        directional_rate = output_rate(directional_spikes, output_ids)
        directional_recovery = rate_similarity(directional_rate, baseline_rate)
        directional_gain = directional_recovery - lesioned_recovery

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "lesioned_recovery": lesioned_recovery,
            "scalar_gain": scalar_gain,
            "directional_gain": directional_gain,
            "directional_recovery": directional_recovery,
            "source_signal_note": "directional (per-candidate) least-squares oracle vs directionless scalar oracle, informative lambda=0 source",
        }

        _save_directional_plot(result_dir / "directional_oracle.png", lesioned_recovery, scalar_gain, directional_gain)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Directional (per-candidate) least-squares oracle vs the directionless scalar oracle, to re-test whether STDP is the bottleneck.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished directional oracle: lesioned_recovery=%.4f, scalar_gain=%.4f, directional_gain=%.4f",
            lesioned_recovery,
            scalar_gain,
            directional_gain,
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 directional-oracle failed")
        raise


def _svd_summary(W: np.ndarray) -> dict:
    sv = np.linalg.svd(W, compute_uv=False)
    if sv[0] == 0.0:
        sv_norm = sv
    else:
        sv_norm = sv / sv[0]
    effective_rank = int(np.count_nonzero(sv_norm > 1e-3))
    return {"singular_values": [float(x) for x in sv_norm], "effective_rank": effective_rank}


def _save_svd_plot(path: Path, svd_dict: dict, W_stdp: np.ndarray, W_scalar: np.ndarray, W_dir: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1])

    ax_spec = fig.add_subplot(gs[:, 0])
    for name, color in [("STDP", "tab:blue"), ("scalar", "tab:orange"), ("directional", "tab:green")]:
        sv = svd_dict[name]["singular_values"]
        rank = svd_dict[name]["effective_rank"]
        ax_spec.semilogy(range(len(sv)), np.maximum(sv, 1e-12), color=color, label=f"{name} (rank {rank})")
    ax_spec.set_xlabel("singular value index")
    ax_spec.set_ylabel("sigma / sigma_1 (log)")
    ax_spec.set_title("Normalized singular value spectra")
    ax_spec.legend()

    heatmaps = [("STDP W_map", W_stdp), ("scalar oracle (rank-1)", W_scalar), ("directional oracle", W_dir)]
    for row, (title, W) in enumerate(heatmaps):
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(W, aspect="auto", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_svd_diagnostic(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus0 = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus0)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus0)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
        missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        source = generate_informative_source(
            n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
            source_onset_ms, source_duration_ms, base_seed,
        )
        source_f = source.astype(np.float64)

        # STDP mapping.
        mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=1.0, seed=base_seed, **mapping_kwargs)
        _, mapping = run_rstdp_training(
            graph=lesioned_graph,
            parameters=parameters,
            mapping=mapping,
            target_node_ids=target_ids,
            source_signal=source,
            mapping_gain=mapping_gain,
            rstdp_lr=rstdp_lr,
            duration_ms=duration_ms,
            baseline_rate=baseline_rate,
            reward_baseline=reward_baseline,
            n_episodes=n_episodes,
            stimulus=stimulus0,
            tau_elig_ms=tau_elig_ms,
        )
        W_stdp = mapping.W_map.astype(np.float64)
        guard.check("after STDP mapping", logger)

        # Scalar oracle (rank-1 outer product).
        w_scalar, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
        W_scalar = np.outer(np.ones(n_target), w_scalar)

        # Directional oracle (per-candidate least squares).
        syn_weight = np.asarray(
            lesioned_graph.weights[output_ids, :][:, target_ids].sum(axis=0)
        ).ravel().astype(np.float64)
        if syn_weight.sum() == 0.0:
            syn_weight = np.full(n_target, 1.0 / n_target, dtype=np.float64)
        weight_fraction = syn_weight / syn_weight.sum()
        W_dir = np.zeros((n_target, n_source), dtype=np.float64)
        for i in range(n_target):
            W_dir[i, :], _, _, _ = np.linalg.lstsq(source_f, missing * weight_fraction[i], rcond=None)
        guard.check("after oracle mappings", logger)

        svd_dict = {
            "STDP": _svd_summary(W_stdp),
            "scalar": _svd_summary(W_scalar),
            "directional": _svd_summary(W_dir),
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M3",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "svd": svd_dict,
            "source_signal_note": "SVD rank diagnostic of STDP vs scalar vs directional mapping, informative lambda=0 source",
        }

        _save_svd_plot(result_dir / "svd_diagnostic.png", svd_dict, W_stdp, W_scalar, W_dir)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "SVD diagnostic: singular-value spectra and effective rank of the STDP, scalar, and directional mappings, to test whether direction (rank) is exploited.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished SVD diagnostic: ranks STDP=%d scalar=%d directional=%d",
            svd_dict["STDP"]["effective_rank"],
            svd_dict["scalar"]["effective_rank"],
            svd_dict["directional"]["effective_rank"],
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
            milestone="M3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M3 SVD diagnostic failed")
        raise


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap confidence interval (percentile) for the mean of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    return (
        float(np.percentile(boot_means, 100.0 * alpha / 2.0)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
    )


def permutation_p(values, n_perm=2000, seed=0):
    """One-sample sign-flip permutation p-value against a zero mean."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    n = values.size
    count = 0
    for _ in range(n_perm):
        signs = rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0
        null_mean = float(np.mean(signs * values))
        if observed > 0.0:
            if null_mean >= observed:
                count += 1
        else:
            if null_mean <= observed:
                count += 1
    return (count + 1.0) / (n_perm + 1.0)


def _save_deepening_plot(path: Path, freq_results: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = [f["frequency"] for f in freq_results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _panel(ax, key, title):
        means = [f[key]["mean"] for f in freq_results]
        los = [f[key]["bootstrap_ci_lo"] for f in freq_results]
        his = [f[key]["bootstrap_ci_hi"] for f in freq_results]
        ps = [f[key]["permutation_p"] for f in freq_results]
        yerr_lo = [m - lo for m, lo in zip(means, los)]
        yerr_hi = [hi - m for m, hi in zip(means, his)]
        ax.errorbar(freqs, means, yerr=[yerr_lo, yerr_hi], fmt="o", capsize=5, color="tab:blue", label="mean ± bootstrap 95% CI")
        for fx, m, p in zip(freqs, means, ps):
            ax.annotate(f"p={p:.3f}", (fx, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("value")
        ax.set_title(title)

    _panel(axes[0], "bypass_gain", "bypass_gain")
    _panel(axes[1], "informativeness_effect", "informativeness_effect")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_statistical_deepening(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])

        # Lesion once (shared across frequencies).
        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        frequencies = list(config["sweep"]["frequencies"])
        freq_results = []

        for freq in frequencies:
            freq = float(freq)
            stim_cfg = dict(config["stimulus"])
            stim_cfg["frequency_hz"] = freq
            stimulus = _build_stimulus(stim_cfg, graph.groups, base_seed, parameters.dt_ms)

            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            if baseline_max > 0:
                command_rate = (baseline_rate / baseline_max).astype(np.float64)
            else:
                command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

            bypass_gains: list[float] = []
            informativeness_effects: list[float] = []

            for rep in range(n_seeds):
                seed = base_seed + rep
                source0 = generate_informative_source(
                    n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                    source_onset_ms, source_duration_ms, seed,
                )
                mapping0 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
                res0, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=mapping0,
                    target_node_ids=target_ids,
                    source_signal=source0,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus,
                    tau_elig_ms=tau_elig_ms,
                )
                bypass_gain = rate_similarity(output_rate(res0, output_ids), baseline_rate) - reward_baseline

                source1 = generate_informative_source(
                    n_steps, n_source, 1.0, command_rate, parameters.dt_ms,
                    source_onset_ms, source_duration_ms, seed,
                )
                mapping1 = MappingInterface(n_source=n_source, n_target=n_target, seed=seed, **mapping_kwargs)
                res1, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=mapping1,
                    target_node_ids=target_ids,
                    source_signal=source1,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus,
                    tau_elig_ms=tau_elig_ms,
                )
                gain_lam1 = rate_similarity(output_rate(res1, output_ids), baseline_rate) - reward_baseline

                bypass_gains.append(float(bypass_gain))
                informativeness_effects.append(float(bypass_gain - gain_lam1))
                guard.check(f"after freq={freq} seed={rep}", logger)

            bypass_arr = np.asarray(bypass_gains, dtype=np.float64)
            info_arr = np.asarray(informativeness_effects, dtype=np.float64)

            bypass_stats = summary_stats(bypass_gains)
            b_lo, b_hi = bootstrap_ci(bypass_arr, seed=base_seed)
            bypass_stats["bootstrap_ci_lo"] = b_lo
            bypass_stats["bootstrap_ci_hi"] = b_hi
            bypass_stats["permutation_p"] = permutation_p(bypass_arr, seed=base_seed)

            info_stats = summary_stats(informativeness_effects)
            i_lo, i_hi = bootstrap_ci(info_arr, seed=base_seed)
            info_stats["bootstrap_ci_lo"] = i_lo
            info_stats["bootstrap_ci_hi"] = i_hi
            info_stats["permutation_p"] = permutation_p(info_arr, seed=base_seed)

            freq_results.append(
                {
                    "frequency": freq,
                    "bypass_gain": bypass_stats,
                    "informativeness_effect": info_stats,
                    "n_seeds": n_seeds,
                }
            )

        passed = bool(
            all(
                f["bypass_gain"]["permutation_p"] < 0.05
                and f["informativeness_effect"]["permutation_p"] < 0.05
                for f in freq_results
            )
        )

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M4",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "frequencies": freq_results,
            "source_signal_note": "cross-stimulus (multi-frequency) statistical validation with bootstrap CI + permutation test",
        }
        metrics["passed"] = passed

        _save_deepening_plot(result_dir / "statistical_deepening.png", freq_results)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M4",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M4 statistical deepening: bootstrap CI + permutation test, with cross-stimulus (multi-frequency) repeats.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info("Finished statistical deepening: passed=%s over %d frequencies", metrics["passed"], len(freq_results))
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
            milestone="M4",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M4 statistical deepening failed")
        raise


def _save_testbed_plot(path: Path, sources: dict, lesioned_recovery: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(sources.keys())
    recoveries = [sources[n]["recovery"] for n in names]
    gains = [sources[n]["gain"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, recoveries, width, label="recovery", color="tab:blue")
    ax.bar(x + width / 2, gains, width, label="gain", color="tab:green")
    ax.axhline(lesioned_recovery, color="gray", linestyle="--", label=f"lesioned_recovery ({lesioned_recovery:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("source type")
    ax.set_ylabel("value")
    ax.set_title("Same-species testbed: source difficulty -> recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _seed_stats(values):
    """Aggregate per-seed scalar measurements into mean/std with seed-0 reference."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "seed0": float(values[0]),
        "per_seed": [float(x) for x in values],
    }


def run_testbed_sources(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "weight_max": float(map_cfg["weight_max"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))

        # Real source neuron populations (seed-independent).
        int_ids = [n.id for n in graph.nodes if n.cell_type == "_Int_"]
        int_ids = sorted(int_ids, key=lambda nid: (-drive_onto[nid], nid))[:min(n_source, len(int_ids))]
        axl_ids = [n.id for n in graph.nodes if n.cell_type == "_Axl_"]

        source_names = ["synthetic", "intact_Int", "donor_Axl"]
        n_src_map = {"synthetic": n_source, "intact_Int": len(int_ids), "donor_Axl": len(axl_ids)}
        per_seed_recovery = {name: [] for name in source_names}
        per_seed_gain = {name: [] for name in source_names}
        seed0_reward_baseline = None

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            intact_spikes = act0.spikes
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            if i == 0:
                seed0_reward_baseline = reward_baseline

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            if baseline_max > 0:
                command_rate = (baseline_rate / baseline_max).astype(np.float64)
            else:
                command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

            source_specs = {
                "synthetic": generate_informative_source(
                    n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                    source_onset_ms, source_duration_ms, seed,
                ),
                "intact_Int": intact_spikes[:, int_ids],
                "donor_Axl": intact_spikes[:, axl_ids],
            }
            for name, src in source_specs.items():
                mapping = MappingInterface(n_source=n_src_map[name], n_target=n_target, seed=seed, **mapping_kwargs)
                res, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=mapping,
                    target_node_ids=target_ids,
                    source_signal=src,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus,
                    tau_elig_ms=tau_elig_ms,
                )
                recovery = rate_similarity(output_rate(res, output_ids), baseline_rate)
                per_seed_recovery[name].append(float(recovery))
                per_seed_gain[name].append(float(recovery - reward_baseline))
            guard.check(f"after seed {i}", logger)

        sources: Dict[str, Any] = {}
        for name in source_names:
            rec = _seed_stats(per_seed_recovery[name])
            gain = _seed_stats(per_seed_gain[name])
            sources[name] = {
                "n_src": n_src_map[name],
                "recovery": rec["mean"],
                "gain": gain["mean"],
                "recovery_std": rec["std"],
                "gain_std": gain["std"],
                "recovery_seed0": rec["seed0"],
                "gain_seed0": gain["seed0"],
            }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M5",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_seeds": n_seeds,
            "lesioned_recovery": seed0_reward_baseline,
            "sources": sources,
            "source_signal_note": "same-species testbed: synthetic informative vs intact _Int_ natural-driver vs donor _Axl_ functional-mismatch sources",
        }

        _save_testbed_plot(result_dir / "testbed_sources.png", sources, seed0_reward_baseline)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M5",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M5 step 1: same-species neural-repair testbed, three graded source types (synthetic / intact _Int_ / donor _Axl_), measuring the source-difficulty -> recovery gradient.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished testbed sources (n_seeds=%d): synthetic=%.4f intact_Int=%.4f donor_Axl=%.4f",
            n_seeds,
            sources["synthetic"]["gain"],
            sources["intact_Int"]["gain"],
            sources["donor_Axl"]["gain"],
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
            milestone="M5",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M5 testbed-sources failed")
        raise


def _convergence_episodes(curve, final_recovery):
    if not curve:
        return None
    threshold = 0.9 * final_recovery
    for index, value in enumerate(curve):
        if value >= threshold:
            return index
    return None


def _truncate_curve(curve, max_points=50):
    if len(curve) <= max_points:
        return [float(x) for x in curve]
    indices = np.linspace(0, len(curve) - 1, max_points).astype(int)
    return [float(curve[i]) for i in indices]


def _save_benchmark_plot(path: Path, algorithms: dict, lesioned_recovery: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if algorithms["online_ls"]["convergence_curve"]:
        ax1.plot(algorithms["online_ls"]["convergence_curve"], label="online LS", color="tab:orange")
    if algorithms["rstdp"]["convergence_curve"]:
        ax1.plot(algorithms["rstdp"]["convergence_curve"], label="R-STDP", color="tab:green")
    offline_rec = algorithms["offline_ls"]["recovery"]
    ax1.axhline(offline_rec, color="tab:blue", linestyle="--", label=f"offline LS ({offline_rec:.3f})")
    ax1.axhline(lesioned_recovery, color="gray", linestyle=":", label=f"lesioned ({lesioned_recovery:.3f})")
    ax1.set_xlabel("epoch / iteration")
    ax1.set_ylabel("recovery")
    ax1.set_title("Convergence curves")
    ax1.legend()

    names = list(algorithms.keys())
    gains = [algorithms[n]["gain"] for n in names]
    bars = ax2.bar(names, gains, color=["tab:blue", "tab:orange", "tab:green"])
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("gain")
    ax2.set_title("Final gain")
    for bar, g in zip(bars, gains):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{g:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_algorithm_benchmark(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))
        weight_max = float(map_cfg["weight_max"])
        delta_lr = float(map_cfg.get("delta_lr", 0.01))
        delta_epochs = int(map_cfg.get("delta_epochs", 300))

        algo_names = ["offline_ls", "online_ls", "rstdp"]
        per_seed_recovery = {name: [] for name in algo_names}
        per_seed_gain = {name: [] for name in algo_names}
        per_seed_lesioned = []
        seed0_convergence: Dict[str, Any] = {name: None for name in algo_names}
        seed0_curve: Dict[str, Any] = {name: [] for name in algo_names}

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)
            per_seed_lesioned.append(float(reward_baseline))

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            if baseline_max > 0:
                command_rate = (baseline_rate / baseline_max).astype(np.float64)
            else:
                command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            source_f = source.astype(np.float64)

            # Offline LS (closed-form).
            w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
            offline_spikes = _run_fixed_drive(
                lesioned_graph, parameters, target_ids, source @ w_offline, mapping_gain, duration_ms, stimulus
            )
            recovery_offline = rate_similarity(output_rate(offline_spikes, output_ids), baseline_rate)
            per_seed_recovery["offline_ls"].append(float(recovery_offline))
            per_seed_gain["offline_ls"].append(float(recovery_offline - reward_baseline))

            # Online LS (delta rule).
            w = np.zeros(n_source, dtype=np.float64)
            online_curve = []
            for epoch in range(delta_epochs):
                error = missing - source_f @ w
                grad = source_f.T @ error / n_steps
                w = w + delta_lr * grad
                np.clip(w, 0.0, weight_max, out=w)
                online_spikes = _run_fixed_drive(
                    lesioned_graph, parameters, target_ids, source @ w, mapping_gain, duration_ms, stimulus
                )
                online_curve.append(float(rate_similarity(output_rate(online_spikes, output_ids), baseline_rate)))
            recovery_online = online_curve[-1]
            per_seed_recovery["online_ls"].append(float(recovery_online))
            per_seed_gain["online_ls"].append(float(recovery_online - reward_baseline))

            # R-STDP.
            mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
            res, _, rstdp_curve = run_rstdp_training(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping,
                target_node_ids=target_ids,
                source_signal=source,
                mapping_gain=mapping_gain,
                rstdp_lr=rstdp_lr,
                duration_ms=duration_ms,
                baseline_rate=baseline_rate,
                reward_baseline=reward_baseline,
                n_episodes=n_episodes,
                stimulus=stimulus,
                tau_elig_ms=tau_elig_ms,
                track_recovery=True,
            )
            recovery_rstdp = rate_similarity(output_rate(res, output_ids), baseline_rate)
            per_seed_recovery["rstdp"].append(float(recovery_rstdp))
            per_seed_gain["rstdp"].append(float(recovery_rstdp - reward_baseline))

            if i == 0:
                seed0_convergence["offline_ls"] = None
                seed0_curve["offline_ls"] = []
                seed0_convergence["online_ls"] = _convergence_episodes(online_curve, recovery_online)
                seed0_curve["online_ls"] = _truncate_curve(online_curve)
                seed0_convergence["rstdp"] = _convergence_episodes(rstdp_curve, recovery_rstdp)
                seed0_curve["rstdp"] = _truncate_curve(rstdp_curve)
            guard.check(f"after seed {i}", logger)

        algorithms: Dict[str, Any] = {}
        for name in algo_names:
            rec = _seed_stats(per_seed_recovery[name])
            gain = _seed_stats(per_seed_gain[name])
            algorithms[name] = {
                "recovery": rec["mean"],
                "gain": gain["mean"],
                "recovery_std": rec["std"],
                "gain_std": gain["std"],
                "recovery_seed0": rec["seed0"],
                "gain_seed0": gain["seed0"],
                "convergence_episodes": seed0_convergence[name],
                "convergence_curve": seed0_curve[name],
            }

        lesioned_stats = _seed_stats(per_seed_lesioned)
        headroom = 1.0 - lesioned_stats["mean"]
        gap = algorithms["offline_ls"]["gain"] - algorithms["rstdp"]["gain"]
        gap_frac = gap / headroom if headroom > 0 else float("inf")

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "lesioned_recovery": lesioned_stats["mean"],
            "lesioned_recovery_std": lesioned_stats["std"],
            "headroom": headroom,
            "algorithms": algorithms,
            "gap": gap,
            "gap_frac": gap_frac,
            "source_signal_note": "M6 benchmark: offline LS vs online LS (delta rule) vs R-STDP on the same-species testbed, informative lambda=0 source",
        }

        _save_benchmark_plot(result_dir / "algorithm_benchmark.png", algorithms, lesioned_stats["mean"])
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
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M6: benchmark three neural-repair algorithms (offline LS, online LS delta rule, R-STDP) on the same-species testbed, reporting recovery + convergence.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished algorithm benchmark (n_seeds=%d): offline=%.4f online=%.4f rstdp=%.4f gap_frac=%.4f",
            n_seeds,
            algorithms["offline_ls"]["gain"],
            algorithms["online_ls"]["gain"],
            algorithms["rstdp"]["gain"],
            gap_frac,
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
        logger.exception("M6 algorithm benchmark failed")
        raise


def _save_credit_plot(path: Path, credits: list, credit_mean: float, broadcast_alignment: float, credit_rel_std: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(credits, bins=20, color="tab:blue", alpha=0.7)
    ax.axvline(credit_mean, color="tab:red", linestyle="--", label=f"mean ({credit_mean:.5f})")
    ax.set_xlabel("per-candidate counterfactual credit")
    ax.set_ylabel("count")
    ax.set_title("Per-unit credit signal")
    ax.text(
        0.98, 0.95, f"broadcast_alignment={broadcast_alignment:.4f}\ncredit_rel_std={credit_rel_std:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_credit_signal_analysis(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))
        weight_max = float(map_cfg["weight_max"])

        # Finite-difference step for the marginal credit (relative, on the
        # trained R-STDP mapping row).
        eps = 0.05

        per_seed_R_full = []
        per_seed_plain_gain = []
        per_seed_credit = []  # list of (n_target,) marginal-credit arrays
        per_seed_sign_zero_gain = []
        per_seed_sign_flip_gain = []
        seed0_reward_baseline = None

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)
            if i == 0:
                seed0_reward_baseline = reward_baseline

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            if baseline_max > 0:
                command_rate = (baseline_rate / baseline_max).astype(np.float64)
            else:
                command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            source_f = source.astype(np.float64)

            # Offline-LS ceiling R_full (for reference).
            w, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
            full_spikes = _run_fixed_drive(
                lesioned_graph, parameters, target_ids, source @ w, mapping_gain, duration_ms, stimulus
            )
            R_full = rate_similarity(output_rate(full_spikes, output_ids), baseline_rate)
            per_seed_R_full.append(float(R_full))

            # Train the R-STDP mapping (plain), keep its W_map + final gain.
            mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
            res, _ = run_rstdp_training(
                graph=lesioned_graph,
                parameters=parameters,
                mapping=mapping,
                target_node_ids=target_ids,
                source_signal=source,
                mapping_gain=mapping_gain,
                rstdp_lr=rstdp_lr,
                duration_ms=duration_ms,
                baseline_rate=baseline_rate,
                reward_baseline=reward_baseline,
                n_episodes=n_episodes,
                stimulus=stimulus,
                tau_elig_ms=tau_elig_ms,
            )
            W_trained = mapping.W_map.astype(np.float64)
            per_seed_plain_gain.append(float(rate_similarity(output_rate(res, output_ids), baseline_rate) - reward_baseline))

            # Per-candidate marginal credit via central finite difference of the
            # recovery w.r.t. candidate k's row multiplier around the trained map.
            credit = np.zeros(n_target, dtype=np.float64)
            for k in range(n_target):
                W_plus = W_trained.copy()
                W_minus = W_trained.copy()
                W_plus[k, :] *= (1.0 + eps)
                W_minus[k, :] *= (1.0 - eps)
                spikes_plus = _run_vector_drive(
                    lesioned_graph, parameters, target_ids, W_plus @ source_f.T, mapping_gain, duration_ms, stimulus
                )
                spikes_minus = _run_vector_drive(
                    lesioned_graph, parameters, target_ids, W_minus @ source_f.T, mapping_gain, duration_ms, stimulus
                )
                R_plus = rate_similarity(output_rate(spikes_plus, output_ids), baseline_rate)
                R_minus = rate_similarity(output_rate(spikes_minus, output_ids), baseline_rate)
                credit[k] = (R_plus - R_minus) / (2.0 * eps)
            per_seed_credit.append(credit)

            # Sign-oracle control: zero / flip the negative-credit candidates.
            neg_mask = credit < 0.0
            zero_pu = np.where(neg_mask, 0.0, 1.0).astype(np.float64)
            flip_pu = np.where(neg_mask, -1.0, 1.0).astype(np.float64)
            for pu, bucket in ((zero_pu, per_seed_sign_zero_gain), (flip_pu, per_seed_sign_flip_gain)):
                m = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
                r, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=m,
                    target_node_ids=target_ids,
                    source_signal=source,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus,
                    tau_elig_ms=tau_elig_ms,
                    per_unit_weight=pu,
                )
                bucket.append(float(rate_similarity(output_rate(r, output_ids), baseline_rate) - reward_baseline))
            guard.check(f"after seed {i}", logger)

        R_full_stats = _seed_stats(per_seed_R_full)
        plain_gain_stats = _seed_stats(per_seed_plain_gain)
        sign_zero_stats = _seed_stats(per_seed_sign_zero_gain)
        sign_flip_stats = _seed_stats(per_seed_sign_flip_gain)

        # Aggregate the per-candidate credit vector across seeds.
        credit_matrix = np.asarray(per_seed_credit, dtype=np.float64)  # (n_seeds, n_target)
        credit_vec = credit_matrix.mean(axis=0)
        credit_vec_std = credit_matrix.std(axis=0)

        credit_mean = float(credit_vec.mean())
        credit_std = float(credit_vec.std())
        credit_rel_std = credit_std / credit_mean if credit_mean != 0.0 else float("inf")
        norm_c = float(np.linalg.norm(credit_vec))
        if norm_c == 0.0:
            broadcast_alignment = 0.0
        else:
            broadcast_alignment = float(np.dot(credit_vec, np.ones(n_target)) / (norm_c * np.sqrt(n_target)))
        n_positive = int(np.count_nonzero(credit_vec > 0.0))
        n_negative = int(np.count_nonzero(credit_vec < 0.0))
        n_neutral = int(n_target - n_positive - n_negative)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M7",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "eps": eps,
            "R_full": R_full_stats["mean"],
            "R_full_std": R_full_stats["std"],
            "R_full_seed0": R_full_stats["seed0"],
            "lesioned_recovery": seed0_reward_baseline,
            "credit_mean": credit_mean,
            "credit_std": credit_std,
            "credit_rel_std": credit_rel_std,
            "broadcast_alignment": broadcast_alignment,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_neutral": n_neutral,
            "per_candidate_credit": [float(x) for x in credit_vec],
            "per_candidate_credit_std": [float(x) for x in credit_vec_std],
            "plain_rstdp_gain": plain_gain_stats["mean"],
            "plain_rstdp_gain_std": plain_gain_stats["std"],
            "sign_oracle": {
                "zero_negative": {"gain": sign_zero_stats["mean"], "gain_std": sign_zero_stats["std"], "gain_seed0": sign_zero_stats["seed0"]},
                "flip_negative": {"gain": sign_flip_stats["mean"], "gain_std": sign_flip_stats["std"], "gain_seed0": sign_flip_stats["seed0"]},
            },
            "source_signal_note": "M7: per-unit marginal credit (finite-difference dR/dw around trained R-STDP map) + broadcast bias + sign-oracle control, informative lambda=0 source",
        }

        _save_credit_plot(result_dir / "credit_signal.png", [float(x) for x in credit_vec], credit_mean, broadcast_alignment, credit_rel_std)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M7",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M7 theory path: per-unit marginal credit (finite-difference) and the broadcast-bias metric, plus a sign-oracle control, connecting the per-unit credit heterogeneity to the R-STDP vs supervised gap.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished credit signal analysis: R_full=%.4f, credit_mean=%.5f, broadcast_alignment=%.4f, sign_zero=%.4f sign_flip=%.4f",
            R_full_stats["mean"],
            credit_mean,
            broadcast_alignment,
            sign_zero_stats["mean"],
            sign_flip_stats["mean"],
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
            milestone="M7",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M7 credit signal analysis failed")
        raise


def _save_synapse_weighted_plot(path: Path, plain_rstdp: dict, sw_rstdp: dict, supervised: dict, lesioned_recovery: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["plain_rstdp", "sw_rstdp", "supervised"]
    gains = [plain_rstdp["gain"], sw_rstdp["gain"], supervised["gain"]]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, gains, color=colors)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("gain")
    ax.set_title("Synapse-weighted R-STDP vs plain R-STDP vs supervised")
    for bar, g in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{g:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_synapse_weighted_rstdp(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        sel = select_bypass_candidates(
            lesioned_graph,
            output_group="output",
            lesion_group="lesion",
            max_candidates=int(data["max_bypass_candidates"]),
            seed=base_seed,
        )
        target_ids = sel["candidate_ids"]
        if not target_ids:
            raise ValueError("no bypass candidates")
        n_target = len(target_ids)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        n_seeds = int(config["reproducibility"]["n_seeds"])

        map_cfg = config["mapping"]
        mapping_kwargs = {
            "weight_min": float(map_cfg["weight_min"]),
            "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
            "tau_post_ms": float(map_cfg["tau_post_ms"]),
            "init_scale": float(map_cfg["init_scale"]),
        }
        mapping_gain = float(map_cfg["gain"])
        rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
        n_episodes = int(map_cfg.get("n_episodes", 20))
        tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))
        weight_max = float(map_cfg["weight_max"])

        # Per-candidate synaptic weight onto output, normalized to mean 1 (seed-independent).
        syn_weight = np.asarray(
            lesioned_graph.weights[output_ids, :][:, target_ids].sum(axis=0)
        ).ravel().astype(np.float64)
        syn_mean = syn_weight.mean()
        if syn_mean == 0.0:
            per_unit_weight = np.ones(n_target, dtype=np.float64)
        else:
            per_unit_weight = syn_weight / syn_mean

        per_seed = {name: {"recovery": [], "gain": []} for name in ("plain_rstdp", "sw_rstdp", "supervised")}
        seed0_reward_baseline = None

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)
            if i == 0:
                seed0_reward_baseline = reward_baseline

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            if baseline_max > 0:
                command_rate = (baseline_rate / baseline_max).astype(np.float64)
            else:
                command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )

            def _run_stdp(pu):
                mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
                res, _ = run_rstdp_training(
                    graph=lesioned_graph,
                    parameters=parameters,
                    mapping=mapping,
                    target_node_ids=target_ids,
                    source_signal=source,
                    mapping_gain=mapping_gain,
                    rstdp_lr=rstdp_lr,
                    duration_ms=duration_ms,
                    baseline_rate=baseline_rate,
                    reward_baseline=reward_baseline,
                    n_episodes=n_episodes,
                    stimulus=stimulus,
                    tau_elig_ms=tau_elig_ms,
                    per_unit_weight=pu,
                )
                recovery = rate_similarity(output_rate(res, output_ids), baseline_rate)
                return recovery, recovery - reward_baseline

            recovery_plain, gain_plain = _run_stdp(None)
            recovery_sw, gain_sw = _run_stdp(per_unit_weight)
            per_seed["plain_rstdp"]["recovery"].append(float(recovery_plain))
            per_seed["plain_rstdp"]["gain"].append(float(gain_plain))
            per_seed["sw_rstdp"]["recovery"].append(float(recovery_sw))
            per_seed["sw_rstdp"]["gain"].append(float(gain_sw))

            # Supervised reference (offline LS).
            w, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
            supervised_spikes = _run_fixed_drive(
                lesioned_graph, parameters, target_ids, source @ w, mapping_gain, duration_ms, stimulus
            )
            recovery_supervised = rate_similarity(output_rate(supervised_spikes, output_ids), baseline_rate)
            per_seed["supervised"]["recovery"].append(float(recovery_supervised))
            per_seed["supervised"]["gain"].append(float(recovery_supervised - reward_baseline))
            guard.check(f"after seed {i}", logger)

        results: Dict[str, Any] = {}
        for name in ("plain_rstdp", "sw_rstdp", "supervised"):
            rec = _seed_stats(per_seed[name]["recovery"])
            gain = _seed_stats(per_seed[name]["gain"])
            results[name] = {
                "recovery": rec["mean"],
                "gain": gain["mean"],
                "recovery_std": rec["std"],
                "gain_std": gain["std"],
                "recovery_seed0": rec["seed0"],
                "gain_seed0": gain["seed0"],
            }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M8",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "lesion_fraction": lesion_fraction,
            "lesion_target_count": len(lesion_target),
            "bypass_candidate_count": n_target,
            "n_source": n_source,
            "n_seeds": n_seeds,
            "lesioned_recovery": seed0_reward_baseline,
            "plain_rstdp": results["plain_rstdp"],
            "sw_rstdp": results["sw_rstdp"],
            "supervised": results["supervised"],
            "per_unit_weight": {
                "min": float(per_unit_weight.min()),
                "max": float(per_unit_weight.max()),
                "mean": float(per_unit_weight.mean()),
                "std": float(per_unit_weight.std()),
            },
            "source_signal_note": "M8: synapse-weighted R-STDP (per-candidate syn_weight credit proxy) vs plain R-STDP vs supervised, informative lambda=0 source",
        }

        _save_synapse_weighted_plot(result_dir / "synapse_weighted.png", results["plain_rstdp"], results["sw_rstdp"], results["supervised"], seed0_reward_baseline)
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M8",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M8 method path: synapse-weighted R-STDP using per-candidate synaptic weight as a local credit proxy, to test whether this structure-prior closes part of the R-STDP->supervised gap.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished synapse-weighted R-STDP (n_seeds=%d): plain=%.4f sw=%.4f supervised=%.4f",
            n_seeds,
            results["plain_rstdp"]["gain"],
            results["sw_rstdp"]["gain"],
            results["supervised"]["gain"],
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
            milestone="M8",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M8 synapse-weighted R-STDP failed")
        raise


def run_source_channel_credit(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = _experiment_id(config)
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
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / parameters.dt_ms))
        output_ids = graph.groups["output"]

        base_seed = int(config["experiment"]["seed"])
        stimulus = _build_stimulus(config["stimulus"], graph.groups, base_seed, parameters.dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
        baseline_rate = output_rate(act0.spikes, output_ids)

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(graph.groups["lesion"], key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(graph.groups["lesion"])))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        R_lesion = rate_similarity(lesioned_rate, baseline_rate)
        m = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        if baseline_max > 0:
            command_rate = (baseline_rate / baseline_max).astype(np.float64)
        else:
            command_rate = np.zeros_like(baseline_rate, dtype=np.float64)

        # lambda=0 informative source raster S (T x n_s).
        S = generate_informative_source(
            n_steps, n_source, 0.0, command_rate, parameters.dt_ms,
            source_onset_ms, source_duration_ms, base_seed,
        ).astype(np.float64)
        T = int(S.shape[0])
        n_s = int(S.shape[1])

        # Optimal full-rank linear mapping (the ceiling residual).
        w_star = np.linalg.lstsq(S, m, rcond=None)[0]
        resid_o = float(np.linalg.norm(m - S @ w_star) / np.linalg.norm(m))

        # Rank-1 candidate directions.
        q_sat = np.ones(n_s)
        q_hebb = S.T @ np.ones(T)

        def project(q):
            Sq = S @ q
            beta = float((m @ Sq) / (Sq @ Sq))
            resid_b = float(np.linalg.norm(m - S @ (beta * q)) / np.linalg.norm(m))
            return beta, resid_b

        beta_sat, resid_b_sat = project(q_sat)
        beta_hebb, resid_b_hebb = project(q_hebb)

        # Source-channel credit c = S^T m.
        c = S.T @ m

        def align(q):
            return float((c @ q) / (np.linalg.norm(c) * np.linalg.norm(q)))

        alpha_sat = align(q_sat)
        alpha_hebb = align(q_hebb)

        H = 1.0 - R_lesion
        Delta_pred_sat = float(H * (resid_b_sat - resid_o))
        Delta_pred_hebb = float(H * (resid_b_hebb - resid_o))

        # Rank diagnostics (confirm S is near-rank-1).
        rank_S = int(np.linalg.matrix_rank(S))
        s_vals = np.linalg.svd(S, compute_uv=False)
        sv_top5 = [float(x) for x in s_vals[:5]]
        sv_ratio = float(s_vals[0] / s_vals[1]) if s_vals[1] > 0 else float("inf")

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M7",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "n_s": n_s,
            "T": T,
            "H": H,
            "R_lesion": R_lesion,
            "resid_o": resid_o,
            "resid_b_sat": resid_b_sat,
            "resid_b_hebb": resid_b_hebb,
            "beta_sat": beta_sat,
            "beta_hebb": beta_hebb,
            "alpha_sat": alpha_sat,
            "alpha_hebb": alpha_hebb,
            "Delta_pred_sat": Delta_pred_sat,
            "Delta_pred_hebb": Delta_pred_hebb,
            "rank_S": rank_S,
            "sv_top5": sv_top5,
            "sv_ratio": sv_ratio,
            "source_signal_note": "M7 theory-path: source-channel credit c = S^T m; rank-1 projection (saturation vs Hebbian) vs optimal LS, to test whether the R-STDP gap is a rank/alignment limit",
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
            milestone="M7",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021", "drosophila_source": "synthetic_poisson"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "M7 theory-path follow-up: source-channel credit c = S^T m and rank-1 projection (saturation / Hebbian) residuals vs the optimal LS residual, to decide whether the R-STDP gap is a rank/alignment limit.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished source-channel credit: n_s=%d, H=%.4f, resid_o=%.4f, resid_b_sat=%.4f, resid_b_hebb=%.4f, alpha_sat=%.4f, alpha_hebb=%.4f, Delta_pred_sat=%.5f, Delta_pred_hebb=%.5f",
            n_s, H, resid_o, resid_b_sat, resid_b_hebb, alpha_sat, alpha_hebb, Delta_pred_sat, Delta_pred_hebb,
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
            milestone="M7",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M7 source-channel credit failed")
        raise
