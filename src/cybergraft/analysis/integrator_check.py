"""Integrator verification: does the intact oculomotor circuit integrate velocity?

Feeds a velocity signal (a step, then a sinusoid) to the vestibular input
(``_DOs_``) and asks whether the abducens motoneuron (``ABD_m``) output exhibits
the signature of a velocity-to-position integrator:

  * step velocity      -> output that ramps (accumulates) and then persists (holds)
  * sinusoidal velocity -> output that lags the input by ~90 degrees

A pure integrator transforms velocity ``v`` into position ``p = integral(v)``.
For a step ``v(t) = v0``, position ramps linearly (``p = v0 * t``) and, once
velocity returns to zero, position is *maintained* (memory). For a sinusoid,
``integral(sin) = -cos``, i.e. a 90-degree phase lag.

This is a precondition for the "computational substitutability" direction
(newIdea.md §7.3): if the intact circuit does NOT integrate under the current
LIF model, then asking whether the surviving circuit can reproduce the
integrator's computation is vacuous.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_training import output_rate
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import DirectCurrentStimulus, RhythmicStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _step_metrics(
    rate: np.ndarray,
    dt_ms: float,
    start_ms: float,
    step_duration_ms: float,
    hold_duration_ms: float,
    settle_ms: float,
) -> Dict[str, Any]:
    """Integrator signature from a step-velocity response.

    ``rate`` is the per-step (smoothed) output rate. We compare the pre-step
    baseline, the during-step level (after an onset transient of ``settle_ms``),
    and the post-step level (velocity now zero). An integrator ramps up during the
    step (positive ``ramp_slope_per_ms``) and then *holds* (``hold_ratio`` ~ 1); a
    leaky pass-through jumps to a level quickly (``ramp_slope`` ~ 0) and decays
    (``hold_ratio`` ~ 0).
    """
    n = int(rate.shape[0])
    step_start = int(round(start_ms / dt_ms))
    settle = step_start + int(round(settle_ms / dt_ms))
    step_end = int(round((start_ms + step_duration_ms) / dt_ms))
    hold_end = min(n, int(round((start_ms + step_duration_ms + hold_duration_ms) / dt_ms)))

    baseline = float(np.mean(rate[:step_start])) if step_start > 0 else 0.0
    during = rate[settle:step_end].astype(np.float64)
    post = rate[step_end:hold_end].astype(np.float64)

    if during.size < 2:
        return {"error": "step window too short", "baseline_rate": baseline}

    t_during = np.arange(during.size, dtype=np.float64) * dt_ms
    ramp_slope = float(np.polyfit(t_during, during, 1)[0])  # output-rate units per ms

    post_decay_slope = float("nan")
    if post.size >= 2:
        t_post = np.arange(post.size, dtype=np.float64) * dt_ms
        post_decay_slope = float(np.polyfit(t_post, post, 1)[0])

    step_level = float(np.mean(during))
    post_level = float(np.mean(post)) if post.size else float("nan")

    elevated_step = step_level - baseline
    elevated_post = post_level - baseline
    hold_ratio = float(elevated_post / elevated_step) if elevated_step > 0 else float("nan")

    return {
        "baseline_rate": baseline,
        "step_rate": step_level,
        "post_rate": post_level,
        "ramp_slope_per_ms": ramp_slope,
        "post_decay_slope_per_ms": post_decay_slope,
        "hold_ratio": hold_ratio,
    }


def _reconstruct_velocity(sine_stim: RhythmicStimulus, dt_ms: float, n_steps: int) -> np.ndarray:
    """The zero-mean sinusoidal component of the rhythmic velocity stimulus."""
    t = np.arange(n_steps, dtype=np.float64) * dt_ms
    v = np.zeros(n_steps, dtype=np.float64)
    mask = (t >= sine_stim.start_ms) & (t < sine_stim.start_ms + sine_stim.duration_ms)
    v[mask] = np.sin(2.0 * np.pi * sine_stim.frequency_hz * (t[mask] - sine_stim.start_ms) / 1000.0)
    return v


def _phase_lag_deg(
    velocity: np.ndarray,
    output: np.ndarray,
    dt_ms: float,
    freq_hz: float,
    start_ms: float,
    duration_ms: float,
    steady_frac: float = 0.5,
) -> float:
    """Phase lag (degrees) of ``output`` behind ``velocity`` in a steady window.

    Positive lag means the output trails the input. A pure integrator lags by
    ~90 degrees; a leaky pass-through by ~0 degrees. The lag is found by
    cross-correlating the two signals inside the central ``steady_frac`` of the
    stimulus window (to avoid onset/offset transients).
    """
    onset = int(round(start_ms / dt_ms))
    end = int(round((start_ms + duration_ms) / dt_ms))
    span = end - onset
    if span < 4:
        return float("nan")
    trim = int(round(span * (1.0 - steady_frac) / 2.0))
    a, b = onset + trim, end - trim
    v = velocity[a:b] - velocity[a:b].mean()
    o = output[a:b] - output[a:b].mean()
    if float(np.linalg.norm(v)) == 0.0 or float(np.linalg.norm(o)) == 0.0:
        return float("nan")
    corr = np.correlate(o, v, mode="full")
    lags = np.arange(-len(v) + 1, len(o))
    best_lag = int(lags[int(np.argmax(corr))])
    phase_deg = best_lag * dt_ms / 1000.0 * freq_hz * 360.0
    return float(phase_deg)


def _verdict(step: Dict[str, Any], ramp_slope: float, phase_lag: float) -> str:
    """Classify the integrator signature.

    A velocity integrator MUST show *graded accumulation*: under a step velocity
    the output ramps linearly (``ramp_slope > 0``) rather than jumping instantly.
    ``hold_ratio ~ 1`` alone is ambiguous---it is shared by a true integrator
    (graded memory) and by a bistable latch (self-sustaining saturation). The
    phase lag further disambiguates: an integrator lags by ~+90 degrees, whereas
    a latch/pass-through does not.
    """
    hr = step.get("hold_ratio", float("nan"))
    holds = (not np.isnan(hr)) and hr > 0.5
    ramp = abs(float(ramp_slope)) if ramp_slope is not None and not np.isnan(ramp_slope) else 0.0
    accumulates = ramp > 1e-3  # rate/ms; a graded integrator ramp is clearly positive
    phase_ok = (not np.isnan(phase_lag)) and (30.0 < phase_lag < 150.0)  # near +90 deg
    if accumulates and holds and phase_ok:
        return "integrator (graded accumulation + memory + ~90 deg lag)"
    if (not accumulates) and holds:
        return "bistable-latch / saturation (self-sustaining, NOT a graded integrator)"
    if (not accumulates) and (not holds):
        return "no-integration (fast leaky pass-through)"
    return "partial / ambiguous"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_integrator_check(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = f"p9-integrator-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

    ic = config["integrator_check"]
    settle_ms = float(ic["settle_ms"])
    hold_duration_ms = float(ic["hold_duration_ms"])
    sine_steady_frac = float(ic.get("sine_steady_frac", 0.5))

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
        dt_ms = parameters.dt_ms
        duration_ms = float(config["simulation"]["duration_ms"])
        n_steps = int(round(duration_ms / dt_ms))

        output_ids = [n.id for n in graph.nodes if n.cell_type == data["readout_type"]]
        input_ids = [n.id for n in graph.nodes if n.cell_type == data["input_type"]]
        groups = {"output": output_ids, "input": input_ids}

        step_cfg = dict(config["stimulus"])
        step_stim = DirectCurrentStimulus.from_config(step_cfg, groups)
        step_start_ms = float(step_cfg["start_ms"])
        step_duration_ms = float(step_cfg["duration_ms"])

        sine_cfg = dict(config["sine_stimulus"])
        sine_stim = RhythmicStimulus.from_config(sine_cfg, groups)
        sine_freq_hz = float(sine_cfg["frequency_hz"])
        sine_start_ms = float(sine_cfg["start_ms"])
        sine_duration_ms = float(sine_cfg["duration_ms"])

        per_seed: Dict[str, list] = {"ramp_slope": [], "hold_ratio": [], "phase_lag": []}
        for i in range(n_seeds):
            seed = base_seed + i
            # NOTE: LIF + DirectCurrent/Rhythmic are deterministic given graph+params;
            # the seed does not affect these stimulus types, but we keep the loop to
            # match the reproducibility conventions and allow future stochastic sources.
            _ = seed

            step_spikes = ReferenceLIFSimulator(graph, parameters).run(duration_ms, step_stim).spikes
            step_rate = output_rate(step_spikes, output_ids)
            step_m = _step_metrics(step_rate, dt_ms, step_start_ms, step_duration_ms, hold_duration_ms, settle_ms)

            sine_spikes = ReferenceLIFSimulator(graph, parameters).run(duration_ms, sine_stim).spikes
            sine_rate = output_rate(sine_spikes, output_ids)
            velocity = _reconstruct_velocity(sine_stim, dt_ms, n_steps)
            phase_lag = _phase_lag_deg(
                velocity, sine_rate, dt_ms, sine_freq_hz, sine_start_ms, sine_duration_ms, sine_steady_frac
            )

            if "error" not in step_m:
                per_seed["ramp_slope"].append(step_m["ramp_slope_per_ms"])
                per_seed["hold_ratio"].append(step_m["hold_ratio"])
            per_seed["phase_lag"].append(phase_lag)
            guard.check(f"after seed {i}", logger)

        def _mean(xs):
            xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
            return float(np.mean(xs)) if xs else float("nan")

        ramp_slope = _mean(per_seed["ramp_slope"])
        hold_ratio = _mean(per_seed["hold_ratio"])
        phase_lag = _mean(per_seed["phase_lag"])

        # re-run once more for the final step metrics (report the seed-0 detailed table)
        step_spikes = ReferenceLIFSimulator(graph, parameters).run(duration_ms, step_stim).spikes
        step_rate = output_rate(step_spikes, output_ids)
        step_m = _step_metrics(step_rate, dt_ms, step_start_ms, step_duration_ms, hold_duration_ms, settle_ms)

        verdict = _verdict({"hold_ratio": hold_ratio}, ramp_slope, phase_lag)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "integrator_signature": {
                "ramp_slope_per_ms": ramp_slope,
                "hold_ratio": hold_ratio,
                "phase_lag_deg": phase_lag,
            },
            "step_response": step_m,
            "verdict": verdict,
            "note": (
                "Integrator expectation: ramp_slope > 0 (accumulates), hold_ratio ~ 1 "
                "(memory), phase_lag ~ 90 deg (sin -> -cos). Fast leaky pass-through: "
                "ramp_slope ~ 0, hold_ratio ~ 0, phase_lag ~ 0."
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
                "description": "Integrator verification: does the intact circuit integrate velocity? (step + sinusoid)",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info(
            "integrator check: ramp_slope=%.4f/ms hold_ratio=%.3f phase_lag=%.1f deg -> %s",
            ramp_slope, hold_ratio, phase_lag, verdict,
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
        logger.exception("integrator check failed")
        raise
