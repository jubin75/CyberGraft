#!/usr/bin/env python
"""V3 positive control: does config B produce a persistent (line-attractor) integrator?

Runs the oculomotor integrator under E/I + slow tau (config B) in TWO conditions:

  * intact   — the full circuit
  * lesioned — top 50% of ``_Int_`` silenced by readout weight (V3 lesion)

In each, drive with a velocity step, cut drive, and measure the readout decay
time constant ``tau_tangent`` over the hold window:

  * gradient (line) attractor : ``tau_tangent >> tau_m`` (holds its integral)
  * leaky / dead integrator    : ``tau_tangent ~ tau_m`` (no memory mode)

Comparing intact vs lesioned directly tests whether the lesion destroys the
integrator's persistent (attractor-like) dynamics.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.attractor_analysis.dynamics import Injection, run_dynamics
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_training import output_rate
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters
from cybergraft.utils.config import load_config


def _fit_decay_tau(t_hold_ms: float, y0: float, y_end: float, baseline: float) -> float:
    """Estimate the hold-window decay time constant (ms) from endpoints."""
    dy0 = y0 - baseline
    dy_end = y_end - baseline
    if dy0 <= 0.0:
        return 0.0
    if dy_end <= 0.0:
        return float(t_hold_ms / 3.0)
    if dy_end >= dy0:
        return float("inf")
    return float(-t_hold_ms / np.log(dy_end / dy0))


def _run_assay(graph, parameters, input_ids, output_ids, amplitude, *, T_pre_ms, T_on_ms, T_hold_ms):
    dt_ms = parameters.dt_ms
    duration_ms = T_pre_ms + T_on_ms + T_hold_ms
    n_steps = int(round(duration_ms / dt_ms))
    t = np.arange(n_steps, dtype=np.float64) * dt_ms
    v = np.zeros(n_steps, dtype=np.float32)
    v[(t >= T_pre_ms) & (t < T_pre_ms + T_on_ms)] = np.float32(1.0)

    weights = np.ones(len(input_ids), dtype=np.float32)
    signal = np.float32(amplitude) * v
    result = run_dynamics(
        graph, parameters, duration_ms=duration_ms,
        inject=Injection(target_ids=np.asarray(input_ids, dtype=np.int64), weights=weights, signal=signal, gain=1.0),
        record_voltage=False,
    )
    R = output_rate(result.spikes, output_ids, win=12).astype(np.float64)

    pre_end = int(round(T_pre_ms / dt_ms))
    on_end = int(round((T_pre_ms + T_on_ms) / dt_ms))
    baseline = float(np.mean(R[:pre_end]))
    R_on_window = R[on_end - 100:on_end]
    hold = R[on_end:]

    ramp = float(R_on_window.mean() - baseline)
    R_hold_start = float(np.mean(hold[:100]))
    R_hold_end = float(np.mean(hold[-200:]))
    tau_tangent = _fit_decay_tau(T_hold_ms, R_hold_start, R_hold_end, baseline)
    return {"ramp": ramp, "baseline": baseline, "hold_start": R_hold_start,
            "hold_end": R_hold_end, "tau_tangent_ms": tau_tangent}


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    data = config["data"]
    mat = project_root / data["mat_file"]

    adapter = ZebrafishAdapter(
        mat_path=mat,
        max_nodes=int(data["max_nodes"]),
        synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
    )
    ei = config.get("ei", {})
    graph, _ = adapter.build_full_subgraph(
        readout_type=data["readout_type"],
        lesion_type=data["lesion_type"],
        input_type=data["input_type"],
        max_nodes=int(data["max_nodes"]),
        inhibitory_fraction=float(ei.get("inhibitory_fraction", 0.0)),
        inhibitory_gain=float(ei.get("inhibitory_gain", 1.0)),
    )
    parameters = LIFParameters.from_config(config["simulation"])
    tau_m = parameters.tau_m_ms

    output_ids = [n.id for n in graph.nodes if n.cell_type == data["readout_type"]]
    input_ids = [n.id for n in graph.nodes if n.cell_type == data["input_type"]]
    lesion_ids = [n.id for n in graph.nodes if n.cell_type == data["lesion_type"]]

    # lesion top 50% _Int_ by readout weight (same as V3 runner)
    drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
    ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
    n_lesion = int(np.ceil(float(data["lesion_fraction"]) * len(lesion_ids)))
    lesion_target = ranked[:n_lesion]
    lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

    amplitude = float(config["velocity"]["amplitude"])
    T_pre_ms, T_on_ms, T_hold_ms = 500.0, 500.0, 4000.0
    print(f"[pc] nodes={graph.node_count} input={len(input_ids)} output={len(output_ids)} "
          f"lesioned={len(lesion_target)} tau_m={tau_m}ms")

    intact = _run_assay(graph, parameters, input_ids, output_ids, amplitude,
                        T_pre_ms=T_pre_ms, T_on_ms=T_on_ms, T_hold_ms=T_hold_ms)
    lesioned = _run_assay(lesioned_graph, parameters, input_ids, output_ids, amplitude,
                          T_pre_ms=T_pre_ms, T_on_ms=T_on_ms, T_hold_ms=T_hold_ms)

    def _verdict(tau: float) -> str:
        return "PERSISTENT (line attractor)" if tau > 5.0 * tau_m else "LEAKY/DEAD (no memory mode)"

    print("\n===== V3 POSITIVE CONTROL (config B, E/I + slow tau) =====")
    for label, r in (("intact  ", intact), ("lesioned", lesioned)):
        print(f"[pc] {label}: ramp={r['ramp']:7.2f}  hold {r['hold_start']:6.2f} -> {r['hold_end']:6.2f}  "
              f"tau_tangent={r['tau_tangent_ms']:8.0f} ms  -> {_verdict(r['tau_tangent_ms'])}")
    print(f"[pc] tau_m = {tau_m} ms;  intact tau_tangent = {intact['tau_tangent_ms']:.0f} ms "
          f"({intact['tau_tangent_ms']/tau_m:.1f}x tau_m)  vs  lesioned = {lesioned['tau_tangent_ms']:.0f} ms "
          f"({lesioned['tau_tangent_ms']/tau_m:.1f}x tau_m)")

    result_dir = project_root / config["experiment"]["output_dir"] / (
        f"v3-pc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    out = {
        "tau_m_ms": tau_m,
        "T_pre_ms": T_pre_ms, "T_on_ms": T_on_ms, "T_hold_ms": T_hold_ms,
        "intact": intact, "lesioned": lesioned,
        "verdict_intact": _verdict(intact["tau_tangent_ms"]),
        "verdict_lesioned": _verdict(lesioned["tau_tangent_ms"]),
    }
    (result_dir / "positive_control.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[pc] wrote {result_dir}/positive_control.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
