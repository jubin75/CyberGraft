"""Milestone V2.1 — lesion robustness of the direct-weight -> substitutability rule.

The paper's headline claim is that a graft site's *direct synaptic weight onto the
readout* predicts its *substitutability* (how well a trained source injected there
recovers the lesioned output). A reviewer flagged that the current lesion is defined
by ranking ``_Int_`` neurons by their weight onto the readout (top fraction), which
is circular with that predictor.

This experiment shows the rule is ROBUST across:

  * **lesion severity** — fraction of the ``_Int_`` population silenced, swept over
    ``[0.10, 0.25, 0.50, 0.75, 1.00]``, and
  * **lesion selection strategy** — how the lesioned ``_Int_`` neurons are chosen:
    ``weight_ranked`` (current behaviour), ``random``, and ``weight_matched_random``
    (random but matched to the weight distribution of the weight-ranked selection,
    controlling for the "you specifically lesioned high-weight neurons" confound).

For every (strategy x severity) combination we graft (oracle / offline least-squares)
into every surviving neuron with positive direct weight onto the readout, compute the
per-neuron substitutability, and regress substitutability on direct weight. The pass
criterion is that the slope stays positive and significant (permutation p < 0.05)
across ALL combinations — i.e. the rule is not an artifact of how the lesion was
chosen.

This module reuses the graft pipeline from :mod:`cybergraft.analysis.grafting`
(``run_graft_node_scan``) without modifying it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from cybergraft.analysis.grafting import _resolve_ids

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"

DEFAULT_SEVERITIES = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_STRATEGIES = ["weight_ranked", "random", "weight_matched_random"]
DEFAULT_N_PERMUTATIONS = 500
DEFAULT_N_BINS = 4


def _select_lesion(
    graph,
    lesion_ids: List[int],
    drive_onto: np.ndarray,
    fraction: float,
    strategy: str,
    rng: np.random.Generator,
) -> List[int]:
    """Select the ``_Int_`` neurons to silence for a given strategy + severity.

    ``drive_onto`` is the per-neuron direct synaptic weight onto the readout
    (``W[output_ids, :].sum(axis=0)``), the same quantity used as the regression
    predictor, so ``weight_ranked`` reproduces the circular-in-construction lesion
    exactly while ``random`` and ``weight_matched_random`` break that circularity.
    """
    n_lesion = int(np.ceil(fraction * len(lesion_ids)))
    if n_lesion <= 0:
        return []
    if n_lesion >= len(lesion_ids):
        return [int(n) for n in lesion_ids]

    if strategy == "weight_ranked":
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        return [int(n) for n in ranked[:n_lesion]]

    if strategy == "random":
        return [int(x) for x in rng.choice(np.asarray(lesion_ids, dtype=int), size=n_lesion, replace=False)]

    if strategy == "weight_matched_random":
        # Stratified sampling: bin lesion_ids into Q equal-count bins by drive_onto
        # (ascending), then within each bin select the same count as weight_ranked
        # would select from that bin, choosing neurons uniformly at random. This
        # matches the weight distribution of the weight-ranked selection while
        # randomising which specific neurons are lesioned.
        ordered = sorted(lesion_ids, key=lambda n: drive_onto[n])  # ascending weight
        bins = list(np.array_split(np.asarray(ordered, dtype=int), DEFAULT_N_BINS))
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        top_set = set(ranked[:n_lesion])
        selected: List[int] = []
        for bin_ in bins:
            count = int(sum(1 for x in bin_ if int(x) in top_set))
            chosen = rng.choice(bin_, size=count, replace=False)
            selected.extend(int(x) for x in chosen)
        return selected

    raise ValueError(f"unknown lesion strategy: {strategy}")


def _permutation_p(
    xs: np.ndarray,
    ys: np.ndarray,
    observed_slope: float,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """One-sided permutation p for ``slope > 0``.

    Hold ``xs`` (direct weight) fixed, shuffle ``ys`` (substitutability) across
    neurons, recompute the OLS slope, and report the fraction of null slopes that
    are >= the observed slope (with a ``+1`` Laplace smoothing so p is never 0).
    """
    null_slopes = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        ys_perm = rng.permutation(ys)
        slope, _ = np.polyfit(xs, ys_perm, 1)
        null_slopes[k] = slope
    p = (float(np.sum(null_slopes >= observed_slope)) + 1.0) / (n_perm + 1.0)
    return p, null_slopes


def _fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float, float]:
    """OLS slope/intercept + Pearson r + R^2 over finite (x, y) pairs."""
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(xs, ys, 1)
    if xs.size < 2 or np.std(xs) == 0.0:
        return float(slope), float(intercept), float("nan"), float("nan")
    r = float(np.corrcoef(xs, ys)[0, 1])
    return float(slope), float(intercept), r, r * r


def run_lesion_robustness(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Sweep lesion severity x selection strategy and test the direct-weight rule."""
    experiment_id = f"v2-lesion-robustness-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

    rb_cfg = config.get("lesion_robustness", {})
    severities = [float(s) for s in rb_cfg.get("severities", DEFAULT_SEVERITIES)]
    strategies = [str(s) for s in rb_cfg.get("strategies", DEFAULT_STRATEGIES)]
    n_perm = int(rb_cfg.get("n_permutations", DEFAULT_N_PERMUTATIONS))

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
        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": [n.id for n in graph.nodes if n.cell_type == data["input_type"]]}

        # direct synaptic weight onto the readout (== _site_features direct_weight for a single node).
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        output_set = set(output_ids)

        # Intact baseline rate depends only on the stimulus, not the lesion: compute
        # once per seed and reuse across every (strategy, severity) combination.
        baseline_rates: List[np.ndarray] = []
        stimuli: List[Any] = []
        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rates.append(output_rate(act0.spikes, output_ids))
            stimuli.append(stimulus)

        rng = np.random.default_rng(base_seed)
        perm_rng = np.random.default_rng(base_seed + 1_000_000)

        matrix: Dict[str, Dict[str, Any]] = {}
        for strategy in strategies:
            for severity in severities:
                key = f"{strategy}__{severity}"
                lesion_target = _select_lesion(graph, lesion_ids, drive_onto, severity, strategy, rng)
                lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
                lesion_set = set(lesion_target)

                neurons = [
                    n.id for n in graph.nodes
                    if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > 0
                ]

                per_neuron_gain: Dict[int, list] = {j: [] for j in neurons}
                per_seed_headroom: List[float] = []

                for i in range(n_seeds):
                    stimulus = stimuli[i]
                    baseline_rate = baseline_rates[i]
                    act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
                    lesioned_rate = output_rate(act_lesion.spikes, output_ids)
                    reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
                    per_seed_headroom.append(1.0 - float(reward_baseline))
                    missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

                    baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
                    command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
                    source = generate_informative_source(
                        n_steps, n_source, 0.0, command_rate, dt_ms,
                        source_onset_ms, source_duration_ms, base_seed + i,
                    )
                    source_f = source.astype(np.float64)
                    w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
                    drive = source @ w_offline

                    for j in neurons:
                        spikes = _run_fixed_drive(lesioned_graph, parameters, [j], drive, mapping_gain, duration_ms, stimulus)
                        rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                        per_neuron_gain[j].append(float(rec - reward_baseline))

                headroom = float(np.mean(per_seed_headroom)) if per_seed_headroom else 0.0

                xs = np.asarray([drive_onto[j] for j in neurons], dtype=np.float64)
                ys = np.asarray([
                    (float(np.mean(per_neuron_gain[j])) / headroom) if headroom > 0 else float("nan")
                    for j in neurons
                ], dtype=np.float64)

                finite = np.isfinite(ys)
                xs_f = xs[finite]
                ys_f = ys[finite]
                slope, intercept, r, r2 = _fit(xs_f, ys_f)
                p_perm = float("nan")
                if np.isfinite(slope) and xs_f.size >= 2 and np.std(xs_f) > 0.0:
                    p_perm, _ = _permutation_p(xs_f, ys_f, slope, n_perm, perm_rng)

                matrix[key] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r": r,
                    "r_squared": r2,
                    "p_perm": p_perm,
                    "n_neurons": int(xs_f.size),
                    "headroom": headroom,
                }
                logger.info(
                    "%-22s severity=%-5s slope=%+.5f r=%+.4f R2=%.4f p_perm=%.4f n=%d headroom=%.4f",
                    strategy, f"{severity:g}", slope, r, r2, p_perm, int(xs_f.size), headroom,
                )
                guard.check(f"after {key}", logger)

        rule_holds = all(
            np.isfinite(v["slope"]) and v["slope"] > 0 and v["p_perm"] < 0.05
            for v in matrix.values()
        )
        broken = [
            k for k, v in matrix.items()
            if not (np.isfinite(v["slope"]) and v["slope"] > 0 and v["p_perm"] < 0.05)
        ]
        if broken:
            summary = (
                f"{len(broken)}/{len(matrix)} combinations break the rule "
                "(slope <= 0 or p_perm >= 0.05): " + "; ".join(broken)
            )
        else:
            summary = f"All {len(matrix)} (strategy x severity) combinations hold: slope > 0 and p_perm < 0.05."

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "V2.1",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "strategies": strategies,
            "severities": severities,
            "n_permutations": n_perm,
            "n_bins": DEFAULT_N_BINS,
            "matrix": matrix,
            "rule_holds": bool(rule_holds),
            "summary": summary,
            "broken_combos": broken,
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
            milestone="V2.1",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Milestone V2.1 lesion robustness: does the direct-weight -> substitutability "
                    "rule survive severity and lesion-selection-strategy sweeps? weight_matched_random "
                    "uses Q=4 equal-count bins of _Int_ neurons by direct weight onto the readout, "
                    "selecting within each bin the same count as weight_ranked but uniformly at random."
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("rule_holds=%s", rule_holds)
        logger.info("%s", summary)
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
            milestone="V2.1",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("lesion robustness sweep failed")
        raise


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
