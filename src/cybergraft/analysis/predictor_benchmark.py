"""Milestone V2.3 — structural predictor benchmark for substitutability.

The reviewer's objection: a correlation (r ~ 0.78, R² ~ 0.60) is not proof that
DIRECT weight is THE structural determinant of graft substitutability. This
experiment computes nine standard per-node structural predictors on the INTACT
graph and compares them head-to-head against the per-neuron substitutability
(weight_ranked @ 50%, 5 Hz, oracle), under univariate association AND 5-fold
held-out prediction.

Predictors (weight matrix ``W[target, source]``, ``C[j] = Σ_o W[o, j]`` = direct
weight, ``A = W``):

  1. ``direct_weight``      = C[i]  (t=0 readout influence)
  2. ``out_degree``         = nnz(W[:, i])
  3. ``in_degree``          = nnz(W[i, :])
  4. ``recurrence``         = fraction of i's outgoing weight landing within i's
                              cell-type population
  5. ``hub_share``          = C[i] / Σ_j C[j]
  6. ``reachability``       = Σ_{t=0}^T ‖C A^t e_i‖²  (path strength, T=10)
  7. ``indirect_influence`` = reachability(i) − direct_weight(i)²  (t≥1 hops)
  8. ``average_controllability`` = Σ_{t=0}^T ‖A^t e_i‖²  (Gu et al. 2015)
  9. ``observability``      = Σ_{t=0}^T ‖(A^T)^t e_i‖²  (Montanari 2022)

All finite-horizon sparse-matrix-power column-norm sums. Reuses the graft node
scan from :mod:`cybergraft.analysis.grafting` and ``_select_lesion`` /
``_permutation_p`` from :mod:`cybergraft.analysis.lesion_robustness`; does not
modify existing modules.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

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
from cybergraft.analysis.lesion_robustness import _select_lesion

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"

PREDICTOR_NAMES = [
    "direct_weight",
    "out_degree",
    "in_degree",
    "recurrence",
    "hub_share",
    "reachability",
    "indirect_influence",
    "average_controllability",
    "observability",
]


def _compute_predictors(graph, output_ids: List[int], horizon: int) -> Dict[str, np.ndarray]:
    """Compute the nine per-node structural predictors on the intact graph."""
    W = graph.weights.tocsr().astype(np.float64)
    n = graph.node_count

    # direct_weight = C = Σ_o W[o, :] (1×n)
    C = np.asarray(W[output_ids, :].sum(axis=0)).ravel().astype(np.float64)

    # out_degree (nnz per column), in_degree (nnz per row)
    out_deg = np.asarray(W.getnnz(axis=0)).astype(np.float64)
    in_deg = np.asarray(W.getnnz(axis=1)).astype(np.float64)

    # recurrence: fraction of outgoing weight landing within the same cell type.
    cell_types = [nd.cell_type for nd in graph.nodes]
    Wc = W.tocsc()
    col_sums = np.asarray(Wc.sum(axis=0)).ravel().astype(np.float64)
    recur = np.zeros(n, dtype=np.float64)
    for ct in set(cell_types):
        idx = np.array([i for i, t in enumerate(cell_types) if t == ct], dtype=int)
        # for each source column j, total weight into cell-type ct
        into_ct = np.asarray(W[idx, :].sum(axis=0)).ravel().astype(np.float64)
        recur[idx] = into_ct[idx]
    recur = np.divide(recur, col_sums, out=np.zeros_like(recur), where=col_sums > 0)

    # hub_share
    total_C = float(C.sum())
    hub_share = C / total_C if total_C > 0 else np.zeros_like(C)

    # reachability: R_0 = C; R_{t+1} = R_t @ A; acc += R_t^2
    R = C.copy()
    reach = R ** 2
    for _ in range(1, horizon + 1):
        R = np.asarray(W.T @ R.reshape(n, 1)).ravel()
        reach += R ** 2
    indirect = reach - C ** 2

    # average_controllability: Q_0 = I; Q_{t+1} = A @ Q_t; acc += col-norms²
    Q = np.eye(n, dtype=np.float64)
    avg_ctrl = np.zeros(n, dtype=np.float64)
    for t in range(horizon + 1):
        avg_ctrl += (Q ** 2).sum(axis=0)
        if t < horizon:
            Q = W @ Q

    # observability: Q_0 = I; Q_{t+1} = A^T @ Q_t
    Wt = W.T
    Q = np.eye(n, dtype=np.float64)
    avg_obs = np.zeros(n, dtype=np.float64)
    for t in range(horizon + 1):
        avg_obs += (Q ** 2).sum(axis=0)
        if t < horizon:
            Q = Wt @ Q

    return {
        "direct_weight": C,
        "out_degree": out_deg,
        "in_degree": in_deg,
        "recurrence": recur,
        "hub_share": hub_share,
        "reachability": reach,
        "indirect_influence": indirect,
        "average_controllability": avg_ctrl,
        "observability": avg_obs,
    }


def _cv_fold_r2(x: np.ndarray, y: np.ndarray, n_folds: int, rng: np.random.Generator) -> Tuple[float, float]:
    """5-fold CV R² for a single predictor (with intercept), mean ± std over folds."""
    n = len(y)
    if n < n_folds or np.std(x) == 0.0:
        return float("nan"), float("nan")
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    r2s: List[float] = []
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        x_tr = x[mask]
        y_tr = y[mask]
        x_te = x[fold]
        y_te = y[fold]
        slope, intercept = np.polyfit(x_tr, y_tr, 1)
        pred = slope * x_te + intercept
        ss_res = float(np.sum((y_te - pred) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        r2s.append((1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))
    return float(np.mean(r2s)), float(np.std(r2s))


def _cv_multivariate_r2(X: np.ndarray, y: np.ndarray, n_folds: int, rng: np.random.Generator) -> Tuple[float, float]:
    """5-fold CV R² for a multivariate OLS model (with intercept)."""
    n, d = X.shape
    if n < n_folds or d == 0:
        return float("nan"), float("nan")
    # z-score each feature for conditioning (R² is affine-invariant).
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    Xz = (X - mu) / sd
    Xd = np.column_stack([Xz, np.ones(n)])
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    r2s: List[float] = []
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        coef, _, _, _ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
        pred = Xd[fold] @ coef
        y_te = y[fold]
        ss_res = float(np.sum((y_te - pred) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        r2s.append((1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))
    return float(np.mean(r2s)), float(np.std(r2s))


def run_predictor_benchmark(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Benchmark 9 structural predictors against per-neuron substitutability."""
    experiment_id = f"v2-predictors-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

    pred_cfg = config.get("predictors", {})
    horizon = int(pred_cfg.get("horizon", 10))
    n_folds = int(pred_cfg.get("n_folds", 5))

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

        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        rng = np.random.default_rng(base_seed)
        lesion_target = _select_lesion(graph, lesion_ids, drive_onto, float(data["lesion_fraction"]), "weight_ranked", rng)
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
        lesion_set = set(lesion_target)
        output_set = set(output_ids)

        neurons = [
            n.id for n in graph.nodes
            if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > 0
        ]

        # ---- per-neuron scan (weight_ranked @ 50%, 5 Hz, oracle) ----
        per_neuron_gain: Dict[int, list] = {j: [] for j in neurons}
        per_seed_headroom: List[float] = []
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
            w_offline, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
            drive = source @ w_offline

            for j in neurons:
                spikes = _run_fixed_drive(lesioned_graph, parameters, [j], drive, mapping_gain, duration_ms, stimulus)
                rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                per_neuron_gain[j].append(float(rec - reward_baseline))
            guard.check(f"after seed {i}", logger)

        headroom = float(np.mean(per_seed_headroom)) if per_seed_headroom else 0.0
        y = np.asarray([
            (float(np.mean(per_neuron_gain[j])) / headroom) if headroom > 0 else float("nan")
            for j in neurons
        ], dtype=np.float64)

        # ---- structural predictors (intact graph) ----
        preds = _compute_predictors(graph, output_ids, horizon)
        # align to surviving neurons
        P: Dict[str, np.ndarray] = {name: preds[name][neurons] for name in PREDICTOR_NAMES}

        # finiteness check for all predictors on the aligned set
        finite_flags = {name: bool(np.isfinite(P[name]).all()) for name in PREDICTOR_NAMES}

        # ---- univariate association ----
        univariate: Dict[str, Dict[str, Any]] = {}
        cv: Dict[str, Dict[str, Any]] = {}
        cv_rng = np.random.default_rng(base_seed + 2_000_000)
        for name in PREDICTOR_NAMES:
            x = P[name]
            mask = np.isfinite(x) & np.isfinite(y)
            xf = x[mask]
            yf = y[mask]
            r = float(np.corrcoef(xf, yf)[0, 1]) if xf.size >= 2 and np.std(xf) > 0 else float("nan")
            rho = float(spearmanr(xf, yf)[0]) if xf.size >= 2 else float("nan")
            univariate[name] = {"r": r, "spearman": rho}

            cv_raw_mean, cv_raw_std = _cv_fold_r2(xf, yf, n_folds, cv_rng)
            logx = np.log1p(xf)
            cv_log_mean, cv_log_std = (float("nan"), float("nan"))
            if np.std(logx) > 0 and np.isfinite(logx).all():
                cv_log_mean, cv_log_std = _cv_fold_r2(logx, yf, n_folds, cv_rng)
            if (not np.isnan(cv_log_mean)) and (np.isnan(cv_raw_mean) or cv_log_mean > cv_raw_mean):
                cv[name] = {
                    "cv_r2_mean": cv_log_mean,
                    "cv_r2_std": cv_log_std,
                    "cv_r2_mean_raw": cv_raw_mean,
                    "cv_r2_mean_log1p": cv_log_mean,
                    "cv_transform": "log1p",
                }
            else:
                cv[name] = {
                    "cv_r2_mean": cv_raw_mean,
                    "cv_r2_std": cv_raw_std,
                    "cv_r2_mean_raw": cv_raw_mean,
                    "cv_r2_mean_log1p": cv_log_mean,
                    "cv_transform": "raw",
                }

        ranking = sorted(PREDICTOR_NAMES, key=lambda n: (-cv[n]["cv_r2_mean"] if not np.isnan(cv[n]["cv_r2_mean"]) else float("-inf")))

        # ---- multivariate model ----
        mv_features = ["direct_weight", "recurrence", "in_degree", "out_degree", "reachability"]
        mask = np.isfinite(y)
        for f in mv_features:
            mask &= np.isfinite(P[f])
        X_mv = np.column_stack([P[f][mask] for f in mv_features])
        y_mv = y[mask]
        mv_mean, mv_std = _cv_multivariate_r2(X_mv, y_mv, n_folds, cv_rng)

        # ---- verdict ----
        dw_cv = cv["direct_weight"]["cv_r2_mean"]
        dw_rank = ranking.index("direct_weight") + 1
        best = ranking[0]
        best_cv = cv[best]["cv_r2_mean"]
        competitive = bool((not np.isnan(dw_cv)) and dw_cv > 0 and dw_rank <= 5)
        verdict = {
            "direct_weight_rank": dw_rank,
            "direct_weight_cv_r2_mean": dw_cv,
            "best_predictor": best,
            "best_cv_r2_mean": best_cv,
            "direct_weight_competitive": competitive,
            "interpretability_note": (
                "direct_weight is the first-order (t=0) readout-synapse term; reachability/"
                "indirect_influence/controllability are higher-order multi-hop composites that "
                "subsume it, so they score >= it but are less interpretable."
            ),
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "V2.3",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "horizon": horizon,
            "n_folds": n_folds,
            "n_neurons": int((~np.isnan(y)).sum()),
            "headroom": headroom,
            "predictors": PREDICTOR_NAMES,
            "predictors_finite": finite_flags,
            "univariate": univariate,
            "cv": cv,
            "ranking": ranking,
            "multivariate": {"features": mv_features, "cv_r2_mean": mv_mean, "cv_r2_std": mv_std},
            "verdict": verdict,
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
            milestone="V2.3",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Milestone V2.3 structural predictor benchmark: 9 per-node predictors "
                    "(direct_weight, out/in_degree, recurrence, hub_share, reachability, "
                    "indirect_influence, average_controllability [Gu et al. 2015], observability "
                    "[Montanari 2022]) vs per-neuron substitutability (weight_ranked @50%, 5Hz, "
                    "oracle). Univariate r/Spearman + 5-fold held-out R2 (raw vs log1p), ranked; "
                    "plus a multivariate model. horizon=10, n_folds=5."
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("ranking (desc by held-out R2):")
        for rank, name in enumerate(ranking, 1):
            logger.info(
                "  %d. %-24s r=%+.4f spearman=%+.4f cv_r2=%+.4f (transform=%s)",
                rank, name, univariate[name]["r"], univariate[name]["spearman"],
                cv[name]["cv_r2_mean"], cv[name]["cv_transform"],
            )
        logger.info("multivariate cv_r2=%.4f +/- %.4f", mv_mean, mv_std)
        logger.info("verdict: %s", verdict)
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
            milestone="V2.3",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("predictor benchmark failed")
        raise


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
