"""Multivariate OLS models for the V3 mechanistic-analysis layer (PRD §19).

Three nested OLS regressions relate per-node graft metrics to wiring/attractor
predictors. Pure NumPy (``np.linalg.lstsq``); no sklearn/pandas/networkx.
Missing/None values are treated as missing and dropped (complete-case), never
zero-filled.
"""

import numpy as np

__all__ = ["design_matrix", "ols_fit", "nested_delta_r2", "run_regressions"]

# (model name, target field, predictor fields, key predictor for nested delta_r2)
MODELS = [
    (
        "model_1",
        "readout_substitutability",
        ["direct_readout_weight", "recurrence", "attractor_support_score", "degree"],
        "direct_readout_weight",
    ),
    (
        "model_2",
        "computational_graftability",
        ["attractor_support_score", "recurrence", "direct_readout_weight", "degree"],
        "attractor_support_score",
    ),
    (
        "model_3",
        "computational_headroom",
        ["attractor_support_score", "recurrence", "direct_readout_weight", "degree"],
        "attractor_support_score",
    ),
]


def _column(records: list[dict], field: str) -> np.ndarray:
    """Extract one record field as a float64 array (``nan`` for missing)."""
    arr = np.full(len(records), np.nan, dtype=np.float64)
    for i, rec in enumerate(records):
        v = rec.get(field)
        if v is not None:
            arr[i] = float(v)
    return arr


def design_matrix(
    records: list[dict],
    predictor_fields: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build an intercept-augmented design matrix and a valid-row mask.

    Returns ``(X, mask)`` where ``X`` has shape ``(n_kept, 1 + p)`` (leading
    column of ones for the intercept) and ``mask`` is a boolean array of length
    ``len(records)`` marking rows where every predictor is present and finite.
    """
    n = len(records)
    mask = np.ones(n, dtype=bool)
    cols: list = []
    for f in predictor_fields:
        col = np.full(n, np.nan, dtype=np.float64)
        for i, rec in enumerate(records):
            v = rec.get(f)
            if v is None:
                mask[i] = False
            else:
                col[i] = float(v)
        cols.append(col)
    for c in cols:
        mask &= np.isfinite(c)
    keep = np.nonzero(mask)[0]
    X = np.column_stack([np.ones(keep.size, dtype=np.float64)] + [c[keep] for c in cols])
    return X, mask


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares via ``np.linalg.lstsq``.

    ``X`` must include the intercept column. Returns the predictor coefficient
    vector (``(p,)``), the intercept, R², sample size, and an empty
    ``predictor_names`` list (populated by the caller when names are known).
    Returns NaN coefficients when there are not more rows than columns.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(X.shape[0])
    p = int(X.shape[1] - 1)
    coef = np.full(p, np.nan, dtype=np.float64)
    intercept = float("nan")
    r2 = float("nan")
    if n > X.shape[1]:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=np.float64)
    return {
        "coef": coef,
        "intercept": intercept,
        "r2": r2,
        "n": n,
        "predictor_names": [],
    }


def nested_delta_r2(X_base: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> dict:
    """R² of a base model vs a full model that adds the last column(s).

    Returns ``{"r2_base", "r2_full", "delta_r2"}`` where ``delta_r2 = r2_full -
    r2_base`` (the incremental variance explained by the added predictors).
    """
    r2_base = ols_fit(X_base, y)["r2"]
    r2_full = ols_fit(X_full, y)["r2"]
    delta = float("nan")
    if not (np.isnan(r2_base) or np.isnan(r2_full)):
        delta = r2_full - r2_base
    return {"r2_base": r2_base, "r2_full": r2_full, "delta_r2": delta}


def run_regressions(records: list[dict]) -> dict:
    """Three OLS models (PRD §19); full fit plus key-predictor nested ΔR².

    Each model is fit on the rows where the target and all predictors are
    present. Returns, per model: the full-model ``ols_fit`` coefficients (with
    ``predictor_names``), R², ``n``, and the nested ``delta_r2`` for the key
    predictor added last vs. a base model without it. Keys: ``model_1``,
    ``model_2``, ``model_3``.
    """
    results: dict = {}
    for name, target, predictors, key in MODELS:
        y_all = _column(records, target)
        pred_cols = {f: _column(records, f) for f in predictors}

        valid = np.isfinite(y_all)
        for f in predictors:
            valid &= np.isfinite(pred_cols[f])
        keep = np.nonzero(valid)[0]
        n = int(keep.size)

        if n == 0:
            results[name] = {
                "coef": np.full(len(predictors), np.nan, dtype=np.float64),
                "intercept": float("nan"),
                "r2": float("nan"),
                "n": 0,
                "predictor_names": list(predictors),
                "key_predictor": key,
                "delta_r2": float("nan"),
                "r2_base": float("nan"),
                "r2_full": float("nan"),
            }
            continue

        y = y_all[keep]
        X_full = np.column_stack(
            [np.ones(n, dtype=np.float64)] + [pred_cols[f][keep] for f in predictors]
        )
        fit = ols_fit(X_full, y)
        fit["predictor_names"] = list(predictors)

        base_fields = [f for f in predictors if f != key]
        X_base = np.column_stack(
            [np.ones(n, dtype=np.float64)] + [pred_cols[f][keep] for f in base_fields]
        )
        key_col = pred_cols[key][keep]
        X_full_delta = np.column_stack([X_base, key_col])
        delta = nested_delta_r2(X_base, X_full_delta, y)

        results[name] = {
            "coef": fit["coef"],
            "intercept": fit["intercept"],
            "r2": fit["r2"],
            "n": n,
            "predictor_names": list(predictors),
            "key_predictor": key,
            "delta_r2": delta["delta_r2"],
            "r2_base": delta["r2_base"],
            "r2_full": delta["r2_full"],
        }
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    direct = rng.gamma(2.0, 0.5, size=n)
    recurrence = rng.beta(2.0, 5.0, size=n)
    degree = rng.poisson(4, size=n).astype(float)
    attractor = 0.3 + 0.6 * direct / direct.max() + 0.2 * recurrence + rng.normal(0, 0.1, size=n)

    records = []
    for i in range(n):
        records.append(
            {
                "node_id": i,
                "direct_readout_weight": float(direct[i]),
                "recurrence": float(recurrence[i]),
                "degree": float(degree[i]),
                "attractor_support_score": float(attractor[i]),
                "readout_substitutability": float(
                    0.5 * direct[i] / direct.max() + 0.3 * attractor[i] + rng.normal(0, 0.1)
                ),
                "computational_graftability": float(
                    0.4 * attractor[i] + 0.2 * recurrence[i] + 0.3 * direct[i] / direct.max() + rng.normal(0, 0.1)
                ),
                "computational_headroom": float(0.3 * attractor[i] + 0.4 * recurrence[i] + rng.normal(0, 0.1)),
            }
        )
    records[3]["attractor_support_score"] = None
    records[5]["direct_readout_weight"] = None
    records[11]["readout_substitutability"] = None
    records[20]["computational_graftability"] = None

    X, mask = design_matrix(
        records, ["direct_readout_weight", "recurrence", "attractor_support_score", "degree"]
    )
    print(f"design_matrix: X.shape={X.shape}, rows kept={int(mask.sum())}/{len(records)}")

    reg = run_regressions(records)
    for name, m in reg.items():
        print(
            f"{name}: r2={m['r2']:.4f} n={m['n']} key={m['key_predictor']} "
            f"delta_r2={m['delta_r2']:.4f}"
        )
    print("multivariate_models OK")
