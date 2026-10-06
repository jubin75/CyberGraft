#!/usr/bin/env python
"""V3 recurrence control + multivariate regressions (caveat 1, PRD §19/§20/§30-F2).

Answers: is attractor-support A_i just a relabeling of recurrence, or does it add
explanatory power for computational graftability beyond recurrent structure?

Loads the existing per-node records, attaches the SCC-based per-node recurrence
(from structural.recurrence_scores — not the degenerate self-loop recurrence),
renames the causal computational-support score to ``computational_graftability``
(per oracle: per-node G_comp is operationalized as silencing dG), and runs the
nested OLS models.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

from cybergraft.analysis import attractor_graft_relationship as ar
from cybergraft.analysis import multivariate_models as mm
from cybergraft.attractor_analysis import structural
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.utils.config import load_config


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    data = config["data"]

    # latest per-node records
    dirs = sorted(glob.glob("results/v3-zfish-pn-*/"))
    if not dirs:
        print("no per-node results found; run scripts/v3_run_zebrafish_pernode.py first")
        return 1
    src_dir = dirs[-1]
    records = json.load(open(Path(src_dir) / "records.json"))
    print(f"[rc] source records: {src_dir} (n={len(records)})")

    # SCC recurrence on the E/I graph
    adapter = ZebrafishAdapter(
        mat_path=project_root / data["mat_file"], max_nodes=int(data["max_nodes"]),
        synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
    )
    ei = config.get("ei", {})
    graph, _ = adapter.build_full_subgraph(
        readout_type=data["readout_type"], lesion_type=data["lesion_type"],
        input_type=data["input_type"], max_nodes=int(data["max_nodes"]),
        inhibitory_fraction=float(ei.get("inhibitory_fraction", 0.0)),
        inhibitory_gain=float(ei.get("inhibitory_gain", 1.0)),
    )
    rec_scores = structural.recurrence_scores(graph)
    scc_rec = rec_scores["recurrence"]  # (N,) SCC-based recurrence

    for r in records:
        nid = int(r["node_id"])
        r["recurrence"] = None if np.isnan(scc_rec[nid]) else float(scc_rec[nid])
        # per-node computational graftability = causal silencing dG (oracle Problem 3)
        r["computational_graftability"] = r.pop("computational_support", None)

    # ---- correlations + recurrence control --------------------------------
    corr_A_rec = ar.correlate(records, "attractor_support_score", "recurrence")
    corr_A_C = ar.correlate(records, "attractor_support_score", "computational_graftability")
    corr_rec_C = ar.correlate(records, "recurrence", "computational_graftability")

    reg = mm.run_regressions(records)

    out = {
        "source_dir": src_dir,
        "n": len(records),
        "corr_A_vs_recurrence": corr_A_rec,
        "corr_A_vs_C": corr_A_C,
        "corr_recurrence_vs_C": corr_rec_C,
        "regressions": reg,
    }
    out_path = Path(src_dir) / "recurrence_control.json"
    out_path.write_text(json.dumps(_to_jsonable(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n===== RECURRENCE CONTROL =====")
    print(f"Corr(A_i, recurrence)        rho={corr_A_rec['rho']:+.3f} p={corr_A_rec['p_perm']:.3f} n={corr_A_rec['n']}")
    print(f"Corr(A_i, C_i^supp)          rho={corr_A_C['rho']:+.3f} p={corr_A_C['p_perm']:.4f} n={corr_A_C['n']}")
    print(f"Corr(recurrence, C_i^supp)   rho={corr_rec_C['rho']:+.3f} p={corr_rec_C['p_perm']:.4f} n={corr_rec_C['n']}")
    for k in ("model_1", "model_2", "model_3"):
        m = reg[k]
        print(f"\n{k}: R²={m['r2']:.3f} n={m['n']}")
        print(f"  key predictor delta_R² = {m['delta_r2']:.4f}")
        for name, coef in zip(m["predictor_names"], m["coef"]):
            print(f"    {name:<24} coef={coef:+.4f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
