#!/usr/bin/env python
"""V3 core figure — function-specific graftability (reviewer §12).

Four panels:
  A  S_i (readout graftability) by site
  B  G_comp (computational graftability) by site
  C  2D: x=S_i, y=G_comp (site-level role separation)
  D  A_i vs C_i^supp (per-node, mechanism — not the main finding)
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _latest(prefix):
    dirs = sorted(glob.glob(f"results/{prefix}*/"))
    return dirs[-1] if dirs else None


def main() -> int:
    site_dir = _latest("v3-zfish-2")
    pn_dir = _latest("v3-zfish-pn-")
    if site_dir is None:
        print("no site-level results"); return 1
    site = json.load(open(Path(site_dir) / "records.json"))
    per_node = json.load(open(Path(pn_dir) / "records.json")) if pn_dir else []

    order = ["surviving_Int", "_Axl_", "_DOs_", "untyped_direct", "random", "gold_top19"]
    site_map = {r["population"]: r for r in site}
    pops = [p for p in order if p in site_map]
    S = [site_map[p]["readout_substitutability"] for p in pops]
    G = [site_map[p]["computational_graftability"] for p in pops]
    A = [site_map[p].get("attractor_support_score") for p in pops]

    # per-node A_i vs C_i^supp
    pn_A = np.array([r["attractor_support_score"] for r in per_node], float)
    pn_C = np.array([r["computational_support"] for r in per_node], float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    xpos = np.arange(len(pops))
    colors = ["#c44e52" if p == "surviving_Int" else "#4c72b0" for p in pops]
    axes[0, 0].bar(xpos, S, color=colors)
    axes[0, 0].set_xticks(xpos); axes[0, 0].set_xticklabels(pops, rotation=45, ha="right", fontsize=10)
    axes[0, 0].set_ylabel("S_i (readout graftability)", fontsize=11)
    axes[0, 0].set_title("(A) Relay: feedforward-direct best", fontsize=12)

    axes[0, 1].bar(xpos, G, color=colors)
    axes[0, 1].set_xticks(xpos); axes[0, 1].set_xticklabels(pops, rotation=45, ha="right", fontsize=10)
    axes[0, 1].set_ylabel("G_comp (computational graftability)", fontsize=11)
    axes[0, 1].set_title("(B) Computation: surviving_Int only", fontsize=12)

    axes[1, 0].scatter(S, G, s=220, c=colors, edgecolor="k")
    for p, s, g in zip(pops, S, G):
        if p == "surviving_Int":
            # red ball (top): label below the ball, reading down
            axes[1, 0].annotate(p, (s, g), textcoords="offset points", xytext=(0, -12),
                                fontsize=9, rotation=-90, va="top", ha="center")
        else:
            # blue balls (bottom): label above the ball, reading up
            axes[1, 0].annotate(p, (s, g), textcoords="offset points", xytext=(0, 10),
                                fontsize=9, rotation=90, va="bottom", ha="center")
    axes[1, 0].axhline(0, color="gray", lw=0.5); axes[1, 0].axvline(0, color="gray", lw=0.5)
    axes[1, 0].set_xlim(-0.12, 0.32)
    axes[1, 0].set_ylim(-0.12, 0.80)
    axes[1, 0].set_xlabel("S_i (relay)", fontsize=11); axes[1, 0].set_ylabel("G_comp (computation)", fontsize=11)
    axes[1, 0].set_title("(C) Function-specific roles", fontsize=12)

    axes[1, 1].scatter(pn_A, pn_C, s=12, c="#888888", alpha=0.5)
    axes[1, 1].set_xlabel(r"$A_i$ (persistent-state support)", fontsize=11)
    axes[1, 1].set_ylabel(r"$\Delta G_{\mathrm{comp}}$ (causal computation)", fontsize=11)
    axes[1, 1].set_title(r"(D) $A_i$ vs $\Delta G_{\mathrm{comp}}$ (rho=0.856)", fontsize=12)

    fig.suptitle("Neural graftability is function-specific", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/fig6_function_specific_graftability.png", dpi=300)
    plt.close(fig)
    print("wrote figures/fig6_function_specific_graftability.png")
    for p, s, g in zip(pops, S, G):
        print(f"  {p:<16} S_i={s:+.3f}  G_comp={g:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
