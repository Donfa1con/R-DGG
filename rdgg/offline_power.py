"""OFFLINE power analysis (plan task 11 / list item 14): how the CI width of the main contrast
AND of the pairwise dyadic differences shrinks as the number of identity clusters k grows (4..13).
Descriptive; recommends how many identities are needed to rank systems.

Pure numpy, reuses the verified loaders/arithmetic from offline_main_test (committed
shift_ablation energies, ONE common mask). No GPU.

  python3 scratchpad/offline_power.py
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from offline_main_test import (  # noqa: E402
    DYADIC, TALKING, cluster_sums, column_energy, common_order, load, net_of,
)

ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(ROOT, "data/GT/_metrics/power_analysis__LS_AL.json")
B_INNER = 4000          # inner identity bootstrap per subsample (CI width)
MAX_SUBSETS = 120       # cap on sampled k-subsets of clusters (use all C(NC,k) if fewer)


def per_cluster_sums_all(cols, stems, ident):
    """Build per-cluster (BR,MR,BN,MN) for every final system; indexed by the FULL cluster list.
    Returns ({sysname: sums}, clusters) with dyadic seed-averaged, talking-head single."""
    clusters = list(dict.fromkeys(ident))
    S = {}
    for m, dirs in DYADIC.items():
        br, mr, bn, mn = column_energy(cols, dirs, stems)
        S[m], _ = cluster_sums(br, mr, bn, mn, ident)
    for m in TALKING:
        br, mr, bn, mn = column_energy(cols, m, stems)
        S[m], _ = cluster_sums(br, mr, bn, mn, ident)
    return S, clusters


def contrast_on(sel, S):
    dy = np.mean([net_of(sel, S[m]) for m in DYADIC])
    th = np.mean([net_of(sel, S[m]) for m in TALKING])
    return (dy - th) * 1e4


def pair_on(sel, S, a, b):
    return (net_of(sel, S[a]) - net_of(sel, S[b])) * 1e4


def ci_width(stat_fn, sub, rng, B):
    """Inner identity bootstrap over the clusters in `sub` -> 95% CI width of stat_fn."""
    k = len(sub)
    sub = np.asarray(sub)
    draws = np.array([stat_fn(sub[rng.integers(0, k, k)]) for _ in range(B)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(hi - lo)


def main():
    cols = load()
    all_final = [l for dirs in DYADIC.values() for l in dirs] + TALKING
    stems, ident = common_order(cols, all_final)
    S, clusters = per_cluster_sums_all(cols, stems, ident)
    NC = len(clusters)
    print(f"  common stems {len(stems)}, identities {NC}", flush=True)

    pairs = list(combinations(DYADIC, 2))   # 3 dyadic pairs
    rng = np.random.default_rng(0)
    rows = []
    for k in range(4, NC + 1):
        allk = list(combinations(range(NC), k))
        subs = allk if len(allk) <= MAX_SUBSETS else [
            tuple(np.sort(rng.choice(NC, k, replace=False))) for _ in range(MAX_SUBSETS)
        ]
        cw = []
        pw = {f"{a}-{b}": [] for a, b in pairs}
        for sub in subs:
            cw.append(ci_width(lambda sel: contrast_on(sel, S), sub, rng, B_INNER))
            for a, b in pairs:
                pw[f"{a}-{b}"].append(ci_width(lambda sel: pair_on(sel, S, a, b), sub, rng, B_INNER))
        row = {"k": k, "n_subsets": len(subs),
               "contrast_ci_width_mean": round(float(np.mean(cw)), 4),
               "contrast_ci_width_sd": round(float(np.std(cw, ddof=1)) if len(cw) > 1 else 0.0, 4),
               "pair_ci_width_mean": {kk: round(float(np.mean(v)), 4) for kk, v in pw.items()}}
        rows.append(row)
        pm = " ".join(f"{kk}={row['pair_ci_width_mean'][kk]:.2f}" for kk in row["pair_ci_width_mean"])
        print(f"  k={k:2d} (n_sub={len(subs):3d})  contrast CI width {row['contrast_ci_width_mean']:.3f}"
              f"  | pairs {pm}", flush=True)

    # 1/sqrt(k) fit on contrast width (expect slope ~ -0.5 in log-log)
    ks = np.array([r["k"] for r in rows], float)
    w = np.array([r["contrast_ci_width_mean"] for r in rows], float)
    slope = float(np.polyfit(np.log(ks), np.log(w), 1)[0])
    # extrapolate: identities needed for the contrast CI width to reach a target
    w13 = w[-1]
    def k_for_width(target):
        # w ~ C * k^slope  ->  k = (target/C)^(1/slope);  C from the k=13 anchor
        C = w13 / (NC ** slope)
        return float((target / C) ** (1.0 / slope))
    print(f"\n  contrast CI width log-log slope {slope:+.3f} (theory -0.5); width@13={w13:.3f}", flush=True)
    for tgt in (0.5, 0.4, 0.3):
        print(f"    to reach contrast CI width {tgt}: ~{k_for_width(tgt):.0f} identities", flush=True)

    result = {
        "metric": "R-DGG power analysis (plan task 11)",
        "definition": ("95% identity-bootstrap CI width of the main dyadic-vs-talking-head contrast "
                       "and of the 3 pairwise dyadic differences, vs number of identity clusters k=4..13; "
                       f"mean over up to {MAX_SUBSETS} random k-subsets of the {NC} clusters, "
                       f"inner bootstrap B={B_INNER}"),
        "n_stems": len(stems), "n_identities_full": NC,
        "per_k": rows,
        "contrast_width_loglog_slope": round(slope, 4), "expected_slope": -0.5,
        "contrast_width_at_full": round(float(w13), 4),
        "identities_for_contrast_width": {str(t): round(k_for_width(t), 1) for t in (0.5, 0.4, 0.3)},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [out] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
