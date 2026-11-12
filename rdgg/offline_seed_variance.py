"""OFFLINE seed variance across the 3 dyadic models x 3 generation seeds (list item 8).
Per model: the 3 per-seed nets (unseeded/seed0/seed1), their mean/SD/range, and the 3 pairwise
seed-delta nets each with an identity-cluster CI (the per-run seed noise, RESUME 11.1). Pooled
seed SD across the 3 dyadic models is the headline reproducibility nuisance.

Pure numpy, reuses the verified loaders/arithmetic from offline_main_test (committed
shift_ablation energies, ONE common mask). No GPU.

  python3 scratchpad/offline_seed_variance.py
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from offline_main_test import (  # noqa: E402
    DYADIC, cluster_sums, column_energy, common_order, load, net_of,
)

ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(ROOT, "data/GT/_metrics/seed_variance_dyadic__LS_AL.json")
B = 20000
SEED_TAG = {0: "unseeded", 1: "seed0", 2: "seed1"}


def main():
    cols = load()
    all_seed_cols = [l for dirs in DYADIC.values() for l in dirs]
    stems, ident = common_order(cols, all_seed_cols)
    NC = len(dict.fromkeys(ident))
    full = np.arange(NC)
    print(f"  common stems {len(stems)}, identities {NC}", flush=True)

    out = {}
    pooled_sd = []
    for model, dirs in DYADIC.items():
        # per-seed cluster sums + net
        S = []
        nets = []
        for d in dirs:
            br, mr, bn, mn = column_energy(cols, d, stems)
            Sd, _ = cluster_sums(br, mr, bn, mn, ident)
            S.append(Sd)
            nets.append(net_of(full, Sd) * 1e4)
        nets = np.array(nets)
        sd = float(nets.std(ddof=1))
        pooled_sd.append(sd)

        # pairwise seed deltas with identity CI
        rng = np.random.default_rng(0)
        pair_rows = []
        for i, j in combinations(range(len(dirs)), 2):
            def dstat(sel, Si=S[i], Sj=S[j]):
                return (net_of(sel, Si) - net_of(sel, Sj)) * 1e4
            point = float(dstat(full))
            draws = np.array([dstat(rng.integers(0, NC, NC)) for _ in range(B)])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            p_two = float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))
            pair_rows.append({"pair": f"{SEED_TAG[i]}-{SEED_TAG[j]}",
                              "delta_net": round(point, 4),
                              "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                              "p_two_sided": round(min(p_two, 1.0), 4)})
        out[model] = {
            "per_seed_net": {SEED_TAG[k]: round(float(nets[k]), 4) for k in range(len(dirs))},
            "mean_net": round(float(nets.mean()), 4),
            "sd_net_n3": round(sd, 4),
            "range_net": round(float(nets.max() - nets.min()), 4),
            "pairwise_seed_deltas": pair_rows,
        }
        print(f"  [{model:14s}] nets {np.round(nets,3)} mean {nets.mean():+.3f} "
              f"SD(n=3) {sd:.3f} range {nets.max()-nets.min():.3f}", flush=True)
        for r in pair_rows:
            print(f"      {r['pair']:16s} d={r['delta_net']:+.3f} CI[{r['lo']:+.2f},{r['hi']:+.2f}] "
                  f"p={r['p_two_sided']:.3f}", flush=True)

    pooled = float(np.sqrt(np.mean(np.square(pooled_sd))))   # RMS of the per-model seed SDs
    print(f"\n  pooled seed SD (RMS over 3 dyadic models): {pooled:.4f} x1e4", flush=True)

    result = {
        "metric": "R-DGG per-run (generation-seed) variance, 3 dyadic x 3 seeds (list item 8)",
        "definition": ("per model: 3 per-seed nets on one common mask, their SD(n=3) and range; "
                       "3 pairwise seed-delta nets each with an identity-cluster CI (two-sided p). "
                       "No pairwise seed delta is expected significant (seed noise sits inside the "
                       "identity CI); the pooled SD is the reproducibility nuisance (RESUME 11.1)."),
        "n_stems": len(stems), "n_identities": NC, "bootstrap": B,
        "per_model": out,
        "pooled_seed_sd_rms": round(pooled, 4),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [out] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
