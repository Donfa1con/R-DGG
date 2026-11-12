"""R-DGG estimator variance vs the COUNT of each Monte-Carlo component (convergence curves).

REWRITTEN (2026-11, user spec): all 15 shift_ablation columns (not 3); even log-grids; K_seeds=100;
per-stem/per-null energies STASHED to npz so any re-aggregation (seed-average, other column subsets)
runs offline. Seed-averaged dyadic slices are computed here as the headline.

NEW file only. Imports existing R-DGG blocks + shift_ablation / dose_curve / metric_seed_sweep helpers;
edits no existing rdgg module, adds no model to _config.MODELS, commits nothing.

Two cheap dimensions on each column's captured energies (one GPU capture of M=500 timeshift nulls
per column; the sweeps themselves are pure numpy):
  1. Timeshift-null count N ∈ {10,20,50,100,200}: K=200 random N-subsets of the M=500 pool ->
     SD(floor)=SD(net) (raw fixed), vs the theoretical SD(pool gains)/sqrt(N) and its finite-pool
     correction. Default N_TIMESHIFT=100 sits mid-grid.
  2. Bootstrap count B ∈ {125,500,2000,8000,20000,32000} (even ×4 log-grid incl. the 20000 default):
     K_seeds=100 bootstrap seeds -> SD of the identity-clustered CI lo/hi. Point is invariant.
Both expected ∝ 1/sqrt(count): slope of log SD vs log count ≈ -0.5.

Columns = the 15 shift_ablation columns (GT, GT×other surrogate, 3 dyadic × 3 seeds, 4 talking-head);
plus seed-averaged dyadic slices (energy-average the 3 seeds per stem, then the two sweeps).

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/count_sweep.py
A 22 GB GPU needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import MINSHIFT, SOURCES, _ident_key
from _features import collect_design, gridded_face_ok, load_all_motion
from _granger import capture_energies
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column, intersect_masks
from shift_ablation import DYADIC_SEEDS, SEED_TAG, TALKING, build_surrogate_cache
from metric_seed_sweep import boot_ci_seeded

NS = [10, 20, 50, 100, 200]
BS = [125, 500, 2000, 8000, 20000, 32000]
K_SUBSETS = 200
K_SEEDS = 100


def build_timeshift_pool(listeners, pairs, SPK_G_AUD, M, base0=2000):
    pool = []
    for k in range(M):
        rng = np.random.default_rng(base0 + k)
        off = {}
        for ls in listeners:
            sp = pairs[ls]
            n = SPK_G_AUD[sp].shape[0] if sp in SPK_G_AUD else 0
            off[ls] = int(rng.integers(MINSHIFT, n - MINSHIFT)) if n > 2 * MINSHIFT + 20 else max(25, MINSHIFT)
        pool.append(off)
    return pool


def pool_null_gains(E, stems):
    MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    BN = np.sum([E[s]["both_n"] for s in stems], 0)
    return np.log(MN / BN) * 1e4


def raw_x1e4(E, stems):
    BR = sum(E[s]["both_r"] for s in stems); MR = sum(E[s]["mot_r"] for s in stems)
    return float(np.log(MR / BR)) * 1e4


def loglog_slope(counts, sds):
    x = np.log(np.asarray(counts, float)); y = np.log(np.asarray(sds, float))
    ok = np.isfinite(y)
    return round(float(np.polyfit(x[ok], y[ok], 1)[0]), 4) if ok.sum() >= 2 else None


def dim1_timeshift(E, stems, M):
    g = pool_null_gains(E, stems); pop_sd = float(g.std(ddof=1)); rng = np.random.default_rng(0)
    rows = []
    for N in [n for n in NS if n <= M]:   # guard: cannot draw N>M without replacement
        floors = np.array([g[rng.choice(M, N, replace=False)].mean() for _ in range(K_SUBSETS)])
        sd_theory = pop_sd / np.sqrt(N)
        fpc = sd_theory * np.sqrt((M - N) / (M - 1))
        rows.append({"N": N, "sd_floor_net_empirical": round(float(floors.std(ddof=1)), 5),
                     "sd_theory_1_over_sqrtN": round(sd_theory, 5),
                     "sd_theory_finite_pool": round(float(fpc), 5), "mean_floor": round(float(floors.mean()), 4)})
    slope = loglog_slope([r["N"] for r in rows if r["N"] <= 100],
                         [r["sd_floor_net_empirical"] for r in rows if r["N"] <= 100])
    return rows, slope


def dim2_bootstrap(E, stems):
    rows = []
    for B in BS:
        los = np.empty(K_SEEDS); his = np.empty(K_SEEDS); pts = np.empty(K_SEEDS)
        for i in range(K_SEEDS):
            d = boot_ci_seeded(E, stems, _ident_key, B, i)
            los[i], his[i], pts[i] = d["lo"], d["hi"], d["point"]
        rows.append({"B": B, "sd_ci_lo": round(float(los.std(ddof=1)), 5),
                     "sd_ci_hi": round(float(his.std(ddof=1)), 5), "point_sd": round(float(pts.std(ddof=1)), 6)})
    slope_lo = loglog_slope([r["B"] for r in rows], [r["sd_ci_lo"] for r in rows])
    slope_hi = loglog_slope([r["B"] for r in rows], [r["sd_ci_hi"] for r in rows])
    return rows, slope_lo, slope_hi


def seed_avg_E(E_list, stems):
    return {s: {"both_r": float(np.mean([E[s]["both_r"] for E in E_list])),
                "mot_r": float(np.mean([E[s]["mot_r"] for E in E_list])),
                "both_n": np.mean([E[s]["both_n"] for E in E_list], axis=0),
                "mot_n": np.mean([E[s]["mot_n"] for E in E_list], axis=0)} for s in stems}


def main():
    ap = argparse.ArgumentParser(description="R-DGG Monte-Carlo count sweep, all 15 columns (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--pool", type=int, default=500)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT
    M = args.pool

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    eval_mask = mask_gtonly
    for d in all_dirs:
        eval_mask = intersect_masks(eval_mask, gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus))

    pool = build_timeshift_pool(corpus.listeners, corpus.pairs, corpus.SPK_G_AUD, M)
    print(f"  [pool] {M} distinct-seed timeshifts; capturing per-null energies per column ...", flush=True)

    # capture every column with the M-null pool
    Es = {}
    def cap(label, per_or_cache, is_cache=True):
        tc = time.time()
        per = collect_design(per_or_cache, ctx, eval_mask) if is_cache else per_or_cache
        E, _ = capture_energies(per, ops, ctx, timeshifts=pool)
        Es[label] = E
        print(f"    captured {label:22s} {len(E)} stems {time.time()-tc:.1f}s", flush=True)

    cap("GT", gt_cache)
    cap("GT×other", build_surrogate_cache(gt_cache, corpus))
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            cap(f"{model} [{SEED_TAG[i]}]", load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus))
    for model, d in TALKING.items():
        cap(model, load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus))

    common = sorted(set.intersection(*[set(E) for E in Es.values()]))
    Es = {k: {s: v[s] for s in common} for k, v in Es.items()}
    print(f"  [mask] shared stems {len(common)}", flush=True)

    # seed-averaged dyadic columns
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        Es[f"{model} [seed-avg]"] = seed_avg_E([Es[l] for l in labels], common)

    # the two sweeps on every column
    result_cols = {}
    npz = {}
    for label, E in Es.items():
        d1, s1 = dim1_timeshift(E, common, M)
        d2, s2lo, s2hi = dim2_bootstrap(E, common)
        result_cols[label] = {
            "raw_x1e4": round(raw_x1e4(E, common), 4),
            "pool_floor_x1e4": round(float(pool_null_gains(E, common).mean()), 4),
            "pool_null_gain_SD_x1e4": round(float(pool_null_gains(E, common).std(ddof=1)), 4),
            "dim1_timeshift": {"per_N": d1, "loglog_slope_Nle100": s1},
            "dim2_bootstrap": {"per_B": d2, "loglog_slope_lo": s2lo, "loglog_slope_hi": s2hi},
        }
        # stash per-stem/per-null energies for offline re-aggregation
        npz[f"{label}::both_r"] = np.array([E[s]["both_r"] for s in common])
        npz[f"{label}::mot_r"] = np.array([E[s]["mot_r"] for s in common])
        npz[f"{label}::both_n"] = np.stack([E[s]["both_n"] for s in common], axis=1)
        npz[f"{label}::mot_n"] = np.stack([E[s]["mot_n"] for s in common], axis=1)
        print(f"  [{label:22s}] dim1 slope={s1} dim2 slope(lo)={s2lo}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "n_stems": len(common), "n_identities": len({_ident_key(s) for s in common}),
        "M_pool": M, "N_grid": NS, "B_grid": BS, "K_subsets": K_SUBSETS, "K_seeds": K_SEEDS,
        "default_N_timeshift": _config.N_TIMESHIFT, "default_bootstrap": _config.BOOTSTRAP,
        "expected_slope": -0.5,
        "columns": result_cols,
        "note": ("per-column convergence of the timeshift floor (N) and identity bootstrap (B). "
                 "Default N=100 / B=20000 sit on the grids. Energies stashed to npz for offline "
                 "re-aggregation (seed subsets etc)."),
    }
    np.savez_compressed(f"{out_dir}/count_sweep__{S}.npz", **npz)
    with open(f"{out_dir}/count_sweep__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] {out_dir}/count_sweep__{S}.{{json,npz}}", flush=True)
    print("COUNT_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
