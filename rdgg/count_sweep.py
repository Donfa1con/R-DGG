"""R-DGG estimator variance vs the COUNT of each Monte-Carlo component (convergence curves).

NEW file only. Imports existing R-DGG blocks and the helpers from dose_curve.py /
cfg0_compare.py / seeded_4way.py / metric_seed_sweep.py (all new); edits no existing rdgg
module, adds no model to _config.MODELS, commits nothing.

Two cheap dimensions (no PCA refit):
  1. Timeshift-null count N. Build M=500 distinct-seed timeshifts (same MINSHIFT + per-
     listener offset logic as compute.py). Capture per-null energies ONCE with the 500-list
     (one capture_energies per column). The pooled per-null gains g_k = log(ΣMN_k/ΣBN_k) are
     then fixed; the floor of any N-subset is mean_{k∈subset} g_k (pure numpy). For
     N∈{10,20,50,100,200,400}: K=200 random N-subsets -> SD(floor)=SD(net) (raw fixed), plus
     the theoretical SD(pool gains)/sqrt(N) and its finite-pool correction.
  2. Bootstrap count B. On the SAME captured energies, for B∈{100,500,2000,10000,20000}:
     K=40 bootstrap seeds -> SD of the identity-clustered CI lo/hi. Point is invariant.

Both expected ∝ 1/sqrt(count): slope of log SD vs log count ≈ -0.5.

Fixed columns GT / AVTR-1_seeded (base0) / AVTR-1_zeroother on the shared mask.
Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/count_sweep.py
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
from dose_curve import build_context_and_operator
from seeded_4way import inter
from metric_seed_sweep import boot_ci_seeded


def build_timeshift_pool(listeners, pairs, SPK_G_AUD, M, base0=2000):
    """M timeshift draws with distinct seeds base0..base0+M-1 (compute.py's offset logic)."""
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
    """Per-null pooled gain g_k = log(ΣMN_k / ΣBN_k) * 1e4 over `stems`; shape (M,)."""
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


def main():
    ap = argparse.ArgumentParser(description="R-DGG variance vs Monte-Carlo count (standalone).")
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
    print(f"  [phase1] done {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    caches = {"GT": gt_cache,
              "seeded": load_all_motion(f"{C}/AVTR-1_seeded/{S}_{suffix}", source, corpus),
              "zero": load_all_motion(f"{C}/AVTR-1_zeroother/{S}_{suffix}", source, corpus)}
    foks = {nm: gridded_face_ok(caches[nm], corpus) for nm in caches}

    pool = build_timeshift_pool(corpus.listeners, corpus.pairs, corpus.SPK_G_AUD, M)
    print(f"  [pool] built {M} distinct-seed timeshifts; capturing per-null energies per column ...", flush=True)
    E = {}
    for nm in ("GT", "seeded", "zero"):
        others = [foks[o] for o in foks if o != nm]
        per = collect_design(caches[nm], ctx, inter(mask_gtonly, *others))
        tc = time.time()
        E[nm], _ = capture_energies(per, ops, ctx, timeshifts=pool)
        print(f"    captured {nm}: {len(E[nm])} stems, {M} nulls, {time.time()-tc:.1f}s", flush=True)
    common = sorted(set.intersection(*[set(E[nm]) for nm in E]))
    E = {nm: {s: E[nm][s] for s in common} for nm in E}
    print(f"  [mask] shared stems {len(common)}", flush=True)

    gains = {nm: pool_null_gains(E[nm], common) for nm in E}      # (M,) per column
    raws = {nm: raw_x1e4(E[nm], common) for nm in E}

    # ---- Dimension 1: timeshift-null count N ----
    Ns = [10, 20, 50, 100, 200, 400]
    rng = np.random.default_rng(0)
    dim1 = {nm: [] for nm in E}
    for nm in E:
        g = gains[nm]; pop_sd = float(g.std(ddof=1))
        for N in Ns:
            floors = np.array([g[rng.choice(M, N, replace=False)].mean() for _ in range(200)])
            sd_emp = float(floors.std(ddof=1))
            sd_theory = pop_sd / np.sqrt(N)
            fpc = sd_theory * np.sqrt((M - N) / (M - 1))     # sampling-without-replacement correction
            dim1[nm].append({"N": N, "sd_floor_net_empirical": round(sd_emp, 5),
                             "sd_theory_1_over_sqrtN": round(sd_theory, 5),
                             "sd_theory_finite_pool": round(float(fpc), 5),
                             "mean_floor": round(float(floors.mean()), 4)})
    # slope on N<=100 (finite-pool correction negligible there)
    slope1 = {nm: loglog_slope([r["N"] for r in dim1[nm] if r["N"] <= 100],
                               [r["sd_floor_net_empirical"] for r in dim1[nm] if r["N"] <= 100]) for nm in E}
    for nm in E:
        print(f"  [dim1 {nm}] " + " ".join(f"N{r['N']}={r['sd_floor_net_empirical']}" for r in dim1[nm])
              + f" | slope(logSD~logN,N<=100)={slope1[nm]}", flush=True)

    # ---- Dimension 2: bootstrap count B ----
    Bs = [100, 500, 2000, 10000, 20000]
    dim2 = {nm: [] for nm in E}
    for nm in E:
        for B in Bs:
            los = np.empty(40); his = np.empty(40); pts = np.empty(40)
            for i, seed in enumerate(range(40)):
                d = boot_ci_seeded(E[nm], common, _ident_key, B, seed)
                los[i], his[i], pts[i] = d["lo"], d["hi"], d["point"]
            dim2[nm].append({"B": B, "sd_ci_lo": round(float(los.std(ddof=1)), 5),
                             "sd_ci_hi": round(float(his.std(ddof=1)), 5),
                             "point_sd": round(float(pts.std(ddof=1)), 6)})
    slope2_lo = {nm: loglog_slope([r["B"] for r in dim2[nm]], [r["sd_ci_lo"] for r in dim2[nm]]) for nm in E}
    slope2_hi = {nm: loglog_slope([r["B"] for r in dim2[nm]], [r["sd_ci_hi"] for r in dim2[nm]]) for nm in E}
    for nm in E:
        print(f"  [dim2 {nm}] " + " ".join(f"B{r['B']}:lo={r['sd_ci_lo']}" for r in dim2[nm])
              + f" | slope(lo)={slope2_lo[nm]} slope(hi)={slope2_hi[nm]}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "n_stems": len(common), "n_identities": len({_ident_key(s) for s in common}),
        "columns": ["GT", "AVTR-1_seeded (base0)", "AVTR-1_zeroother"],
        "raw_x1e4": {nm: round(raws[nm], 4) for nm in E},
        "pool_null_gain_SD_x1e4": {nm: round(float(gains[nm].std(ddof=1)), 4) for nm in E},
        "pool_floor_x1e4": {nm: round(float(gains[nm].mean()), 4) for nm in E},
        "dim1_timeshift_count": {
            "M_pool": M, "K_subsets": 200,
            "note": "SD(floor)=SD(net) since raw is fixed; finite-pool correction shrinks SD as N->M",
            "per_column": dim1, "loglog_slope_Nle100": slope1,
            "expected_slope": -0.5},
        "dim2_bootstrap_count": {
            "K_seeds": 40,
            "per_column": dim2, "loglog_slope_lo": slope2_lo, "loglog_slope_hi": slope2_hi,
            "expected_slope": -0.5},
    }
    with open(f"{out_dir}/count_sweep__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/count_sweep__{S}.json", flush=True)
    print("COUNT_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
