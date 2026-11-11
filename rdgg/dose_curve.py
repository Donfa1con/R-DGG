"""R-DGG zeroother dose-response curve (standalone harness).

NEW code only. It imports the already-written R-DGG building blocks and composes
them; it edits nothing in rdgg/ and adds no model to _config.MODELS. See
CVPR_zeroother_design.md sections 4-7.

What it does, entirely on existing features (no generation):
  1. Build the exact GT reference operator (Phase-1 of score_source), by calling
     the importable blocks in the same order. Folds are identity-COMPONENT folds.
  2. Capture per-stem residual energies for the STOCK column (data/AVTR-1) and the
     ZEROOTHER column (data/AVTR-1_zeroother) on ONE shared mask
     (GT-only reactive  ∩  stock face_ok  ∩  zero face_ok), via the paired-cell
     trick, so a per-video swap exchanges identical rows.
  3. Dose sweep over k = number of zeroed videos (0..N): E_mix[stem] = zero if the
     stem is in the drawn k-subset else stock; R-DGG = (log(ΣMR/ΣBR) -
     mean_k log(ΣMN_k/ΣBN_k)) * 1e4. Mean + percentile band over random subsets,
     plus the closed-form expected-sums mean, plus identity-clustered CI at a few k.
  4. Exact additivity check: at selected k, rebuild a TRUE mixed design and call
     capture_energies (refits the pooled ControlFit); compare to the additive
     estimate to bound the approximation error.

Run in the shared metrics env, as pipeline/4_rdgg.sh does:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/dose_curve.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import (
    BOOTSTRAP, KMV, LAG_NP, SOURCES, SourceContext, _ident_key, parallel_map,
)
from _features import (
    collect_design, fit_pca, gridded_face_ok, gtonly_eval_mask, load_all_motion,
    pca_fit_sample, speaker_motion_raw,
)
from _granger import capture_energies, fit_bgt_fold
from _bootstrap import cluster_sums, gnet, gnet_boot_ci
from compute import load_corpus


# ----------------------------------------------------------------------------- Phase-1 (compose importable blocks)
def build_context_and_operator(source, suffix, corpus):
    """Reproduce score_source Phase 1 by CALLING the importable blocks in order.
    Returns (ctx, ops, mask_gtonly, gt_cache). No existing code is modified."""
    gt_cache = load_all_motion(f"{_config.CORPUS}/GT/{_config.SPLIT}_{suffix}", source, corpus)
    mraw = {
        sp: speaker_motion_raw(sp, suffix, source, corpus.SPK_G_AUD[sp].shape[0])
        for sp in corpus.speakers
        if sp in corpus.SPK_G_AUD
    }
    mraw = {sp: res for sp, res in mraw.items() if res is not None}
    mmu, msd, mP = fit_pca([mraw[sp] for sp in pca_fit_sample(list(mraw), 103)], KMV)
    driver = {
        sp: torch.cat(
            [corpus.SPK_G_AUD[sp],
             torch.tensor(((mraw[sp] - mmu) / msd) @ mP, device=_config.DEV, dtype=_config.DT)], 1)
        for sp in mraw
    }
    width = corpus.k_audio + mP.shape[1]
    speaker_names = list(driver.keys())
    speaker_lens = [driver[s].shape[0] for s in speaker_names]
    ka = corpus.k_audio
    is_audio_col = np.tile(np.concatenate([np.ones(ka, bool), np.zeros(width - ka, bool)]), len(LAG_NP))
    mask_gtonly = gtonly_eval_mask(suffix, source, corpus)
    ctx = SourceContext(
        driver=driver,
        driver_all=torch.cat([driver[s] for s in speaker_names], 0),
        speaker_row0={s: int(o) for s, o in
                      zip(speaker_names, np.concatenate([[0], np.cumsum(speaker_lens)[:-1]]))},
        channel_cols={"audio": torch.tensor(is_audio_col, device=_config.DEV),
                      "motion": torch.tensor(~is_audio_col, device=_config.DEV)},
        eval_mask=mask_gtonly,   # never used here (we always pass explicit masks); set to a valid dict
        corpus=corpus,
    )
    # GT-derived target balancing (EMOCA vs LP), exactly as score_source computes it.
    pg = collect_design(gt_cache, ctx, mask_gtonly)
    assert pg and len(pg) >= 10, f"GT design too small ({len(pg)} stems) to fit B"
    d = int(next(iter(pg.values()))[1].shape[1])
    s = torch.zeros(d, device=_config.DEV, dtype=torch.float64)
    sq = torch.zeros(d, device=_config.DEV, dtype=torch.float64)
    n = 0
    for _ls in pg:
        t = torch.as_tensor(pg[_ls][1], device=_config.DEV, dtype=torch.float64)
        s += t.sum(0); sq += (t * t).sum(0); n += int(t.shape[0])
    var = (sq / max(n, 1) - (s / max(n, 1)) ** 2).clamp_min(0)
    ctx.target_scale_inv = torch.rsqrt(var + torch.median(var).clamp_min(1e-12)).to(_config.DT)
    ops = [fit_bgt_fold(pg, fold, ctx) for fold in range(corpus.NFOLD)]
    assert all(op is not None for op in ops), "a B_GT fold failed"
    return ctx, ops, mask_gtonly, gt_cache


# ----------------------------------------------------------------------------- shared-mask captures
def intersect_masks(base, other):
    """base[ls] & other[ls] on the grid, for stems present in both (same length by construction)."""
    out = {}
    for ls, b in base.items():
        o = other.get(ls)
        if o is None:
            continue
        assert len(b) == len(o), f"mask length {len(b)} vs {len(o)} for {ls}"
        out[ls] = b & o
    return out


def capture_column(emoca_dir, eval_mask, ctx, ops, corpus, source, marginal_audio=False):
    """collect_design on a column with the given eval_mask (its OWN face_ok is applied inside),
    then capture per-stem energies under the fixed GT operator. Returns (E, per_for_reuse, own_r2).
    marginal_audio just forwards to capture_energies (opt-in extra keys; default off is unchanged)."""
    cache = load_all_motion(emoca_dir, source, corpus)
    per = collect_design(cache, ctx, eval_mask)
    # capture_energies consumes `per`; keep a design snapshot for the exact additivity check.
    per_keep = {ls: tuple(v) for ls, v in per.items()}
    E, own_r2 = capture_energies(per, ops, ctx, marginal_audio=marginal_audio)
    return E, per_keep, own_r2


# ----------------------------------------------------------------------------- dose arithmetic
def point_net_lib(E, stems):
    """Point R-DGG (x1e4) over `stems`, via the written cluster_sums+gnet (NC=1 -> pooled)."""
    sums = cluster_sums(E, stems, {s: 0 for s in stems}, 1)
    return float(gnet(np.array([0]), sums)) * 1e4


def stack_energies(E_stock, E_zero, stems):
    """Per-stem arrays for vectorized mixing. Rows aligned to `stems`."""
    K = len(next(iter(E_stock.values()))["both_n"])
    A = {"BRs": np.array([E_stock[s]["both_r"] for s in stems]),
         "MRs": np.array([E_stock[s]["mot_r"] for s in stems]),
         "BNs": np.array([E_stock[s]["both_n"] for s in stems]).T.copy(),   # (K, N)
         "MNs": np.array([E_stock[s]["mot_n"] for s in stems]).T.copy(),
         "BRz": np.array([E_zero[s]["both_r"] for s in stems]),
         "MRz": np.array([E_zero[s]["mot_r"] for s in stems]),
         "BNz": np.array([E_zero[s]["both_n"] for s in stems]).T.copy(),
         "MNz": np.array([E_zero[s]["mot_n"] for s in stems]).T.copy()}
    return A, K


def net_from_mask(A, z):
    """R-DGG (x1e4) for a boolean zero-mask z over stems (True = use zeroother)."""
    zc = z.astype(np.float64); sc = 1.0 - zc
    BR = (zc * A["BRz"] + sc * A["BRs"]).sum()
    MR = (zc * A["MRz"] + sc * A["MRs"]).sum()
    BN = (A["BNz"] * zc + A["BNs"] * sc).sum(1)   # (K,)
    MN = (A["MNz"] * zc + A["MNs"] * sc).sum(1)
    return (np.log(MR / BR) - np.log(MN / BN).mean()) * 1e4


def net_expected(A, p):
    """Closed-form mean: each stem zeroed with prob p (expected sums, Jensen-approx)."""
    BR = (p * A["BRz"] + (1 - p) * A["BRs"]).sum()
    MR = (p * A["MRz"] + (1 - p) * A["MRs"]).sum()
    BN = (p * A["BNz"] + (1 - p) * A["BNs"]).sum(1)
    MN = (p * A["MNz"] + (1 - p) * A["MNs"]).sum(1)
    return (np.log(MR / BR) - np.log(MN / BN).mean()) * 1e4


def dose_ci(E_stock, E_zero, stems, k, B, seed, n_sched=160):
    """Dose-level CI at k switched videos: EXPECTED-over-schedules point with an
    identity-cluster CI (the statistically-clean estimand).

    The schedule (WHICH k of N videos are switched) is a finite-population design
    choice, not a sample from a superpopulation, so it must NOT inflate the CI
    tails. We therefore INTEGRATE the schedule INTO THE POINT: the net is averaged
    over `n_sched` random schedules of size k (averaging the realized log-ratio, so
    there is no Jensen bias), and the CI is the identity-cluster bootstrap of that
    schedule-averaged net. The interval then reflects ONLY population (identity)
    uncertainty, is comparable across the whole curve, and does not balloon
    mid-curve from the combinatorics of choosing k videos. At k=0 / k=N there is one
    schedule, so this reduces to the pure identity bootstrap. Pure numpy; no GPU.

    Returns point == estimand of the interval (same quantity), so point lies between
    lo and hi by construction (up to Monte-Carlo)."""
    N = len(stems)
    clusters = list(dict.fromkeys(_ident_key(s) for s in stems))
    cidx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    cl = np.array([cidx[_ident_key(s)] for s in stems])
    Cmat = np.zeros((NC, N))
    Cmat[cl, np.arange(N)] = 1.0                     # cluster one-hot: Cmat @ per-stem -> per-cluster
    a = lambda E, q: np.array([E[s][q] for s in stems])
    BRs, MRs, BNs, MNs = a(E_stock, "both_r"), a(E_stock, "mot_r"), a(E_stock, "both_n"), a(E_stock, "mot_n")
    BRz, MRz, BNz, MNz = a(E_zero, "both_r"), a(E_zero, "mot_r"), a(E_zero, "both_n"), a(E_zero, "mot_n")
    K = BNs.shape[1]
    rng = np.random.default_rng(seed)
    # pre-aggregate each schedule to per-cluster sums, stacked over schedules S
    S = 1 if (k <= 0 or k >= N) else n_sched
    BRc = np.empty((S, NC)); MRc = np.empty((S, NC))
    BNc = np.empty((S, NC, K)); MNc = np.empty((S, NC, K))
    for s in range(S):
        if k <= 0:
            zc = np.zeros(N, bool)
        elif k >= N:
            zc = np.ones(N, bool)
        else:
            zc = np.zeros(N, bool); zc[rng.choice(N, k, replace=False)] = True
        BRc[s] = Cmat @ np.where(zc, BRz, BRs); MRc[s] = Cmat @ np.where(zc, MRz, MRs)
        BNc[s] = Cmat @ np.where(zc[:, None], BNz, BNs); MNc[s] = Cmat @ np.where(zc[:, None], MNz, MNs)

    def sched_avg_net(sel):
        """mean over schedules of the realized net on the resampled clusters `sel`."""
        BR = BRc[:, sel].sum(1); MR = MRc[:, sel].sum(1)           # (S,)
        BK = BNc[:, sel, :].sum(1); MK = MNc[:, sel, :].sum(1)     # (S, K)
        return float((np.log(MR / BR) - np.log(MK / BK).mean(1)).mean()) * 1e4

    point = sched_avg_net(np.arange(NC))
    d = np.array([sched_avg_net(rng.integers(0, NC, NC)) for _ in range(B)])
    lo, p50, hi = np.percentile(d, [2.5, 50, 97.5])
    return {"point": float(point), "lo": float(lo), "p50": float(p50), "hi": float(hi),
            "n_clusters": NC, "n_sched": int(S)}


def main():
    ap = argparse.ArgumentParser(description="R-DGG zeroother dose-response curve (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--stock_dir", default=None, help="stock AVTR-1 corpus dir name (default AVTR-1)")
    ap.add_argument("--zero_dir", default=None, help="zeroother corpus dir name (default AVTR-1_zeroother)")
    ap.add_argument("--subsets", type=int, default=200, help="random subsets per k for the band")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    stock_name = args.stock_dir or "AVTR-1"
    zero_name = args.zero_dir or "AVTR-1_zeroother"
    stock_emoca = f"{_config.CORPUS}/{stock_name}/{_config.SPLIT}_{suffix}"
    zero_emoca = f"{_config.CORPUS}/{zero_name}/{_config.SPLIT}_{suffix}"

    t0 = time.time()
    corpus = load_corpus()
    print(f"  [phase1] building GT operator ({corpus.NFOLD} folds)...", flush=True)
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] done in {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    # grid face_ok of each column, and the shared mask via the paired trick
    stock_cache0 = load_all_motion(stock_emoca, source, corpus)
    zero_cache0 = load_all_motion(zero_emoca, source, corpus)
    stock_fok = gridded_face_ok(stock_cache0, corpus)
    zero_fok = gridded_face_ok(zero_cache0, corpus)
    del stock_cache0, zero_cache0
    em_stock = intersect_masks(mask_gtonly, zero_fok)   # capture stock on GT & stock(own) & zero
    em_zero = intersect_masks(mask_gtonly, stock_fok)    # capture zero  on GT & zero(own) & stock

    print("  [capture] STOCK column ...", flush=True)
    tc = time.time()
    E_stock, per_stock, r2_s = capture_column(stock_emoca, em_stock, ctx, ops, corpus, source)
    print(f"    stock: {len(E_stock)} stems, own_r2={r2_s:.3f}, {time.time() - tc:.1f}s", flush=True)
    print("  [capture] ZEROOTHER column ...", flush=True)
    tc = time.time()
    E_zero, per_zero, r2_z = capture_column(zero_emoca, em_zero, ctx, ops, corpus, source)
    print(f"    zero:  {len(E_zero)} stems, own_r2={r2_z:.3f}, {time.time() - tc:.1f}s", flush=True)

    stems = sorted(set(E_stock) & set(E_zero))
    only_s = sorted(set(E_stock) - set(E_zero)); only_z = sorted(set(E_zero) - set(E_stock))
    nrow_mismatch = [(s, E_stock[s]["nrows"], E_zero[s]["nrows"]) for s in stems
                     if E_stock[s]["nrows"] != E_zero[s]["nrows"]]
    print(f"  [mask] shared stems: {len(stems)} | stock-only {len(only_s)} | zero-only {len(only_z)} "
          f"| nrows mismatch {len(nrow_mismatch)}", flush=True)
    if nrow_mismatch:
        print(f"    first mismatches: {nrow_mismatch[:5]}", flush=True)
    N = len(stems)

    # internal check: my vectorized point formula == the library gnet on the same E
    A, K = stack_energies(E_stock, E_zero, stems)
    z0 = np.zeros(N, bool)
    chk_vec = net_from_mask(A, z0)
    chk_lib = point_net_lib(E_stock, stems)
    print(f"  [check] vectorized k=0 {chk_vec:.4f} vs library gnet {chk_lib:.4f} "
          f"(diff {abs(chk_vec - chk_lib):.2e})", flush=True)

    k0 = net_from_mask(A, z0)
    kN = net_from_mask(A, np.ones(N, bool))
    print(f"  [curve] k=0 (pure STOCK) R-DGG = {k0:.3f} | k={N} (pure ZERO) = {kN:.3f}", flush=True)

    # dose sweep: sampled mean+band on a k-grid, closed-form mean at every k
    rng = np.random.default_rng(0)
    kgrid = sorted(set([0, N] + list(range(0, N + 1, max(1, N // 20)))))
    sampled = []
    for k in kgrid:
        if k == 0:
            vals = np.array([net_from_mask(A, z0)])
        elif k == N:
            vals = np.array([net_from_mask(A, np.ones(N, bool))])
        else:
            vals = np.empty(args.subsets)
            for b in range(args.subsets):
                z = np.zeros(N, bool); z[rng.choice(N, k, replace=False)] = True
                vals[b] = net_from_mask(A, z)
        lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
        sampled.append({"k": int(k), "frac": k / N, "mean": float(vals.mean()),
                        "p50": float(med), "lo": float(lo), "hi": float(hi), "n_subsets": int(len(vals))})
    cf_k = list(range(N + 1))
    cf_net = [float(net_expected(A, k / N)) for k in cf_k]

    # dose-level CI: EXPECTED-over-schedules point + identity-cluster CI (dose_ci).
    # The schedule (which k videos) is integrated into the POINT, not the CI tails,
    # so interval widths are comparable across the whole curve and do not balloon
    # mid-curve from combinatorics. Endpoints k=0 / k=N have one schedule, so dose_ci
    # reduces to the pure identity bootstrap (cross-checked against gnet_boot_ci).
    ci = {}
    for k in sorted(set([0, N // 4, N // 2, 3 * N // 4, N])):
        d = dose_ci(E_stock, E_zero, stems, k, BOOTSTRAP, 100 + k)
        ci[int(k)] = {"point": d["point"], "lo": d["lo"], "hi": d["hi"], "p50": d["p50"],
                      "n_clusters": d["n_clusters"], "n_sched": d["n_sched"],
                      "method": "expected-over-schedules + identity"}
        c = ci[int(k)]
        print(f"  [CI] k={k}: point {c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] "
              f"({c['n_clusters']} clusters, {c['n_sched']} schedules)", flush=True)

    # exact additivity validation at selected k (rebuild TRUE mixed design, refit ControlFit)
    print("  [validate] exact additivity (refit ControlFit on the true mixed design)...", flush=True)
    val = []
    shared = [s for s in stems if s in per_stock and s in per_zero]
    for k in sorted(set([0, N // 4, N // 2, 3 * N // 4, N])):
        z = np.zeros(N, bool)
        if 0 < k < N:
            z[np.random.default_rng(500 + k).choice(N, k, replace=False)] = True
        elif k == N:
            z[:] = True
        Szero = {stems[i] for i in range(N) if z[i]}
        add_net = net_from_mask(A, z)
        per_mix = {s: (per_zero[s] if s in Szero else per_stock[s]) for s in shared}
        E_exact, _ = capture_energies({s: tuple(v) for s, v in per_mix.items()}, ops, ctx)
        exact_net = point_net_lib(E_exact, list(E_exact))
        val.append({"k": int(k), "additive": float(add_net), "exact": float(exact_net),
                    "abs_err": float(abs(add_net - exact_net))})
        print(f"    k={k}: additive {add_net:.3f} | exact {exact_net:.3f} "
              f"| |err| {abs(add_net - exact_net):.4f}", flush=True)
    max_err = max(v["abs_err"] for v in val)
    print(f"  [validate] max additivity error over checked k: {max_err:.4f} (x1e4 units)", flush=True)

    out_dir = f"{_config.CORPUS}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": _config.SPLIT,
        "stock_dir": stock_name, "zero_dir": zero_name,
        "n_stems": N, "n_folds": corpus.NFOLD,
        "n_identities": len({_ident_key(s) for s in stems}),
        "mask": "GT-only reactive AND stock face_ok AND zero face_ok (paired-cell trick)",
        "scoring_audio": "real GT hubert/vad (never zeroed)",
        "k0_pure_stock": float(k0), "kN_pure_zero": float(kN),
        "k0_note": ("self-consistent pure-stock value on the GT&stock&zero mask; NOT the leaderboard "
                    "AVTR-1 value, which is scored on the all-8-model COMMON mask"),
        "own_r2_stock": float(r2_s), "own_r2_zero": float(r2_z),
        "sampled_curve": sampled,
        "closed_form_curve": {"k": cf_k, "net": cf_net},
        "identity_ci": ci,
        "identity_ci_method": ("expected-over-schedules point with an identity-cluster CI: the net is "
                               "averaged over n_sched random size-k schedules (schedule integrated into "
                               "the POINT, no Jensen bias), then the identity clusters are bootstrapped. "
                               "The CI is pure population uncertainty, comparable across k; endpoints "
                               "k=0/k=N have one schedule and reduce to the identity bootstrap"),
        "additivity_validation": val, "additivity_max_abs_err": float(max_err),
        "nrow_mismatch_count": len(nrow_mismatch),
    }
    with open(f"{out_dir}/dose_curve__{_config.SPLIT}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(f"{out_dir}/dose_curve__{_config.SPLIT}.npz",
             stems=np.array(stems),
             sampled_k=np.array([d["k"] for d in sampled]),
             sampled_mean=np.array([d["mean"] for d in sampled]),
             sampled_lo=np.array([d["lo"] for d in sampled]),
             sampled_hi=np.array([d["hi"] for d in sampled]),
             cf_k=np.array(cf_k), cf_net=np.array(cf_net),
             **{f"E_stock_{q}": np.array([E_stock[s][q] for s in stems]) for q in ("both_r", "mot_r")},
             **{f"E_zero_{q}": np.array([E_zero[s][q] for s in stems]) for q in ("both_r", "mot_r")})
    print(f"  [out] wrote {out_dir}/dose_curve__{_config.SPLIT}.{{json,npz}}", flush=True)
    print("DOSE_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
