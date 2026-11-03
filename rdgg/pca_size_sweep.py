"""R-DGG PCA-basis variance vs SUBSAMPLE SIZE (not seed). Convergence curve: as the
PCA stem-subsample n -> all stems and the row-subsample r -> all rows, the basis
becomes deterministic and the net variance -> 0.

NEW file. Reuses metric_seed_sweep (mss) by monkeypatching ONLY the two size-bearing
fitters (pca_fit_sample's n=60, fit_pca's 40000-row cap) to read module globals
N_STEMS / N_ROWS. Everything else (make_corpus/make_ctx/capture3/pca_diag/B refit)
is mss's validated code. Edits no existing rdgg module; commits nothing.

At each size: K draws of the SWEPT knob's seed ONLY (the other knob held fixed) -> net SD,
plus mean subspace overlap vs that dim's deterministic-endpoint basis.
  dim A  stem-subsample n in {30,60,90,120,ALL=184}  (rows held at 40000, row_seed 0)
  dim B  row-subsample  r in {10k,20k,40k,80k,ALL}   (stems held at 60, stem_seed 201/211)
At each dim's endpoint the swept seed is inert AND the other is held, so the basis is
genuinely deterministic -> net SD = 0 and overlap = 1 (measured, not assumed).

Run:  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      pixi run --manifest-path rdgg/pixi.toml python rdgg/pca_size_sweep.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import DEV, K_A_TARGET, KMV, SOURCES, _ident_key, parallel_map
from _features import (collect_design, gridded_face_ok, gtonly_eval_mask, load_all_motion,
                       speaker_motion_raw, speaker_raw)
from _granger import fit_bgt_fold
from compute import load_corpus
import metric_seed_sweep as mss

# ---- size-parametrized fitters (monkeypatched into mss; read these globals) ----
N_STEMS = 60
N_ROWS = 40000


def sized_sample(stems, seed):
    stems = list(stems)
    if len(stems) <= N_STEMS:
        return stems
    idx = np.random.default_rng(seed).choice(len(stems), size=N_STEMS, replace=False)
    return [stems[i] for i in sorted(idx)]


def sized_fit_pca(mats, k, row_seed):
    X = np.concatenate(mats, 0)
    X = X[np.isfinite(X).all(1)]
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-8)
    Xc = (X - mu) / sd
    idx = np.random.default_rng(row_seed).choice(len(Xc), size=min(N_ROWS, len(Xc)), replace=False)
    _, sv, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    kk = min(k, Vt.shape[0])
    return mu, sd, Vt[:kk].T


mss.pca_fit_sample = sized_sample          # mss.audio_pca/motion_pca resolve these at call time
mss.fit_pca_seeded = sized_fit_pca


def build_ops(a_seed, m_seed, r_seed, PG, TSI, NFOLD):
    """Refit PCA(size)+B at this (seed) and capture the 3 columns. Returns (nets, aud, mot, ops, E, common)."""
    corpus_v, aud = mss.make_corpus(a_seed, r_seed, 2000)
    ctx_v, mot = mss.make_ctx(corpus_v, m_seed, r_seed, TSI)
    ops_v = [fit_bgt_fold(PG, f, ctx_v) for f in range(NFOLD)]
    Ev, com = mss.capture3(ctx_v, ops_v)
    nets = mss.col_nets(Ev, com)
    del ctx_v
    torch.cuda.empty_cache()
    return nets, aud, mot


def sweep_dim(sizes, set_size, REF_AUD, REF_MOT, vary, PG, TSI, NFOLD, K, key):
    """For each size: K draws of the SWEPT knob's seed only (the other knob's seed held fixed)
    -> net SD + mean overlap vs this dim's deterministic-endpoint basis (REF)."""
    out = []
    for sz in sizes:
        set_size(sz)
        nets_k = {nm: [] for nm in ("GT", "seeded", "zero")}
        aud_ov, mot_ov, aud_ang, mot_ang = [], [], [], []
        k_here = 1 if sz["deterministic"] else K       # endpoint is genuinely deterministic now
        for d in range(k_here):
            # vary ONLY the swept knob's seed; hold the other fixed (isolates the knob)
            a_seed, m_seed, r_seed = (201 + d, 211 + d, 0) if vary == "stem" else (201, 211, d)
            nets, aud, mot = build_ops(a_seed, m_seed, r_seed, PG, TSI, NFOLD)
            for nm in nets_k:
                nets_k[nm].append(nets[nm])
            da = mss.pca_diag(aud, REF_AUD, mss.XAUD); dm = mss.pca_diag(mot, REF_MOT, mss.XMOT)
            aud_ov.append(da["mean_cos2_overlap"]); mot_ov.append(dm["mean_cos2_overlap"])
            aud_ang.append(da["max_principal_angle_deg"]); mot_ang.append(dm["max_principal_angle_deg"])
        net_sd = {nm: round(float(np.std(nets_k[nm], ddof=1)), 5) if len(nets_k[nm]) > 1 else 0.0
                  for nm in nets_k}
        row = {"size": sz["val"], "label": sz["label"], "K": k_here,
               "net_SD": net_sd,
               "net_mean": {nm: round(float(np.mean(nets_k[nm])), 4) for nm in nets_k},
               "aud_overlap_mean": round(float(np.mean(aud_ov)), 5),
               "mot_overlap_mean": round(float(np.mean(mot_ov)), 5),
               "aud_max_angle_mean": round(float(np.mean(aud_ang)), 3),
               "mot_max_angle_mean": round(float(np.mean(mot_ang)), 3)}
        out.append(row)
        print(f"  [{key}] {sz['label']} (K={k_here}): net_SD {net_sd} | aud_overlap {row['aud_overlap_mean']} "
              f"aud_angle {row['aud_max_angle_mean']}deg", flush=True)
    return out


def main():
    global N_STEMS, N_ROWS
    ap = argparse.ArgumentParser(description="R-DGG PCA variance vs subsample size (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--k", type=int, default=5, help="seeds per non-deterministic size")
    ap.add_argument("--smoke", action="store_true", help="1 size x 1 seed per dim, to check end-to-end")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only"
    _config.set_paths(args.corpus, args.split)
    SRC, SUF = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    # ---- setup -> set mss globals (mirror metric_seed_sweep.main) ----
    mss.CORPUS0 = load_corpus()
    mss.SRC, mss.SUF = SRC, SUF
    mss.RAW_AUD = {sp: r for sp, r in zip(mss.CORPUS0.speakers, parallel_map(speaker_raw, mss.CORPUS0.speakers))
                   if r is not None}
    mss.GT_CACHE = load_all_motion(f"{C}/GT/{S}_{SUF}", SRC, mss.CORPUS0)
    mss.MRAW = {sp: speaker_motion_raw(sp, SUF, SRC, mss.CORPUS0.SPK_G_AUD[sp].shape[0])
                for sp in mss.CORPUS0.speakers if sp in mss.CORPUS0.SPK_G_AUD}
    mss.MRAW = {sp: r for sp, r in mss.MRAW.items() if r is not None}
    mss.MASK = gtonly_eval_mask(SUF, SRC, mss.CORPUS0)
    mss.CACHES = {"GT": mss.GT_CACHE,
                  "seeded": load_all_motion(f"{C}/AVTR-1_seeded/{S}_{SUF}", SRC, mss.CORPUS0),
                  "zero": load_all_motion(f"{C}/AVTR-1_zeroother/{S}_{SUF}", SRC, mss.CORPUS0)}
    mss.FOKS = {nm: gridded_face_ok(mss.CACHES[nm], mss.CORPUS0) for nm in mss.CACHES}
    mss.XAUD = np.concatenate([mss.RAW_AUD[sp] for sp in sorted(mss.RAW_AUD)], 0)[:5000]
    mss.XMOT = np.concatenate([mss.MRAW[sp] for sp in sorted(mss.MRAW)], 0)[:5000]
    NFOLD = mss.CORPUS0.NFOLD
    n_speakers, n_rows_aud = len(mss.RAW_AUD), mss.XAUD.shape and sum(mss.RAW_AUD[s].shape[0] for s in mss.RAW_AUD)
    ALL_STEMS = len(mss.MRAW)
    print(f"  [caches] {time.time()-t0:.1f}s; NFOLD={NFOLD}; speakers(motion)={ALL_STEMS}; "
          f"aud rows~{n_rows_aud}", flush=True)

    # ---- PG/TSI (PCA-independent enough; built at default size) ----
    N_STEMS, N_ROWS = 60, 40000
    ctx_tmp, _ = mss.make_ctx(mss.CORPUS0, 103, 0, None)
    PG = collect_design(mss.GT_CACHE, ctx_tmp, mss.MASK)
    TSI = mss.compute_tsi(PG)
    del ctx_tmp
    # per-dim deterministic-ENDPOINT reference bases, so overlap -> 1 as that dim -> full.
    # dim A endpoint = all stems (seed inert), rows fixed 40000, row_seed 0.
    N_STEMS, N_ROWS = ALL_STEMS, 40000
    REF_A_AUD, REF_A_MOT = mss.audio_pca(201, 0), mss.motion_pca(211, 0)
    # dim B endpoint = fixed 60 stems (seed 201/211), all rows (row seed inert).
    N_STEMS, N_ROWS = 60, 10 ** 9
    REF_B_AUD, REF_B_MOT = mss.audio_pca(201, 0), mss.motion_pca(211, 0)

    def set_stem(sz):
        global N_STEMS, N_ROWS
        N_STEMS, N_ROWS = sz["val"], 40000

    def set_row(sz):
        global N_STEMS, N_ROWS
        N_STEMS, N_ROWS = 60, sz["val"]

    stem_sizes = [{"val": 30, "label": "n=30", "deterministic": False},
                  {"val": 60, "label": "n=60 (default)", "deterministic": False},
                  {"val": 90, "label": "n=90", "deterministic": False},
                  {"val": 120, "label": "n=120", "deterministic": False},
                  {"val": ALL_STEMS, "label": f"n={ALL_STEMS} (ALL)", "deterministic": True}]
    row_sizes = [{"val": 10000, "label": "r=10k", "deterministic": False},
                 {"val": 20000, "label": "r=20k", "deterministic": False},
                 {"val": 40000, "label": "r=40k (default)", "deterministic": False},
                 {"val": 80000, "label": "r=80k", "deterministic": False},
                 {"val": 10 ** 9, "label": "r=ALL", "deterministic": True}]
    if args.smoke:
        stem_sizes, row_sizes, args.k = stem_sizes[:1], row_sizes[:1], 1

    dimA = sweep_dim(stem_sizes, set_stem, REF_A_AUD, REF_A_MOT, "stem", PG, TSI, NFOLD, args.k, "stem")
    dimB = sweep_dim(row_sizes, set_row, REF_B_AUD, REF_B_MOT, "row", PG, TSI, NFOLD, args.k, "row")

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {"source": SRC, "suffix": SUF, "split": S, "n_folds": NFOLD,
              "n_speakers_motion": ALL_STEMS,
              "columns": ["GT", "AVTR-1_seeded (base0 stock)", "AVTR-1_zeroother"],
              "note": "overlap is vs the FULL-set (deterministic) basis; net_SD is across K seeds at that size",
              "default_metric_seed_net_SD_ref": {"GT": 0.0534, "seeded": 0.0775, "zero": 0.0634},
              "generation_seed_SD_for_comparison": 0.111,
              "dimA_stem_subsample_size": {"rows_fixed": 40000, "per_size": dimA},
              "dimB_row_subsample_size": {"stems_fixed": 60, "per_size": dimB}}
    with open(f"{out_dir}/pca_size_sweep__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/pca_size_sweep__{S}.json", flush=True)
    print("PCA_SIZE_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
