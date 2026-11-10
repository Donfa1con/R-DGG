"""Shift-correction ablation for R-DGG: is the time-shift correction needed?

NEW code only. It imports the already-written R-DGG blocks and the dose_curve.py
helpers; it edits nothing in rdgg/, adds no model to _config.MODELS, and commits
nothing of its own. See CVPR_zeroother_design.md section 10 (g_shift vs g_a_m).

The metric is net = g_a_m - g_shift (R-DGG). This harness reports, for every model,
every available seed, and the two GT controls (GT positive, GT*other surrogate
negative):
    g_a_m   = 1e4 * log(mot_r / both_r)                 (UNCORRECTED aligned gain)
    g_shift = 1e4 * mean_k log(mot_n[k] / both_n[k])    (time-shift floor, 100 nulls)
    net     = g_a_m - g_shift                           (CORRECTED = R-DGG)
with an identity-clustered bootstrap CI (one-sided p = P(draw <= 0)) on BOTH g_a_m
and net, so the reader sees whether dropping the correction destroys the separation
between reactive models and the controls.

Each column is scored on its OWN GT&model marginal mask (RESULTS.md-comparable),
the same arithmetic as leaderboard_stats / cfg0_compare.marginal.

Run in the shared metrics env, as pipeline/4_rdgg.sh does:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/shift_ablation.py
A 22 GB GPU needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to hold the
four fold operators without fragmenting.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import BOOTSTRAP, SOURCES, SURROGATE, LSPK, _ident_key
from _features import collect_design, load_all_motion
from _granger import capture_energies
from _bootstrap import cluster_sums
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column


# ---- columns: every model x every available seed, plus the two GT controls ----
DYADIC_SEEDS = {
    "AVTR-1": ["AVTR-1", "AVTR-1_seeded", "AVTR-1_seeded_b1"],
    "DyStream": ["dystream", "dystream_seeded", "dystream_seeded_b1"],
    "AvatarForcing": ["AvatarForcing", "AvatarForcing_seeded", "AvatarForcing_seeded_b1"],
}
TALKING = {  # one seed each; non-dyadic -> genuine near-zero reaction (model-side controls)
    "SoulX Lite": "SoulX_FlashHead_Lite",
    "SoulX Pro": "SoulX_FlashHead_Pro",
    "Ditto": "ditto",
    "FLOAT": "FLOAT",
}
SEED_TAG = {0: "unseeded", 1: "seed0", 2: "seed1"}


# ---- raw vs net arithmetic (pooled over clusters `sel`; x1, callers scale by 1e4) ----
def net_of(sel, sums):
    br, mr, bn, mn = sums
    return np.log(mr[sel].sum() / br[sel].sum()) - np.log(mn[:, sel].sum(1) / bn[:, sel].sum(1)).mean()


def raw_of(sel, sums):
    br, mr, bn, mn = sums
    return np.log(mr[sel].sum() / br[sel].sum())


def boot_ci(E, stems, corrected):
    """Identity-clustered bootstrap CI (x1e4) of the pooled gain. corrected=True ->
    net = raw - floor (R-DGG); corrected=False -> raw g_a_m (no shift correction).
    one-sided p = P(draw <= 0), same convention as the leaderboard."""
    clusters = list(dict.fromkeys(_ident_key(s) for s in stems))
    idx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    sums = cluster_sums(E, stems, {s: idx[_ident_key(s)] for s in stems}, NC)
    stat = net_of if corrected else raw_of
    point = float(stat(np.arange(NC), sums)) * 1e4
    rng = np.random.default_rng(0)
    d = np.array([stat(rng.integers(0, NC, size=NC), sums) for _ in range(BOOTSTRAP)]) * 1e4
    lo, p50, hi = np.percentile(d, [2.5, 50, 97.5])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "p50": round(float(p50), 4),
            "hi": round(float(hi), 4), "p": round(float((d <= 0).mean()), 4), "n_clusters": NC}


def decompose(E, stems):
    """g_a_m, g_shift, net, floor_sd (x1e4) pooled over stems, plus per-stem raw/net arrays."""
    BR = sum(E[s]["both_r"] for s in stems)
    MR = sum(E[s]["mot_r"] for s in stems)
    BN = np.sum([E[s]["both_n"] for s in stems], 0)
    MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    g_a_m = float(np.log(MR / BR)) * 1e4
    floor_k = np.log(MN / BN) * 1e4
    g_shift = float(floor_k.mean())
    raw_i = np.array([np.log(E[s]["mot_r"] / E[s]["both_r"]) for s in stems]) * 1e4
    net_i = raw_i - np.array([np.log(E[s]["mot_n"] / E[s]["both_n"]).mean() for s in stems]) * 1e4
    return {"g_a_m": round(g_a_m, 4), "g_shift": round(g_shift, 4),
            "net": round(g_a_m - g_shift, 4), "floor_sd": round(float(floor_k.std()), 4)}, raw_i, net_i


def build_surrogate_cache(gt_cache, corpus):
    """GT*other negative control: each listener wears another person's GT motion
    (corpus.SURRPERM), wrapped to length with the wrap seams blinded. Same as
    score_source's surrogate, so the control matches the leaderboard."""
    mcache = {ls: gt_cache[corpus.SURRPERM[ls]] for ls in corpus.listeners if corpus.SURRPERM[ls] in gt_cache}
    surr = {}
    for tgt, (regions, _fok, fps) in mcache.items():
        if tgt not in gt_cache:
            continue
        n_tgt = len(gt_cache[tgt][1])
        n_src = len(mcache[tgt][1])
        if n_src < 2:
            continue
        off = int(np.random.default_rng(90000 + corpus.listeners.index(tgt)).integers(0, n_src))
        wrap = (off + np.arange(n_tgt)) % n_src
        fok = mcache[tgt][1][wrap].copy()
        for seam in np.flatnonzero(np.diff(wrap) != 1) + 1:
            fok[seam:seam + LSPK] = False
        surr[tgt] = ({rn: rm[wrap] for rn, rm in regions.items()}, fok, fps)
    return surr


def score(E, own_r2):
    """One column's full record from its captured per-listener energies."""
    stems = sorted(E)
    dec, raw_i, net_i = decompose(E, stems)
    rec = {
        "n_stems": len(stems),
        "n_identities": len({_ident_key(s) for s in stems}),
        "own_r2": round(float(own_r2), 4),
        **dec,
        "ci_net": boot_ci(E, stems, corrected=True),       # R-DGG (corrected)
        "ci_raw": boot_ci(E, stems, corrected=False),       # UNCORRECTED g_a_m
    }
    arrays = {"stems": np.array(stems), "raw_i": raw_i, "net_i": net_i,
              "ident": np.array([_ident_key(s) for s in stems])}
    return rec, arrays


def main():
    ap = argparse.ArgumentParser(description="R-DGG shift-correction ablation (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] operator built {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    columns = {}      # label -> record (all numbers)
    npz = {}          # flat arrays for redraw (per-dyad raw/net)
    order = []        # presentation order

    def add(label, model, group, seed_tag, E, own_r2):
        rec, arr = score(E, own_r2)
        rec.update({"model": model, "group": group, "seed": seed_tag})
        columns[label] = rec
        for k, v in arr.items():
            npz[f"{label}::{k}"] = v
        order.append(label)
        c = rec["ci_net"]; r = rec["ci_raw"]
        print(f"  [{label:22s}] n={rec['n_stems']:3d} g_a_m={rec['g_a_m']:+.3f} g_shift={rec['g_shift']:+.3f} "
              f"net={rec['net']:+.3f} | CI_net[{c['lo']:+.2f},{c['hi']:+.2f}]p={c['p']:.3f} "
              f"CI_raw[{r['lo']:+.2f},{r['hi']:+.2f}]p={r['p']:.3f}", flush=True)

    # --- GT positive control ---
    print("\n  [controls] GT (positive) + GT*other surrogate (negative) ...", flush=True)
    E_gt, own_gt = capture_energies(collect_design(gt_cache, ctx, mask_gtonly), ops, ctx)
    add("GT", "GT", "GT-control", "GT", E_gt, own_gt)

    # --- GT*other surrogate negative control ---
    surr = build_surrogate_cache(gt_cache, corpus)
    E_sur, own_sur = capture_energies(collect_design(surr, ctx, mask_gtonly), ops, ctx)
    add(SURROGATE, SURROGATE, "GT-control", "surrogate", E_sur, own_sur)

    # --- dyadic models, every seed ---
    print("\n  [dyadic] AVTR-1 / DyStream / AvatarForcing, 3 seeds each ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", mask_gtonly, ctx, ops, corpus, source)
            add(f"{model} [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], E, r2)

    # --- talking-head models (non-dyadic model-side controls), one seed each ---
    print("\n  [non-dyadic] talking-head models, one seed each ...", flush=True)
    for model, d in TALKING.items():
        E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", mask_gtonly, ctx, ops, corpus, source)
        add(model, model, "non-dyad", "single", E, r2)

    # --- seed-averaged dyadic summary (mean/SD of per-seed net over the 3 seeds) ---
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        nets = np.array([columns[l]["net"] for l in labels])
        raws = np.array([columns[l]["g_a_m"] for l in labels])
        seed_avg[model] = {
            "seeds": labels,
            "net_mean": round(float(nets.mean()), 4), "net_sd": round(float(nets.std(ddof=1)), 4),
            "net_min": round(float(nets.min()), 4), "net_max": round(float(nets.max()), 4),
            "g_a_m_mean": round(float(raws.mean()), 4), "g_a_m_sd": round(float(raws.std(ddof=1)), 4),
        }

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG shift-correction ablation",
        "question": "is the time-shift correction (net = g_a_m - g_shift) needed?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask": "each column on its OWN GT-reactive & model face_ok marginal mask (RESULTS.md-comparable)",
        "arithmetic": {
            "g_a_m": "1e4 * log(mot_r/both_r)  -- aligned gain, UNCORRECTED",
            "g_shift": "1e4 * mean_k log(mot_n[k]/both_n[k])  -- mean of 100 time-shift nulls",
            "net": "g_a_m - g_shift  -- R-DGG, CORRECTED",
            "ci": f"identity-clustered bootstrap, B={BOOTSTRAP}, one-sided p=P(draw<=0)",
        },
        "controls": {
            "GT": "held-out positive control (must react, net>0)",
            SURROGATE: "GT motion under a wrong pairing (negative control, must read ~0)",
            "non-dyad": "talking-head models with real audio but no dyadic reaction (must read ~0)",
        },
        "order": order,
        "columns": columns,
        "seed_averaged_dyadic": seed_avg,
    }
    jpath = f"{outdir}/shift_ablation__{S}.json"
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    npz_path = f"{outdir}/shift_ablation__{S}.npz"
    np.savez_compressed(npz_path, **npz)
    print(f"\n  [out] {jpath}\n        {npz_path}", flush=True)
    print("SHIFT_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
