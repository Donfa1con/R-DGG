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
from _features import collect_design, gridded_face_ok, load_all_motion
from _granger import capture_energies
from _bootstrap import cluster_sums
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column, intersect_masks


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


def stash_energies(npz, label, E, stems):
    """Save per-stem residual energies so seed-averaging and the EXACT identity bootstrap
    run offline (no GPU): both_r, mot_r (n_stems,) and both_n, mot_n (K, n_stems)."""
    npz[f"{label}::both_r"] = np.array([E[s]["both_r"] for s in stems], dtype=np.float64)
    npz[f"{label}::mot_r"] = np.array([E[s]["mot_r"] for s in stems], dtype=np.float64)
    npz[f"{label}::both_n"] = np.stack([E[s]["both_n"] for s in stems], axis=1)
    npz[f"{label}::mot_n"] = np.stack([E[s]["mot_n"] for s in stems], axis=1)


def seed_avg_ci(E_list, corrected):
    """Seed-average over the N seed columns, then the EXACT identity-cluster bootstrap.
    On the common mask the seeds share stems, so average each stem's energies over the
    seeds, then boot_ci on the averaged energies (RESUME §12: average the value over seeds,
    THEN identity bootstrap; the seed noise lives inside this CI). corrected -> net; else raw."""
    stems = sorted(set.intersection(*[set(E) for E in E_list]))
    E_avg = {s: {"both_r": float(np.mean([E[s]["both_r"] for E in E_list])),
                 "mot_r": float(np.mean([E[s]["mot_r"] for E in E_list])),
                 "both_n": np.mean([E[s]["both_n"] for E in E_list], axis=0),
                 "mot_n": np.mean([E[s]["mot_n"] for E in E_list], axis=0)}
             for s in stems}
    d = boot_ci(E_avg, stems, corrected)
    return {"point": d["point"], "lo": d["lo"], "hi": d["hi"], "p": d["p"],
            "n_stems": len(stems), "n_identities": d["n_clusters"]}


def main():
    ap = argparse.ArgumentParser(description="R-DGG shift-correction ablation (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--mask", choices=["marginal", "common"], default="common",
                    help="common (default): ONE shared mask = GT-reactive & every column's "
                         "face_ok, so all columns score on IDENTICAL frames (leaderboard rule); "
                         "writes shift_ablation__<split>.*. marginal: each column on its OWN "
                         "GT&model mask; writes shift_ablation_marginal__<split>.*.")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] operator built {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    # Which mask every column is scored on. marginal = each column's own GT&model mask.
    # common = GT-reactive & EVERY model-seed column's face_ok (the surrogate is excluded
    # from the intersection, as in the leaderboard, because its seam-blinding removes rows).
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    if args.mask == "common":
        eval_mask = mask_gtonly
        for d in all_dirs:
            fok = gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus)
            eval_mask = intersect_masks(eval_mask, fok)
        n_fr = sum(int(v.sum()) for v in eval_mask.values())
        print(f"  [common-mask] {len(eval_mask)} stems, {n_fr} grid frames valid in GT & all "
              f"{len(all_dirs)} model-seed columns", flush=True)
    else:
        eval_mask = mask_gtonly

    columns = {}      # label -> record (all numbers)
    npz = {}          # flat arrays for redraw (per-dyad raw/net + raw energies)
    order = []        # presentation order
    E_keep = {}       # retain dyadic-seed energies in RAM for the offline seed-average + CI

    def add(label, model, group, seed_tag, E, own_r2):
        rec, arr = score(E, own_r2)
        rec.update({"model": model, "group": group, "seed": seed_tag})
        columns[label] = rec
        for k, v in arr.items():
            npz[f"{label}::{k}"] = v
        stash_energies(npz, label, E, sorted(E))
        if group == "dyadic":
            E_keep[label] = E
        order.append(label)
        c = rec["ci_net"]; r = rec["ci_raw"]
        print(f"  [{label:22s}] n={rec['n_stems']:3d} g_a_m={rec['g_a_m']:+.3f} g_shift={rec['g_shift']:+.3f} "
              f"net={rec['net']:+.3f} | CI_net[{c['lo']:+.2f},{c['hi']:+.2f}]p={c['p']:.3f} "
              f"CI_raw[{r['lo']:+.2f},{r['hi']:+.2f}]p={r['p']:.3f}", flush=True)

    # --- GT positive control ---
    print("\n  [controls] GT (positive) + GT*other surrogate (negative) ...", flush=True)
    E_gt, own_gt = capture_energies(collect_design(gt_cache, ctx, eval_mask), ops, ctx)
    add("GT", "GT", "GT-control", "GT", E_gt, own_gt)

    # --- GT*other surrogate negative control ---
    surr = build_surrogate_cache(gt_cache, corpus)
    E_sur, own_sur = capture_energies(collect_design(surr, ctx, eval_mask), ops, ctx)
    add(SURROGATE, SURROGATE, "GT-control", "surrogate", E_sur, own_sur)

    # --- dyadic models, every seed ---
    print("\n  [dyadic] AVTR-1 / DyStream / AvatarForcing, 3 seeds each ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", eval_mask, ctx, ops, corpus, source)
            add(f"{model} [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], E, r2)

    # --- talking-head models (non-dyadic model-side controls), one seed each ---
    print("\n  [non-dyadic] talking-head models, one seed each ...", flush=True)
    for model, d in TALKING.items():
        E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", eval_mask, ctx, ops, corpus, source)
        add(model, model, "non-dyad", "single", E, r2)

    # --- seed-averaged dyadic summary: average each stem's energies over the 3 seeds, then
    # the EXACT identity bootstrap -> seed-averaged net (corrected) AND raw g_a_m (uncorrected)
    # each with its identity CI, plus the per-seed point spread. ---
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        E_list = [E_keep[l] for l in labels]
        nets = np.array([columns[l]["net"] for l in labels])
        raws = np.array([columns[l]["g_a_m"] for l in labels])
        net_ci = seed_avg_ci(E_list, corrected=True)
        raw_ci = seed_avg_ci(E_list, corrected=False)
        seed_avg[model] = {
            "seeds": labels, "n_seeds": len(labels),
            "net": net_ci,          # seed-averaged R-DGG net + identity CI
            "g_a_m": raw_ci,        # seed-averaged uncorrected gain + identity CI
            "per_seed_net": [round(float(x), 4) for x in nets],
            "per_seed_net_sd": round(float(nets.std(ddof=1)), 4),
            "per_seed_g_a_m": [round(float(x), 4) for x in raws],
            "per_seed_g_a_m_sd": round(float(raws.std(ddof=1)), 4),
        }
        n = net_ci
        print(f"  [seed-avg {model:13s}] net={n['point']:+.3f} CI[{n['lo']:+.2f},{n['hi']:+.2f}] "
              f"p={n['p']:.3f} (per-seed SD {seed_avg[model]['per_seed_net_sd']:.3f})", flush=True)

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG shift-correction ablation",
        "question": "is the time-shift correction (net = g_a_m - g_shift) needed?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask_mode": args.mask,
        "mask": ("ONE shared mask = GT-reactive & every model-seed column's face_ok; all columns "
                 "score on IDENTICAL frames (surrogate excluded from the intersection, like the "
                 "leaderboard; its seam-blinding then applies on top)"
                 if args.mask == "common"
                 else "each column on its OWN GT-reactive & model face_ok marginal mask (RESULTS.md-comparable)"),
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
    tag = "" if args.mask == "common" else f"_{args.mask}"   # common is canonical
    jpath = f"{outdir}/shift_ablation{tag}__{S}.json"
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    npz_path = f"{outdir}/shift_ablation{tag}__{S}.npz"
    np.savez_compressed(npz_path, **npz)
    print(f"\n  [out] {jpath}\n        {npz_path}", flush=True)
    print("SHIFT_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
