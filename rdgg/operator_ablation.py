"""Reference-operator ablation for R-DGG: GT reference operator vs the model's own.

Question: does R-DGG need the GT REFERENCE reaction operator, or would a self-fitted
operator do? R-DGG scores every model's reaction through ONE operator B_GT fit on
ground-truth reaction (the "Reference-Based" in the name). This harness re-scores
every column through a SELF operator B_self, cross-fit on THAT column's own reaction,
and compares:
    net_ref  = g_a_m - g_shift  with B_GT   (R-DGG, reference)
    net_self = g_a_m - g_shift  with B_self  (self-Granger, fit on the model itself)
with an identity-clustered bootstrap CI on each. B_self is leave-one-identity-fold
cross-fit (fit_bgt_fold holds the scored fold out), so a row is never scored by an
operator that saw it -- the self gain is still honest, just self-referential.

The point: B_self is free to fit whatever speaker->motion structure a column has, so
a lip-sync talking-head (real audio, no dyadic reaction) can score a positive self
gain; B_GT credits only motion that matches GENUINE human reaction, so the controls
read ~0. If the reference operator is needed, the controls move from ~0 (reference)
to positive (self) while GT is unchanged (B_GT IS GT's own operator).

NEW code only. It imports the already-written R-DGG blocks and composes them; it
edits nothing in rdgg/, adds no model to _config.MODELS, and commits nothing of its
own. Every column scores on the COMMON mask (identical frames), like the leaderboard.

Run (shared metrics env, GPU):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    pixi run --manifest-path rdgg/pixi.toml python rdgg/operator_ablation.py
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
from _granger import capture_energies, fit_bgt_fold
from _bootstrap import cluster_sums, gnet_boot_ci
from compute import load_corpus
from dose_curve import build_context_and_operator, intersect_masks


DYADIC_SEEDS = {
    "AVTR-1": ["AVTR-1", "AVTR-1_seeded", "AVTR-1_seeded_b1"],
    "DyStream": ["dystream", "dystream_seeded", "dystream_seeded_b1"],
    "AvatarForcing": ["AvatarForcing", "AvatarForcing_seeded", "AvatarForcing_seeded_b1"],
}
TALKING = {"SoulX Lite": "SoulX_FlashHead_Lite", "SoulX Pro": "SoulX_FlashHead_Pro",
           "Ditto": "ditto", "FLOAT": "FLOAT"}
SEED_TAG = {0: "unseeded", 1: "seed0", 2: "seed1"}


# ---- raw/net arithmetic + identity-clustered CI (same as shift_ablation) ----
def net_of(sel, sums):
    br, mr, bn, mn = sums
    return np.log(mr[sel].sum() / br[sel].sum()) - np.log(mn[:, sel].sum(1) / bn[:, sel].sum(1)).mean()


def raw_of(sel, sums):
    br, mr, bn, mn = sums
    return np.log(mr[sel].sum() / br[sel].sum())


def boot_ci(E, stems, corrected):
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
    BR = sum(E[s]["both_r"] for s in stems)
    MR = sum(E[s]["mot_r"] for s in stems)
    BN = np.sum([E[s]["both_n"] for s in stems], 0)
    MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    g_a_m = float(np.log(MR / BR)) * 1e4
    floor_k = np.log(MN / BN) * 1e4
    raw_i = np.array([np.log(E[s]["mot_r"] / E[s]["both_r"]) for s in stems]) * 1e4
    net_i = raw_i - np.array([np.log(E[s]["mot_n"] / E[s]["both_n"]).mean() for s in stems]) * 1e4
    return {"g_a_m": round(g_a_m, 4), "g_shift": round(float(floor_k.mean()), 4),
            "net": round(g_a_m - float(floor_k.mean()), 4), "floor_sd": round(float(floor_k.std()), 4)}, raw_i, net_i


def operator_record(E, own_r2):
    stems = sorted(E)
    dec, raw_i, net_i = decompose(E, stems)
    rec = {"n_stems": len(stems), "n_identities": len({_ident_key(s) for s in stems}),
           "own_r2": round(float(own_r2), 4), **dec,
           "ci_net": boot_ci(E, stems, corrected=True),
           "ci_raw": boot_ci(E, stems, corrected=False)}
    return rec, raw_i, net_i, stems


def seed_avg_ci(E_list):
    """Seed-average over the N seed columns, then the EXACT identity-cluster bootstrap.
    On the common mask the seeds share stems, so average the per-stem energies
    (both_r, mot_r, both_n[K], mot_n[K]) over the seeds, then run gnet_boot_ci on the
    averaged energies -- RESUME §12: average each stem's value over seeds, THEN identity
    bootstrap (do NOT add seed SD in quadrature; the seed noise is inside this CI)."""
    stems = sorted(set.intersection(*[set(E) for E in E_list]))
    E_avg = {s: {"both_r": float(np.mean([E[s]["both_r"] for E in E_list])),
                 "mot_r": float(np.mean([E[s]["mot_r"] for E in E_list])),
                 "both_n": np.mean([E[s]["both_n"] for E in E_list], axis=0),
                 "mot_n": np.mean([E[s]["mot_n"] for E in E_list], axis=0)}
             for s in stems}
    d = gnet_boot_ci(E_avg, stems, _ident_key, BOOTSTRAP, one_sided=True)
    return {"net": round(d["point"], 4), "lo": round(d["lo"], 4), "hi": round(d["hi"], 4),
            "p": round(d["p"], 4), "n_stems": len(stems), "n_identities": d["n_clusters"]}


def stash_energies(npz, label, oper, E, stems):
    """Save the per-stem residual energies so seed-averaging and the EXACT identity
    bootstrap can run offline (no GPU). These four arrays are the full raw material:
    gnet = log(sum mot_r / sum both_r) - mean_k log(sum mot_n[k] / sum both_n[k])."""
    npz[f"{label}::{oper}::both_r"] = np.array([E[s]["both_r"] for s in stems], dtype=np.float64)
    npz[f"{label}::{oper}::mot_r"] = np.array([E[s]["mot_r"] for s in stems], dtype=np.float64)
    npz[f"{label}::{oper}::both_n"] = np.stack([E[s]["both_n"] for s in stems], axis=1)  # (K, n_stems)
    npz[f"{label}::{oper}::mot_n"] = np.stack([E[s]["mot_n"] for s in stems], axis=1)    # (K, n_stems)


def build_surrogate_cache(gt_cache, corpus):
    """GT*other negative control (same construction as compute.score_source)."""
    mcache = {ls: gt_cache[corpus.SURRPERM[ls]] for ls in corpus.listeners if corpus.SURRPERM[ls] in gt_cache}
    surr = {}
    for tgt, (regions, _fok, fps) in mcache.items():
        if tgt not in gt_cache:
            continue
        n_tgt = len(gt_cache[tgt][1]); n_src = len(mcache[tgt][1])
        if n_src < 2:
            continue
        off = int(np.random.default_rng(90000 + corpus.listeners.index(tgt)).integers(0, n_src))
        wrap = (off + np.arange(n_tgt)) % n_src
        fok = mcache[tgt][1][wrap].copy()
        for seam in np.flatnonzero(np.diff(wrap) != 1) + 1:
            fok[seam:seam + LSPK] = False
        surr[tgt] = ({rn: rm[wrap] for rn, rm in regions.items()}, fok, fps)
    return surr


def fit_self_ops(mcache, ctx, fit_mask, corpus):
    """B_self: the reaction operator fit on THIS column's own (listener motion) design,
    leave-one-identity-fold cross-fit. Driver is still the GT speaker (ctx.driver); only
    the target reaction is the column's own. Returns ops list or None if a fold is too small."""
    per = collect_design(mcache, ctx, fit_mask)
    if not per or len(per) < 10:
        return None
    ops = [fit_bgt_fold(per, fold, ctx) for fold in range(corpus.NFOLD)]
    return None if any(op is None for op in ops) else ops


def main():
    ap = argparse.ArgumentParser(description="R-DGG reference-vs-self operator ablation (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops_gt, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] GT reference operator built {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    # COMMON mask (identical frames): GT-reactive & every model-seed column's face_ok.
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    common = mask_gtonly
    for d in all_dirs:
        common = intersect_masks(common, gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus))
    print(f"  [common-mask] {len(common)} stems, {sum(int(v.sum()) for v in common.values())} frames", flush=True)

    columns, npz, order = {}, {}, []
    E_keep = {}   # retain dyadic-seed energies in RAM for the offline seed-average + CI

    def score_column(label, model, group, seed_tag, mcache, allow_self=True):
        # REFERENCE: B_GT applied (R-DGG)
        E_ref, r2_ref = capture_energies(collect_design(mcache, ctx, common), ops_gt, ctx)
        ref, raw_i_r, net_i_r, stems = operator_record(E_ref, r2_ref)
        stash_energies(npz, label, "ref", E_ref, stems)
        # SELF: B_self fit on this column's own mask_gtonly design, scored on common (cross-fit)
        self_rec = None
        E_self = None
        if allow_self:
            ops_self = fit_self_ops(mcache, ctx, mask_gtonly, corpus)
            if ops_self is not None:
                E_self, r2_self = capture_energies(collect_design(mcache, ctx, common), ops_self, ctx)
                self_rec, raw_i_s, net_i_s, _ = operator_record(E_self, r2_self)
                npz[f"{label}::net_i_self"] = net_i_s
                stash_energies(npz, label, "self", E_self, stems)
                del ops_self
        columns[label] = {"model": model, "group": group, "seed": seed_tag,
                          "reference": ref, "self": self_rec}
        if group == "dyadic":
            E_keep[label] = {"ref": E_ref, "self": E_self}
        npz[f"{label}::stems"] = np.array(stems)
        npz[f"{label}::net_i_ref"] = net_i_r
        npz[f"{label}::ident"] = np.array([_ident_key(s) for s in stems])
        order.append(label)
        torch.cuda.empty_cache()                               # bound transient peaks (2 captures + B-fit/column)
        sn = self_rec["net"] if self_rec else float("nan")
        sp = self_rec["ci_net"]["p"] if self_rec else float("nan")
        print(f"  [{label:22s}] n={ref['n_stems']:3d} | REF net={ref['net']:+.3f} "
              f"p={ref['ci_net']['p']:.3f} | SELF net={sn:+.3f} p={sp:.3f}", flush=True)

    print("\n  [controls] GT + GT*other surrogate ...", flush=True)
    score_column("GT", "GT", "GT-control", "GT", gt_cache)                 # B_self(GT) == B_GT by construction
    score_column(SURROGATE, SURROGATE, "GT-control", "surrogate", build_surrogate_cache(gt_cache, corpus))

    print("\n  [dyadic] 3 seeds each ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            mcache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
            score_column(f"{model} [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], mcache)

    print("\n  [non-dyadic] talking-head controls ...", flush=True)
    for model, d in TALKING.items():
        mcache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
        score_column(model, model, "non-dyad", "single", mcache)

    # seed-averaged dyadic summary: average each stem's energies over the 3 seeds, then the
    # EXACT identity bootstrap -> seed-averaged net + identity CI, for BOTH operators.
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        ref_pts = np.array([columns[l]["reference"]["net"] for l in labels])
        entry = {"n_seeds": len(labels),
                 "reference": {**seed_avg_ci([E_keep[l]["ref"] for l in labels]),
                               "per_seed_net": [round(float(x), 4) for x in ref_pts],
                               "per_seed_sd": round(float(ref_pts.std(ddof=1)), 4)}}
        E_self_seeds = [E_keep[l]["self"] for l in labels]
        if all(e is not None for e in E_self_seeds):
            self_pts = np.array([columns[l]["self"]["net"] for l in labels])
            entry["self"] = {**seed_avg_ci(E_self_seeds),
                             "per_seed_net": [round(float(x), 4) for x in self_pts],
                             "per_seed_sd": round(float(self_pts.std(ddof=1)), 4)}
        seed_avg[model] = entry
        r = entry["reference"]; sf = entry.get("self")
        print(f"  [seed-avg {model:13s}] REF net={r['net']:+.3f} CI[{r['lo']:+.2f},{r['hi']:+.2f}] p={r['p']:.3f}"
              + (f" | SELF net={sf['net']:+.3f} CI[{sf['lo']:+.2f},{sf['hi']:+.2f}] p={sf['p']:.3f}" if sf else ""),
              flush=True)

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG reference-vs-self operator ablation",
        "question": "does R-DGG need the GT REFERENCE reaction operator, or would a self-fitted operator do?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask": "ONE shared COMMON mask = GT-reactive & every model-seed column's face_ok (identical frames)",
        "operators": {
            "reference": "B_GT fit on ground-truth reaction, applied to every column (R-DGG)",
            "self": "B_self fit on THAT column's own reaction, leave-one-identity-fold cross-fit, applied to itself",
        },
        "arithmetic": {
            "net": "g_a_m - g_shift (x1e4); g_a_m=1e4*log(mot_r/both_r); g_shift=mean of 100 time-shift nulls",
            "ci": f"identity-clustered bootstrap, B={BOOTSTRAP}, one-sided p=P(draw<=0)",
        },
        "note_GT": "GT's self operator IS the reference operator, so GT ref==self by construction (a sanity check).",
        "npz_contents": ("per column: <label>::stems, ::ident, ::net_i_{ref,self}; and the raw per-stem "
                         "energies <label>::{ref,self}::{both_r,mot_r,both_n,mot_n} (both_n/mot_n are (K,n)). "
                         "These energies let seed-averaging and the EXACT identity bootstrap run offline, no GPU."),
        "order": order,
        "columns": columns,
        "seed_averaged_dyadic": seed_avg,
    }
    with open(f"{outdir}/operator_ablation__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(f"{outdir}/operator_ablation__{S}.npz", **npz)
    print(f"\n  [out] {outdir}/operator_ablation__{S}.json (+ .npz)", flush=True)
    print("OPERATOR_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
