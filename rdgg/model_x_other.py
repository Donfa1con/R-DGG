"""R-DGG model x other: the specificity control (plan task 9c / list item 10).

NEW standalone harness. Imports the written R-DGG blocks + shift_ablation / dose_curve
helpers; edits nothing in rdgg/, adds no model to _config.MODELS, commits nothing.

GT x other (the surrogate) pairs each listener with ANOTHER conversation's GT motion
(corpus.SURRPERM) and reads ~0 -- the metric does not credit a wrong pairing. This harness
runs the SAME scramble for EVERY model: each model column's listener motion is replaced by
that same model's motion from SURRPERM[ls] (wrapped to length, seams blinded, exactly like
the surrogate), then scored against the real GT speaker audio on the shared common mask.

Reading: a dyadic model's aligned gain should VANISH under the scramble (its reactivity is
specific to the true partner); a talking-head already reads ~0 aligned, so it stays ~0.
The paired drop (aligned net - model x other net) per model is the specificity evidence.

Columns = the 15 shift_ablation columns' models: 3 dyadic x 3 seeds (seed-averaged) + 4
talking-head, each SCRAMBLED, plus GT (aligned, reference) and GT x other (the original
surrogate, = the GT row of this table). Same common mask as shift_ablation.

Run in the shared metrics env:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/model_x_other.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import SOURCES, SURROGATE, LSPK
from _features import collect_design, gridded_face_ok, load_all_motion
from _granger import capture_energies
from compute import load_corpus
from dose_curve import build_context_and_operator, intersect_masks
from shift_ablation import (
    DYADIC_SEEDS, SEED_TAG, TALKING, build_surrogate_cache, score, seed_avg_ci, stash_energies,
)


def scramble_cache(cache, corpus):
    """model x other: each listener ls wears model-motion from SURRPERM[ls], wrapped to the
    target length with the wrap seams blinded -- IDENTICAL transform to build_surrogate_cache,
    but on an arbitrary model cache instead of gt_cache. The target length is the listener's
    own frame count in THIS cache (so the row grid matches the aligned column)."""
    src = {ls: cache[corpus.SURRPERM[ls]] for ls in corpus.listeners
           if corpus.SURRPERM[ls] in cache and ls in cache}
    out = {}
    for ls, (regions, _fok, fps) in src.items():
        n_tgt = len(cache[ls][1])            # listener's own length (its face_ok grid)
        n_src = len(cache[corpus.SURRPERM[ls]][1])
        if n_src < 2:
            continue
        off = int(np.random.default_rng(90000 + corpus.listeners.index(ls)).integers(0, n_src))
        wrap = (off + np.arange(n_tgt)) % n_src
        fok = cache[corpus.SURRPERM[ls]][1][wrap].copy()
        for seam in np.flatnonzero(np.diff(wrap) != 1) + 1:
            fok[seam:seam + LSPK] = False
        out[ls] = ({rn: rm[wrap] for rn, rm in regions.items()}, fok, fps)
    return out


def main():
    ap = argparse.ArgumentParser(description="R-DGG model x other specificity control (standalone).")
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

    # same common mask as shift_ablation: GT-reactive & every aligned model-seed column's face_ok
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    caches = {d: load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus) for d in all_dirs}
    eval_mask = mask_gtonly
    for d in all_dirs:
        eval_mask = intersect_masks(eval_mask, gridded_face_ok(caches[d], corpus))
    n_fr = sum(int(v.sum()) for v in eval_mask.values())
    print(f"  [common-mask] {len(eval_mask)} stems, {n_fr} grid frames (GT & all {len(all_dirs)} columns)", flush=True)

    columns = {}
    npz = {}
    order = []
    E_keep = {}

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
        c = rec["ci_net"]
        print(f"  [{label:26s}] n={rec['n_stems']:3d} net={rec['net']:+.3f} "
              f"CI[{c['lo']:+.2f},{c['hi']:+.2f}] p={c['p']:.3f}", flush=True)

    # GT aligned (reference) + GT x other (the original surrogate = GT row of this table)
    print("\n  [reference] GT aligned + GT x other surrogate ...", flush=True)
    E_gt, own_gt = capture_energies(collect_design(gt_cache, ctx, eval_mask), ops, ctx)
    add("GT (aligned)", "GT", "reference", "GT", E_gt, own_gt)
    E_sur, own_sur = capture_energies(collect_design(build_surrogate_cache(gt_cache, corpus), ctx, eval_mask), ops, ctx)
    add(SURROGATE, SURROGATE, "reference", "surrogate", E_sur, own_sur)

    # model x other, every dyadic seed
    print("\n  [model x other] dyadic, 3 seeds each (scrambled) ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            E, own = capture_energies(collect_design(scramble_cache(caches[d], corpus), ctx, eval_mask), ops, ctx)
            add(f"{model} x other [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], E, own)

    print("\n  [model x other] talking-head (scrambled) ...", flush=True)
    for model, d in TALKING.items():
        E, own = capture_energies(collect_design(scramble_cache(caches[d], corpus), ctx, eval_mask), ops, ctx)
        add(f"{model} x other", model, "non-dyad", "single", E, own)

    # seed-averaged dyadic model x other (energy-average 3 seeds, then identity bootstrap)
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} x other [{SEED_TAG[i]}]" for i in range(len(dirs))]
        E_list = [E_keep[l] for l in labels]
        nets = np.array([columns[l]["net"] for l in labels])
        net_ci = seed_avg_ci(E_list, corrected=True)
        seed_avg[model] = {"seeds": labels, "net": net_ci,
                           "per_seed_net": [round(float(x), 4) for x in nets],
                           "per_seed_net_sd": round(float(nets.std(ddof=1)), 4)}
        n = net_ci
        print(f"  [seed-avg {model:13s} x other] net={n['point']:+.3f} CI[{n['lo']:+.2f},{n['hi']:+.2f}] p={n['p']:.3f}", flush=True)

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG model x other (specificity control)",
        "question": "does a model's aligned gain vanish when its listener wears another "
                    "conversation's motion (wrong pairing)?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask": "ONE shared common mask = GT-reactive & every aligned model-seed column's face_ok",
        "note": ("each column's listener motion is replaced by SURRPERM[ls]'s motion FROM THAT MODEL "
                 "(wrapped, seams blinded, like the GT x other surrogate), scored vs the real GT "
                 "speaker audio. Compare each model's net here to its ALIGNED net in shift_ablation: "
                 "a dyadic drop toward 0 is the specificity evidence."),
        "order": order,
        "columns": columns,
        "seed_averaged_dyadic": seed_avg,
    }
    with open(f"{outdir}/model_x_other__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(f"{outdir}/model_x_other__{S}.npz", **npz)
    print(f"\n  [out] {outdir}/model_x_other__{S}.{{json,npz}}", flush=True)
    print("MODEL_X_OTHER_DONE", flush=True)


if __name__ == "__main__":
    main()
