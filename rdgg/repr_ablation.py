"""R-DGG on a SINGLE motion representation: EMOCA-only and LivePortrait-only (plan task 10 /
list item 9). The whole pipeline -- scaling, PCA, reference operator, projections, scoring --
is rebuilt on one representation instead of the combo (emoca (+) liveportrait) default.

NEW standalone harness. Imports the written R-DGG blocks + shift_ablation / dose_curve helpers;
edits nothing on disk. For LP-only it installs a TRANSIENT monkeypatch of _features.load_regions_full
(restored in finally) that returns the liveportrait half ALONE, mirroring combo's LP branch
(_features.py:101-116). EMOCA-only needs no patch: source='emoca' is combo's existing inner half.

Only the MOTION representation changes; the speaker audio driver (HuBERT) is unchanged, so this
isolates "does R-DGG's directed-audio result survive on each visible-motion representation?"
(answer wanted: main contrast > 0 and GT > 0 in BOTH).

Columns = the 15 shift_ablation columns (GT, GT*other, 3 dyadic x 3 seeds seed-averaged, 4
talking-head), on one shared common mask per representation. Writes
repr_ablation__{repr}__LS_AL.{json,npz}.

Run (two reprs) in the shared metrics env:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/repr_ablation.py --repr emoca
    pixi run --manifest-path rdgg/pixi.toml python rdgg/repr_ablation.py --repr lp
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import _config
import _features
from _config import SURROGATE
from _features import REGION, collect_design, gridded_face_ok, load_all_motion
from _granger import capture_energies
from compute import load_corpus
from dose_curve import build_context_and_operator, intersect_masks
from shift_ablation import (
    DYADIC_SEEDS, SEED_TAG, TALKING, build_surrogate_cache, score, seed_avg_ci, stash_energies,
)

_ORIG_LRF = _features.load_regions_full


def _lp_only_regions(npz_path, source):
    """liveportrait half ALONE (no emoca concat), returned at the LP fps so collect_design grids
    it like any source. Mirrors combo's LP branch (_features.py:101-116) exactly, minus the emoca
    concat/align. The LP run uses suffix='liveportrait', so npz_path ALREADY points at the
    _liveportrait npz (no replace needed) -- and _src_fok then reads LP face_ok directly, no patch.
    Any non-'lp' source delegates to the original loader."""
    if source != "lp":
        return _ORIG_LRF(npz_path, source)
    lp = np.load(npz_path)
    Tl = int(lp["n_frames"])
    lfok = lp["face_ok"][:Tl].astype(bool)
    Rr = lp["R"][:Tl].astype(np.float64)
    expo = lp["exp"][:Tl].astype(np.float64)
    so3 = np.zeros((Tl, 3))
    if lfok.any():
        so3[lfok] = Rotation.from_matrix(Rr[lfok]).as_rotvec()
    t_ = lp["t"][:Tl].astype(np.float64)
    sc = lp["scale"][:Tl].astype(np.float64).reshape(Tl, 1)
    lov = np.concatenate([so3, expo.reshape(Tl, -1), t_, sc], 1)
    lp_fps = float(lp["fps"]) if "fps" in lp.files else None
    return {REGION: lov}, lfok, lp_fps


def run(source, suffix, tag):
    C, S = _config.CORPUS, _config.SPLIT
    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] {tag}: operator built {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}; "
          f"driver width (audio {corpus.k_audio} + motion)", flush=True)

    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    eval_mask = mask_gtonly
    for d in all_dirs:
        eval_mask = intersect_masks(eval_mask, gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus))
    n_fr = sum(int(v.sum()) for v in eval_mask.values())
    print(f"  [common-mask] {tag}: {len(eval_mask)} stems, {n_fr} grid frames", flush=True)

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
        print(f"  [{label:22s}] n={rec['n_stems']:3d} net={rec['net']:+.3f} "
              f"CI[{c['lo']:+.2f},{c['hi']:+.2f}] p={c['p']:.3f}", flush=True)

    print(f"\n  [controls] {tag}: GT + GT*other ...", flush=True)
    E_gt, own_gt = capture_energies(collect_design(gt_cache, ctx, eval_mask), ops, ctx)
    add("GT", "GT", "GT-control", "GT", E_gt, own_gt)
    E_sur, own_sur = capture_energies(collect_design(build_surrogate_cache(gt_cache, corpus), ctx, eval_mask), ops, ctx)
    add(SURROGATE, SURROGATE, "GT-control", "surrogate", E_sur, own_sur)

    print(f"\n  [dyadic] {tag}: 3 models x 3 seeds ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            cache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
            E, own = capture_energies(collect_design(cache, ctx, eval_mask), ops, ctx)
            add(f"{model} [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], E, own)

    print(f"\n  [non-dyadic] {tag}: talking-head ...", flush=True)
    for model, d in TALKING.items():
        cache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
        E, own = capture_energies(collect_design(cache, ctx, eval_mask), ops, ctx)
        add(model, model, "non-dyad", "single", E, own)

    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        E_list = [E_keep[l] for l in labels]
        nets = np.array([columns[l]["net"] for l in labels])
        net_ci = seed_avg_ci(E_list, corrected=True)
        seed_avg[model] = {"seeds": labels, "net": net_ci,
                           "per_seed_net": [round(float(x), 4) for x in nets],
                           "per_seed_net_sd": round(float(nets.std(ddof=1)), 4)}
        n = net_ci
        print(f"  [seed-avg {model:13s}] net={n['point']:+.3f} CI[{n['lo']:+.2f},{n['hi']:+.2f}] p={n['p']:.3f}", flush=True)

    # (the main dyadic-vs-talking-head contrast for this representation is computed offline from
    #  the stashed npz by the same recipe as offline_main_test, to keep this harness single-purpose.)
    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": f"R-DGG on {tag} representation only",
        "representation": tag,
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask": "ONE shared common mask = GT-reactive & every model-seed column's face_ok",
        "note": ("whole pipeline (scaling, PCA, reference operator, projections, scoring) rebuilt on "
                 f"the {tag} motion representation; speaker audio driver unchanged. EMOCA-only = combo's "
                 "inner 'emoca' half; LP-only via a transient load_regions_full patch (liveportrait half alone)."),
        "order": order,
        "columns": columns,
        "seed_averaged_dyadic": seed_avg,
    }
    with open(f"{outdir}/repr_ablation__{tag}__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(f"{outdir}/repr_ablation__{tag}__{S}.npz", **npz)
    print(f"\n  [out] {outdir}/repr_ablation__{tag}__{S}.{{json,npz}}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="R-DGG on a single motion representation (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--repr", choices=["emoca", "lp"], required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    try:
        if args.repr == "lp":
            # suffix='liveportrait' -> every path points straight at the _liveportrait npz, so
            # _src_fok reads LP face_ok directly (no patch) and the patched loader reads it directly.
            _features.load_regions_full = _lp_only_regions
            run("lp", "liveportrait", "lp")
        else:
            run("emoca", "emoca", "emoca")   # combo's existing inner emoca half; no patch
    finally:
        _features.load_regions_full = _ORIG_LRF
    assert _features.load_regions_full is _ORIG_LRF, "load_regions_full not restored"
    print(f"REPR_ABLATION_DONE {args.repr}", flush=True)


if __name__ == "__main__":
    main()
