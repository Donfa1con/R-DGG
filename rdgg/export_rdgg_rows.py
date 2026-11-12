"""Export the exact R-DGG rows as a per-stem keep mask, for the classic-metrics comparison
(plan item 13). The classic code (classic_metrics/*/compute.py) is left untouched except for a
small --rdgg_mask_dir that consumes this file; the loader and formulas stay the repo's.

R-DGG's rows are the `tid` returned by collect_design on the common 15-column mask: the grid
frames (25 fps, length n_grid=len(SPK_CACHE[sp])) inside reactive segments ∩ every column's
face_ok ∩ the lagged MINSEG design. We write keep_grid[n_grid] (bool, True on tid) per listener.

Axis note (validated empirically, 2026-11): face_ok is ~full on SI-184 (median 1.000) and motion
is 25 fps (models) / 30→25 (GT), so the classic loader's frame axis and this grid axis are
near-identical; the classic side aligns keep_grid by min-length, with ≤1-frame slack at the GT
resample edges (the user accepted ±1 frame as immaterial).

Run in the shared metrics env:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/export_rdgg_rows.py
"""
import argparse
import os
import time

import numpy as np
import torch

import _config
from _config import SOURCES
from _features import collect_design, gridded_face_ok, load_all_motion
from compute import load_corpus
from dose_curve import build_context_and_operator, intersect_masks
from shift_ablation import DYADIC_SEEDS, TALKING


def main():
    ap = argparse.ArgumentParser(description="Export exact R-DGG rows as per-stem keep masks.")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs the metrics env"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    # the common 15-column mask, same as shift_ablation / the leaderboard
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    eval_mask = mask_gtonly
    for d in all_dirs:
        eval_mask = intersect_masks(eval_mask, gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus))

    # R-DGG rows = tid per listener from collect_design on the common mask
    per = collect_design(gt_cache, ctx, eval_mask)
    masks = {}
    n_kept = 0
    for ls, built in per.items():
        tid = np.asarray(built[2])
        sp = corpus.pairs[ls]
        n_grid = len(corpus.SPK_CACHE[sp])
        keep = np.zeros(n_grid, dtype=bool)
        keep[tid] = True
        masks[ls] = keep
        n_kept += int(keep.sum())
    out = f"{C}/GT/_metrics/rdgg_rows_mask__{S}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **masks)
    print(f"  [out] {out}: {len(masks)} listeners, {n_kept} total R-DGG frames "
          f"({len({_config._ident_key(s) for s in masks})} identities)", flush=True)
    print("EXPORT_RDGG_ROWS_DONE", flush=True)


if __name__ == "__main__":
    main()
