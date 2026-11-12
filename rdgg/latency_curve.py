"""R-DGG latency curve: R-DGG net as a function of the speaker-driver offset, +-1 s.

NEW standalone harness. It imports the already-written R-DGG blocks and the
shift_ablation / dose_curve helpers; it edits nothing in rdgg/, adds no model to
_config.MODELS, and commits nothing. See CVPR_zeroother_design.md section 15.

This REPLACES the narrow figures/latency_sweep.py (which swept only +-5 slots = +-200 ms
on the leaderboard base-seed columns). Here:
  * range is OFFSETS = [-25..+25] slots x 40 ms = +-1 s (user decision);
  * all 15 columns are swept (GT, GT*other surrogate, 3 dyadic x 3 seeds, 4 talking-head),
    on ONE shared common mask, exactly like shift_ablation.py;
  * seed-averaged dyadic curves are built by energy-averaging the 3 seeds at EACH offset
    (RESUME invariant: average per-stem energies over the 3 seeds, THEN the identity
    bootstrap), so the user gets the seed-avg curve without any manual post-step;
  * per-stem aligned energies at every offset AND the fixed floor nulls are stashed to npz,
    so any re-aggregation (peak CI, other seed subsets) runs offline.

The mechanism reuses capture_energies' exact internal fit (pool_design -> DriverPredictor ->
own_block_standardised -> ControlFit), then sweeps the offset through pred.predict. The
driver offset is applied as a circular wrap modulo the speaker length in gather_driver, so
the design-row set is IDENTICAL at every offset (the curve tails are directly comparable)
and large +-25 shifts are as safe as the time-shift-floor nulls (which use the same path).

net(delta) uses the SAME fixed floor as the reported number (the delta=0 time-shift nulls),
so net(0) reproduces the committed shift_ablation value per column -- a built-in verify gate.

Run in the shared metrics env:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/latency_curve.py
A 22 GB GPU needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import DEV, DT, SOURCES, SURROGATE
from _features import collect_design, gridded_face_ok, load_all_motion
from _granger import ControlFit, DriverPredictor, own_block_standardised, pool_design
from compute import load_corpus
from dose_curve import build_context_and_operator, intersect_masks
from shift_ablation import (
    DYADIC_SEEDS, SEED_TAG, TALKING, boot_ci, build_surrogate_cache, decompose, seed_avg_ci,
)

OFFSETS = list(range(-25, 26))     # grid slots; 40 ms each @ 25 fps -> +-1 s (user decision)
MS_PER_SLOT = 40


def fit_column(per, ops, ctx):
    """Reproduce capture_energies' internal fit (consumes `per`), returning the handles the
    offset sweep reuses: the ControlFit, the DriverPredictor, and the listener order."""
    order, XO, Y, tids, depths = pool_design(per, consume=True)
    pred = DriverPredictor(ops, ctx, order, tids, depths)
    Xz_o = own_block_standardised(ctx, order, tids, XO, pred.lag_is_in_segment)
    del XO
    fit = ControlFit(Xz_o, Y, order, pred.rows_per_listener, target_scale_inv=ctx.target_scale_inv)
    return fit, pred, order


def sweep_column(cache, ctx, ops, eval_mask):
    """Per-stem aligned energies at every offset in OFFSETS, plus the fixed floor nulls
    (the delta=0 time-shift nulls). Returns (E_by_delta, stems):
        E_by_delta[d] = {stem: {both_r, mot_r, both_n[K], mot_n[K]}}  -- same format as
    capture_energies / shift_ablation, so decompose / boot_ci / seed_avg_ci apply verbatim.
    The floor nulls (both_n, mot_n) are identical across d (offset-independent by construction),
    shared into every delta's record so net(d) = g_a_m(d) - g_shift uses the one reported floor."""
    per = collect_design(cache, ctx, eval_mask)
    if not per or len(per) < 5:
        return None, None
    fit, pred, order = fit_column(per, ops, ctx)
    n_ls = len(order)
    rl = fit.row_listener

    def per_listener(row_energy):
        return torch.zeros(n_ls, device=DEV, dtype=DT).index_add_(0, rl, row_energy).detach().cpu().numpy()

    def aligned(off):
        """per-stem (e_both, e_mot) at speaker-driver offset `off` (None = aligned, delta=0)."""
        shat = pred.predict(ctx.corpus.pairs, off)
        e_both = fit.row_energy(fit.residual_with([shat["audio"], shat["motion"]]))
        e_mot = fit.row_energy(fit.residual_with([shat["motion"]]))
        return per_listener(e_both), per_listener(e_mot)

    # fixed floor nulls (delta=0 time-shift nulls) -- offset-independent, computed once
    ts = ctx.corpus.timeshifts
    K = len(ts)
    both_n = np.empty((K, n_ls)); mot_n = np.empty((K, n_ls))
    for k in range(K):
        both_n[k], mot_n[k] = aligned(ts[k])

    E_by_delta = {}
    for d in OFFSETS:
        off = None if d == 0 else {ls: d for ls in order}     # circular wrap in gather_driver
        b, mm = aligned(off)
        E_by_delta[d] = {
            order[i]: {"both_r": float(b[i]), "mot_r": float(mm[i]),
                       "both_n": both_n[:, i].astype(np.float64),
                       "mot_n": mot_n[:, i].astype(np.float64)}
            for i in range(n_ls)
        }
    return E_by_delta, sorted(order)


def curve_from(E_by_delta):
    """[[offset_ms, net_x1e4], ...] over OFFSETS, pooled over all stems (decompose = g_a_m - g_shift)."""
    out = []
    for d in OFFSETS:
        stems = sorted(E_by_delta[d])
        dec, _raw_i, _net_i = decompose(E_by_delta[d], stems)
        out.append([d * MS_PER_SLOT, dec["net"]])
    return out


def seed_avg_point(E_list):
    """Seed-avg net (x1e4) at one offset: energy-average the seeds per stem, then pooled net.
    No bootstrap (point only) -- cheap, for the whole curve."""
    stems = sorted(set.intersection(*[set(E) for E in E_list]))
    E_avg = {s: {"both_r": float(np.mean([E[s]["both_r"] for E in E_list])),
                 "mot_r": float(np.mean([E[s]["mot_r"] for E in E_list])),
                 "both_n": np.mean([E[s]["both_n"] for E in E_list], axis=0),
                 "mot_n": np.mean([E[s]["mot_n"] for E in E_list], axis=0)}
             for s in stems}
    dec, _raw_i, _net_i = decompose(E_avg, stems)
    return dec["net"]


def seed_avg_curve(E_by_delta_list):
    """Seed-averaged dyadic curve: at EACH offset, energy-average the seeds then the pooled net
    (point only, cheap). The full identity bootstrap runs ONCE, at the peak offset. Returns
    ([[ms, net], ...], peak_ci)."""
    curve = []
    best_d, best_val = None, -np.inf
    for d in OFFSETS:
        E_list = [E_by_delta[d] for E_by_delta in E_by_delta_list]
        pt = seed_avg_point(E_list)
        curve.append([d * MS_PER_SLOT, pt])
        if pt > best_val:
            best_d, best_val = d, pt
    ci = seed_avg_ci([E_by_delta[best_d] for E_by_delta in E_by_delta_list], corrected=True)
    peak_ci = {"offset_ms": best_d * MS_PER_SLOT, **ci}
    return curve, peak_ci


def main():
    ap = argparse.ArgumentParser(description="R-DGG latency curve, +-1 s (standalone).")
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
    print(f"  [phase1] operator built {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}; "
          f"OFFSETS {OFFSETS[0]}..{OFFSETS[-1]} slots = {OFFSETS[0]*MS_PER_SLOT}..{OFFSETS[-1]*MS_PER_SLOT} ms", flush=True)

    # ONE shared common mask = GT-reactive & every model-seed column's face_ok (surrogate excluded
    # from the intersection, as in shift_ablation / the leaderboard).
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    eval_mask = mask_gtonly
    for d in all_dirs:
        fok = gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus)
        eval_mask = intersect_masks(eval_mask, fok)
    n_fr = sum(int(v.sum()) for v in eval_mask.values())
    print(f"  [common-mask] {len(eval_mask)} stems, {n_fr} grid frames valid in GT & all "
          f"{len(all_dirs)} model-seed columns", flush=True)

    curves = {}         # label -> [[ms, net], ...]
    peaks = {}          # label -> {offset_ms, net@peak}
    npz = {}            # per-stem aligned energies at every offset + floor nulls, for offline re-agg
    E_dyadic = {}       # label -> E_by_delta, kept for the seed-avg curves

    def sweep_and_record(label, cache, group, model=None):
        tc = time.time()
        E_by_delta, stems = sweep_column(cache, ctx, ops, eval_mask)
        if E_by_delta is None:
            print(f"  [{label:22s}] SKIPPED (empty design)", flush=True)
            curves[label] = None
            return
        cur = curve_from(E_by_delta)
        curves[label] = cur
        pk = max(cur, key=lambda c: c[1])
        v0 = dict((int(p[0]), p[1]) for p in cur)[0]
        peaks[label] = {"offset_ms": int(pk[0]), "net_at_peak": pk[1], "net_at_0": v0}
        # stash energies: aligned both/mot per stem at each offset, plus the shared floor nulls once
        npz[f"{label}::stems"] = np.array(stems)
        npz[f"{label}::offsets_ms"] = np.array([d * MS_PER_SLOT for d in OFFSETS])
        npz[f"{label}::both_r"] = np.array([[E_by_delta[d][s]["both_r"] for s in stems] for d in OFFSETS])
        npz[f"{label}::mot_r"] = np.array([[E_by_delta[d][s]["mot_r"] for s in stems] for d in OFFSETS])
        npz[f"{label}::both_n"] = np.stack([E_by_delta[OFFSETS[0]][s]["both_n"] for s in stems], axis=1)
        npz[f"{label}::mot_n"] = np.stack([E_by_delta[OFFSETS[0]][s]["mot_n"] for s in stems], axis=1)
        if group == "dyadic":
            E_dyadic[label] = E_by_delta
        print(f"  [{label:22s}] n={len(stems):3d} net@0={v0:+.3f} peak={pk[1]:+.3f}@{int(pk[0]):+d}ms "
              f"({time.time()-tc:.1f}s)", flush=True)

    # --- GT positive control + GT*other surrogate negative control ---
    print("\n  [controls] GT + GT*other surrogate ...", flush=True)
    sweep_and_record("GT", gt_cache, "GT-control")
    sweep_and_record(SURROGATE, build_surrogate_cache(gt_cache, corpus), "GT-control")

    # --- dyadic models, every seed ---
    print("\n  [dyadic] AVTR-1 / DyStream / AvatarForcing, 3 seeds each ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            cache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
            sweep_and_record(f"{model} [{SEED_TAG[i]}]", cache, "dyadic", model)

    # --- talking-head models, one seed each ---
    print("\n  [non-dyadic] talking-head models ...", flush=True)
    for model, d in TALKING.items():
        cache = load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus)
        sweep_and_record(model, cache, "non-dyad", model)

    # --- seed-averaged dyadic curves (energy-average the 3 seeds at EACH offset) ---
    print("\n  [seed-avg] dyadic curves (energy-average over 3 seeds per offset) ...", flush=True)
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        if not all(l in E_dyadic for l in labels):
            print(f"  [seed-avg {model}] SKIPPED (a seed column missing)", flush=True)
            continue
        cur, peak_ci = seed_avg_curve([E_dyadic[l] for l in labels])
        curves[f"{model} [seed-avg]"] = cur
        seed_avg[model] = {"seeds": labels, "curve_label": f"{model} [seed-avg]",
                           "peak": peak_ci,
                           "net_at_0": dict((int(p[0]), p[1]) for p in cur)[0]}
        pc = peak_ci
        print(f"  [seed-avg {model:13s}] peak net={pc['point']:+.3f} CI[{pc['lo']:+.2f},{pc['hi']:+.2f}] "
              f"@{pc['offset_ms']:+d}ms p={pc['p']:.3f}", flush=True)

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG latency curve",
        "question": "where (in ms) does the speaker-driver offset peak the R-DGG net?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "offsets_ms": [d * MS_PER_SLOT for d in OFFSETS],
        "range_ms": [OFFSETS[0] * MS_PER_SLOT, OFFSETS[-1] * MS_PER_SLOT],
        "step_ms": MS_PER_SLOT,
        "mask": "ONE shared common mask = GT-reactive & every model-seed column's face_ok (surrogate excluded)",
        "note": ("net(delta) = g_a_m(delta) - g_shift, with the fixed delta=0 time-shift floor; "
                 "net(0) reproduces the committed shift_ablation net per column (verify gate). "
                 "Offset is a circular wrap modulo speaker length (gather_driver), so the row set "
                 "is identical at every offset and large +-25 shifts match the floor nulls' path."),
        "curves": curves,       # includes per-seed, GT, surrogate, talking-head, and [seed-avg] dyadic
        "peaks": peaks,
        "seed_averaged_dyadic": seed_avg,
    }
    with open(f"{outdir}/latency_curve__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(f"{outdir}/latency_curve__{S}.npz", **npz)
    # also refresh the figures/ json consumed by plot_latency.py (per-system [[ms, net]])
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(figdir, "latency_curves.json"), "w") as f:
        json.dump({k: v for k, v in curves.items() if v is not None}, f, indent=1)
    print(f"\n  [out] {outdir}/latency_curve__{S}.{{json,npz}}", flush=True)
    print(f"        {figdir}/latency_curves.json (for plot_latency.py)", flush=True)
    print("LATENCY_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
