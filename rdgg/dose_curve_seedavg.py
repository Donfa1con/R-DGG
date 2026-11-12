"""Dose-response curve with a SEED-AVERAGED stock baseline (plan item 11).

NEW standalone harness. Reuses dose_curve.py's verified machinery; edits nothing in rdgg/.

The shipped dose_curve.py anchors k=0 on a SINGLE stock seed -- a seed lottery (RESUME 11.1:
per-run SD ~0.14). This rebuilds both dose curves (stock->zeroother and stock->cfg0) with the
baseline SEED-AVERAGED over the 3 AVTR-1 generation seeds (AVTR-1 / _seeded / _seeded_b1):
capture each seed's per-stem energies on ONE shared mask, average them per stem (RESUME 12),
then run the dose mix against the single-seed ablation column (zeroother / cfg0 have one seed).

Shared mask per ablation = GT-reactive ∩ face_ok(3 stock seeds) ∩ face_ok(ablation), so a video
swap exchanges identical rows and the 3 stock seeds + ablation all score on the same stems.
The dose CI is the expected-over-schedules identity CI (dose_curve.dose_ci), unchanged.

Run in the shared metrics env:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/dose_curve_seedavg.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import BOOTSTRAP, SOURCES, _ident_key
from _features import gridded_face_ok, load_all_motion
from compute import load_corpus
from dose_curve import (
    build_context_and_operator, capture_column, dose_ci, intersect_masks,
    net_expected, net_from_mask, point_net_lib, stack_energies,
)

STOCK_SEEDS = ["AVTR-1", "AVTR-1_seeded", "AVTR-1_seeded_b1"]
ABLATIONS = {"zeroother": "AVTR-1_zeroother", "cfg0": "AVTR-1_cfgother0"}


def seed_avg_energies(E_list, stems):
    """Per-stem energy average over the stock seeds (RESUME 12: average energies, THEN downstream)."""
    return {s: {"both_r": float(np.mean([E[s]["both_r"] for E in E_list])),
                "mot_r": float(np.mean([E[s]["mot_r"] for E in E_list])),
                "both_n": np.mean([E[s]["both_n"] for E in E_list], axis=0),
                "mot_n": np.mean([E[s]["mot_n"] for E in E_list], axis=0)}
            for s in stems}


def build_curve(E_stock_avg, E_abl, corpus):
    """Dose curve + CI from seed-avg stock vs single-seed ablation. Returns a JSON-able dict."""
    stems = sorted(set(E_stock_avg) & set(E_abl))
    N = len(stems)
    A, K = stack_energies(E_stock_avg, E_abl, stems)
    k0 = float(net_from_mask(A, np.zeros(N, bool)))
    kN = float(net_from_mask(A, np.ones(N, bool)))

    # closed-form expected-sums curve over every k (cheap, Jensen-approx mean)
    cf_k = list(range(N + 1))
    cf_net = [float(net_expected(A, k / N)) for k in cf_k]

    # sampled mean+band on a k-grid
    rng = np.random.default_rng(0)
    kgrid = sorted(set([0, N] + list(range(0, N + 1, max(1, N // 20)))))
    sampled = []
    for k in kgrid:
        if k == 0:
            vals = np.array([net_from_mask(A, np.zeros(N, bool))])
        elif k == N:
            vals = np.array([net_from_mask(A, np.ones(N, bool))])
        else:
            vals = np.empty(200)
            for b in range(200):
                z = np.zeros(N, bool); z[rng.choice(N, k, replace=False)] = True
                vals[b] = net_from_mask(A, z)
        lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
        sampled.append({"k": int(k), "frac": k / N, "mean": float(vals.mean()),
                        "lo": float(lo), "hi": float(hi), "n_subsets": int(len(vals))})

    # expected-over-schedules identity CI at selected k
    ci = {}
    for k in sorted(set([0, N // 4, N // 2, 3 * N // 4, N])):
        d = dose_ci(E_stock_avg, E_abl, stems, k, BOOTSTRAP, 100 + k)
        ci[int(k)] = {"point": d["point"], "lo": d["lo"], "hi": d["hi"], "p50": d["p50"],
                      "n_clusters": d["n_clusters"], "n_sched": d["n_sched"]}
    return {"n_stems": N, "n_identities": len({_ident_key(s) for s in stems}),
            "k0_seedavg_stock": k0, "kN_ablation": kN,
            "closed_form_curve": {"k": cf_k, "net": cf_net},
            "sampled_curve": sampled, "identity_ci": ci}, A, stems


def main():
    ap = argparse.ArgumentParser(description="Dose curve with seed-averaged stock baseline (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    ap.add_argument("--ablation", choices=list(ABLATIONS) + ["both"], default="both")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] operator built {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    targets = list(ABLATIONS) if args.ablation == "both" else [args.ablation]
    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)

    for abl_name in targets:
        abl_dir = ABLATIONS[abl_name]
        print(f"\n  ===== ablation {abl_name} ({abl_dir}) =====", flush=True)
        # shared mask: GT-reactive ∩ face_ok(3 stock seeds) ∩ face_ok(ablation)
        em = mask_gtonly
        for d in STOCK_SEEDS + [abl_dir]:
            em = intersect_masks(em, gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus))
        n_fr = sum(int(v.sum()) for v in em.values())
        print(f"    [mask] {len(em)} stems, {n_fr} grid frames", flush=True)

        # capture 3 stock seeds + the ablation on the SAME mask
        E_stocks = []
        for d in STOCK_SEEDS:
            E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", em, ctx, ops, corpus, source)
            print(f"    [capture {d:18s}] {len(E)} stems own_r2={r2:.3f}", flush=True)
            E_stocks.append(E)
        E_abl, _per_a, r2_a = capture_column(f"{C}/{abl_dir}/{S}_{suffix}", em, ctx, ops, corpus, source)
        print(f"    [capture {abl_dir:18s}] {len(E_abl)} stems own_r2={r2_a:.3f}", flush=True)

        stems = sorted(set.intersection(*[set(E) for E in E_stocks], set(E_abl)))
        E_stock_avg = seed_avg_energies(E_stocks, stems)
        E_abl = {s: E_abl[s] for s in stems}
        print(f"    [seed-avg] {len(stems)} shared stems across 3 stock seeds + {abl_name}", flush=True)

        curve, _A, _stems = build_curve(E_stock_avg, E_abl, corpus)
        curve.update({
            "metric": "R-DGG dose curve, SEED-AVERAGED stock baseline",
            "ablation": abl_name, "ablation_dir": abl_dir,
            "stock_seeds": STOCK_SEEDS, "source": source, "suffix": suffix, "split": S,
            "mask": "GT-reactive ∩ face_ok(3 stock seeds) ∩ face_ok(ablation)",
            "note": ("k=0 is the SEED-AVERAGED stock baseline (3 AVTR-1 seeds, per-stem energy "
                     "average); k=N is the single-seed ablation. Dose mixes whole videos; CI is the "
                     "expected-over-schedules identity bootstrap (dose_ci)."),
        })
        c0 = curve["identity_ci"][0]; cN = curve["identity_ci"][len(stems)]
        print(f"    [curve] k=0 seed-avg stock {curve['k0_seedavg_stock']:+.3f} "
              f"[{c0['lo']:+.2f},{c0['hi']:+.2f}] -> k=N {abl_name} {curve['kN_ablation']:+.3f} "
              f"[{cN['lo']:+.2f},{cN['hi']:+.2f}]", flush=True)
        path = f"{outdir}/dose_curve_seedavg__{abl_name}__{S}.json"
        with open(path, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"    [out] {path}", flush=True)

    print("DOSE_CURVE_SEEDAVG_DONE", flush=True)


if __name__ == "__main__":
    main()
