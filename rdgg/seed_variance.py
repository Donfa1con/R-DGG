"""Seed-induced variance of R-DGG across three stock noise realizations (+ full 5-column table).

NEW code only. Imports the already-written R-DGG blocks and the helpers from
dose_curve.py / cfg0_compare.py / seeded_4way.py (all new); edits nothing in the rdgg
metric modules, adds no model to _config.MODELS, commits nothing.

Three stock runs of the SAME model + audio + cfg_other=3, differing ONLY in generation
noise seed:
    unseeded  = AVTR-1            (default, no per-video seed)
    seed0     = AVTR-1_seeded     (--seed_per_video SEED_BASE=0)
    seed1     = AVTR-1_seeded_b1  (--seed_per_video SEED_BASE=1, distinct seeds)

Part 1: the three stock runs on ONE common mask (GT reactive & the three stock face_ok),
identical frames -> net + identity CI each, their spread (mean/SD/range), and the three
identity-clustered paired deltas (expected all n.s.).
Part 2 (combined): all FIVE columns (3 stock seeds + zeroother + cfg0) on one common mask
(GT & all five face_ok) -> net + CI each, full picture on identical frames.

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/seed_variance.py
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
from _bootstrap import pair_stats
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column
from cfg0_compare import marginal, ci
from seeded_4way import inter


def capture_set(names, dirs, fok, mask_gtonly, ctx, ops, corpus, source):
    """Capture each column in `names` on the common mask = GT & (all `names` face_ok),
    via the paired-cell trick (pass the OTHER columns' face_ok; own applied inside)."""
    E, r2 = {}, {}
    for nm in names:
        others = [fok[o] for o in names if o != nm]
        E[nm], _, r2[nm] = capture_column(dirs[nm], inter(mask_gtonly, *others), ctx, ops, corpus, source)
        print(f"    captured {nm}: {len(E[nm])} stems own_r2={r2[nm]:.3f}", flush=True)
    common = sorted(set.intersection(*[set(E[nm]) for nm in names]))
    Ec = {nm: {s: E[nm][s] for s in common} for nm in names}
    nmm = sum(1 for s in common if len({Ec[nm][s]["nrows"] for nm in names}) != 1)
    return Ec, common, r2, nmm


def main():
    ap = argparse.ArgumentParser(description="Seed-induced variance of R-DGG (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT
    dirs = {
        "unseeded": f"{C}/AVTR-1/{S}_{suffix}",
        "seed0": f"{C}/AVTR-1_seeded/{S}_{suffix}",
        "seed1": f"{C}/AVTR-1_seeded_b1/{S}_{suffix}",
        "zeroother": f"{C}/AVTR-1_zeroother/{S}_{suffix}",
        "cfg0": f"{C}/AVTR-1_cfgother0/{S}_{suffix}",
    }

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, _gt = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] done {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)
    fok = {nm: gridded_face_ok(load_all_motion(d, source, corpus), corpus) for nm, d in dirs.items()}

    # ---- Part 1: three stock noise realizations on their common mask ----
    print("  [part1] three stock seeds on GT & unseeded & seed0 & seed1 ...", flush=True)
    stock = ["unseeded", "seed0", "seed1"]
    Ec, common3, r2_3, nmm3 = capture_set(stock, dirs, fok, mask_gtonly, ctx, ops, corpus, source)
    print(f"    common stems {len(common3)}; nrow mismatch {nmm3}", flush=True)
    marg3 = {nm: {**marginal(Ec[nm], common3), "ci": ci(Ec[nm], common3), "own_r2": round(r2_3[nm], 4)}
             for nm in stock}
    nets = np.array([marg3[nm]["net"] for nm in stock], dtype=float)
    spread = {"nets": {nm: marg3[nm]["net"] for nm in stock},
              "mean": round(float(nets.mean()), 4),
              "sd_sample": round(float(nets.std(ddof=1)), 4),   # n=3 sample SD
              "sd_pop": round(float(nets.std(ddof=0)), 4),
              "min": round(float(nets.min()), 4), "max": round(float(nets.max()), 4),
              "range": round(float(nets.max() - nets.min()), 4)}
    deltas3 = {}
    for a, b in [("unseeded", "seed0"), ("unseeded", "seed1"), ("seed0", "seed1")]:
        d = pair_stats(Ec[a], Ec[b], BOOTSTRAP)
        deltas3[f"{a}_minus_{b}"] = {"delta": d["delta"], "lo": d["lo"], "hi": d["hi"],
                                     "p": d["p"], "sig": d["sig"]}
    for nm in stock:
        print(f"    [net] {nm:9s} {marg3[nm]['net']:+.4f} CI[{marg3[nm]['ci']['lo']:+.3f},"
              f"{marg3[nm]['ci']['hi']:+.3f}] p={marg3[nm]['ci']['p']}", flush=True)
    print(f"    [spread] mean={spread['mean']} SD(n=3)={spread['sd_sample']} range={spread['range']} "
          f"[{spread['min']},{spread['max']}]", flush=True)
    for nm, d in deltas3.items():
        print(f"    [paired] {nm}: {d['delta']:+.4f} CI[{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"p={d['p']} sig={d['sig']}", flush=True)

    # ---- Part 2: all five columns on one common mask ----
    print("  [part2] five columns on GT & all-five face_ok ...", flush=True)
    five = ["unseeded", "seed0", "seed1", "zeroother", "cfg0"]
    Ec5, common5, r2_5, nmm5 = capture_set(five, dirs, fok, mask_gtonly, ctx, ops, corpus, source)
    print(f"    common stems {len(common5)}; nrow mismatch {nmm5}", flush=True)
    marg5 = {nm: {**marginal(Ec5[nm], common5), "ci": ci(Ec5[nm], common5), "own_r2": round(r2_5[nm], 4)}
             for nm in five}
    for nm in five:
        d = marg5[nm]
        print(f"    [5col] {nm:9s} net={d['net']:+.4f} CI[{d['ci']['lo']:+.3f},{d['ci']['hi']:+.3f}] "
              f"p={d['ci']['p']}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "scoring_audio": "real GT hubert/vad (never zeroed) for ALL columns",
        "columns": {
            "unseeded": "AVTR-1 (cfg_other=3, no per-video seed)",
            "seed0": "AVTR-1_seeded (cfg_other=3, SEED_BASE=0)",
            "seed1": "AVTR-1_seeded_b1 (cfg_other=3, SEED_BASE=1, distinct seeds)",
            "zeroother": "AVTR-1_zeroother (waveform zeroed)",
            "cfg0": "AVTR-1_cfgother0 (cfg_other=0)",
        },
        "part1_three_stock_seeds": {
            "mask": "GT reactive & unseeded & seed0 & seed1 face_ok (identical frames)",
            "n_stems": len(common3), "n_identities": len({_ident_key(s) for s in common3}),
            "nrow_mismatch": nmm3,
            "marginals": marg3, "seed_spread": spread, "paired_deltas": deltas3,
        },
        "part2_five_columns": {
            "mask": "GT reactive & all five face_ok (identical frames)",
            "n_stems": len(common5), "n_identities": len({_ident_key(s) for s in common5}),
            "nrow_mismatch": nmm5, "marginals": marg5,
        },
        "reference_marginals": {"GT": 0.695, "non_dyad_floor": "~0 (FLOAT +0.03, Ditto 0.00, SoulX-L -0.21)"},
    }
    with open(f"{out_dir}/seed_variance__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/seed_variance__{S}.json", flush=True)
    print("SEED_VARIANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
