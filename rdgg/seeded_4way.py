"""Seed-matched 4-way R-DGG comparison: seeded-stock / zeroother / cfg0 (+ unseeded cross-check).

NEW code only. Imports the already-written R-DGG blocks and the helpers from
dose_curve.py and cfg0_compare.py (also new); edits nothing in rdgg/, adds no model
to _config.MODELS, commits nothing.

AVTR-1_seeded = true stock (real audio, cfg_other=3) generated with --seed_per_video
SEED_BASE=0, so its per-video seeds are IDENTICAL to zeroother and cfg0. All three
ablations now share the same noise realization, so the per-stem paired difference
removes the noise-realization variance.

Outputs (all scored vs real GT audio):
  1. On ONE shared mask (GT reactive & seeded & zero & cfg0 face_ok): marginal net +
     identity CI for seeded-stock, zeroother, cfg0; plus original UNSEEDED stock on the
     same mask as a cross-check (net should ~= seeded-stock).
  2. Seed-matched paired deltas (identity-clustered, paired bootstrap on identical
     frames + identical noise): seeded-cfg0, seeded-zero, zero-cfg0.
  3. Whether seed-matching tightened the CIs vs the earlier unseeded comparison and
     whether any paired drop now reaches significance.

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/seeded_4way.py
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
from dose_curve import build_context_and_operator, capture_column, intersect_masks
from cfg0_compare import marginal, ci


def inter(*masks):
    out = masks[0]
    for m in masks[1:]:
        out = intersect_masks(out, m)
    return out


# earlier unseeded-stock paired deltas (from cfg0_compare), for the tightening check
PRIOR = {
    "seeded_minus_cfg0": {"delta": 0.307, "lo": -0.114, "hi": 0.794, "p": 0.159},   # was stock(unseeded)-cfg0
    "zero_minus_cfg0":   {"delta": 0.241, "lo": -0.412, "hi": 0.843, "p": 0.433},
}


def main():
    ap = argparse.ArgumentParser(description="Seed-matched 4-way R-DGG comparison (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT
    seeded_e = f"{C}/AVTR-1_seeded/{S}_{suffix}"
    zero_e = f"{C}/AVTR-1_zeroother/{S}_{suffix}"
    cfg0_e = f"{C}/AVTR-1_cfgother0/{S}_{suffix}"
    unseed_e = f"{C}/AVTR-1/{S}_{suffix}"

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, _gt = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] done {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    seeded_fok = gridded_face_ok(load_all_motion(seeded_e, source, corpus), corpus)
    zero_fok = gridded_face_ok(load_all_motion(zero_e, source, corpus), corpus)
    cfg0_fok = gridded_face_ok(load_all_motion(cfg0_e, source, corpus), corpus)
    unseed_fok = gridded_face_ok(load_all_motion(unseed_e, source, corpus), corpus)

    # paired-trick captures onto base4 = GT reactive & seeded & zero & cfg0 (own face_ok applied inside)
    print("  [capture] seeded / zero / cfg0 on shared 4-way mask ...", flush=True)
    E_se, _, r2_se = capture_column(seeded_e, inter(mask_gtonly, zero_fok, cfg0_fok), ctx, ops, corpus, source)
    E_ze, _, r2_z = capture_column(zero_e, inter(mask_gtonly, seeded_fok, cfg0_fok), ctx, ops, corpus, source)
    E_cf, _, r2_c = capture_column(cfg0_e, inter(mask_gtonly, seeded_fok, zero_fok), ctx, ops, corpus, source)
    # unseeded stock on the same 4-way eval mask (its own face_ok applied inside) -> cross-check
    E_un, _, r2_u = capture_column(unseed_e, inter(mask_gtonly, seeded_fok, zero_fok, cfg0_fok),
                                   ctx, ops, corpus, source)

    common = sorted(set(E_se) & set(E_ze) & set(E_cf))
    Ese = {s: E_se[s] for s in common}
    Eze = {s: E_ze[s] for s in common}
    Ecf = {s: E_cf[s] for s in common}
    nrow_mismatch = sum(1 for s in common
                        if not (Ese[s]["nrows"] == Eze[s]["nrows"] == Ecf[s]["nrows"]))
    print(f"  [mask] shared stems {len(common)} (seeded {len(E_se)}, zero {len(E_ze)}, cfg0 {len(E_cf)}); "
          f"nrow mismatch {nrow_mismatch}", flush=True)

    marg = {
        "seeded_stock": {**marginal(Ese, common), "ci": ci(Ese, common), "own_r2": round(r2_se, 4)},
        "zeroother": {**marginal(Eze, common), "ci": ci(Eze, common), "own_r2": round(r2_z, 4)},
        "cfg0": {**marginal(Ecf, common), "ci": ci(Ecf, common), "own_r2": round(r2_c, 4)},
    }
    # unseeded cross-check on common stems that survived its own face_ok too
    un_stems = sorted(set(E_un) & set(common))
    unseeded_cc = {**marginal({s: E_un[s] for s in un_stems}, un_stems),
                   "ci": ci({s: E_un[s] for s in un_stems}, un_stems),
                   "own_r2": round(r2_u, 4), "n_stems": len(un_stems)}
    for nm, d in marg.items():
        print(f"    [shared] {nm:13s} net={d['net']:+.4f} CI[{d['ci']['lo']:+.3f},{d['ci']['hi']:+.3f}] "
              f"p={d['ci']['p']}", flush=True)
    print(f"    [xcheck] unseeded_stock net={unseeded_cc['net']:+.4f} "
          f"CI[{unseeded_cc['ci']['lo']:+.3f},{unseeded_cc['ci']['hi']:+.3f}] "
          f"(n={unseeded_cc['n_stems']}) vs seeded {marg['seeded_stock']['net']:+.4f} "
          f"-> diff {unseeded_cc['net'] - marg['seeded_stock']['net']:+.4f}", flush=True)

    # seed-matched paired deltas (identical frames + identical noise realization)
    pairs = {
        "seeded_minus_cfg0": pair_stats(Ese, Ecf, BOOTSTRAP),
        "seeded_minus_zero": pair_stats(Ese, Eze, BOOTSTRAP),
        "zero_minus_cfg0": pair_stats(Eze, Ecf, BOOTSTRAP),
    }
    tighten = {}
    for nm, d in pairs.items():
        width = d["hi"] - d["lo"]
        line = {"delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "p": d["p"], "sig": d["sig"],
                "ci_width": round(width, 4)}
        if nm in PRIOR:
            pw = PRIOR[nm]["hi"] - PRIOR[nm]["lo"]
            line["prior_ci_width"] = round(pw, 4)
            line["tightened"] = bool(width < pw)
            line["prior_delta"] = PRIOR[nm]["delta"]
            line["prior_p"] = PRIOR[nm]["p"]
        tighten[nm] = line
        extra = (f" | prior width {line.get('prior_ci_width')} -> tightened={line.get('tightened')}"
                 if nm in PRIOR else "")
        print(f"    [paired] {nm}: delta={d['delta']:+.4f} CI[{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"p={d['p']} sig={d['sig']} width={width:.3f}{extra}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "scoring_audio": "real GT hubert/vad (never zeroed) for ALL columns",
        "columns": {
            "seeded_stock": "AVTR-1_seeded (real audio, cfg_other=3, seed_per_video SEED_BASE=0)",
            "zeroother": "AVTR-1_zeroother (waveform zeroed, same seeds)",
            "cfg0": "AVTR-1_cfgother0 (real audio, cfg_other=0, same seeds)",
            "unseeded_stock": "AVTR-1 (real audio, cfg_other=3, unseeded) -- cross-check only",
        },
        "note": "seeded_stock, zeroother, cfg0 share IDENTICAL per-video seeds (noise realization matched)",
        "mask": "GT reactive & seeded & zero & cfg0 face_ok (paired-cell trick; identical frames per stem)",
        "n_stems": len(common), "n_identities": len({_ident_key(s) for s in common}),
        "nrow_mismatch": nrow_mismatch,
        "marginals": marg,
        "unseeded_cross_check": unseeded_cc,
        "paired_deltas": tighten,
        "prior_unseeded_deltas": PRIOR,
        "reference_marginals": {"GT": 0.695, "non_dyad_floor": "~0 (FLOAT +0.03, Ditto 0.00, SoulX-L -0.21)"},
    }
    with open(f"{out_dir}/seeded_4way__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/seeded_4way__{S}.json", flush=True)
    print("SEEDED_4WAY_DONE", flush=True)


if __name__ == "__main__":
    main()
