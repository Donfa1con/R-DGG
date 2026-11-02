"""Four AVTR-1 columns on ONE fully-common mask (incl. unseeded stock): fair seed test.

NEW code only. Imports the already-written R-DGG blocks and the helpers from
dose_curve.py / cfg0_compare.py / seeded_4way.py (all new); edits nothing in the rdgg
metric modules, adds no model to _config.MODELS, commits nothing.

The audit flagged that unseeded stock was previously scored on a subset mask (its own
face_ok was applied on top of the 4-way mask, so it sat on fewer frames than seeded).
Here ALL FOUR columns are captured on the SAME intersection
    T = GT reactive & unseeded & seeded & zeroother & cfg0 face_ok
via the paired-cell trick, so every column is on identical frames per stem. Then the
Δ(unseeded - seeded) seed test is same model / same audio / same cfg, differing ONLY in
the generation seed, on identical frames + identity-clustered paired bootstrap.

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/common4way.py
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


# earlier (non-common-mask) seeded_4way paired deltas, to flag any significance change
PRIOR = {
    "seeded_minus_cfg0": {"delta": 0.144, "lo": -0.275, "hi": 0.563, "p": 0.486, "sig": False},
    "seeded_minus_zero": {"delta": -0.094, "lo": -0.604, "hi": 0.572, "p": 0.697, "sig": False},
    "zero_minus_cfg0":   {"delta": 0.238, "lo": -0.415, "hi": 0.841, "p": 0.438, "sig": False},
}


def main():
    ap = argparse.ArgumentParser(description="4 AVTR-1 columns on one fully-common mask (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT
    dirs = {
        "unseeded_stock": f"{C}/AVTR-1/{S}_{suffix}",
        "seeded_stock": f"{C}/AVTR-1_seeded/{S}_{suffix}",
        "zeroother": f"{C}/AVTR-1_zeroother/{S}_{suffix}",
        "cfg0": f"{C}/AVTR-1_cfgother0/{S}_{suffix}",
    }

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, _gt = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] done {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    fok = {nm: gridded_face_ok(load_all_motion(d, source, corpus), corpus) for nm, d in dirs.items()}

    # paired-trick capture of each column onto T = GT & (all four face_ok): pass the OTHER
    # three columns' face_ok as eval_mask; the column's own face_ok is applied inside collect_design.
    print("  [capture] four columns on the fully-common mask ...", flush=True)
    E = {}
    r2 = {}
    for nm in dirs:
        others = [fok[o] for o in dirs if o != nm]
        eval_mask = inter(mask_gtonly, *others)
        E[nm], _, r2[nm] = capture_column(dirs[nm], eval_mask, ctx, ops, corpus, source)
        print(f"    captured {nm}: {len(E[nm])} stems, own_r2={r2[nm]:.3f}", flush=True)

    common = sorted(set.intersection(*[set(E[nm]) for nm in dirs]))
    Ec = {nm: {s: E[nm][s] for s in common} for nm in dirs}
    nrow_mismatch = sum(1 for s in common if len({Ec[nm][s]["nrows"] for nm in dirs}) != 1)
    print(f"  [mask] fully-common stems {len(common)} "
          f"({', '.join(f'{nm} {len(E[nm])}' for nm in dirs)}); nrow mismatch {nrow_mismatch}", flush=True)

    marg = {nm: {**marginal(Ec[nm], common), "ci": ci(Ec[nm], common), "own_r2": round(r2[nm], 4)}
            for nm in dirs}
    for nm in dirs:
        d = marg[nm]
        print(f"    [marg] {nm:15s} net={d['net']:+.4f} CI[{d['ci']['lo']:+.3f},{d['ci']['hi']:+.3f}] "
              f"p={d['ci']['p']}", flush=True)

    delta_defs = {
        "unseeded_minus_seeded": ("unseeded_stock", "seeded_stock"),   # THE seed test
        "seeded_minus_cfg0": ("seeded_stock", "cfg0"),
        "seeded_minus_zero": ("seeded_stock", "zeroother"),
        "zero_minus_cfg0": ("zeroother", "cfg0"),
        "unseeded_minus_cfg0": ("unseeded_stock", "cfg0"),
    }
    deltas = {}
    for nm, (a, b) in delta_defs.items():
        d = pair_stats(Ec[a], Ec[b], BOOTSTRAP)
        row = {"delta": d["delta"], "lo": d["lo"], "hi": d["hi"], "p": d["p"], "sig": d["sig"],
               "ci_width": round(d["hi"] - d["lo"], 4), "n_clusters": d["n_clusters"]}
        if nm in PRIOR:
            row["prior"] = PRIOR[nm]
            row["sig_changed"] = bool(d["sig"] != PRIOR[nm]["sig"])
        deltas[nm] = row
        extra = f" | prior sig={PRIOR[nm]['sig']} changed={row['sig_changed']}" if nm in PRIOR else ""
        print(f"    [paired] {nm}: delta={d['delta']:+.4f} CI[{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"p={d['p']} sig={d['sig']}{extra}", flush=True)

    seed_test = deltas["unseeded_minus_seeded"]
    verdict = ("SEED GAP SIGNIFICANT (CI excludes 0): the ~0.16 unseeded-vs-seeded difference is real"
               if seed_test["sig"] else
               "SEED GAP NOT SIGNIFICANT (CI includes 0): the unseeded-vs-seeded difference is within noise")
    print(f"  [verdict] {verdict}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "scoring_audio": "real GT hubert/vad (never zeroed) for ALL columns",
        "mask": "GT reactive & unseeded & seeded & zeroother & cfg0 face_ok (ALL four; identical frames per stem)",
        "columns": {
            "unseeded_stock": "AVTR-1 (real audio, cfg_other=3, UNSEEDED)",
            "seeded_stock": "AVTR-1_seeded (real audio, cfg_other=3, seed_per_video SEED_BASE=0)",
            "zeroother": "AVTR-1_zeroother (waveform zeroed, same seeds as seeded/cfg0)",
            "cfg0": "AVTR-1_cfgother0 (real audio, cfg_other=0, same seeds)",
        },
        "n_stems": len(common), "n_identities": len({_ident_key(s) for s in common}),
        "nrow_mismatch": nrow_mismatch,
        "marginals": marg,
        "paired_deltas": deltas,
        "seed_test": {"delta_name": "unseeded_minus_seeded", **seed_test, "verdict": verdict},
        "reference_marginals": {"GT": 0.695, "non_dyad_floor": "~0 (FLOAT +0.03, Ditto 0.00, SoulX-L -0.21)"},
    }
    with open(f"{out_dir}/common4way__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/common4way__{S}.json", flush=True)
    print("COMMON4WAY_DONE", flush=True)


if __name__ == "__main__":
    main()
