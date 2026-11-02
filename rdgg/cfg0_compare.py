"""cfg_other_audio=0 (CFG-level reaction ablation) R-DGG, vs stock and zeroother.

NEW code only. Imports the already-written R-DGG blocks AND the helpers from
dose_curve.py (also new); edits nothing in rdgg/, adds no model to _config.MODELS,
commits nothing.

cfg0 = AVTR-1 generated with REAL speaker audio but cfg_other_audio=0 (speaker-audio
guidance OFF), seed-matched to zeroother. It ablates the reaction pathway at the CFG
level instead of zeroing the waveform.

Outputs:
  1. cfg0 MARGINAL net on cfg0's own mask (GT reactive & cfg0 face_ok) -- the
     RESULTS.md-comparable number: g_a_m, g_shift, net, identity CI, p.
  2. stock / zeroother / cfg0 on ONE shared mask (GT reactive & stock & zero & cfg0
     face_ok): three nets + identity CIs, and paired deltas (stock-cfg0), (zero-cfg0).

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/cfg0_compare.py
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
from _bootstrap import gnet_boot_ci, pair_stats
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column, intersect_masks


def marginal(E, stems):
    """g_a_m, g_shift, net (x1e4) pooled over `stems`, same arithmetic as leaderboard_stats."""
    BR = sum(E[s]["both_r"] for s in stems)
    MR = sum(E[s]["mot_r"] for s in stems)
    BN = np.sum([E[s]["both_n"] for s in stems], 0)
    MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    g_a_m = float(np.log(MR / BR)) * 1e4
    g_shift = float(np.mean(np.log(MN / BN))) * 1e4
    return {"g_a_m": round(g_a_m, 4), "g_shift": round(g_shift, 4), "net": round(g_a_m - g_shift, 4)}


def ci(E, stems):
    d = gnet_boot_ci(E, stems, _ident_key, BOOTSTRAP, one_sided=True)
    return {"point": round(d["point"], 4), "lo": round(d["lo"], 4), "hi": round(d["hi"], 4),
            "p": round(d["p"], 4), "n_clusters": d["n_clusters"]}


def intersect3(a, b, c):
    return intersect_masks(intersect_masks(a, b), c)


def main():
    ap = argparse.ArgumentParser(description="cfg_other_audio=0 R-DGG vs stock/zeroother (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only metric"
    _config.set_paths(args.corpus, args.split)
    source, suffix = SOURCES[0]
    C = _config.CORPUS
    S = _config.SPLIT
    stock_e = f"{C}/AVTR-1/{S}_{suffix}"
    zero_e = f"{C}/AVTR-1_zeroother/{S}_{suffix}"
    cfg0_e = f"{C}/AVTR-1_cfgother0/{S}_{suffix}"

    t0 = time.time()
    corpus = load_corpus()
    ctx, ops, mask_gtonly, gt_cache = build_context_and_operator(source, suffix, corpus)
    print(f"  [phase1] done {time.time()-t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    stock_fok = gridded_face_ok(load_all_motion(stock_e, source, corpus), corpus)
    zero_fok = gridded_face_ok(load_all_motion(zero_e, source, corpus), corpus)
    cfg0_fok = gridded_face_ok(load_all_motion(cfg0_e, source, corpus), corpus)

    # ---- TASK 1: cfg0 MARGINAL on its own mask (GT reactive & cfg0 face_ok) ----
    print("  [task1] cfg0 marginal (own GT&model mask) ...", flush=True)
    E_cfg0_m, _, r2m = capture_column(cfg0_e, mask_gtonly, ctx, ops, corpus, source)
    stems_m = sorted(E_cfg0_m)
    m_cfg0 = marginal(E_cfg0_m, stems_m)
    ci_cfg0 = ci(E_cfg0_m, stems_m)
    print(f"    cfg0 marginal: n={len(stems_m)} own_r2={r2m:.3f} "
          f"g_a_m={m_cfg0['g_a_m']} g_shift={m_cfg0['g_shift']} net={m_cfg0['net']} "
          f"CI[{ci_cfg0['lo']},{ci_cfg0['hi']}] p={ci_cfg0['p']}", flush=True)

    # ---- TASK 2: stock / zero / cfg0 on ONE shared mask (GT & stock & zero & cfg0) ----
    print("  [task2] shared-mask capture of stock / zeroother / cfg0 ...", flush=True)
    em_stock = intersect3(mask_gtonly, zero_fok, cfg0_fok)   # stock's own face_ok applied inside
    em_zero = intersect3(mask_gtonly, stock_fok, cfg0_fok)
    em_cfg0 = intersect3(mask_gtonly, stock_fok, zero_fok)
    E_stock, _, r2s = capture_column(stock_e, em_stock, ctx, ops, corpus, source)
    E_zero, _, r2z = capture_column(zero_e, em_zero, ctx, ops, corpus, source)
    E_cfg0, _, r2c = capture_column(cfg0_e, em_cfg0, ctx, ops, corpus, source)
    common = sorted(set(E_stock) & set(E_zero) & set(E_cfg0))
    print(f"    shared stems: {len(common)} "
          f"(stock {len(E_stock)}, zero {len(E_zero)}, cfg0 {len(E_cfg0)})", flush=True)
    Es = {s: E_stock[s] for s in common}
    Ez = {s: E_zero[s] for s in common}
    Ec = {s: E_cfg0[s] for s in common}
    shared = {
        "n_stems": len(common),
        "n_identities": len({_ident_key(s) for s in common}),
        "stock": {**marginal(Es, common), "ci": ci(Es, common), "own_r2": round(r2s, 4)},
        "zeroother": {**marginal(Ez, common), "ci": ci(Ez, common), "own_r2": round(r2z, 4)},
        "cfg0": {**marginal(Ec, common), "ci": ci(Ec, common), "own_r2": round(r2c, 4)},
        "delta_stock_minus_cfg0": pair_stats(Es, Ec, BOOTSTRAP),
        "delta_zero_minus_cfg0": pair_stats(Ez, Ec, BOOTSTRAP),
    }
    for nm in ("stock", "zeroother", "cfg0"):
        d = shared[nm]
        print(f"    [shared] {nm:9s} net={d['net']:+.4f} "
              f"CI[{d['ci']['lo']:+.3f},{d['ci']['hi']:+.3f}] p={d['ci']['p']}", flush=True)
    for nm in ("delta_stock_minus_cfg0", "delta_zero_minus_cfg0"):
        d = shared[nm]
        print(f"    [paired] {nm}: delta={d['delta']:+.4f} CI[{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"p={d['p']} sig={d['sig']}", flush=True)

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "source": source, "suffix": suffix, "split": S,
        "scoring_audio": "real GT hubert/vad (never zeroed) for ALL columns",
        "columns": {"stock": "AVTR-1 (unseeded)",
                    "zeroother": "AVTR-1_zeroother (waveform zeroed, seed-matched to cfg0)",
                    "cfg0": "AVTR-1_cfgother0 (real audio, cfg_other_audio=0, seed-matched to zeroother)"},
        "n_folds": corpus.NFOLD,
        "task1_cfg0_marginal": {
            "mask": "GT reactive & cfg0 face_ok (own GT&model mask; RESULTS.md-comparable)",
            "n_stems": len(stems_m), "own_r2": round(r2m, 4),
            **m_cfg0, "ci": ci_cfg0,
        },
        "task2_shared_mask": {
            "mask": "GT reactive & stock face_ok & zero face_ok & cfg0 face_ok",
            **shared,
        },
        "results_md_reference_marginals": {
            "GT": 0.695, "AVTR-1": 0.565, "zeroother": 0.497,
            "FLOAT": 0.026, "Ditto": 0.003, "SoulX-Lite": -0.206,
        },
    }
    with open(f"{out_dir}/cfgother0_rdgg__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [out] wrote {out_dir}/cfgother0_rdgg__{S}.json", flush=True)
    print("CFG0_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
