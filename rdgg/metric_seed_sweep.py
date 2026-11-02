"""Metric-INTERNAL seed sensitivity of R-DGG: does the metric's own randomness move the
fitted object (PCA basis, B operator, timeshift floor, bootstrap) -- and the final net?

NEW file only. Imports the existing R-DGG blocks and writes small SEEDED variants where a
seed is currently hardcoded (fit_pca row seed, timeshift base, bootstrap seed); it edits NO
existing rdgg module, adds no model to _config.MODELS, and commits nothing.

HEADLINE = source-level stability (low-noise): for each knob, how much does the LEARNED
object change vs the default? The final net (wide n=13 CI) is reported but secondary.

Fixed generated columns (no regeneration), one shared mask (GT reactive & their face_ok):
    GT (anchor), AVTR-1_seeded (base0 stock), AVTR-1_zeroother (control).

Knobs, each ~5 values incl. default, others held at default:
  1  PCA stem-subsample seed   (pca_fit_sample: audio 101.., motion 103.., paired)
  2  PCA row-subsample seed    (fit_pca's hardcoded rng(0) -> rng(row_seed), audio+motion)
  3  B operator                (diagnostic across the PCA-seed variants above)
  4  timeshift-floor base      (rng(2000+k) -> rng(base+k))
  5  identity-bootstrap seed   (rng(0) -> rng(seed); point fixed, CI only)

Run:  pixi run --manifest-path rdgg/pixi.toml python rdgg/metric_seed_sweep.py
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

import _config
from _config import (
    BOOTSTRAP, DEV, DT, K_A_TARGET, KMV, LAG_NP, MINSHIFT, N_TIMESHIFT, SOURCES,
    SourceContext, _ident_key, parallel_map,
)
from _features import (
    collect_design, gridded_face_ok, gtonly_eval_mask, load_all_motion,
    pca_fit_sample, speaker_motion_raw, speaker_raw,
)
from _granger import capture_energies, fit_bgt_fold
from _bootstrap import cluster_sums, gnet
from compute import load_corpus
from seeded_4way import inter


# ---- seeded variant of fit_pca (_features.py:26-35): only the row-subsample rng changes ----
def fit_pca_seeded(mats, k, row_seed):
    X = np.concatenate(mats, 0)
    X = X[np.isfinite(X).all(1)]
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-8)
    Xc = (X - mu) / sd
    idx = np.random.default_rng(row_seed).choice(len(Xc), size=min(40000, len(Xc)), replace=False)
    _, sv, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    kk = min(k, Vt.shape[0])
    return mu, sd, Vt[:kk].T


def build_timeshifts(listeners, pairs, SPK_G_AUD, base):
    ts = []
    for k in range(N_TIMESHIFT):
        rng = np.random.default_rng(base + k)
        off = {}
        for ls in listeners:
            sp = pairs[ls]
            n = SPK_G_AUD[sp].shape[0] if sp in SPK_G_AUD else 0
            off[ls] = int(rng.integers(MINSHIFT, n - MINSHIFT)) if n > 2 * MINSHIFT + 20 else max(25, MINSHIFT)
        ts.append(off)
    return ts


# ---- seeded identity-bootstrap CI (mirrors _bootstrap.gnet_boot_ci; only rng seed varies) ----
def boot_ci_seeded(E, stems, key_of, B, seed):
    clusters = list(dict.fromkeys(key_of(s) for s in stems))
    idx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    sums = cluster_sums(E, stems, {s: idx[key_of(s)] for s in stems}, NC)
    point = float(gnet(np.arange(NC), sums)) * 1e4
    rng = np.random.default_rng(seed)
    d = np.array([gnet(rng.integers(0, NC, size=NC), sums) for _ in range(B)]) * 1e4
    lo, p50, hi = np.percentile(d, [2.5, 50, 97.5])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}


def marginal(E, stems):
    BR = sum(E[s]["both_r"] for s in stems); MR = sum(E[s]["mot_r"] for s in stems)
    BN = np.sum([E[s]["both_n"] for s in stems], 0); MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    g_a_m = float(np.log(MR / BR)) * 1e4
    g_shift = float(np.mean(np.log(MN / BN))) * 1e4
    return {"g_a_m": round(g_a_m, 4), "g_shift": round(g_shift, 4), "net": round(g_a_m - g_shift, 4)}


def floor_stats(E, stems):
    BN = np.sum([E[s]["both_n"] for s in stems], 0); MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    fk = np.log(MN / BN) * 1e4
    return {"floor": round(float(fk.mean()), 4),
            "floor_se_within_draw": round(float(fk.std(ddof=1) / np.sqrt(len(fk))), 4)}


def pca_diag(seed_pca, def_pca, Xfixed):
    mus, sds, Vs = seed_pca
    mud, sdd, Vd = def_pca
    sv = np.clip(np.linalg.svd(Vd.T @ Vs, compute_uv=False), -1.0, 1.0)   # cos(principal angles)
    ang = np.degrees(np.arccos(sv))
    ps = ((Xfixed - mus) / sds) @ Vs
    pd = ((Xfixed - mud) / sdd) @ Vd
    return {"max_principal_angle_deg": round(float(ang.max()), 4),
            "mean_cos2_overlap": round(float((sv ** 2).mean()), 6),
            "rel_mu": round(float(np.linalg.norm(mus - mud) / (np.linalg.norm(mud) + 1e-12)), 6),
            "rel_sd": round(float(np.linalg.norm(sds - sdd) / (np.linalg.norm(sdd) + 1e-12)), 6),
            "rel_proj_fixed_features": round(float(np.linalg.norm(ps - pd) / (np.linalg.norm(pd) + 1e-12)), 6)}


def bd_frobenius(ops_s, ops_d):
    out = {}
    for ch in ("audio", "motion"):
        num = den = 0.0
        for f in range(len(ops_d)):
            A = ops_s[f][2][ch].double(); Bm = ops_d[f][2][ch].double()
            num += float(((A - Bm) ** 2).sum()); den += float((Bm ** 2).sum())
        out[ch] = round((num / den) ** 0.5, 6) if den > 0 else None
    return out


# globals set in main (caches shared across all settings)
CORPUS0 = RAW_AUD = GT_CACHE = MRAW = MASK = PG = TSI = None
CACHES = FOKS = None
SRC = SUF = None
XAUD = XMOT = None   # fixed held-out feature matrices for the functional PCA check


def audio_pca(audio_seed, row_seed):
    sample = pca_fit_sample(CORPUS0.speakers, audio_seed)
    return fit_pca_seeded([RAW_AUD[sp] for sp in sample if sp in RAW_AUD], K_A_TARGET, row_seed)


def motion_pca(motion_seed, row_seed):
    sample = pca_fit_sample(list(MRAW), motion_seed)
    return fit_pca_seeded([MRAW[sp] for sp in sample], KMV, row_seed)


def make_corpus(audio_seed, row_seed, ts_base):
    amu, asd, aP = audio_pca(audio_seed, row_seed)
    SPK_CACHE = {sp: ((RAW_AUD[sp] - amu) / asd) @ aP for sp in RAW_AUD}
    SPK_G_AUD = {sp: torch.tensor(v, device=DEV, dtype=DT) for sp, v in SPK_CACHE.items()}
    ts = build_timeshifts(CORPUS0.listeners, CORPUS0.pairs, SPK_G_AUD, ts_base)
    corpus = dataclasses.replace(CORPUS0, SPK_CACHE=SPK_CACHE, SPK_G_AUD=SPK_G_AUD,
                                 k_audio=aP.shape[1], timeshifts=ts)
    return corpus, (amu, asd, aP)


def make_ctx(corpus, motion_seed, row_seed, tsi):
    mmu, msd, mP = motion_pca(motion_seed, row_seed)
    driver = {sp: torch.cat([corpus.SPK_G_AUD[sp],
                             torch.tensor(((MRAW[sp] - mmu) / msd) @ mP, device=DEV, dtype=DT)], 1)
              for sp in MRAW}
    width = corpus.k_audio + mP.shape[1]
    names = list(driver)
    lens = [driver[s].shape[0] for s in names]
    ka = corpus.k_audio
    iac = np.tile(np.concatenate([np.ones(ka, bool), np.zeros(width - ka, bool)]), len(LAG_NP))
    ctx = SourceContext(
        driver=driver, driver_all=torch.cat([driver[s] for s in names], 0),
        speaker_row0={s: int(o) for s, o in zip(names, np.concatenate([[0], np.cumsum(lens)[:-1]]))},
        channel_cols={"audio": torch.tensor(iac, device=DEV), "motion": torch.tensor(~iac, device=DEV)},
        eval_mask=MASK, corpus=corpus, target_scale_inv=tsi)
    return ctx, (mmu, msd, mP)


def capture3(ctx, ops):
    E = {}
    for nm in ("GT", "seeded", "zero"):
        others = [FOKS[o] for o in FOKS if o != nm]
        per = collect_design(CACHES[nm], ctx, inter(MASK, *others))
        E[nm], _ = capture_energies(per, ops, ctx)
    common = sorted(set.intersection(*[set(E[nm]) for nm in E]))
    return {nm: {s: E[nm][s] for s in common} for nm in E}, common


def compute_tsi(pg):
    d = int(next(iter(pg.values()))[1].shape[1])
    s = torch.zeros(d, device=DEV, dtype=torch.float64)
    sq = torch.zeros(d, device=DEV, dtype=torch.float64); n = 0
    for ls in pg:
        t = torch.as_tensor(pg[ls][1], device=DEV, dtype=torch.float64)
        s += t.sum(0); sq += (t * t).sum(0); n += int(t.shape[0])
    var = (sq / max(n, 1) - (s / max(n, 1)) ** 2).clamp_min(0)
    return torch.rsqrt(var + torch.median(var).clamp_min(1e-12)).to(DT)


def col_nets(Ec, common):
    return {nm: marginal(Ec[nm], common)["net"] for nm in Ec}


def main():
    global CORPUS0, RAW_AUD, GT_CACHE, MRAW, MASK, PG, TSI, CACHES, FOKS, SRC, SUF, XAUD, XMOT
    ap = argparse.ArgumentParser(description="Metric-internal seed sensitivity of R-DGG (standalone).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS))
    ap.add_argument("--split", default=_config.SPLIT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-only"
    _config.set_paths(args.corpus, args.split)
    SRC, SUF = SOURCES[0]
    C, S = _config.CORPUS, _config.SPLIT

    t0 = time.time()
    CORPUS0 = load_corpus()
    RAW_AUD = {sp: r for sp, r in zip(CORPUS0.speakers, parallel_map(speaker_raw, CORPUS0.speakers)) if r is not None}
    GT_CACHE = load_all_motion(f"{C}/GT/{S}_{SUF}", SRC, CORPUS0)
    MRAW = {sp: speaker_motion_raw(sp, SUF, SRC, CORPUS0.SPK_G_AUD[sp].shape[0])
            for sp in CORPUS0.speakers if sp in CORPUS0.SPK_G_AUD}
    MRAW = {sp: r for sp, r in MRAW.items() if r is not None}
    MASK = gtonly_eval_mask(SUF, SRC, CORPUS0)
    CACHES = {"GT": GT_CACHE,
              "seeded": load_all_motion(f"{C}/AVTR-1_seeded/{S}_{SUF}", SRC, CORPUS0),
              "zero": load_all_motion(f"{C}/AVTR-1_zeroother/{S}_{SUF}", SRC, CORPUS0)}
    FOKS = {nm: gridded_face_ok(CACHES[nm], CORPUS0) for nm in CACHES}
    XAUD = np.concatenate([RAW_AUD[sp] for sp in sorted(RAW_AUD)], 0)[:5000]
    XMOT = np.concatenate([MRAW[sp] for sp in sorted(MRAW)], 0)[:5000]
    print(f"  [caches] loaded in {time.time()-t0:.1f}s; NFOLD={CORPUS0.NFOLD}", flush=True)

    DEF_AUD = audio_pca(101, 0)
    DEF_MOT = motion_pca(103, 0)

    # default build + pg + tsi + ops0 + default captures
    ctx_tmp, _ = make_ctx(CORPUS0, 103, 0, None)
    PG = collect_design(GT_CACHE, ctx_tmp, MASK)
    TSI = compute_tsi(PG)
    del ctx_tmp
    ctx0, _ = make_ctx(CORPUS0, 103, 0, TSI)
    ops0 = [fit_bgt_fold(PG, f, ctx0) for f in range(CORPUS0.NFOLD)]
    assert all(o is not None for o in ops0)
    E0, common = capture3(ctx0, ops0)
    nets0 = col_nets(E0, common)
    print(f"  [default] stems={len(common)} nets={nets0}", flush=True)

    result = {"source": SRC, "suffix": SUF, "split": S, "n_folds": CORPUS0.NFOLD,
              "n_stems": len(common), "n_identities": len({_ident_key(s) for s in common}),
              "columns": ["GT", "AVTR-1_seeded (base0 stock)", "AVTR-1_zeroother"],
              "default_nets": nets0,
              "generation_seed_SD_for_comparison": 0.111,
              "knob1_pca_stem_seed": {}, "knob2_pca_row_seed": {},
              "knob4_timeshift_base": {}, "knob5_bootstrap_seed": {}}

    def record_pca_knob(key, settings):
        rows = []
        net_series = {nm: [nets0[nm]] for nm in nets0}
        for (a_seed, m_seed, r_seed, label) in settings:
            corpus_v, aud = make_corpus(a_seed, r_seed, 2000)
            ctx_v, mot = make_ctx(corpus_v, m_seed, r_seed, TSI)
            ops_v = [fit_bgt_fold(PG, f, ctx_v) for f in range(corpus_v.NFOLD)]
            Ev, com = capture3(ctx_v, ops_v)
            nets_v = col_nets(Ev, com)
            row = {"label": label, "audio_pca_vs_default": pca_diag(aud, DEF_AUD, XAUD),
                   "motion_pca_vs_default": pca_diag(mot, DEF_MOT, XMOT),
                   "B_operator_rel_frobenius_vs_default": bd_frobenius(ops_v, ops0),
                   "nets": nets_v}
            for nm in nets_v:
                net_series[nm].append(nets_v[nm])
            rows.append(row)
            print(f"  [{key}] {label}: aud angle {row['audio_pca_vs_default']['max_principal_angle_deg']}deg "
                  f"overlap {row['audio_pca_vs_default']['mean_cos2_overlap']} | "
                  f"mot angle {row['motion_pca_vs_default']['max_principal_angle_deg']}deg | "
                  f"B dF {row['B_operator_rel_frobenius_vs_default']} | nets {nets_v}", flush=True)
            del ctx_v, ops_v
            torch.cuda.empty_cache()
        net_sd = {nm: round(float(np.std(net_series[nm], ddof=1)), 4) for nm in net_series}
        net_rng = {nm: round(float(np.max(net_series[nm]) - np.min(net_series[nm])), 4) for nm in net_series}
        return {"settings": rows, "net_point_SD_incl_default": net_sd, "net_point_range_incl_default": net_rng}

    result["knob1_pca_stem_seed"] = record_pca_knob(
        "knob1", [(101 + d, 103 + d, 0, f"stem+{d}") for d in (1, 2, 3, 4)])
    result["knob2_pca_row_seed"] = record_pca_knob(
        "knob2", [(101, 103, r, f"row={r}") for r in (1, 2, 3, 4)])

    # knob 4: timeshift base -> floor (reuse default B/ctx; only nulls change)
    floor_series = {nm: [] for nm in nets0}
    k4 = {}
    for base in (2000, 3000, 4000, 5000, 6000):
        if base == 2000:
            Eb, comb = E0, common
        else:
            corpus_b = dataclasses.replace(CORPUS0, timeshifts=build_timeshifts(
                CORPUS0.listeners, CORPUS0.pairs, CORPUS0.SPK_G_AUD, base))
            ctx_b = dataclasses.replace(ctx0, corpus=corpus_b)
            Eb, comb = capture3(ctx_b, ops0)
        fs = {nm: floor_stats(Eb[nm], comb) for nm in Eb}
        nets_b = col_nets(Eb, comb)
        k4[str(base)] = {"floor": {nm: fs[nm]["floor"] for nm in fs},
                         "floor_se_within_draw": {nm: fs[nm]["floor_se_within_draw"] for nm in fs},
                         "nets": nets_b}
        for nm in fs:
            floor_series[nm].append(fs[nm]["floor"])
        print(f"  [knob4] base {base}: floors {k4[str(base)]['floor']} nets {nets_b}", flush=True)
    result["knob4_timeshift_base"] = {
        "per_base": k4,
        "floor_SD_across_bases": {nm: round(float(np.std(floor_series[nm], ddof=1)), 4) for nm in floor_series},
        "floor_range_across_bases": {nm: round(float(np.max(floor_series[nm]) - np.min(floor_series[nm])), 4)
                                     for nm in floor_series}}

    # knob 5: bootstrap seed (point fixed; CI only), on default E
    k5 = {}
    lo_series = {nm: [] for nm in nets0}; hi_series = {nm: [] for nm in nets0}; pt = {}
    for seed in (0, 1, 2, 3, 4):
        per = {}
        for nm in E0:
            d = boot_ci_seeded(E0[nm], common, _ident_key, BOOTSTRAP, seed)
            per[nm] = d
            lo_series[nm].append(d["lo"]); hi_series[nm].append(d["hi"]); pt[nm] = d["point"]
        k5[str(seed)] = per
    result["knob5_bootstrap_seed"] = {
        "per_seed": k5, "point_fixed": pt,
        "point_SD_across_seeds": {nm: round(float(np.std([k5[str(s)][nm]["point"] for s in (0, 1, 2, 3, 4)], ddof=1)), 6)
                                  for nm in nets0},
        "ci_lo_SD": {nm: round(float(np.std(lo_series[nm], ddof=1)), 4) for nm in nets0},
        "ci_hi_SD": {nm: round(float(np.std(hi_series[nm], ddof=1)), 4) for nm in nets0}}
    print(f"  [knob5] point SD across boot seeds {result['knob5_bootstrap_seed']['point_SD_across_seeds']} "
          f"| CI-lo SD {result['knob5_bootstrap_seed']['ci_lo_SD']}", flush=True)

    # combined metric-seed SD per column (knobs 1,2,4 point SDs in quadrature)
    comb_sd = {}
    for nm in nets0:
        parts = [result["knob1_pca_stem_seed"]["net_point_SD_incl_default"][nm],
                 result["knob2_pca_row_seed"]["net_point_SD_incl_default"][nm],
                 result["knob4_timeshift_base"]["floor_SD_across_bases"][nm]]
        comb_sd[nm] = round(float(np.sqrt(np.sum(np.square(parts)))), 4)
    result["combined_metric_seed_net_SD"] = comb_sd
    result["comparison"] = {
        "generation_seed_net_SD": 0.111,
        "note": "combined metric-seed net SD (knobs 1,2,4 in quadrature) vs the generation-seed SD ~0.11"}

    out_dir = f"{C}/GT/_metrics"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/metric_seed_sweep__{S}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [combined] metric-seed net SD {comb_sd} vs generation-seed SD ~0.111", flush=True)
    print(f"  [out] wrote {out_dir}/metric_seed_sweep__{S}.json", flush=True)
    print("METRIC_SEED_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
