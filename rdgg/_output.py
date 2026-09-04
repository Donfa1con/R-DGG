"""Result output for R-DGG: the leaderboard, the 8x8 pairwise matrix, and the JSON/npz writers.

report_matrix writes into `_config.CORPUS`/GT/_metrics.
"""
import json
import os

import numpy as np

import _config
from _config import FIDFLOOR, MODELS_ALL, N_TIMESHIFT, PAIRED_NULLS, SURROGATE, _ident_key, model_kind
from _features import collect_design
from _granger import capture_energies
from _bootstrap import gnet_boot_ci, pair_stats, t_stat


def _flip_cell(st):
    """MAT[b][a] is MAT[a][b] with the sign of the delta flipped: Δ_ba = -Δ_ab, and its CI is [-hi, -lo]."""
    if st is None:
        return None
    return {"delta": round(-st["delta"], 3), "lo": round(-st["hi"], 3), "p50": round(-st["p50"], 3),
            "hi": round(-st["lo"], 3), "p": st["p"], "sig": st["sig"],
            "n_listeners": st["n_listeners"], "n_clusters": st["n_clusters"]}


def report_matrix(source, E_by_model, caches, fok_grid, mask_gtonly, ctx, BGT, B):
    """The pairwise 8x8. B is fit ONCE on GT-only and shared. The MARGINAL row uses each system's own GT&model
    capture (E_by_model). Each PAIRWISE cell (A,B) RE-CAPTURES A and B on the FRAME-EXACT paired mask GT&A&B (so
    Δ = G_A - G_B compares the two on identical frames), using PAIRED_NULLS timeshifts (the Δ cancels most floor
    noise). Saves json (matrix + marginals) + npz (marginal energies) under the corpus _metrics/."""
    SYS = [s for s in E_by_model if E_by_model.get(s)]
    if len(SYS) < 2:
        print("\n  [matrix] fewer than 2 systems captured -- skipped")
        return
    ALIAS = {"AvatarForcing": "AvatarF", "SoulX Lite": "SoulX-L", "SoulX Pro": "SoulX-P"}
    al = lambda s: ALIAS.get(s, s)
    marg = {s: gnet_boot_ci(E_by_model[s], list(E_by_model[s]), _ident_key, B, one_sided=False) for s in SYS}
    row = sorted(SYS, key=lambda s: -marg[s]["point"])
    paired_ts = ctx.corpus.timeshifts[:PAIRED_NULLS]

    def paired_capture(target, other):                     # target's energies on GT & target & other frames
        om = fok_grid.get(other, {})
        em = {ls: mask_gtonly[ls] & om[ls] for ls in mask_gtonly
              if ls in om and len(om[ls]) == len(mask_gtonly[ls])}
        per = collect_design(caches[target], ctx, em)      # target's own face_ok is intersected inside -> GT&A&B
        if not per or len(per) < 5:
            return None
        return capture_energies(per, BGT, ctx, timeshifts=paired_ts)[0]

    def paired_cell(a, b):
        if a not in caches or b not in caches:
            return pair_stats(E_by_model[a], E_by_model[b], B)   # fallback: no cache -> per-model (marginal) masks
        EA = paired_capture(a, b)
        EB = paired_capture(b, a)
        if not EA or not EB:
            return None
        return pair_stats(EA, EB, B)

    MAT = {a: {b: None for b in SYS} for a in SYS}
    for i, a in enumerate(SYS):
        for b in SYS[i + 1:]:
            st = paired_cell(a, b)                          # capture each unordered pair once; mirror with sign flip
            MAT[a][b] = st
            MAT[b][a] = _flip_cell(st)

    print(f"\n=== {source}: per-system marginal G_net(aud|mot) x1e4  point [2.5% p50 97.5%] (own max GT&model set; n varies; {N_TIMESHIFT} nulls) ===")
    for s in row:
        c = marg[s]
        print(f"    {al(s):9s} {c['point']:+7.3f}  [{c['lo']:+.3f} p50={c['p50']:+.3f} {c['hi']:+.3f}]  (n={len(E_by_model[s])})")

    LEFTW, W = 10, 14
    print(f"\n=== {source}: PAIRWISE Δ = G_row − G_col x1e4 (GT&A&B frame-exact paired; {PAIRED_NULLS} nulls) ; cell = Δ(*) / p50 / [95% CI] / p(n) ; * CI excludes 0 ===")
    print(" " * LEFTW + "".join(f"{al(s):>{W}s}" for s in row))

    def cell(st):
        if st is None:
            return ["—".rjust(W)] + ["".rjust(W)] * 3
        return [f"{st['delta']:+.2f}{'*' if st['sig'] else ' '}".rjust(W),
                f"{st['p50']:+.2f}".rjust(W),
                f"[{st['lo']:+.2f},{st['hi']:+.2f}]".rjust(W),
                f"p{st['p']:.3f}({st['n_listeners']})".rjust(W)]

    for r in row:
        cols = [cell(MAT[r][c] if r != c else None) for c in row]
        for li in range(4):
            print((f"{al(r):<{LEFTW}s}" if li == 0 else " " * LEFTW) + "".join(c[li] for c in cols))
        print()

    print("=== significant pairs (CI excludes 0), stronger row > weaker col ===")
    seen = set()
    for a in row:
        for b in row:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            st = MAT[a][b]
            if st and st["sig"]:
                hi_, lo_ = (a, b) if st["delta"] >= 0 else (b, a)   # label ">" by the sign of the paired Δ, not marginal row order
                d = MAT[hi_][lo_]
                print(f"    {al(hi_):9s} > {al(lo_):9s}  Δ={d['delta']:+.3f} [{d['lo']:+.3f} p50={d['p50']:+.3f} "
                      f"{d['hi']:+.3f}] p={d['p']:.3f}  n={d['n_listeners']}/{d['n_clusters']}cl")

    outdir = f"{_config.CORPUS}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/reactivity_matrix__{_config.SPLIT}.json", "w") as fh:
        json.dump({"systems": SYS, "marginal": marg, "B": B, "matrix": MAT}, fh, indent=1)
    saved = {}
    for s in SYS:
        stems = list(E_by_model[s])
        saved[f"{s}__stems"] = np.array(stems)
        saved[f"{s}__idents"] = np.array([_ident_key(st) for st in stems])
        saved[f"{s}__both_r"] = np.array([E_by_model[s][st]["both_r"] for st in stems])
        saved[f"{s}__mot_r"] = np.array([E_by_model[s][st]["mot_r"] for st in stems])
        saved[f"{s}__both_n"] = np.stack([E_by_model[s][st]["both_n"] for st in stems], 1)
        saved[f"{s}__mot_n"] = np.stack([E_by_model[s][st]["mot_n"] for st in stems], 1)
    np.savez_compressed(f"{outdir}/reactivity_matrix__{_config.SPLIT}.npz", **saved)
    print(f"\n  wrote {outdir}/reactivity_matrix__{_config.SPLIT}.{{json,npz}}  (re-bootstrappable)")


def report_leaderboard(source, stats, fidelity, coverage, grid_alignment):
    """Leaderboard on the COMMON (all-model) eval set: tracking fidelity, eval coverage, grid alignment, then the
    per-model aud|mot net with its raw/floor decomposition, identity- and interaction-clustered CIs (+p50), and
    rank stability vs the median null. `stats[lbl]` is leaderboard_stats(...) or None for a skipped model."""
    print(f"\n=== {source}: per-model tracking fidelity (mean face_ok; floor={FIDFLOOR * 100:.0f}%) ===")
    print(f"    {'model':14s} {'kind':9s} {'face_ok':>8s} {'n_stem':>7s} {'flag':>6s}")
    for lbl, mdl_dir in MODELS_ALL:
        fid, ns, _n_face = fidelity.get(lbl, (float("nan"), 0, 0))
        kind = model_kind(lbl)
        if ns == 0:
            print(f"    {lbl:14s} {kind:9s} {'n/a':>8s} {ns:7d}")
            continue
        print(f"    {lbl:14s} {kind:9s} {fid * 100:7.1f}% {ns:7d} {'LOW' if fid < FIDFLOOR else '':>6s}")
    if coverage:
        distinct_coverage = {vec for k, vec in coverage.items() if k != SURROGATE}
        print(f"\n=== {source}: EVAL-SET COVERAGE (must be IDENTICAL across models) ===")
        print(f"    {'model':14s} {'lis':>4s} {'segs':>6s} {'rows':>8s} {'hours':>6s}")
        for lbl_, _md in MODELS_ALL:
            if lbl_ not in coverage:
                continue
            l_, s_, r_ = coverage[lbl_]
            print(f"    {lbl_:14s} {l_:>4d} {s_:>6d} {r_:>8d} {r_ / 25 / 3600:>6.2f}")
        n_models = sum(1 for lbl_ in coverage if lbl_ != SURROGATE)
        agreed = len(distinct_coverage) == 1 and len(coverage) == len(MODELS_ALL)
        print("    --> " + (f"IDENTICAL across all {n_models} models ({SURROGATE} excluded by design: blinding its "
              f"wrap seams removes rows)" if agreed else f"NOT COMPARABLE: {len(distinct_coverage)} distinct row-set(s) "
              f"over {len(coverage)} of {len(MODELS_ALL)} — cross-model comparison does not mean anything here"))
        print(f"\n=== {source}: GRID ALIGNMENT (how each source's frames land on the 25fps metric grid) ===")
        print(f"    {'model':14s} {'fps':>8s} {'frames vs grid':>26s}   slots past end of source")
        for lbl_, _md in MODELS_ALL:
            if lbl_ not in grid_alignment:
                continue
            rates_, offs_, unc_ = grid_alignment[lbl_]
            span = f"median {int(np.median(offs_)):+d}  range {min(offs_):+d}..{max(offs_):+d}" if offs_ else "-"
            beyond = f"median {int(np.median(unc_))}  max {max(unc_)}" if unc_ else "-"
            rate = ",".join(f"{r:g}" for r in sorted(rates_)) if rates_ else "?"
            print(f"    {lbl_:14s} {rate:>8s} {span:>26s}   {beyond}")

    valid = {l: s for l, s in stats.items() if s is not None}
    print(f"\n=== {source} — R-DGG  (cross-fit; ×1e4 = gain above the timeshift floor; "
          f"COMMON mask; {N_TIMESHIFT} timeshifts) ===")
    if not valid:
        print("    (all skipped)")
        return
    n_ident = next(iter(valid.values()))["ci_ident"]["n_clusters"]
    print(f"    significance = {n_ident} identity clusters   (GT = held-out anchor: must score & be significant;\n"
          "    every non-dyadic row and the surrogate must read ≈0 — a number is a valid R-DGG value only where they do.)")
    print(f"    {'model':14s} {'kind':9s} {'channel':>9s} {'value':>7s} {'[lo p50 hi]':>26s} {'p':>7s} | {'ownR2':>6s}")
    order = sorted(stats, key=lambda l: (stats[l]["net"] if stats[l] is not None else -1e18), reverse=True)
    for lbl in order:
        s = stats[lbl]
        kind = model_kind(lbl)
        if s is None:
            print(f"    {lbl:14s} {kind:9s}   (skipped)")
            continue
        ci = s["ci_ident"]
        cistr = f"[{ci['lo']:+.2f} p50={ci['p50']:+.2f} {ci['hi']:+.2f}]"
        print(f"    {lbl:14s} {kind:9s}*{'aud|mot':>8s} {s['net']:+7.2f} {cistr:>26s} {ci['p']:>7.3f} | {s['own_r2']:>6.3f}")
        print(f"    {'':14s} {'':9s} {'decomp':>9s} aud|mot: raw{s['raw']:+7.2f} floor{s['floor']:+7.2f}"
              f"(sd{s['floor_sd']:5.2f}) net{s['net']:+7.2f}")
        print(f"        └ {'aud|mot':11s}: per-dyad t={t_stat(s['dyad']):+5.1f} "
              f"%>0={100 * float((s['dyad'] > 0).mean()):3.0f} | per-inter t={t_stat(s['intr']):+5.1f} "
              f"%>0={100 * float((s['intr'] > 0).mean()):3.0f}")
        ic = s["ci_intr"]
        print(f"        └ boot[intr({ic['n_clusters']})]: aud|mot[{ic['lo']:+.2f} p50={ic['p50']:+.2f} "
              f"{ic['hi']:+.2f}] p={ic['p']:.3f}")
    valid_list = [(l, stats[l]) for l in order if stats[l] is not None]
    if len(valid_list) >= 3:
        by_mean = [l for l, _ in sorted(valid_list, key=lambda it: it[1]["net"], reverse=True)]
        by_med = [l for l, _ in sorted(valid_list, key=lambda it: it[1]["net_med"], reverse=True)]
        same = "SAME" if by_mean[:5] == by_med[:5] else "DIFFERS"
        print(f"    [rank-stability] aud|mot, mean-null vs median-null top5: {same} | "
              f"median order: {', '.join(by_med[:5])}")
    print("\n  READ the aud|mot column: the speaker's AUDIO gain conditioned on the speaker's own MOTION, so visible\n"
          "  co-speech is regressed out. GT is the held-out cross-fit anchor and must be credited; every non-dyadic\n"
          "  row and the surrogate must read ≈0. Significance is the identity-clustered bootstrap. ownR2 is how much\n"
          "  of the target the controls alone explain (control block refit per model).")
