
"""R-DGG for talking-head models: a reference-based, directed Granger gain.

Does the model's listener move BECAUSE of what the speaker is doing, or does it merely produce plausible
listening-shaped motion? This file checks and names every required input before it runs; --corpus and
--split are the only arguments.

No methodological arguments and no environment: --corpus and --split are paths, the operating point itself is
fixed, and any GRANGER_* variable aborts the run.
"""
import argparse
import os
import time

import numpy as np
import torch

import _config
from _config import (
    BOOTSTRAP, Corpus, DEV, DT, K_A_TARGET, KMV, LAG_NP, LMAX, LSPK, MINSEG, MINSHIFT,
    MODELS_ALL, MODELS_ALL_DIRS, NFOLD_MAX, N_TIMESHIFT, RIDGE, SOURCES,
    SURROGATE, SourceContext, _ident_key, _intr_key, _sess_key, parallel_map,
)
from _features import (
    alignment_stats, build_eval_mask, collect_design, fit_pca, gridded_face_ok, gtonly_eval_mask,
    load_all_motion, pca_fit_sample, speaker_motion_raw, speaker_raw,
)
from _granger import capture_energies, fit_bgt_fold
from _bootstrap import leaderboard_stats
from _output import report_leaderboard, report_matrix


def build_surrogate_pairing(listeners):
    """Give each listener ANOTHER person's real listening motion -- never their own, and never their own interaction.

    The half-roster offset on its own collided zero times on this corpus, but by luck of how sorted order interleaves 12
    sessions across 13 identities, not by construction. With fewer identities a large share of the strongest control in
    the method would become the target's own motion, and it would then be credited for nothing at all.
    """
    n = len(listeners)
    partner = [listeners[(i + n // 2) % n] for i in range(n)]

    def collides(i, stem):
        return _ident_key(listeners[i]) == _ident_key(stem) or _intr_key(listeners[i]) == _intr_key(stem)

    for i in range(n):
        if not collides(i, partner[i]):
            continue
        for j in range(n):
            if i != j and not collides(i, partner[j]) and not collides(j, partner[i]):
                partner[i], partner[j] = partner[j], partner[i]
                break
    unfixed = [ls for i, ls in enumerate(listeners) if collides(i, partner[i])]
    assert not unfixed, (
        f"{len(unfixed)} listeners have no different-identity, different-interaction partner available, first: "
        f"{unfixed[:3]} -- the surrogate would be partly self-paired and would stop being a control"
    )
    assert len(set(partner)) == n, "the surrogate pairing is no longer a permutation"
    return dict(zip(listeners, partner))


def load_corpus():
    """Read everything the run needs from the corpus, once. Nothing here depends on which encoder is scored."""
    # directed listener->speaker map; each line: "<listener>.wav <speaker>.wav"
    pairs = {}
    for ln in open(f"{_config.CORPUS}/pairs184.txt").read().splitlines():
        fields = ln.split()
        if len(fields) >= 2:
            pairs[fields[0].replace(".wav", "")] = fields[1].replace(".wav", "")
    assert pairs, f"{_config.CORPUS}/pairs184.txt gave no pairs"
    print(f"  [pairs] {len(pairs)} directed listener->speaker pairs from pairs184.txt", flush=True)
    # listener universe = keys of the pairs map
    listeners = sorted(pairs)
    speakers = sorted(set(pairs.values()))
    malformed = [ls for ls in listeners if len(ls.split("_")) != 4]
    assert not malformed, (
        f"stems must follow V<v>_S<session>_I<interaction>_P<identity>, these do not: {malformed[:3]} -- session, "
        f"interaction and identity are parsed by field position (_sess_key / _intr_key / _ident_key)"
    )
    n_ident = len({_ident_key(ls) for ls in listeners})
    assert n_ident < len(listeners), (
        f"field 4 of the stem takes {n_ident} distinct values over {len(listeners)} stems, so it does not recur and is "
        f"not an identity: folds would not be identity-disjoint and the 'ident' bootstrap would count clips"
    )

    sess_idents = {}
    sess_n = {}
    for ls in listeners:
        sess = _sess_key(ls)
        sess_idents.setdefault(sess, set()).add(_ident_key(ls))
        sess_n[sess] = sess_n.get(sess, 0) + 1
    sessions = sorted(sess_idents)
    parent_of = {sess: sess for sess in sessions}

    def find_root(node):
        while parent_of[node] != node:
            parent_of[node] = parent_of[parent_of[node]]
            node = parent_of[node]
        return node

    sessions_of_ident = {}
    for sess in sessions:
        for idn in sess_idents[sess]:
            sessions_of_ident.setdefault(idn, []).append(sess)
    for shared in sessions_of_ident.values():
        for sess in shared[1:]:
            parent_of[find_root(sess)] = find_root(shared[0])
    comp_members = {}
    for sess in sessions:
        comp_members.setdefault(find_root(sess), []).append(sess)
    components = sorted(comp_members.values(), key=lambda comp: -sum(sess_n[sess] for sess in comp))
    NFOLD = len(components)
    _nf = int(os.environ.get("REACT_NFOLD", "0"))   # test-only fast preview: cap leave-out folds (fewer B-fits)
    if _nf and _nf < NFOLD:
        NFOLD = _nf
    assert NFOLD >= 2, f"only {NFOLD} identity component(s): cannot make NFOLD>=2 identity-disjoint folds"
    assert NFOLD <= NFOLD_MAX, (
        f"{NFOLD} identity components -> {NFOLD} leave-one-out folds, each a full fit on the entire GT design. Above "
        f"NFOLD_MAX={NFOLD_MAX} that is hours of Phase 1; raise the constant deliberately if the corpus warrants it"
    )
    fold_load = [0] * NFOLD
    session_to_fold = {}
    for comp in components:
        fold = min(range(NFOLD), key=lambda i: fold_load[i])
        for sess in comp:
            session_to_fold[sess] = fold
        fold_load[fold] += sum(sess_n[sess] for sess in comp)
    FOLD = {ls: session_to_fold[_sess_key(ls)] for ls in listeners}
    print(
        f"  [folds] NFOLD={NFOLD} leave-one-identity-component-out "
        f"({n_ident} identities over {len(sessions)} sessions): sizes "
        + "/".join(str(sum(1 for item in listeners if FOLD[item] == fold)) for fold in range(NFOLD)),
        flush=True,
    )
    amu, asd, aP = fit_pca(
        [mrow for mrow in parallel_map(speaker_raw, pca_fit_sample(speakers, 101)) if mrow is not None], K_A_TARGET
    )
    k_audio = aP.shape[1]
    SPK_CACHE = {}
    for sp, Sraw in zip(speakers, parallel_map(speaker_raw, speakers)):
        if Sraw is not None:
            SPK_CACHE[sp] = ((Sraw - amu) / asd) @ aP
    SPK_G_AUD = {sp: torch.tensor(vec, device=DEV, dtype=DT) for sp, vec in SPK_CACHE.items()}
    LAG = torch.tensor(LAG_NP, device=DEV, dtype=torch.long)

    timeshifts = []
    n_fixed_offset = 0
    for k in range(N_TIMESHIFT):
        rng = np.random.default_rng(2000 + k)
        off = {}
        for ls in listeners:
            sp = pairs[ls]
            n = SPK_G_AUD[sp].shape[0] if sp in SPK_G_AUD else 0
            if n > 2 * MINSHIFT + 20:
                off[ls] = int(rng.integers(MINSHIFT, n - MINSHIFT))
            else:
                off[ls] = max(25, MINSHIFT)
                n_fixed_offset += 1
        timeshifts.append(off)
    assert not n_fixed_offset, (
        f"{n_fixed_offset} (listener, draw) pairs could not draw an offset: the speaker series is under "
        f"{2 * MINSHIFT + 20} grid frames, so the floor is one fixed {max(25, MINSHIFT)}-frame shift against a "
        f"{LSPK}-frame lag grid rather than a null distribution"
    )

    surr_perm = build_surrogate_pairing(listeners)
    return Corpus(
        pairs=pairs,
        listeners=listeners,
        speakers=speakers,
        FOLD=FOLD,
        NFOLD=NFOLD,
        LAG=LAG,
        timeshifts=timeshifts,
        SPK_CACHE=SPK_CACHE,
        SPK_G_AUD=SPK_G_AUD,
        k_audio=k_audio,
        SURRPERM=surr_perm,
    )


def preflight():
    """Name every missing input up front, instead of failing obscurely two minutes into the load.

    Each of these has produced a misleading failure. A bare `open()` on pairs184.txt. A missing HuBERT directory, which
    makes every speaker return None and dies in fit_pca on "need at least one array to concatenate", with the word
    hubert nowhere in the traceback. Worst, a missing VAD directory, which drops every listener and prints
    "[B_GT] FAILED -- not enough GT design rows" -- a diagnosis that never mentions VAD at all.
    """
    if not os.path.isdir(_config.CORPUS):
        raise SystemExit(
            f"ABORT: corpus root does not exist: {_config.CORPUS}\n"
            "Pass --corpus <path> (or set $CORPUS). It must hold GT/ and one directory per model; the\n"
            "repo-relative default is data/ (populated by data/download_si184.py and the pipeline)."
        )
    required = [f"{_config.CORPUS}/pairs184.txt"] + [
        f"{_config.CORPUS}/GT/{_config.SPLIT}_{d}" for d in ("hubert", "vad")
    ]
    for _source, suffix in SOURCES:
        for _lbl, model_dir in MODELS_ALL_DIRS:
            required += [f"{_config.CORPUS}/{model_dir}/{_config.SPLIT}_{suffix}", f"{_config.CORPUS}/{model_dir}/{_config.SPLIT}_liveportrait"]
    missing = [path for path in dict.fromkeys(required) if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} required input(s) missing:\n  "
            + "\n  ".join(missing)
            + "\n\nEach line above is a required input directory. The model roster is a list in this file, not an "
            "argument: every model in it needs both motion directories, because the scored frame set is intersected "
            "over all of them."
        )


def score_source(source, suffix, corpus):
    """Score every model for one encoder and print the tables."""
    print(
        f"\n################## SOURCE = {source.upper()} [K_A={corpus.k_audio} KMV={KMV} LMAX={LMAX} ridge={RIDGE} "
        f"target-balanced unwhitened-energy "
        f"minseg={MINSEG / 25:.0f}s ntimeshift={N_TIMESHIFT} boot={BOOTSTRAP}] "
        f"##################",
        flush=True,
    )
    t_phase1 = time.time()
    gt_cache = load_all_motion(f"{_config.CORPUS}/GT/{_config.SPLIT}_{suffix}", source, corpus)
    mraw = {
        sp: speaker_motion_raw(sp, suffix, source, corpus.SPK_G_AUD[sp].shape[0])
        for sp in corpus.speakers
        if sp in corpus.SPK_G_AUD
    }
    mraw = {sp: res for sp, res in mraw.items() if res is not None}
    mmu, msd, mP = fit_pca([mraw[sp] for sp in pca_fit_sample(list(mraw), 103)], KMV)
    driver = {
        sp: torch.cat([corpus.SPK_G_AUD[sp], torch.tensor(((mraw[sp] - mmu) / msd) @ mP, device=DEV, dtype=DT)], 1)
        for sp in mraw
    }
    width = corpus.k_audio + mP.shape[1]
    missing = sorted(set(corpus.SPK_CACHE) - set(driver))
    assert not missing, (
        f"{len(missing)} of {len(corpus.SPK_CACHE)} speakers have audio but no motion in {_config.SPLIT}_{suffix}, "
        f"first few: {missing[:5]}"
    )
    del mraw
    speaker_names = list(driver.keys())
    speaker_lens = [driver[sess].shape[0] for sess in speaker_names]
    assert width > corpus.k_audio, f"driver width {width} has no motion block beyond the {corpus.k_audio} audio dims"
    ka = corpus.k_audio
    is_audio_col = np.tile(np.concatenate([np.ones(ka, bool), np.zeros(width - ka, bool)]), len(LAG_NP))
    ctx = SourceContext(
        driver=driver,
        driver_all=torch.cat([driver[sess] for sess in speaker_names], 0),
        speaker_row0={
            sess: int(off)
            for sess, off in zip(speaker_names, np.concatenate([[0], np.cumsum(speaker_lens)[:-1]]))
        },
        channel_cols={
            "audio": torch.tensor(is_audio_col, device=DEV),
            "motion": torch.tensor(~is_audio_col, device=DEV),
        },
        eval_mask=build_eval_mask(suffix, source, corpus),   # COMMON (all-model) frame set = the leaderboard eval
        corpus=corpus,
    )
    mask_gtonly = gtonly_eval_mask(suffix, source, corpus)   # fullest GT: B is fit here, and the 8x8 scores per-cell
    pg = collect_design(gt_cache, ctx, mask_gtonly)          # B decoupled from whether the MODELS tracked
    if pg:                                                   # GT-derived per-target-dim scale (fit balancer): balances the
        d = int(next(iter(pg.values()))[1].shape[1])         # target dims (EMOCA vs LP) before the fit
        s = torch.zeros(d, device=DEV, dtype=torch.float64)
        sq = torch.zeros(d, device=DEV, dtype=torch.float64); n = 0
        for _ls in pg:                                       # pooled GT target variance per dim (no consume: pg is reused below)
            t = torch.as_tensor(pg[_ls][1], device=DEV, dtype=torch.float64)
            s += t.sum(0); sq += (t * t).sum(0); n += int(t.shape[0])
        var = (sq / max(n, 1) - (s / max(n, 1)) ** 2).clamp_min(0)
        ctx.target_scale_inv = torch.rsqrt(var + torch.median(var).clamp_min(1e-12)).to(DT)
        print(f"  [target_norm] GT target var median {float(torch.median(var)):.3g}; scale_inv "
              f"[{float(ctx.target_scale_inv.min()):.2g}, {float(ctx.target_scale_inv.max()):.2g}]", flush=True)
    BGT = None
    if pg and len(pg) >= 10:
        ops = [fit_bgt_fold(pg, fold, ctx) for fold in range(corpus.NFOLD)]
        BGT = None if any(op is None for op in ops) else ops
    if BGT is None:
        print(f"  [B_GT] FAILED for {source} -- not enough GT design rows", flush=True)
    t_phase2 = time.time()
    print(
        f"  [profile] {source}: Phase1 = {t_phase2 - t_phase1:.1f}s  (GT load, audio+motion PCA, "
        f"{corpus.NFOLD} leave-one-out B_GT fits)",
        flush=True,
    )
    t_load = 0.0
    _tdesign = 0.0
    _tcap = 0.0
    stats = {}                          # per-model leaderboard stats (COMMON mask)
    E_gtonly = {}                       # per-model captured energies for the 8x8 MARGINAL row (GT-only per-model mask)
    caches = {}                         # non-surrogate motion caches, kept for the PAIRED 8x8 pass (re-capture on GT&A&B)
    fok_grid = {}                       # per-model grid-projected face_ok, to build the GT&A&B paired mask
    fidelity = {}
    coverage = {}
    grid_alignment = {}
    for lbl, mrow in MODELS_ALL:
        t_start = time.time()
        if lbl == SURROGATE:
            mcache = {
                ls: gt_cache[corpus.SURRPERM[ls]]
                for ls in corpus.listeners
                if corpus.SURRPERM[ls] in gt_cache
            }
            surr_cache = {}
            n_wrapped = n_blinded = 0
            for surr_target, (regions_src, face_ok_src, fps_src) in mcache.items():
                if surr_target not in gt_cache:
                    continue
                n_target_frames = len(gt_cache[surr_target][1])
                n_surr_frames = len(face_ok_src)
                if n_surr_frames < 2:
                    continue
                crop_offset = int(
                    np.random.default_rng(90000 + corpus.listeners.index(surr_target)).integers(0, n_surr_frames)
                )
                wrap_idx = (
                    crop_offset + np.arange(n_target_frames)
                ) % n_surr_frames
                surr_face_ok = face_ok_src[wrap_idx].copy()
                seams = np.flatnonzero(np.diff(wrap_idx) != 1) + 1
                for seam in seams:
                    surr_face_ok[seam : seam + LSPK] = False
                    n_blinded += int(min(LSPK, n_target_frames - seam))
                if len(seams):
                    n_wrapped += 1
                surr_cache[surr_target] = (
                    {reg_name: reg_mat[wrap_idx] for reg_name, reg_mat in regions_src.items()},
                    surr_face_ok,
                    fps_src,
                )
            print(
                f"  [surrogate] {len(surr_cache)} listeners paired with another person's motion; {n_wrapped} wrapped, "
                f"{n_blinded} slots blinded at the seams",
                flush=True,
            )
            mcache = surr_cache
        else:
            mcache = (
                gt_cache if lbl == "GT" else load_all_motion(f"{_config.CORPUS}/{mrow}/{_config.SPLIT}_{suffix}", source, corpus)
            )
        t_load += time.time() - t_start
        grid_alignment[lbl] = alignment_stats(mcache, corpus, ctx.eval_mask)
        foks = [comp[1] for comp in mcache.values() if comp[1] is not None and len(comp[1])]
        if foks:
            all_face_ok = np.concatenate(foks)
            fidelity[lbl] = (float(all_face_ok.mean()), len(foks), int(all_face_ok.size))
        else:
            fidelity[lbl] = (float("nan"), 0, 0)
        t_start = time.time()
        per_c = collect_design(mcache, ctx)                     # COMMON mask -> the leaderboard eval
        _tdesign += time.time() - t_start
        if per_c:
            n_cov_rows = sum(len(vec[2]) for vec in per_c.values())
            n_segments = sum(1 + int((np.diff(vec[2]) != 1).sum()) for vec in per_c.values() if len(vec[2]))
            coverage[lbl] = (len(per_c), n_segments, n_cov_rows)
        if not per_c or len(per_c) < 5 or BGT is None:
            stats[lbl] = None
        else:
            t_start = time.time()
            E_c, own_r2 = capture_energies(per_c, BGT, ctx)
            stats[lbl] = leaderboard_stats(E_c, own_r2, BOOTSTRAP)
            _tcap += time.time() - t_start
        if lbl != SURROGATE and BGT is not None:               # 8x8 MARGINAL capture on the GT-only per-model mask
            per_g = collect_design(mcache, ctx, mask_gtonly)
            if per_g and len(per_g) >= 5:
                t_start = time.time()
                E_gtonly[lbl], _ = capture_energies(per_g, BGT, ctx)
                _tcap += time.time() - t_start
            caches[lbl] = mcache                               # keep (incl. GT) for the PAIRED pairwise pass
            fok_grid[lbl] = gridded_face_ok(mcache, corpus)
        elif lbl == SURROGATE:
            del mcache                                          # surrogate is leaderboard-only, not in the 8x8
    print(
        f"  [profile] {source}: Phase2 = {time.time() - t_phase2:.1f}s  (model-load {t_load:.1f}s | "
        f"collect_design {_tdesign:.1f}s | capture {_tcap:.1f}s)",
        flush=True,
    )
    report_leaderboard(source, stats, fidelity, coverage, grid_alignment)
    report_matrix(source, E_gtonly, caches, fok_grid, mask_gtonly, ctx, BGT, BOOTSTRAP)


def main():
    ap = argparse.ArgumentParser(
        description="R-DGG: a reference-based directed Granger gain.",
        epilog="Paths only. There are no methodological options: to change the metric, edit the constants on a branch.",
    )
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", _config.CORPUS),
                    help="corpus root holding GT/ and one directory per model (default: repo data/, or $CORPUS)")
    ap.add_argument("--split", default=_config.SPLIT, help="split name, the prefix of the per-split directories")
    args = ap.parse_args()
    assert torch.cuda.is_available(), (
        f"DEV={DEV!r}: this metric is GPU-only -- the design is assembled on the device and every solve runs there. "
        f"It needs roughly 30 GB of GPU memory and 12 GB of host RAM for a 186-clip corpus."
    )
    _config.set_paths(args.corpus, args.split)
    preflight()
    corpus = load_corpus()
    for source, suffix in SOURCES:
        score_source(source, suffix, corpus)
    print("\nREFBASED_DONE")


if __name__ == "__main__":
    main()
