"""Motion-conditioning ablation for R-DGG: is partialling out the speaker motion needed?

NEW code only. It imports the already-written R-DGG blocks and the shift_ablation.py
helpers; it edits nothing in rdgg/ except the opt-in `marginal_audio` flag already
added to capture_energies / capture_column, adds no model to _config.MODELS, and
commits nothing of its own. See CVPR_zeroother_design.md section 13.1.

R-DGG reports the CONDITIONAL channel aud|mot, the speaker-audio gain net of the
speaker's visible motion. This harness scores, for every model, every seed, and the
two GT controls, FOUR channels on ONE common mask, all from one capture and all with
an identity-clustered bootstrap CI (see CHANNELS below):

    conditional  aud|mot : add speaker AUDIO over (own + speaker MOTION)   (= R-DGG, audio-specific)
    marginal     aud     : add speaker AUDIO over own                      (speaker motion NOT removed)
    motion       mot     : add speaker MOTION over own                     (the visible-motion channel)
    mot|aud              : add speaker MOTION over (own + speaker AUDIO)

Every channel re-uses the SAME validated net_of/decompose/boot_ci by relabelling the
per-stem energies (channel_view); e_own is constant across the time-shift nulls, so it
cancels in the net exactly as the own-past cancels in aud|mot.

The finding (two parts, full text in FINDING_TEXT): (1) the MOTION channel does NOT
discriminate -- its point estimates are large and positive for every column (surrogate,
talking-heads, GT, dyadic alike) with a scrambled ordering (talking-heads >= some dyadic
seeds), AND its identity CIs are wide so nothing is significant (GT p=0.056), so reading
reactivity off visible motion mis-ranks and certifies nothing. Only the AUDIO channel
separates dyadic/GT (sig.) from the controls (~0), which is why the metric is audio-
specific BY DEFINITION, not a bolt-on. (2) aud|mot == marginal aud to |delta|<=0.03 on
SI-184, because audio and motion are near-orthogonal in the reaction, so conditioning is a
no-op for the number here (robustness). The pre-registered prediction (marginal inflates a
control) was NOT observed. Baseline aud|mot kept (rubric rule 4). See 13.1.

Run in the shared metrics env, as pipeline/4_rdgg.sh does:
    pixi run --manifest-path rdgg/pixi.toml python rdgg/motion_cond_ablation.py
A 22 GB GPU needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _config
from _config import SOURCES, SURROGATE, _ident_key
from _features import collect_design, gridded_face_ok, load_all_motion
from _granger import capture_energies
from compute import load_corpus
from dose_curve import build_context_and_operator, capture_column, intersect_masks
from shift_ablation import (
    DYADIC_SEEDS,
    SEED_TAG,
    TALKING,
    build_surrogate_cache,
    score,
    seed_avg_ci,
    stash_energies,
)


# The four channels this ablation reports. Each is the SAME net_of/decompose/boot_ci
# applied to a relabelling of the per-stem energies captured with marginal_audio=True.
# gain = log(mr/br) - floor; a channel reads "the extra gain of adding X over the baseline":
#   conditional aud|mot = add speaker AUDIO over (own + speaker MOTION)   = R-DGG
#   marginal    aud     = add speaker AUDIO over own                      (motion NOT removed)
#   motion      mot     = add speaker MOTION over own                     (the visible-motion channel)
#   mot|aud             = add speaker MOTION over (own + speaker AUDIO)
# "OWN" means the own-block energy e_own, constant across the time-shift nulls (the own block
# never shifts), so it cancels in the net exactly as the own-past cancels in aud|mot.
CHANNELS = {
    # json_key: (display, (br, mr, bn, mn))
    "conditional":        ("aud|mot (R-DGG)",       ("both_r", "mot_r", "both_n", "mot_n")),
    "marginal":           ("aud (no motion cond.)", ("aud_r",  "own_r", "aud_n",  "OWN")),
    "motion":             ("mot (motion only)",     ("mot_r",  "own_r", "mot_n",  "OWN")),
    "motion_given_audio": ("mot|aud",               ("both_r", "aud_r", "both_n", "aud_n")),
}


def channel_view(E, spec):
    """Relabel per-stem energies into one channel's (baseline vs baseline+driver) contrast,
    so the SAME validated net_of/decompose/boot_ci compute its net. Requires
    capture_energies(marginal_audio=True), which supplies aud_r/aud_n/own_r."""
    br, mr, bn, mn = spec
    out = {}
    for s, d in E.items():
        # size the constant OWN null stack from the PAIRED null array, so a future
        # mismatch in null counts fails loud here, not as an opaque broadcast later
        mn_arr = np.full(len(d[bn]), d["own_r"], dtype=np.float64) if mn == "OWN" else d[mn]
        out[s] = {"both_r": d[br], "mot_r": d[mr], "both_n": d[bn],
                  "mot_n": mn_arr, "nrows": d["nrows"]}
    return out


# JSON metadata, as module constants so the GPU script and the offline rebuilder write ONE text.
CHANNELS_META = {
    "conditional": "aud|mot = add speaker AUDIO over (own + speaker MOTION) -- R-DGG, audio-specific",
    "marginal": "aud = add speaker AUDIO over own -- speaker motion NOT partialled out",
    "motion": "mot = add speaker MOTION over own -- the visible-motion channel",
    "motion_given_audio": "mot|aud = add speaker MOTION over (own + speaker AUDIO)",
    "ci": "identity-clustered bootstrap, one-sided p=P(draw<=0), same as the leaderboard",
}

FINDING_TEXT = (
    "Two results. (1) The MOTION channel does NOT discriminate and is unusable as a reactivity "
    "signal, two ways. By point estimate it is large and positive for EVERY column -- surrogate "
    "+0.34, talking-heads +0.35..+0.78, GT +0.92, dyadic +0.37..+1.19 (x1e4) -- and the ordering "
    "is scrambled: talking-heads score as high as or above genuine dyadic seeds (e.g. SoulX Pro "
    "+0.78 > AVTR-1 seed1 +0.40), so motion magnitude does not rank real listeners above non-"
    "reactive ones. By significance its identity CIs are wide and NOTHING is significant, GT "
    "included (p=0.056); surrogate p=0.12, talking-heads p>=0.08. So a metric read off visible "
    "motion would both mis-rank and certify nothing. Only the AUDIO channel separates: GT (p=0.003) "
    "and the dyadic models >0, surrogate and talking-heads ~0 and n.s. The conditional aud|mot IS "
    "the audio-specific definition, not a bolt-on. (2) On SI-184 aud|mot == marginal aud "
    "(|delta|<=0.03): audio and motion are near-orthogonal in the reaction (V(both)~V(aud)+V(mot), "
    "seen in aud|mot~=aud AND mot|aud~=mot), so partialling motion out is a no-op for the NUMBER "
    "here -- robustness, and free insurance where audio and motion are collinear. The pre-registered "
    "prediction (marginal inflates a control) was NOT observed; the justification is the motion "
    "channel's non-discrimination. Baseline aud|mot kept (rubric rule 4). See "
    "CVPR_zeroother_design.md 13.1."
)


def main():
    ap = argparse.ArgumentParser(description="R-DGG motion-conditioning ablation (standalone).")
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
    print(f"  [phase1] operator built {time.time() - t0:.1f}s; NFOLD={corpus.NFOLD}", flush=True)

    # ONE shared common mask = GT-reactive & every model-seed column's face_ok (surrogate
    # excluded from the intersection, as in the leaderboard). Identical frames for every column.
    all_dirs = [d for dirs in DYADIC_SEEDS.values() for d in dirs] + list(TALKING.values())
    eval_mask = mask_gtonly
    for d in all_dirs:
        fok = gridded_face_ok(load_all_motion(f"{C}/{d}/{S}_{suffix}", source, corpus), corpus)
        eval_mask = intersect_masks(eval_mask, fok)
    n_fr = sum(int(v.sum()) for v in eval_mask.values())
    print(f"  [common-mask] {len(eval_mask)} stems, {n_fr} grid frames valid in GT & all "
          f"{len(all_dirs)} model-seed columns", flush=True)

    columns = {}      # label -> {model, group, seed, <each channel>: rec}
    npz = {}          # per-stem energies (per channel) for the offline bootstrap
    order = []        # presentation order
    E_keep = {}       # dyadic: {label: {channel: relabelled E}} for the seed-average + CI

    def add(label, model, group, seed_tag, E, own_r2):
        rec = {"model": model, "group": group, "seed": seed_tag}
        keep = {}
        for ck, (_disp, spec) in CHANNELS.items():
            Ev = channel_view(E, spec)
            rec[ck], _ = score(Ev, own_r2)
            stash_energies(npz, f"{label}::{ck}", Ev, sorted(Ev))
            if group == "dyadic":
                keep[ck] = Ev
        stems = sorted(E)
        npz[f"{label}::stems"] = np.array(stems)
        npz[f"{label}::ident"] = np.array([_ident_key(s) for s in stems])
        columns[label] = rec
        if group == "dyadic":
            E_keep[label] = keep
        order.append(label)
        print(f"  [{label:22s}] aud|mot={rec['conditional']['net']:+.3f} "
              f"aud={rec['marginal']['net']:+.3f} | mot={rec['motion']['net']:+.3f} "
              f"mot|aud={rec['motion_given_audio']['net']:+.3f}", flush=True)

    # --- GT positive control ---
    print("\n  [controls] GT (positive) + GT*other surrogate (negative) ...", flush=True)
    E_gt, own_gt = capture_energies(collect_design(gt_cache, ctx, eval_mask), ops, ctx, marginal_audio=True)
    add("GT", "GT", "GT-control", "GT", E_gt, own_gt)

    # --- GT*other surrogate negative control ---
    surr = build_surrogate_cache(gt_cache, corpus)
    E_sur, own_sur = capture_energies(collect_design(surr, ctx, eval_mask), ops, ctx, marginal_audio=True)
    add(SURROGATE, SURROGATE, "GT-control", "surrogate", E_sur, own_sur)

    # --- dyadic models, every seed ---
    print("\n  [dyadic] AVTR-1 / DyStream / AvatarForcing, 3 seeds each ...", flush=True)
    for model, dirs in DYADIC_SEEDS.items():
        for i, d in enumerate(dirs):
            E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", eval_mask, ctx, ops, corpus, source,
                                         marginal_audio=True)
            add(f"{model} [{SEED_TAG[i]}]", model, "dyadic", SEED_TAG[i], E, r2)

    # --- talking-head models (non-dyadic model-side controls), one seed each ---
    print("\n  [non-dyadic] talking-head models, one seed each ...", flush=True)
    for model, d in TALKING.items():
        E, _per, r2 = capture_column(f"{C}/{d}/{S}_{suffix}", eval_mask, ctx, ops, corpus, source,
                                     marginal_audio=True)
        add(model, model, "non-dyad", "single", E, r2)

    # --- seed-averaged dyadic summary: average each stem's energies over the 3 seeds, then the
    # EXACT identity bootstrap -> seed-averaged net + CI, for BOTH channels. ---
    seed_avg = {}
    for model, dirs in DYADIC_SEEDS.items():
        labels = [f"{model} [{SEED_TAG[i]}]" for i in range(len(dirs))]
        rec = {"seeds": labels, "n_seeds": len(labels)}
        for ck in CHANNELS:
            ci = seed_avg_ci([E_keep[l][ck] for l in labels], corrected=True)
            nets = np.array([columns[l][ck]["net"] for l in labels])
            rec[ck] = {**ci, "per_seed_net": [round(float(x), 4) for x in nets],
                       "per_seed_sd": round(float(nets.std(ddof=1)), 4)}
        seed_avg[model] = rec
        c, m = rec["conditional"], rec["motion"]
        print(f"  [seed-avg {model:13s}] aud|mot={c['point']:+.3f} p={c['p']:.3f} | "
              f"mot={m['point']:+.3f} p={m['p']:.3f}", flush=True)

    outdir = f"{C}/GT/_metrics"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "metric": "R-DGG motion-conditioning ablation",
        "question": "is partialling out the speaker motion (aud|mot vs marginal aud) needed?",
        "source": source, "suffix": suffix, "split": S, "n_folds": corpus.NFOLD,
        "mask": ("ONE shared mask = GT-reactive & every model-seed column's face_ok; all columns "
                 "score on IDENTICAL frames (surrogate excluded from the intersection, like the "
                 "leaderboard; its seam-blinding then applies on top)"),
        "channels": CHANNELS_META,
        "finding": FINDING_TEXT,
        "controls": {
            "GT": "held-out positive control (must react, net>0)",
            SURROGATE: "GT motion under a wrong pairing (negative control, must read ~0)",
            "non-dyad": "talking-head models with real audio but no dyadic reaction (must read ~0)",
        },
        "order": order,
        "columns": columns,
        "seed_averaged_dyadic": seed_avg,
    }
    jpath = f"{outdir}/motion_cond_ablation__{S}.json"
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    npz_path = f"{outdir}/motion_cond_ablation__{S}.npz"
    np.savez_compressed(npz_path, **npz)
    print(f"\n  [out] {jpath}\n        {npz_path}", flush=True)
    print("MOTION_COND_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
