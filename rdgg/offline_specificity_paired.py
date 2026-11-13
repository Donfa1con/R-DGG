"""OFFLINE paired specificity test: Delta = aligned R-DGG net - (model x other) net.

The model x other JSON reports, per system, the scrambled net with a CI and a p-value.
That p-value tests the scrambled net against ZERO. It does NOT test the drop from the
aligned score. To claim that a wrong pairing SIGNIFICANTLY reduces R-DGG, we need the
paired difference Delta = aligned - (x other) with its own interval.

This script computes Delta per system with ONE identity-cluster bootstrap in which the
SAME resampled clusters are applied to the aligned energies and to the scrambled energies,
so the shared between-subject variance cancels (the paired design in _bootstrap.pair_stats,
here across two separate runs on the ONE common mask).

  aligned energies  <- data/GT/_metrics/shift_ablation__LS_AL.npz   (dyadic seed-averaged)
  x other energies  <- data/GT/_metrics/model_x_other__LS_AL.npz    (dyadic seed-averaged)

Both runs use the SAME common mask and the SAME GT reference operator; the aligned GT
column stored in BOTH files agrees to float32 rounding (~1e-6 relative), which the script
re-checks before it trusts the pairing. Pure numpy, no GPU, no metric import.

  python3 rdgg/offline_specificity_paired.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFT_NPZ = os.path.join(ROOT, "data/GT/_metrics/shift_ablation__LS_AL.npz")
MODELX_NPZ = os.path.join(ROOT, "data/GT/_metrics/model_x_other__LS_AL.npz")
SHIFT_JSON = os.path.join(ROOT, "data/GT/_metrics/shift_ablation__LS_AL.json")
MODELX_JSON = os.path.join(ROOT, "data/GT/_metrics/model_x_other__LS_AL.json")
OUT_JSON = os.path.join(ROOT, "data/GT/_metrics/specificity_paired__LS_AL.json")
B = 20000  # same as the metric bootstrap

# (display, role, aligned labels in shift npz, x other labels in model_x_other npz)
SEEDS = ["unseeded", "seed0", "seed1"]
SYSTEMS = [
    ("GT", "positive", ["GT"], ["GT×other"]),
    ("AVTR-1", "dyadic", [f"AVTR-1 [{s}]" for s in SEEDS],
     [f"AVTR-1 x other [{s}]" for s in SEEDS]),
    ("DyStream", "dyadic", [f"DyStream [{s}]" for s in SEEDS],
     [f"DyStream x other [{s}]" for s in SEEDS]),
    ("AvatarForcing", "dyadic", [f"AvatarForcing [{s}]" for s in SEEDS],
     [f"AvatarForcing x other [{s}]" for s in SEEDS]),
    ("FLOAT", "talking-head", ["FLOAT"], ["FLOAT x other"]),
    ("SoulX Pro", "talking-head", ["SoulX Pro"], ["SoulX Pro x other"]),
    ("SoulX Lite", "talking-head", ["SoulX Lite"], ["SoulX Lite x other"]),
    ("Ditto", "talking-head", ["Ditto"], ["Ditto x other"]),
]


def load(npz):
    d = np.load(npz, allow_pickle=True)
    labels = sorted(set(k.split("::")[0] for k in d.files))
    cols = {}
    for l in labels:
        cols[l] = {
            "stems": list(d[f"{l}::stems"]), "ident": list(d[f"{l}::ident"]),
            "both_r": d[f"{l}::both_r"], "mot_r": d[f"{l}::mot_r"],
            "both_n": d[f"{l}::both_n"], "mot_n": d[f"{l}::mot_n"],  # (K, n_stems)
        }
    return cols


def seed_avg_energy(cols, labels, stems):
    """Per-stem (both_r, mot_r, both_n[K], mot_n[K]) aligned to `stems`, averaged over the
    seed labels FIRST (RESUME invariant 12), then returned."""
    br = mr = bn = mn = None
    for l in labels:
        c = cols[l]
        pos = {s: i for i, s in enumerate(c["stems"])}
        idx = [pos[s] for s in stems]
        e = (c["both_r"][idx], c["mot_r"][idx], c["both_n"][:, idx], c["mot_n"][:, idx])
        if br is None:
            br, mr, bn, mn = (x.copy() for x in e)
        else:
            br += e[0]; mr += e[1]; bn += e[2]; mn += e[3]
    n = len(labels)
    return br / n, mr / n, bn / n, mn / n


def cluster_sums(br, mr, bn, mn, ident):
    """Aggregate per-stem energies to per-cluster sums. Returns ((BR,MR,BN,MN), NC)."""
    clusters = list(dict.fromkeys(ident))
    cidx = {c: i for i, c in enumerate(clusters)}
    cl = np.array([cidx[i] for i in ident])
    NC = len(clusters)
    K = bn.shape[0]
    BR = np.zeros(NC); MR = np.zeros(NC); BN = np.zeros((K, NC)); MN = np.zeros((K, NC))
    np.add.at(BR, cl, br); np.add.at(MR, cl, mr)
    for k in range(K):
        np.add.at(BN[k], cl, bn[k]); np.add.at(MN[k], cl, mn[k])
    return (BR, MR, BN, MN), NC


def net_of(sel, S):
    BR, MR, BN, MN = S
    return np.log(MR[sel].sum() / BR[sel].sum()) - np.log(MN[:, sel].sum(1) / BN[:, sel].sum(1)).mean()


def common_stems_ident(align_cols, alabels, other_cols, olabels):
    """Stems present in every aligned and every x other label, in sorted order, with ident."""
    sets = [set(align_cols[l]["stems"]) for l in alabels] + [set(other_cols[l]["stems"]) for l in olabels]
    stems = sorted(set.intersection(*sets))
    ref = align_cols[alabels[0]]
    pos = {s: i for i, s in enumerate(ref["stems"])}
    ident = [ref["ident"][pos[s]] for s in stems]
    return stems, ident


def main():
    align = load(SHIFT_NPZ)
    other = load(MODELX_NPZ)
    shift_json = json.load(open(SHIFT_JSON))
    modelx_json = json.load(open(MODELX_JSON))

    # mask/operator check: aligned GT stored in BOTH npz must agree (same mask, same operator)
    a = dict(zip(align["GT"]["stems"], align["GT"]["both_r"]))
    b = dict(zip(other["GT (aligned)"]["stems"], other["GT (aligned)"]["both_r"]))
    rel = max(abs(a[s] - b[s]) / abs(a[s]) for s in set(a) & set(b))
    assert rel < 1e-4, f"aligned GT differs across runs (rel={rel:.2e}) -- masks/operators do not match"
    print(f"  mask/operator check OK: aligned GT agrees across runs (max rel diff {rel:.1e})", flush=True)

    rng = np.random.default_rng(0)
    results = {}
    print(f"\n  {'system':14s} {'aligned':>8s} {'xother':>8s} {'Delta':>8s}  {'95% CI':>18s}  {'p(1s)':>6s}", flush=True)
    for disp, role, alabels, olabels in SYSTEMS:
        stems, ident = common_stems_ident(align, alabels, other, olabels)
        Sa, NC = cluster_sums(*seed_avg_energy(align, alabels, stems), ident)
        So, _ = cluster_sums(*seed_avg_energy(other, olabels, stems), ident)
        full = np.arange(NC)
        a_net = float(net_of(full, Sa)) * 1e4
        o_net = float(net_of(full, So)) * 1e4
        point = a_net - o_net

        # ONE resample shared by aligned and x other -> paired difference
        rng = np.random.default_rng(0)
        d = np.empty(B)
        for i in range(B):
            sel = rng.integers(0, NC, size=NC)
            d[i] = (net_of(sel, Sa) - net_of(sel, So)) * 1e4
        lo, p50, hi = np.percentile(d, [2.5, 50, 97.5])
        p_one = float((d <= 0).mean())          # H1: aligned > x other (mismatch REDUCES R-DGG)
        p_two = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
        results[disp] = {
            "role": role, "n_stems": len(stems), "n_identities": int(NC),
            "aligned_net": round(a_net, 4), "x_other_net": round(o_net, 4),
            "delta": round(point, 4), "lo": round(float(lo), 4), "p50": round(float(p50), 4),
            "hi": round(float(hi), 4), "p_one_sided": round(p_one, 4), "p_two_sided": round(p_two, 4),
            "sig_reduces": bool(lo > 0),
        }
        print(f"  {disp:14s} {a_net:+8.3f} {o_net:+8.3f} {point:+8.3f}  "
              f"[{lo:+6.3f},{hi:+6.3f}]  {p_one:6.3f}  {'REDUCES*' if lo > 0 else ''}", flush=True)

    # self-verify aligned + x other marginals against the committed JSONs
    print("\n  self-verify (recomputed marginals vs committed):", flush=True)
    ok = True
    for disp, role, alabels, olabels in SYSTEMS:
        if role == "dyadic":
            wa = shift_json["seed_averaged_dyadic"][disp]["net"]["point"]
            wo = modelx_json["seed_averaged_dyadic"][disp]["net"]["point"]
        elif role == "positive":
            wa = shift_json["columns"]["GT"]["net"]
            wo = modelx_json["columns"]["GT×other"]["net"]
        else:
            wa = shift_json["columns"][disp]["net"]
            wo = modelx_json["columns"][f"{disp} x other"]["net"]
        da = abs(results[disp]["aligned_net"] - wa)
        do = abs(results[disp]["x_other_net"] - wo)
        ok &= da < 0.02 and do < 0.02
        print(f"    {disp:14s} aligned {results[disp]['aligned_net']:+.3f} vs {wa:+.3f} "
              f"|x other {results[disp]['x_other_net']:+.3f} vs {wo:+.3f}", flush=True)
    if not ok:
        print("  !! SELF-VERIFY FAILED -- aborting", flush=True)
        sys.exit(1)
    print("  self-verify OK", flush=True)

    # pooled dyadic specificity contrast: mean over the 3 dyadic systems of (aligned - x other),
    # ONE shared identity resample (the powerful, item-12-style group test; single systems are
    # underpowered -- RESUME item 14 / power analysis).
    DY = ["AVTR-1", "DyStream", "AvatarForcing"]
    sysmap = {d: (al, ol) for d, role, al, ol in SYSTEMS}
    # common stems across ALL dyadic aligned + x other labels
    allab = [l for d in DY for l in sysmap[d][0]]
    ollab = [l for d in DY for l in sysmap[d][1]]
    stems, ident = common_stems_ident(align, allab, other, ollab)
    Sa = {d: cluster_sums(*seed_avg_energy(align, sysmap[d][0], stems), ident)[0] for d in DY}
    So = {d: cluster_sums(*seed_avg_energy(other, sysmap[d][1], stems), ident)[0] for d in DY}
    NC = len(dict.fromkeys(ident))
    full = np.arange(NC)

    def group_delta(sel):
        return float(np.mean([(net_of(sel, Sa[d]) - net_of(sel, So[d])) for d in DY])) * 1e4

    g_point = group_delta(full)
    rng = np.random.default_rng(0)
    gd = np.array([group_delta(rng.integers(0, NC, NC)) for _ in range(B)])
    glo, gp50, ghi = np.percentile(gd, [2.5, 50, 97.5])
    gp_one = float((gd <= 0).mean())
    pooled = {
        "systems": DY, "n_stems": len(stems), "n_identities": int(NC),
        "delta": round(g_point, 4), "lo": round(float(glo), 4), "p50": round(float(gp50), 4),
        "hi": round(float(ghi), 4), "p_one_sided": round(gp_one, 4),
        "sig_reduces": bool(glo > 0),
    }
    print(f"\n  POOLED dyadic specificity (mean aligned - x other, 3 systems): "
          f"{g_point:+.4f} CI[{glo:+.4f},{ghi:+.4f}] one-sided p={gp_one:.4f} "
          f"{'REDUCES*' if glo > 0 else '(CI includes 0)'}", flush=True)

    out = {
        "metric": "R-DGG paired specificity test (aligned - model x other)",
        "definition": ("Delta = aligned net - (model x other) net per system, ONE identity-cluster "
                       "bootstrap with the SAME resample applied to aligned and x other (paired); "
                       "one-sided p = P(Delta <= 0) tests H1: a wrong pairing REDUCES R-DGG. "
                       "Dyadic systems are seed-averaged (energies averaged over 3 seeds first)."),
        "note": ("the model_x_other p-value tests x other vs ZERO; THIS test is the difference from "
                 "the aligned score, which is what the specificity claim requires."),
        "bootstrap": B, "systems": results, "pooled_dyadic": pooled,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\n  [out] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
