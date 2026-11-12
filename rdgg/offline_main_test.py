"""OFFLINE main statistical test (plan task 4 / list item 12): ONE pre-registered contrast
   C = mean R-DGG net over the 3 dyadic systems (seed-averaged) - mean net over the 4
   talking-head systems, with a single identity-cluster bootstrap in which ALL systems share
   the same resampled clusters. Separate systems stay descriptive; NO multiplicity correction
   (user decision; xlsx task 4).

Pure numpy, reads the committed shift_ablation energies (per-stem both_r/mot_r/both_n/mot_n on
the ONE common mask). No GPU, no metric import. Self-verifies the per-system seed-avg nets
against shift_ablation.json.

  python3 scratchpad/offline_main_test.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data/GT/_metrics/shift_ablation__LS_AL.npz")
OUT_JSON = os.path.join(ROOT, "data/GT/_metrics/main_contrast__LS_AL.json")
B = 20000  # same as metric BOOTSTRAP

DYADIC = {
    "AVTR-1": ["AVTR-1 [unseeded]", "AVTR-1 [seed0]", "AVTR-1 [seed1]"],
    "DyStream": ["DyStream [unseeded]", "DyStream [seed0]", "DyStream [seed1]"],
    "AvatarForcing": ["AvatarForcing [unseeded]", "AvatarForcing [seed0]", "AvatarForcing [seed1]"],
}
TALKING = ["SoulX Lite", "SoulX Pro", "Ditto", "FLOAT"]


def load():
    d = np.load(NPZ, allow_pickle=True)
    labels = sorted(set(k.split("::")[0] for k in d.keys()))
    cols = {}
    for l in labels:
        cols[l] = {
            "stems": list(d[f"{l}::stems"]),
            "ident": list(d[f"{l}::ident"]),
            "both_r": d[f"{l}::both_r"], "mot_r": d[f"{l}::mot_r"],
            "both_n": d[f"{l}::both_n"], "mot_n": d[f"{l}::mot_n"],  # (K, n_stems)
        }
    return cols


def common_order(cols, labels):
    """Shared stems across the given labels, in a fixed sorted order; returns (stems, ident)."""
    sets = [set(cols[l]["stems"]) for l in labels]
    stems = sorted(set.intersection(*sets))
    # ident is stem-derived and identical across columns; take from the first
    ref = cols[labels[0]]
    pos = {s: i for i, s in enumerate(ref["stems"])}
    ident = [ref["ident"][pos[s]] for s in stems]
    return stems, ident


def column_energy(cols, label_or_list, stems):
    """Per-stem (both_r, mot_r, both_n[K], mot_n[K]) aligned to `stems`. A list -> seed-average
    the per-stem energies over the seed columns FIRST (RESUME 12), then return the averaged energy."""
    labels = [label_or_list] if isinstance(label_or_list, str) else label_or_list
    br = mr = bn = mn = None
    for l in labels:
        c = cols[l]
        pos = {s: i for i, s in enumerate(c["stems"])}
        idx = [pos[s] for s in stems]
        e_br = c["both_r"][idx]; e_mr = c["mot_r"][idx]
        e_bn = c["both_n"][:, idx]; e_mn = c["mot_n"][:, idx]
        if br is None:
            br, mr, bn, mn = e_br.copy(), e_mr.copy(), e_bn.copy(), e_mn.copy()
        else:
            br += e_br; mr += e_mr; bn += e_bn; mn += e_mn
    n = len(labels)
    return br / n, mr / n, bn / n, mn / n


def cluster_sums(br, mr, bn, mn, ident):
    """Aggregate per-stem energies to per-cluster sums. Returns (BR[NC], MR[NC], BN[K,NC], MN[K,NC])."""
    clusters = list(dict.fromkeys(ident))
    cidx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    cl = np.array([cidx[i] for i in ident])
    K = bn.shape[0]
    BR = np.zeros(NC); MR = np.zeros(NC); BN = np.zeros((K, NC)); MN = np.zeros((K, NC))
    np.add.at(BR, cl, br); np.add.at(MR, cl, mr)
    for k in range(K):
        np.add.at(BN[k], cl, bn[k]); np.add.at(MN[k], cl, mn[k])
    return (BR, MR, BN, MN), NC


def net_of(sel, S):
    BR, MR, BN, MN = S
    return np.log(MR[sel].sum() / BR[sel].sum()) - np.log(MN[:, sel].sum(1) / BN[:, sel].sum(1)).mean()


def main():
    cols = load()
    all_final = [l for dirs in DYADIC.values() for l in dirs] + TALKING
    stems, ident = common_order(cols, all_final)
    NC = len(dict.fromkeys(ident))
    print(f"  common stems {len(stems)}, identities {NC}", flush=True)

    # per-system cluster sums (dyadic seed-averaged; talking-head single)
    S_dyadic = {}
    for m, dirs in DYADIC.items():
        br, mr, bn, mn = column_energy(cols, dirs, stems)
        S_dyadic[m], _ = cluster_sums(br, mr, bn, mn, ident)
    S_th = {}
    for m in TALKING:
        br, mr, bn, mn = column_energy(cols, m, stems)
        S_th[m], _ = cluster_sums(br, mr, bn, mn, ident)

    # self-verify: per-system seed-avg / single nets vs shift_ablation.json
    ref = json.load(open(os.path.join(ROOT, "data/GT/_metrics/shift_ablation__LS_AL.json")))
    full = np.arange(NC)
    print("  self-verify (recomputed vs committed):", flush=True)
    ok = True
    for m in DYADIC:
        got = net_of(full, S_dyadic[m]) * 1e4
        want = ref["seed_averaged_dyadic"][m]["net"]["point"]
        d = abs(got - want); ok &= d < 0.02
        print(f"    {m:14s} seed-avg net {got:+.4f} vs {want:+.4f}  |d|={d:.4f}", flush=True)
    for m in TALKING:
        got = net_of(full, S_th[m]) * 1e4
        want = ref["columns"][m]["net"]
        d = abs(got - want); ok &= d < 0.02
        print(f"    {m:14s} net          {got:+.4f} vs {want:+.4f}  |d|={d:.4f}", flush=True)
    if not ok:
        print("  !! SELF-VERIFY FAILED (recomputed nets differ from committed) -- aborting", flush=True)
        sys.exit(1)
    print("  self-verify OK", flush=True)

    # the contrast on a cluster selection `sel`
    def contrast(sel):
        dy = np.mean([net_of(sel, S_dyadic[m]) for m in DYADIC])
        th = np.mean([net_of(sel, S_th[m]) for m in TALKING])
        return (dy - th) * 1e4

    point = float(contrast(full))
    rng = np.random.default_rng(0)
    draws = np.array([contrast(rng.integers(0, NC, NC)) for _ in range(B)])
    lo, p50, hi = np.percentile(draws, [2.5, 50, 97.5])
    p_one = float((draws <= 0).mean())
    mean_dy = float(np.mean([net_of(full, S_dyadic[m]) for m in DYADIC]) * 1e4)
    mean_th = float(np.mean([net_of(full, S_th[m]) for m in TALKING]) * 1e4)

    print(f"\n  MAIN CONTRAST (dyadic - talking-head): {point:+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}] one-sided p={p_one:.4f}", flush=True)
    print(f"    mean dyadic net {mean_dy:+.4f} | mean talking-head net {mean_th:+.4f}", flush=True)

    result = {
        "metric": "R-DGG main pre-registered contrast (plan task 4)",
        "definition": ("C = mean net over 3 dyadic (seed-averaged) - mean net over 4 talking-head; "
                       "ONE identity-cluster bootstrap, all systems share the resampled clusters; "
                       "one-sided p = P(draw <= 0); NO multiplicity correction (separate systems descriptive)"),
        "n_stems": len(stems), "n_identities": NC, "bootstrap": B,
        "dyadic": list(DYADIC), "talking_head": TALKING,
        "mean_dyadic_net": round(mean_dy, 4), "mean_talking_head_net": round(mean_th, 4),
        "contrast": {"point": round(point, 4), "lo": round(float(lo), 4), "p50": round(float(p50), 4),
                     "hi": round(float(hi), 4), "p_one_sided": round(p_one, 4)},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [out] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
