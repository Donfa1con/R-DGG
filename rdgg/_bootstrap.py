"""Identity-clustered bootstrap for R-DGG: the point gain, its CI, and the paired 8x8 cell statistic.

Pure numpy over the per-listener energies that _granger captured.
"""
import numpy as np

from _config import _ident_key, _intr_key


# ---- aud|mot net from captured energies (pure numpy; own cancels: both - motion_only = log(mot/both)) ----
def cluster_sums(E, stems, cl_of, NC):
    """Sum each stem's four energies into its cluster row. cl_of: stem -> cluster index in [0, NC)."""
    K = len(next(iter(E.values()))["both_n"])
    br = np.zeros(NC); mr = np.zeros(NC); bn = np.zeros((K, NC)); mn = np.zeros((K, NC))
    for st in stems:
        c = cl_of[st]; d = E[st]
        br[c] += d["both_r"]; mr[c] += d["mot_r"]; bn[:, c] += d["both_n"]; mn[:, c] += d["mot_n"]
    return br, mr, bn, mn


def gnet(sel, sums):
    """aud|mot net (x1) over the clusters `sel`: log(mot/both)_real - mean_k log(mot/both)_null_k."""
    br, mr, bn, mn = sums
    BR = br[sel].sum(); MR = mr[sel].sum(); BK = bn[:, sel].sum(1); MK = mn[:, sel].sum(1)
    return np.log(MR / BR) - np.log(MK / BK).mean()


def gnet_boot_ci(E, stems, key_of, B, one_sided):
    """Point + cluster-bootstrap CI (x1e4) of the pooled aud|mot net; key_of(stem) -> cluster key. one_sided p =
    P(draw <= 0) (leaderboard 'reacts > 0'). Point == gnet over all clusters == raw - mean_k floor_k."""
    clusters = list(dict.fromkeys(key_of(s) for s in stems))
    idx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    sums = cluster_sums(E, stems, {s: idx[key_of(s)] for s in stems}, NC)
    point = float(gnet(np.arange(NC), sums)) * 1e4
    rng = np.random.default_rng(0)
    d = np.array([gnet(rng.integers(0, NC, size=NC), sums) for _ in range(B)]) * 1e4
    lo, p50, hi = np.percentile(d, [2.5, 50, 97.5])
    p = float((d <= 0).mean()) if one_sided else float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"point": point, "lo": float(lo), "p50": float(p50), "hi": float(hi), "p": p, "n_clusters": NC}


def leaderboard_stats(E, own_r2, B):
    """Everything the leaderboard prints for one model, from its captured energies: the aud|mot net with its
    decomposition (raw / floor), the identity- and interaction-clustered CIs, and per-video / per-interaction nets."""
    stems = list(E)
    BR = sum(E[s]["both_r"] for s in stems)
    MR = sum(E[s]["mot_r"] for s in stems)
    BN = np.sum([E[s]["both_n"] for s in stems], 0)
    MN = np.sum([E[s]["mot_n"] for s in stems], 0)
    raw = np.log(MR / BR)
    floor_k = np.log(MN / BN)                              # (K,)
    per_net = np.array([np.log(E[s]["mot_r"] / E[s]["both_r"]) - np.log(E[s]["mot_n"] / E[s]["both_n"]).mean()
                        for s in stems])
    by_intr = {}
    for s, v in zip(stems, per_net):
        by_intr.setdefault(_intr_key(s), []).append(v)
    return {
        "net": float(raw - floor_k.mean()) * 1e4,
        "net_med": float(raw - np.median(floor_k)) * 1e4,
        "raw": float(raw) * 1e4,
        "floor": float(floor_k.mean()) * 1e4,
        "floor_sd": float(floor_k.std()) * 1e4,
        "ci_ident": gnet_boot_ci(E, stems, _ident_key, B, one_sided=True),
        "ci_intr": gnet_boot_ci(E, stems, _intr_key, B, one_sided=True),
        "dyad": per_net,
        "intr": np.array([float(np.mean(v)) for v in by_intr.values()]),
        "own_r2": own_r2,
        "n_lis": len(stems),
    }


def pair_stats(EA, EB, B):
    """One 8x8 cell: aud|mot-net difference G_A - G_B on the listeners GT & A & B tracked, identity-cluster
    paired-bootstrapped -- the SAME cluster resample is applied to A and B so their shared between-subject
    variance cancels (more powerful than comparing the two marginal CIs)."""
    common = sorted(set(EA) & set(EB))
    clusters = list(dict.fromkeys(_ident_key(s) for s in common))
    idx = {c: i for i, c in enumerate(clusters)}
    NC = len(clusters)
    cl_of = {s: idx[_ident_key(s)] for s in common}
    Asum = cluster_sums(EA, common, cl_of, NC)
    Bsum = cluster_sums(EB, common, cl_of, NC)
    gA = float(gnet(np.arange(NC), Asum)) * 1e4
    gB = float(gnet(np.arange(NC), Bsum)) * 1e4
    rng = np.random.default_rng(0)
    d = np.empty(B)
    for b in range(B):
        sel = rng.integers(0, NC, size=NC)                 # same resample for A and B -> paired within cell
        d[b] = (gnet(sel, Asum) - gnet(sel, Bsum)) * 1e4
    lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
    return {"delta": round(gA - gB, 3), "lo": round(float(lo), 3), "p50": round(float(med), 3),
            "hi": round(float(hi), 3), "p": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 3),
            "sig": bool(lo > 0 or hi < 0), "n_listeners": len(common), "n_clusters": NC}


def t_stat(arr):
    sem = float(arr.std()) / np.sqrt(max(1, len(arr)))
    return float(arr.mean()) / (sem if sem else 1)
