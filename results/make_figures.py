#!/usr/bin/env python3
"""Rebuild the paper figures from results/tables/*.csv (no GPU, no metric rerun).
    pixi run --manifest-path rdgg/pixi.toml python results/make_figures.py

Writes results/figures/*.pdf (vector). Colours = Okabe-Ito (colourblind-safe by
design; the dataviz validator needs node, which is absent here, so we use this
established CVD-safe set). All R-DGG values are the net directed gain x1e4.
Edit this file to restyle; the CSVs are the source of truth.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# Okabe-Ito (CVD-safe)
OK = {"green": "#009E73", "blue": "#0072B2", "orange": "#E69F00",
      "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999"}
INK, MUTE, GRID = "#222222", "#555555", "#DDDDDD"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTE, "axes.linewidth": 0.8,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE,
                     "ytick.color": MUTE, "savefig.bbox": "tight", "figure.dpi": 150})


def rows(name):
    with open(TABLES / name) as f:
        return [{k: v for k, v in r.items()} for r in csv.DictReader(f)]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


TYPE_C = {"GT": OK["green"], "dyadic": OK["blue"], "non-dyad": OK["orange"], "control": OK["purple"]}


def fig_leaderboard():
    d = [r for r in rows("leaderboard.csv")]
    d.sort(key=lambda r: fnum(r["net_x1e4"]))
    y = range(len(d))
    net = [fnum(r["net_x1e4"]) for r in d]
    lo = [fnum(r["net_x1e4"]) - fnum(r["ci_lo_x1e4"]) for r in d]
    hi = [fnum(r["ci_hi_x1e4"]) - fnum(r["net_x1e4"]) for r in d]
    cols = [TYPE_C.get(r["type"], OK["grey"]) for r in d]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axvline(0, color=MUTE, lw=1, zorder=1)
    ax.barh(list(y), net, color=cols, height=0.6, zorder=2)
    ax.errorbar(net, list(y), xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1,
                capsize=2.5, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["system"] for r in d])
    ax.set_xlabel("R-DGG net (aud|mot) ×1e4   — 95% identity CI")
    tidy(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_C.values()]
    ax.legend(handles, TYPE_C.keys(), frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGS / "fig_leaderboard.pdf")
    plt.close(fig)


def fig_ablation():
    d = rows("ablation_marginals.csv")
    labels = [r["column"] for r in d]
    net = [fnum(r["net_x1e4"]) for r in d]
    lo = [fnum(r["net_x1e4"]) - fnum(r["ci_lo_x1e4"]) for r in d]
    hi = [fnum(r["ci_hi_x1e4"]) - fnum(r["net_x1e4"]) for r in d]
    cond = {"unseeded": OK["blue"], "seed0": OK["blue"], "seed1": OK["blue"],
            "zeroother": OK["orange"], "cfg0": OK["vermillion"]}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(0, color=MUTE, lw=1)
    ax.axhline(0.70, color=OK["green"], lw=1, ls="--", label="GT anchor (0.70)")
    ax.bar(labels, net, color=[cond[c] for c in labels], width=0.62, zorder=2)
    ax.errorbar(labels, net, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1,
                capsize=3, zorder=3)
    ax.set_ylabel("R-DGG net ×1e4   — 95% identity CI")
    ax.set_title("AVTR-1 ablations (one common mask, n=178, 13 identities)", fontsize=9)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / "fig_ablation.pdf")
    plt.close(fig)


def fig_seed_variance():
    d = rows("seed_variance.csv")
    s = rows("seed_variance_summary.csv")[0]
    labels = [r["stock_run"] for r in d]
    net = [fnum(r["net_x1e4"]) for r in d]
    lo = [fnum(r["net_x1e4"]) - fnum(r["ci_lo_x1e4"]) for r in d]
    hi = [fnum(r["ci_hi_x1e4"]) - fnum(r["net_x1e4"]) for r in d]
    mean, sd = fnum(s["mean_x1e4"]), fnum(s["sd_sample_x1e4"])
    x = range(len(d))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.axhspan(mean - sd, mean + sd, color=OK["blue"], alpha=0.12,
               label=f"mean ± SD ({mean:.2f} ± {sd:.2f})")
    ax.axhline(mean, color=OK["blue"], lw=1, ls="--")
    ax.errorbar(list(x), net, yerr=[lo, hi], fmt="o", color=OK["blue"], ecolor=INK,
                elinewidth=1, capsize=3, markersize=6, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("R-DGG net ×1e4 (same model+audio, 3 noise seeds)")
    ax.set_title("Stock AVTR-1: seed-induced variation", fontsize=9)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / "fig_seed_variance.pdf")
    plt.close(fig)


# shift-correction ablation: uncorrected g_a_m (left) vs corrected net = R-DGG (right)
SHIFT_GROUP_C = {"GT-control": OK["green"], "dyadic": OK["blue"], "non-dyad": OK["orange"]}
SHIFT_LOWFID = set()   # all columns n~178 after the DyStream seed re-extraction (det_thresh 0.1)


def fig_shift_ablation(csv="shift_ablation.csv", out="fig_shift_ablation",
                       footnote="All columns on ONE shared mask (identical frames): GT-reactive & every "
                                "model-seed column's face_ok. n≈177, 13 identities.",
                       xlim_left=(2.6, 5.1), floorband=(3.3, 3.5), xlim_right=(-0.7, 1.8)):
    d = rows(csv)
    recs, labels = [], []                                  # insert a spacer between groups
    prev = None
    for r in d:
        if prev is not None and r["group"] != prev:
            recs.append(None); labels.append("")
        recs.append(r); labels.append(r["column"]); prev = r["group"]
    y = list(range(len(recs)))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 6.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})

    def draw(ax, vk, lok, hik, xlim, hollow):
        ax.axvline(0, color=MUTE, lw=1, zorder=1)
        for yi, r in zip(y, recs):
            if r is None:
                continue
            gc = SHIFT_GROUP_C.get(r["group"], OK["grey"])
            v, lo, hi = fnum(r[vk]), fnum(r[lok]), fnum(r[hik])
            low = r["column"] in SHIFT_LOWFID
            if v > xlim[1]:                                # off-scale: clip to edge + label
                ax.plot(xlim[1], yi, marker=">", ms=7, color=gc,
                        alpha=(0.55 if low else 1.0), clip_on=False, zorder=3)
                ax.text(xlim[1] - 0.03, yi + 0.42, f"$g_{{a\\,m}}$={v:.1f}", ha="right",
                        va="bottom", fontsize=6.8, color=gc)
                continue
            kw = dict(mfc="white", mec=gc, mew=1.4) if hollow else dict(mfc=gc, mec=gc)
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", ms=6,
                        ecolor=(gc if hollow else INK), elinewidth=1, capsize=2,
                        alpha=(0.55 if low else 1.0), zorder=3, **kw)
        ax.set_xlim(*xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    draw(axL, "g_a_m_x1e4", "raw_ci_lo_x1e4", "raw_ci_hi_x1e4", xlim_left, hollow=True)
    axL.set_xlabel("uncorrected $g_{a\\,m}$ ×1e4   (95% identity CI)")
    axL.set_title("WITHOUT shift correction", fontsize=9.5)
    axL.axvspan(floorband[0], floorband[1], color=OK["grey"], alpha=0.12, zorder=0)
    axL.text((floorband[0] + floorband[1]) / 2, len(recs) - 0.4, "common\nfloor",
             ha="center", va="top", fontsize=7, color=MUTE)

    gt_net = next((fnum(r["net_x1e4"]) for r in d if r["column"] == "GT"), None)
    draw(axR, "net_x1e4", "net_ci_lo_x1e4", "net_ci_hi_x1e4", xlim_right, hollow=False)
    if gt_net is not None:
        axR.axvline(gt_net, color=OK["green"], lw=1, ls="--", zorder=1)
    axR.set_xlabel("corrected net = $g_{a\\,m}-g_{shift}$ ×1e4   (R-DGG)")
    axR.set_title("WITH shift correction (R-DGG)", fontsize=9.5)

    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    handles = [Line2D([0], [0], marker="s", mfc=c, mec=c, ls="none", ms=7, label=g)
               for g, c in SHIFT_GROUP_C.items()]
    handles += [
        Line2D([0], [0], marker="o", mfc="white", mec=INK, mew=1.4, ls="none", ms=6, label="uncorrected (left)"),
        Line2D([0], [0], marker="o", mfc=INK, mec=INK, ls="none", ms=6, label="corrected (right)"),
        Line2D([0], [0], color=OK["green"], ls="--", lw=1, label="GT net anchor"),
    ]
    axR.legend(handles=handles, frameon=False, fontsize=7.3, loc="lower right")
    fig.suptitle("Is the time-shift correction needed? R-DGG ablation — all models, all seeds, GT controls",
                 fontsize=10.5, y=0.98)
    fig.text(0.5, 0.005, footnote, ha="center", fontsize=7, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


def fig_operator_ablation(csv="operator_ablation.csv", out="fig_operator_ablation",
                          footnote="All columns on ONE shared mask (identical frames): GT-reactive & every "
                                   "model-seed column's face_ok. n≈177, 13 identities. B_self is a "
                                   "leave-one-identity-fold cross-fit on each column's own reaction.",
                          xlim=(-0.7, 1.8)):
    # reshape: one record per column, carrying both operators ------------------
    d = rows(csv)
    order, bycol = [], {}
    for r in d:
        c = r["column"]
        if c not in bycol:
            order.append(c)
            bycol[c] = {"group": r["group"]}
        bycol[c][r["operator"]] = r
    recs, labels, prev = [], [], None
    for c in order:
        g = bycol[c]["group"]
        if prev is not None and g != prev:
            recs.append(None); labels.append("")
        recs.append(bycol[c]); labels.append(c); prev = g
    y = list(range(len(recs)))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 6.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})

    def draw(ax, oper, hollow):
        ax.axvline(0, color=MUTE, lw=1, zorder=1)
        for yi, rc in zip(y, recs):
            if rc is None:
                continue
            o = rc.get(oper)
            gc = SHIFT_GROUP_C.get(rc["group"], OK["grey"])
            if o is None or fnum(o["net_x1e4"]) is None:      # self operator not fit
                ax.plot(xlim[0] + 0.04, yi, marker="x", ms=6, color=OK["grey"],
                        clip_on=False, zorder=3)
                continue
            v, lo, hi = fnum(o["net_x1e4"]), fnum(o["net_ci_lo_x1e4"]), fnum(o["net_ci_hi_x1e4"])
            if v > xlim[1]:                                   # off-scale: clip + label
                ax.plot(xlim[1], yi, marker=">", ms=7, color=gc, clip_on=False, zorder=3)
                ax.text(xlim[1] - 0.03, yi + 0.42, f"net={v:.1f}", ha="right",
                        va="bottom", fontsize=6.8, color=gc)
                continue
            kw = dict(mfc="white", mec=gc, mew=1.4) if hollow else dict(mfc=gc, mec=gc)
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", ms=6,
                        ecolor=(gc if hollow else INK), elinewidth=1, capsize=2, zorder=3, **kw)
        ax.set_xlim(*xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    gt_net = fnum(bycol.get("GT", {}).get("reference", {}).get("net_x1e4"))
    draw(axL, "reference", hollow=False)
    if gt_net is not None:
        axL.axvline(gt_net, color=OK["green"], lw=1, ls="--", zorder=1)
    axL.set_xlabel("net with GT reference operator $B_{GT}$ ×1e4   (R-DGG)")
    axL.set_title("REFERENCE operator (R-DGG)", fontsize=9.5)

    draw(axR, "self", hollow=True)
    if gt_net is not None:
        axR.axvline(gt_net, color=OK["green"], lw=1, ls="--", zorder=1)
    axR.set_xlabel("net with each column's OWN operator $B_{self}$ ×1e4")
    axR.set_title("SELF operator (cross-fit on own reaction)", fontsize=9.5)

    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    handles = [Line2D([0], [0], marker="s", mfc=c, mec=c, ls="none", ms=7, label=g)
               for g, c in SHIFT_GROUP_C.items()]
    handles += [
        Line2D([0], [0], marker="o", mfc=INK, mec=INK, ls="none", ms=6, label="reference $B_{GT}$ (left)"),
        Line2D([0], [0], marker="o", mfc="white", mec=INK, mew=1.4, ls="none", ms=6, label="self $B_{self}$ (right)"),
        Line2D([0], [0], marker=">", color=INK, ls="none", ms=7, label="off scale (see label)"),
        Line2D([0], [0], color=OK["green"], ls="--", lw=1, label="GT net anchor"),
    ]
    axR.legend(handles=handles, frameon=False, fontsize=7.0, loc="lower right")
    fig.suptitle("Reference vs self operator: does R-DGG need the GT reference projection?",
                 fontsize=10.5, y=0.98)
    fig.text(0.5, 0.005, footnote, ha="center", fontsize=6.6, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


def fig_motion_cond_ablation(csv="motion_cond_ablation.csv", out="fig_motion_cond_ablation",
                             footnote="All columns on ONE shared mask (identical frames): GT-reactive & every "
                                      "model-seed column's face_ok. n≈177, 13 identities. Main point: the AUDIO "
                                      "channel separates dyadic/GT (>0) from the controls (~0); the MOTION channel "
                                      "is positive for EVERY column, controls included — so reactivity must be read "
                                      "off audio, net of motion. Conditioning on motion (filled vs hollow, left) "
                                      "barely moves the audio net (|Δ|≤0.03): audio ⊥ motion here.",
                             xlim=(-0.8, 1.9), dy=0.15):
    # one record per column, carrying all channels
    d = rows(csv)
    order, bycol = [], {}
    for r in d:
        c = r["column"]
        if c not in bycol:
            order.append(c); bycol[c] = {"group": r["group"]}
        bycol[c][r["channel"]] = r
    recs, labels, prev = [], [], None
    for c in order:
        g = bycol[c]["group"]
        if prev is not None and g != prev:
            recs.append(None); labels.append("")
        recs.append(bycol[c]); labels.append(c); prev = g
    y = list(range(len(recs)))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 6.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})

    def mark(ax, yi, r, hollow, off):
        v, lo, hi = fnum(r["net_x1e4"]), fnum(r["net_ci_lo_x1e4"]), fnum(r["net_ci_hi_x1e4"])
        gc = SHIFT_GROUP_C.get(r["group"], OK["grey"])
        kw = dict(mfc="white", mec=gc, mew=1.4) if hollow else dict(mfc=gc, mec=gc)
        ax.errorbar(v, yi + off, xerr=[[v - lo], [hi - v]], fmt="o", ms=5.5,
                    ecolor=(gc if hollow else INK), elinewidth=0.9, capsize=2, zorder=3, **kw)

    def frame(ax):
        ax.set_xlim(*xlim)
        ax.axvline(0, color=MUTE, lw=1, zorder=1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

    gt_aud = fnum(bycol.get("GT", {}).get("conditional", {}).get("net_x1e4"))
    gt_mot = fnum(bycol.get("GT", {}).get("motion", {}).get("net_x1e4"))
    # LEFT: audio channel — aud|mot (filled, R-DGG) over aud marginal (hollow)
    for yi, rc in zip(y, recs):
        if rc is None:
            continue
        if "conditional" in rc:
            mark(axL, yi, rc["conditional"], hollow=False, off=+dy)
        if "marginal" in rc:
            mark(axL, yi, rc["marginal"], hollow=True, off=-dy)
    frame(axL)
    if gt_aud is not None:
        axL.axvline(gt_aud, color=OK["green"], lw=1, ls="--", zorder=1)
    axL.set_xlabel("AUDIO net ×1e4   (95% identity CI)")
    axL.set_title("AUDIO channel — separates (controls ~0)", fontsize=9.5)

    # RIGHT: motion channel — mot (marginal), the visible-motion coupling
    for yi, rc in zip(y, recs):
        if rc is None:
            continue
        if "motion" in rc:
            mark(axR, yi, rc["motion"], hollow=False, off=0.0)
    frame(axR)
    if gt_mot is not None:
        axR.axvline(gt_mot, color=OK["green"], lw=1, ls="--", zorder=1)
    axR.set_xlabel("MOTION net ×1e4   (95% identity CI)")
    axR.set_title("MOTION channel — does NOT separate (all >0)", fontsize=9.5)

    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    handles = [Line2D([0], [0], marker="s", mfc=c, mec=c, ls="none", ms=7, label=g)
               for g, c in SHIFT_GROUP_C.items()]
    handles += [
        Line2D([0], [0], marker="o", mfc=INK, mec=INK, ls="none", ms=6, label="conditional $aud|mot$ (R-DGG)"),
        Line2D([0], [0], marker="o", mfc="white", mec=INK, mew=1.4, ls="none", ms=6, label="marginal $aud$"),
        Line2D([0], [0], color=OK["green"], ls="--", lw=1, label="GT net anchor"),
    ]
    axR.legend(handles=handles, frameon=False, fontsize=7.0, loc="lower right")
    fig.suptitle("Why R-DGG reads the AUDIO channel, net of motion: motion coupling is strong but does not discriminate",
                 fontsize=10.0, y=0.98)
    fig.text(0.5, 0.004, footnote, ha="center", fontsize=6.4, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


LMAX_WIN_C = {"L48": OK["sky"], "L100": OK["blue"], "L150": OK["vermillion"]}
LMAX_WIN_LABEL = {"L48": "window 48 fr (1.9 s)", "L100": "window 100 fr (4 s, default)",
                  "L150": "window 150 fr (6 s)"}


def fig_lmax_ablation(csv="lmax_ablation.csv", out="fig_lmax_ablation", xlim=(-0.7, 1.9), dy=0.26):
    # one record per column, carrying the 3 windows
    d = rows(csv)
    order, bycol = [], {}
    for r in d:
        c = r["column"]
        if c not in bycol:
            order.append(c); bycol[c] = {"group": r["group"]}
        bycol[c][r["window"]] = r
    recs, labels, prev = [], [], None
    for c in order:
        g = bycol[c]["group"]
        if prev is not None and g != prev:
            recs.append(None); labels.append("")
        recs.append(bycol[c]); labels.append(c); prev = g
    y = list(range(len(recs)))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    ax.axvline(0, color=MUTE, lw=1, zorder=1)
    gt_net = fnum(bycol.get("GT", {}).get("L100", {}).get("net_x1e4"))
    if gt_net is not None:
        ax.axvline(gt_net, color=OK["green"], lw=1, ls="--", zorder=1)

    offs = {"L48": +dy, "L100": 0.0, "L150": -dy}
    for yi, rc in zip(y, recs):
        if rc is None:
            continue
        for win, off in offs.items():
            r = rc.get(win)
            if r is None:
                continue
            v, lo, hi = fnum(r["net_x1e4"]), fnum(r["net_ci_lo_x1e4"]), fnum(r["net_ci_hi_x1e4"])
            gc = LMAX_WIN_C[win]
            ax.errorbar(v, yi + off, xerr=[[v - lo], [hi - v]], fmt="o", ms=5, color=gc,
                        ecolor=gc, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("net ×1e4   (95% identity CI)")
    handles = [Line2D([0], [0], marker="o", mfc=LMAX_WIN_C[w], mec=LMAX_WIN_C[w], ls="none",
                      ms=6, label=LMAX_WIN_LABEL[w]) for w in ("L48", "L100", "L150")]
    handles += [Line2D([0], [0], color=OK["green"], ls="--", lw=1, label="GT net anchor (default)")]
    ax.legend(handles=handles, frameon=False, fontsize=7.3, loc="lower right")
    fig.suptitle("Lag-window robustness: R-DGG net at 1.9 / 4 / 6 s windows (the three nearly coincide)",
                 fontsize=10.0, y=0.98)
    fig.text(0.5, 0.004, "All on the common mask (identical frames), n≈177, 13 identities. GT stays "
             "significant and controls stay ~0 at every window; the default 4 s sits in the middle. "
             "GT net declines mildly as the window widens (short lags carry most of the coupling).",
             ha="center", fontsize=6.4, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


HUBERT_LAYER_C = {"6": OK["sky"], "9": OK["blue"], "12": OK["vermillion"]}
HUBERT_LAYER_LABEL = {"6": "layer 6", "9": "layer 9 (default)", "12": "layer 12"}


def fig_hubert_ablation(out="fig_hubert_ablation"):
    """Full layer sweep 6..12: GT anchor net (filled=significant) + the control envelope on the
    left axis, the effective dimensionality (participation ratio) on the right axis."""
    hub = rows("hubert_ablation.csv")
    eff = {int(r["hubert_layer"]): r for r in rows("hubert_ablation_effdim.csv")}
    sa = rows("hubert_ablation_seed_avg.csv")
    layers = sorted(eff)
    # control envelope: surrogate + talking-heads (exclude GT and dyadic)
    ctrl = {L: [] for L in layers}
    for r in hub:
        if r["column"] == "GT×other" or r["group"] == "non-dyad":
            ctrl[int(r["hubert_layer"])].append(fnum(r["net_x1e4"]))
    dyad = {}
    for r in sa:
        dyad.setdefault(r["model"], {})[int(r["hubert_layer"])] = fnum(r["net_x1e4"])

    fig, axL = plt.subplots(figsize=(7.4, 5.2))
    axL.axhline(0, color=MUTE, lw=1, zorder=1)
    axL.axvspan(6.5, 11.5, color=OK["green"], alpha=0.06, zorder=0)  # working band 7-11
    # control envelope band
    clo = [min(ctrl[L]) for L in layers]; chi = [max(ctrl[L]) for L in layers]
    axL.fill_between(layers, clo, chi, color=OK["grey"], alpha=0.25, zorder=1, label="controls (envelope)")
    # dyadic faint lines
    for m, c in zip(dyad, (OK["blue"], OK["sky"], OK["purple"])):
        axL.plot(layers, [dyad[m][L] for L in layers], color=c, lw=1, alpha=0.5, zorder=2, label=m)
    # GT anchor: line + filled(sig)/hollow(n.s.) markers
    gnet = [fnum(eff[L]["GT_net_x1e4"]) for L in layers]
    axL.plot(layers, gnet, color=OK["green"], lw=2, zorder=4, label="GT anchor")
    for L in layers:
        sig = fnum(eff[L]["GT_net_p"]) < 0.05
        axL.plot(L, fnum(eff[L]["GT_net_x1e4"]), marker="o", ms=8, zorder=5,
                 mfc=(OK["green"] if sig else "white"), mec=OK["green"], mew=1.8)
    axL.set_xlabel("HuBERT transformer layer"); axL.set_ylabel("R-DGG net ×1e4", color=INK)
    axL.set_xticks(layers); axL.spines["top"].set_visible(False)
    axL.grid(axis="y", color=GRID, lw=0.6, zorder=0); axL.set_axisbelow(True)
    # right axis: effective dimensionality
    axR = axL.twinx()
    axR.plot(layers, [fnum(eff[L]["eff_dim_participation_ratio"]) for L in layers],
             color=INK, lw=1.3, ls="--", marker="s", ms=4, zorder=3)
    axR.set_ylabel("effective dim (participation ratio)", color=INK)
    axR.spines["top"].set_visible(False)
    axL.set_title("HuBERT layer sweep: GT anchor plateaus at 7–11, fails at both ends as the "
                  "representation specialises", fontsize=9.5)
    h, l = axL.get_legend_handles_labels()
    h += [Line2D([0], [0], color=INK, ls="--", marker="s", ms=4, label="eff. dim (right axis)"),
          Line2D([0], [0], marker="o", mfc=OK["green"], mec=OK["green"], ls="none", ms=7, label="GT sig. (p<0.05)"),
          Line2D([0], [0], marker="o", mfc="white", mec=OK["green"], mew=1.8, ls="none", ms=7, label="GT n.s.")]
    axL.legend(handles=h, frameon=False, fontsize=6.8, loc="upper right", ncol=2)
    fig.text(0.5, -0.02, "Common mask, n≈177, 13 identities. Layer 6 fails by lit controls (too acoustic, "
             "high rank); layers 7–11 work (anchor sig., controls ~0); layer 12 fails as the final layer "
             "collapses to low effective rank. Dyadic AVTR-1/DyStream are robust across all layers.",
             ha="center", fontsize=6.3, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


def fig_pca_dim_ablation(csv="pca_dim_ablation.csv", out="fig_pca_dim_ablation", xlim=(-0.7, 1.9), dy=0.26):
    d = rows(csv)
    # split by knob; within each, one record per column carrying its 3 dims
    panels = {"audio_K_A": {}, "motion_KMV": {}}
    orders = {"audio_K_A": [], "motion_KMV": []}
    dims = {"audio_K_A": ["32", "64", "96"], "motion_KMV": ["16", "32", "64"]}
    for r in d:
        k = r["knob"]; c = r["column"]
        if c not in panels[k]:
            orders[k].append(c); panels[k][c] = {"group": r["group"]}
        panels[k][c][r["dim"]] = r
    C = {0: OK["sky"], 1: OK["blue"], 2: OK["vermillion"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, knob in zip(axes, ("audio_K_A", "motion_KMV")):
        bycol = panels[knob]; dd = dims[knob]
        recs, labels, prev = [], [], None
        for c in orders[knob]:
            g = bycol[c]["group"]
            if prev is not None and g != prev:
                recs.append(None); labels.append("")
            recs.append(bycol[c]); labels.append(c); prev = g
        y = list(range(len(recs)))[::-1]
        ax.axvline(0, color=MUTE, lw=1, zorder=1)
        gt = fnum(bycol.get("GT", {}).get(dd[1], {}).get("net_x1e4"))
        if gt is not None:
            ax.axvline(gt, color=OK["green"], lw=1, ls="--", zorder=1)
        offs = {dd[0]: +dy, dd[1]: 0.0, dd[2]: -dy}
        for yi, rc in zip(y, recs):
            if rc is None:
                continue
            for j, dm in enumerate(dd):
                r = rc.get(dm)
                if r is None:
                    continue
                v, lo, hi = fnum(r["net_x1e4"]), fnum(r["net_ci_lo_x1e4"]), fnum(r["net_ci_hi_x1e4"])
                ax.errorbar(v, yi + offs[dm], xerr=[[v - lo], [hi - v]], fmt="o", ms=5, color=C[j],
                            ecolor=C[j], elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xlim(*xlim)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        name = "AUDIO PCA dim $K_A$" if knob == "audio_K_A" else "MOTION PCA dim $K_{MV}$"
        sub = "64 is the sweet spot; both sides worse" if knob == "audio_K_A" else "no effect (flat)"
        ax.set_title(f"{name} — {sub}", fontsize=9.5)
        ax.set_xlabel("net ×1e4   (95% identity CI)")
        hh = [Line2D([0], [0], marker="o", mfc=C[j], mec=C[j], ls="none", ms=6,
                     label=f"{'default ' if j == 1 else ''}{dd[j]}") for j in range(3)]
        ax.legend(handles=hh, frameon=False, fontsize=7.2, loc="lower right", title=name.split("—")[0])
    fig.suptitle("PCA output-dimension ablation (layer 9): audio K_A has an optimum at 64, motion K_MV is immaterial",
                 fontsize=10.0, y=0.99)
    fig.text(0.5, 0.004, "Common mask, n≈177, 13 identities, seed-averaged dyadic. Narrow K_A inflates the "
             "controls; wide K_A drops GT and turns a talking-head control significant. The motion dim changes nothing.",
             ha="center", fontsize=6.4, color=MUTE)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


def fig_dose_curve(tag="", xlabel="number of videos switched to zeroother (k)",
                   title="Dose curve: stock → zeroother (k=0..178)", out="fig_dose_curve",
                   color=OK["blue"]):
    cf = rows(f"dose_curve{tag}_closedform.csv")
    ic = rows(f"dose_curve{tag}_identity_ci.csv")
    k = [fnum(r["k"]) for r in cf]
    net = [fnum(r["net_x1e4"]) for r in cf]
    ik = [fnum(r["k"]) for r in ic]
    ip = [fnum(r["point_x1e4"]) for r in ic]
    ilo = [fnum(r["point_x1e4"]) - fnum(r["ci_lo_x1e4"]) for r in ic]
    ihi = [fnum(r["ci_hi_x1e4"]) - fnum(r["point_x1e4"]) for r in ic]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(0, color=MUTE, lw=1)
    ax.plot(k, net, color=color, lw=2, label="closed-form mean")
    ax.errorbar(ik, ip, yerr=[ilo, ihi], fmt="o", color=color, ecolor=INK,
                elinewidth=1, capsize=3, markersize=5,
                label="expected net ± 95% identity CI", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("R-DGG net ×1e4")
    ax.set_title(title, fontsize=9)
    tidy(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / f"{out}.pdf")
    plt.close(fig)


def fig_metric_seed():
    d = rows("metric_seed_sweep_summary.csv")
    knobs = [r for r in d if r["source"] != "generation_seed (ref)"]
    gen = next(fnum(r["net_SD_GT_x1e4"]) for r in d if r["source"] == "generation_seed (ref)")
    cols = [("net_SD_GT_x1e4", "GT", OK["green"]),
            ("net_SD_seeded_x1e4", "stock", OK["blue"]),
            ("net_SD_zero_x1e4", "zeroother", OK["orange"])]
    x = range(len(knobs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (key, lab, c) in enumerate(cols):
        ax.bar([xx + (i - 1) * w for xx in x], [fnum(r[key]) for r in knobs],
               width=w, color=c, label=lab, zorder=2)
    ax.axhline(gen, color=OK["vermillion"], lw=1.4, ls="--",
               label=f"generation-seed SD ({gen:.3f})")
    LBL = {"knob1": "PCA stem\nsubsample", "knob2": "PCA row\nsubsample",
           "knob4_floor_SD": "timeshift\nfloor", "combined_metric_seed": "combined"}
    ax.set_xticks(list(x))
    ax.set_xticklabels([LBL.get(r["source"], r["source"]) for r in knobs], fontsize=8)
    ax.set_ylabel("metric-seed net SD ×1e4")
    ax.set_title("Metric-internal seed sensitivity vs generation seed", fontsize=9)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIGS / "fig_metric_seed.pdf")
    plt.close(fig)


COLC = {"GT": OK["green"], "seeded": OK["blue"], "zero": OK["orange"]}


def fig_count_sweep():
    ts, bs = rows("count_sweep_timeshift.csv"), rows("count_sweep_bootstrap.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for col in COLC:
        r = [x for x in ts if x["column"] == col]
        a1.plot([fnum(x["n_nulls"]) for x in r], [fnum(x["sd_net_empirical_x1e4"]) for x in r],
                "o-", color=COLC[col], ms=4, lw=1.5, label=col)
    gt = [x for x in ts if x["column"] == "GT"]
    a1.plot([fnum(x["n_nulls"]) for x in gt], [fnum(x["sd_theory_1_over_sqrtN_x1e4"]) for x in gt],
            "--", color=MUTE, lw=1, label="1/√N (theory)")
    a1.set(xscale="log", yscale="log", xlabel="timeshift nulls N",
           ylabel="SD(net) ×1e4")
    a1.set_title("floor Monte-Carlo noise", fontsize=9)
    for col in COLC:
        r = [x for x in bs if x["column"] == col]
        a2.plot([fnum(x["n_bootstrap"]) for x in r], [fnum(x["sd_ci_lo_x1e4"]) for x in r],
                "o-", color=COLC[col], ms=4, lw=1.5, label=col)
    a2.set(xscale="log", yscale="log", xlabel="bootstrap resamples B",
           ylabel="SD(CI lower) ×1e4")
    a2.set_title("CI Monte-Carlo noise", fontsize=9)
    for ax in (a1, a2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="both", color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Metric Monte-Carlo noise vs count (∝ 1/√count)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_count_sweep.pdf")
    plt.close(fig)


def fig_pca_size():
    st, ro = rows("pca_size_stem.csv"), rows("pca_size_row.csv")
    data = {"stem": st, "row": ro}

    def szlabel(v, dim):
        v = fnum(v)
        if dim == "stem":
            return "all" if v >= 184 else str(int(v))
        return "all" if v >= 1e8 else f"{int(v / 1000)}k"

    gen = 0.111
    TOTAL_ROWS = 1069850   # full PCA rows; the "all" row point sits at its true magnitude
    xvals = {"stem": [fnum(r["n_stems"]) for r in st],
             "row": [min(fnum(r["n_rows"]), TOTAL_ROWS) for r in ro]}
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.0))
    titles = {"stem": "stem subsample (rows=40k)", "row": "row subsample (stems=60)"}
    xlab = {"stem": "stems used in the PCA fit (log)", "row": "rows used in the PCA fit (log)"}
    for j, dim in enumerate(("stem", "row")):
        d = data[dim]
        x = xvals[dim]                                 # log axis, true magnitudes
        labels = [szlabel(r["n_stems" if dim == "stem" else "n_rows"], dim) for r in d]
        for key, lab, c in (("net_SD_GT_x1e4", "GT", OK["green"]),
                            ("net_SD_seeded_x1e4", "stock", OK["blue"]),
                            ("net_SD_zero_x1e4", "zeroother", OK["orange"])):
            ax[0][j].plot(x, [fnum(r[key]) for r in d], "o-", color=c, ms=4, lw=1.4, label=lab)
        ax[0][j].axhline(gen, color=OK["vermillion"], lw=1.2, ls="--", label="gen-seed SD")
        ax[0][j].set_title(titles[dim], fontsize=9)
        ax[0][j].set_ylabel("net SD ×1e4" if j == 0 else "")
        ax[1][j].plot(x, [fnum(r["aud_overlap"]) for r in d], "s-", color=OK["sky"], ms=4, lw=1.4, label="audio")
        ax[1][j].plot(x, [fnum(r["mot_overlap"]) for r in d], "^-", color=OK["purple"], ms=4, lw=1.4, label="motion")
        ax[1][j].set_ylabel("basis overlap (cos²)" if j == 0 else "")
        ax[1][j].set_xlabel(xlab[dim])
        ax[1][j].set_ylim(0.9, 1.005)
        for a in (ax[0][j], ax[1][j]):
            a.set_xscale("log")
            a.minorticks_off()                         # kill log minor ticks + their stray labels
            a.set_xticks(x); a.set_xticklabels(labels)
    for a in ax.flat:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    ax[0][0].legend(frameon=False, fontsize=7.5, ncol=2)
    ax[1][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("PCA subsample size: net effect (top) vs basis stability (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pca_size.pdf")
    plt.close(fig)


def main():
    fig_leaderboard()
    fig_ablation()
    fig_shift_ablation()
    fig_operator_ablation()
    fig_motion_cond_ablation()
    fig_lmax_ablation()
    fig_hubert_ablation()
    fig_pca_dim_ablation()
    fig_seed_variance()
    fig_dose_curve()   # stock -> zeroother
    fig_dose_curve(tag="_cfg0", xlabel="number of videos switched to cfg_other=0 (k)",
                   title="Dose curve: stock → cfg0 (k=0..178)", out="fig_dose_curve_cfg0",
                   color=OK["vermillion"])
    fig_count_sweep()
    # fig_metric_seed() and fig_pca_size() are retained above as code + their CSVs stay
    # as provenance for the full-data PCA default, but are NOT rendered into the paper
    # figure set (the PCA seed no longer affects the deterministic full-data basis).
    print("  figures: leaderboard, ablation, shift_ablation, operator_ablation, motion_cond_ablation, "
          "lmax_ablation, hubert_ablation, seed_variance, dose_curve, dose_curve_cfg0, count_sweep "
          "-> results/figures/*.pdf")
    print("done.")


if __name__ == "__main__":
    main()
