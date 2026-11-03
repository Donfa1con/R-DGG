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


def fig_dose_curve():
    cf = rows("dose_curve_closedform.csv")
    ic = rows("dose_curve_identity_ci.csv")
    k = [fnum(r["k"]) for r in cf]
    net = [fnum(r["net_x1e4"]) for r in cf]
    ik = [fnum(r["k"]) for r in ic]
    ip = [fnum(r["point_x1e4"]) for r in ic]
    ilo = [fnum(r["point_x1e4"]) - fnum(r["ci_lo_x1e4"]) for r in ic]
    ihi = [fnum(r["ci_hi_x1e4"]) - fnum(r["point_x1e4"]) for r in ic]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(0, color=MUTE, lw=1)
    ax.plot(k, net, color=OK["blue"], lw=2, label="closed-form mean")
    ax.errorbar(ik, ip, yerr=[ilo, ihi], fmt="o", color=OK["blue"], ecolor=INK,
                elinewidth=1, capsize=3, markersize=5, label="identity-CI points", zorder=3)
    ax.set_xlabel("number of videos with zeroed speaker audio (k)")
    ax.set_ylabel("R-DGG net ×1e4")
    ax.set_title("Dose curve: stock → zeroother (k=0..178)", fontsize=9)
    tidy(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / "fig_dose_curve.pdf")
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
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["source"].replace("_", "\n") for r in knobs], fontsize=7.5)
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


def main():
    for fn in (fig_leaderboard, fig_ablation, fig_seed_variance, fig_dose_curve,
               fig_metric_seed, fig_count_sweep):
        fn()
        print(f"  figures/{fn.__name__.replace('fig_', 'fig_')}.pdf")
    print("done.")


if __name__ == "__main__":
    main()
