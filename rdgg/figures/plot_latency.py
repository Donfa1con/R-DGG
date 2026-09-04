"""Redraw ../latency.{png,pdf} (Figure 1 of RESULTS.md) from latency_curves.json.

The json holds, per system, the R-DGG gain G_{a|m} (x1e-4 nats) at each speaker-driver
offset (ms). This script only DRAWS it and needs nothing but matplotlib:

    python figures/plot_latency.py

To regenerate the values (needs the metric env + a GPU) run figures/latency_sweep.py first.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402

for _f in ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
           "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = json.load(open(os.path.join(HERE, "latency_curves.json")))
OUT_PNG = os.path.join(HERE, "..", "latency.png")
OUT_PDF = os.path.join(HERE, "..", "latency.pdf")

GT = "GT"
SURR = "GT×other"
DYADIC = {"AVTR-1", "DyStream", "AvatarForcing"}
# seaborn "colorblind" palette, inlined so the script depends only on matplotlib
CB = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
      "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]


def v0(k):
    return dict((int(p[0]), p[1]) for p in CURVES[k]).get(0, 0.0)


systems = sorted(CURVES, key=lambda k: -v0(k))  # legend order: gain at zero offset, descending

plt.rcParams.update({
    "font.family": "Nimbus Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",   # sans math to match Nimbus Sans body
    "axes.linewidth": 0.7, "axes.labelsize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.6, "figure.dpi": 300,
})

colors, ci = {}, 0
for k in systems:
    if k in (GT, SURR):
        continue
    colors[k] = CB[ci % len(CB)]
    ci += 1


def style(k):
    if k == GT:
        return dict(color="black", lw=2.4, ls="-", marker="o", ms=3.6, zorder=10)
    if k == SURR:
        return dict(color="0.5", lw=1.5, ls=":", marker="o", ms=2.8, zorder=3)
    if k in DYADIC:
        return dict(color=colors[k], lw=1.8, ls="-", marker="o", ms=3.2, zorder=6)
    return dict(color=colors[k], lw=1.4, ls="--", marker="o", ms=2.8, alpha=0.95, zorder=4)


fig, ax = plt.subplots(figsize=(6.2, 3.5))
ax.set_axisbelow(True)
ax.grid(True, which="major", color="0.9", lw=0.5, zorder=0)
ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
ax.axvline(0.0, color="0.6", lw=0.8, ls=(0, (4, 4)), zorder=1)
xmax = max(int(p[0]) for k in CURVES for p in CURVES[k])

for k in systems:
    pts = sorted(CURVES[k], key=lambda p: p[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            label=("ground truth" if k == GT else k), **style(k))

ax.set_xlabel("speaker-driver offset (ms)")
ax.set_ylabel(r"$G_{a\mid m}\ \ (\times 10^{-4}\ \mathrm{nats})$")
ax.set_xticks(range(-xmax, xmax + 1, 40))
ax.margins(x=0.02)
ax.set_ylim(top=ax.get_ylim()[1] + 0.18)   # headroom so the in-plot legend clears the curves
ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, edgecolor="0.85",
          fontsize=7.2, handlelength=1.6, columnspacing=1.0, labelspacing=0.35, borderaxespad=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PDF, bbox_inches="tight")
print("wrote", OUT_PNG, "and", OUT_PDF)
