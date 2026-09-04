#!/usr/bin/env python3
"""Aggregate SI-184 original-definition classic-metric CSVs into paper tables.

Reads the per-model metric CSVs written by pipeline/5_classic.sh under

    <corpus>/<model>/_metrics/<metric>_<src>[_listening]__LS_AL.csv

and emits, for each (src in {emoca, lp}) x (variant in {full, listening}),
a markdown table over the 8 model rows (GT anchor first). Missing models /
files are shown as an em-dash; the run may still be in progress.

Aggregation rules (per the metric contracts):
  rpcc      per-stem CSV (184 rows)  -> nanmean over stems of rPCC_<region>
  tlcc      per-stem CSV (184 rows)  -> nanmean over stems of |tlcc_offset_<region>|
                                        (headline; offset is SIGNED in the CSV,
                                        abs() is taken BEFORE averaging)
                                     -> nanmean of tlcc_peak_<region> (signed, aux)
  pfd       1 pooled row             -> pfd_<region>
  sid       1 pooled row             -> sid_<region>
  variance  1 pooled row             -> var_<region>

No metric compute happens here; this only reads CSVs.

Stdlib + numpy only.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CORPUS = os.path.join(REPO, "data")

SPLIT = "LS_AL"

# Model dir name -> display label. Order defines table row order (GT first).
MODEL_LABELS = [
    ("GT", "GT"),
    ("AVTR-1", "AVTR-1"),
    ("AvatarForcing", "AvatarForcing"),
    ("FLOAT", "FLOAT"),
    ("SoulX_FlashHead_Lite", "SoulX Lite"),
    ("SoulX_FlashHead_Pro", "SoulX Pro"),
    ("ditto", "Ditto"),
    ("dystream", "DyStream"),
]

SRCS = ["emoca", "lp"]
VARIANTS = ["full", "listening"]

# region groups per source: (overall + primary region + exp)
REGIONS = {
    "emoca": ["overall", "pose", "exp"],
    "lp":    ["overall", "rot", "exp"],
}
PRIMARY = {"emoca": "pose", "lp": "rot"}


# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------
def _csv_path(corpus, model, metric, src, variant):
    suffix = "_listening" if variant == "listening" else ""
    fname = f"{metric}_{src}{suffix}__{SPLIT}.csv"
    return os.path.join(corpus, model, "_metrics", fname)


def _read_rows(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s):
    if s is None:
        return math.nan
    s = s.strip()
    if s == "" or s.lower() in ("nan", "na", "none"):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _col_nanmean(rows, col):
    """nanmean of a per-stem column; None if the column is absent."""
    if rows is None or len(rows) == 0 or col not in rows[0]:
        return None
    vals = np.array([_to_float(r.get(col)) for r in rows], dtype=float)
    if np.all(np.isnan(vals)):
        return math.nan
    return float(np.nanmean(vals))


def _col_abs_nanmean(rows, col):
    """nanmean of |per-stem column|; None if absent. Used for signed tlcc offset."""
    if rows is None or len(rows) == 0 or col not in rows[0]:
        return None
    vals = np.array([_to_float(r.get(col)) for r in rows], dtype=float)
    if np.all(np.isnan(vals)):
        return math.nan
    return float(np.nanmean(np.abs(vals)))


def _pooled_val(rows, col):
    """single pooled-row value; None if file/column absent."""
    if rows is None or len(rows) == 0 or col not in rows[0]:
        return None
    return _to_float(rows[0].get(col))


# ----------------------------------------------------------------------------
# Build the data cube: data[src][variant][model][metric_region] = value|None
# ----------------------------------------------------------------------------
def build_data(corpus):
    data = {}
    for src in SRCS:
        data[src] = {}
        regions = REGIONS[src]
        for variant in VARIANTS:
            data[src][variant] = {}
            for model, _label in MODEL_LABELS:
                d = {}

                # rpcc: nanmean over stems
                rpcc_rows = _read_rows(_csv_path(corpus, model, "rpcc", src, variant))
                for reg in regions:
                    d[f"rpcc_{reg}"] = _col_nanmean(rpcc_rows, f"rPCC_{reg}")

                # tlcc: headline nanmean|offset|, aux nanmean peak (signed)
                tlcc_rows = _read_rows(_csv_path(corpus, model, "tlcc", src, variant))
                for reg in regions:
                    d[f"tlcc_absoff_{reg}"] = _col_abs_nanmean(
                        tlcc_rows, f"tlcc_offset_{reg}")
                    d[f"tlcc_peak_{reg}"] = _col_nanmean(
                        tlcc_rows, f"tlcc_peak_{reg}")

                # pfd / sid / variance: single pooled row
                pfd_rows = _read_rows(_csv_path(corpus, model, "pfd", src, variant))
                sid_rows = _read_rows(_csv_path(corpus, model, "sid", src, variant))
                var_rows = _read_rows(_csv_path(corpus, model, "variance", src, variant))
                for reg in regions:
                    d[f"pfd_{reg}"] = _pooled_val(pfd_rows, f"pfd_{reg}")
                    d[f"sid_{reg}"] = _pooled_val(sid_rows, f"sid_{reg}")
                    d[f"var_{reg}"] = _pooled_val(var_rows, f"var_{reg}")

                data[src][variant][model] = d
    return data


# ----------------------------------------------------------------------------
# Formatting / markdown rendering
# ----------------------------------------------------------------------------
DASH = "—"  # em-dash


_SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def _superscript(n):
    return "".join(_SUP[c] for c in str(n))


def _is_num(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def column_scale(values):
    """Choose a per-column power-of-ten scale n and a fixed decimal count so the
    displayed cells (= raw / 10**n) are clean fixed-decimals in a readable range,
    never scientific notation.

    - normal range (0.1 <= max|.| < 1000): n = 0, no header annotation.
    - otherwise: n = floor(log10(max|.|)) so the column's max lands in [1, 10).
    - decimals ~3 sig figs, ONE count per column: 3 if scaled max < 10,
      2 if < 100, 1 if < 1000.
    Returns (n, decimals).
    """
    finite = [abs(v) for v in values if _is_num(v) and v != 0.0]
    if not finite:
        return 0, 3  # all missing or all exactly zero
    M = max(finite)
    if 0.1 <= M < 1000.0:
        n = 0
    else:
        n = int(math.floor(math.log10(M)))
    scaled_max = M / (10.0 ** n)
    if scaled_max < 10.0:
        decimals = 3
    elif scaled_max < 100.0:
        decimals = 2
    else:
        decimals = 1
    return n, decimals


def format_column(base_label, values):
    """Return (annotated_header, [formatted_cell, ...]) for one column, applying a
    single per-column scale + decimal count. Missing/NaN -> em-dash."""
    n, decimals = column_scale(values)
    header = base_label if n == 0 else f"{base_label} (×10{_superscript(n)})"
    factor = 10.0 ** n
    cells = []
    for v in values:
        if not _is_num(v):
            cells.append(DASH)
        else:
            cells.append(f"{v / factor:.{decimals}f}")
    return header, cells


def render_table(headers, rows):
    """rows: list of lists of already-formatted strings. First col left-aligned,
    rest right-aligned. Returns a padded markdown table string."""
    ncol = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for j in range(ncol):
            widths[j] = max(widths[j], len(row[j]))

    def line(cells):
        out = []
        for j, c in enumerate(cells):
            if j == 0:
                out.append(c.ljust(widths[j]))
            else:
                out.append(c.rjust(widths[j]))
        return "| " + " | ".join(out) + " |"

    sep_cells = []
    for j in range(ncol):
        if j == 0:
            sep_cells.append(":" + "-" * (widths[j] - 1) if widths[j] > 1 else "-")
        else:
            sep_cells.append("-" * (widths[j] - 1) + ":" if widths[j] > 1 else "-")
    sep = "| " + " | ".join(sep_cells) + " |"

    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def _build_block(dsub, col_specs):
    """col_specs: list of (base_label, data_key). Returns a rendered markdown
    table string. Each column is scaled/formatted independently across the 8
    model rows so cells line up (fixed decimals, no scientific notation)."""
    headers = ["Model"]
    columns = []  # one formatted-cell list per data column
    for base_label, key in col_specs:
        raw = [dsub[m].get(key) for m, _label in MODEL_LABELS]
        header, cells = format_column(base_label, raw)
        headers.append(header)
        columns.append(cells)
    rows = []
    for i, (_model, label) in enumerate(MODEL_LABELS):
        rows.append([label] + [columns[j][i] for j in range(len(col_specs))])
    return render_table(headers, rows)


def build_markdown(src, variant, data):
    reg_p = PRIMARY[src]          # primary region label: pose / rot
    dsub = data[src][variant]

    # --- main table: region columns (primary + exp) for the 5 metrics ---
    main_specs = [
        (f"rPCC_{reg_p}", f"rpcc_{reg_p}"), ("rPCC_exp", "rpcc_exp"),
        (f"TLCCabs_{reg_p}", f"tlcc_absoff_{reg_p}"),
        ("TLCCabs_exp", "tlcc_absoff_exp"),
        (f"PFD_{reg_p}", f"pfd_{reg_p}"), ("PFD_exp", "pfd_exp"),
        (f"SID_{reg_p}", f"sid_{reg_p}"), ("SID_exp", "sid_exp"),
        (f"Var_{reg_p}", f"var_{reg_p}"), ("Var_exp", "var_exp"),
    ]
    # --- secondary block: overall columns (dimensionally imbalanced) ---
    over_specs = [
        ("rPCC_all", "rpcc_overall"), ("TLCCabs_all", "tlcc_absoff_overall"),
        ("PFD_all", "pfd_overall"), ("SID_all", "sid_overall"),
        ("Var_all", "var_overall"),
    ]
    # --- auxiliary: TLCC peak correlation (signed, no abs) ---
    aux_specs = [
        (f"TLCCpeak_{reg_p}", f"tlcc_peak_{reg_p}"),
        ("TLCCpeak_exp", "tlcc_peak_exp"),
        ("TLCCpeak_all", "tlcc_peak_overall"),
    ]

    legend = (
        f"_Legend ({src}, {variant}). GT is the reference. "
        f"rPCC / PFD: distance from GT (0 = identical to GT). "
        f"TLCCabs: mean |cross-corr lag offset| in frames (lead/lag magnitude). "
        f"SID: bits, higher = more diverse. Var: pooled per-dim variance (diversity). "
        f"Region '{reg_p}' and 'exp' are the balanced read; 'all'/overall is "
        f"dimensionally imbalanced (exp dominates) and shown for reference only. "
        f"TLCCpeak (auxiliary) is the mean signed peak cross-correlation. "
        f"Per-column ×10ⁿ scale factors are folded into the headers so every cell "
        f"is a plain fixed-decimal number._"
    )

    parts = []
    parts.append(f"### {src} — {variant}")
    parts.append("")
    parts.append(_build_block(dsub, main_specs))
    parts.append("")
    parts.append("_overall block (secondary — dimensionally imbalanced):_")
    parts.append("")
    parts.append(_build_block(dsub, over_specs))
    parts.append("")
    parts.append("_TLCC peak (auxiliary — signed, no abs):_")
    parts.append("")
    parts.append(_build_block(dsub, aux_specs))
    parts.append("")
    parts.append(legend)
    return "\n".join(parts)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="corpus root holding <model>/_metrics/*.csv (default: <repo>/data)")
    ap.add_argument("--json_out", default=None,
                    help="machine-readable dump path "
                         "(default: <corpus>/GT/_metrics/classic_metrics_si184_table.json)")
    args = ap.parse_args()
    corpus = args.corpus
    json_out = args.json_out or os.path.join(
        corpus, "GT", "_metrics", "classic_metrics_si184_table.json")

    data = build_data(corpus)

    # machine-readable dump: {src: {variant: {model: {metric_region: value}}}}
    # (None stays None so the JSON is valid; NaN -> None as well.)
    dump = {}
    for src in SRCS:
        dump[src] = {}
        for variant in VARIANTS:
            dump[src][variant] = {}
            for model, _label in MODEL_LABELS:
                d = data[src][variant][model]
                clean = {}
                for k, v in d.items():
                    if isinstance(v, float) and math.isnan(v):
                        clean[k] = None
                    else:
                        clean[k] = v
                dump[src][variant][model] = clean
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(dump, fh, indent=2, allow_nan=False)

    # print the 4 tables
    blocks = []
    for src in SRCS:
        for variant in VARIANTS:
            blocks.append(build_markdown(src, variant, data))
    print("\n\n---\n\n".join(blocks))
    print()
    print(f"[read CSVs from {corpus}/<model>/_metrics/  "
          f"|  wrote machine-readable dump -> {json_out}]")


if __name__ == "__main__":
    main()
