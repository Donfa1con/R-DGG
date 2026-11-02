#!/usr/bin/env python3
"""Build the paper-ready tidy CSVs (and copy the authoritative JSON) from the
R-DGG metric outputs. Run from anywhere:  python results/make_tables.py

Reads  <repo>/data/GT/_metrics/*.json  (gitignored, authoritative) and
       <repo>/RESULTS.md               (committed leaderboard table)
Writes <repo>/results/metrics/*.json   (small committed copies)
       <repo>/results/tables/*.csv      (one row per data point; CIs as lo/hi;
                                          units in the column names: _x1e4, _deg)

All R-DGG values are the net directed gain ×1e4. No GPU needed.
"""
import csv
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "data" / "GT" / "_metrics"
METRICS = HERE / "metrics"
TABLES = HERE / "tables"
SPLIT = "LS_AL"

METRICS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

JSON = ["common4way", "seed_variance", "dose_curve", "cfgother0_rdgg", "metric_seed_sweep"]


def load(name):
    return json.load(open(SRC / f"{name}__{SPLIT}.json"))


def copy_json():
    for name in JSON:
        src = SRC / f"{name}__{SPLIT}.json"
        if src.exists():
            shutil.copy2(src, METRICS / src.name)


def write_csv(fname, header, rows):
    with open(TABLES / fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  tables/{fname}  ({len(rows)} rows)")


# ---- leaderboard: parse the committed RESULTS.md table ---------------------
def num(s):
    s = s.replace("−", "-").replace("*", "").replace("+", "").strip()
    return float(s)


def build_leaderboard():
    md = (REPO / "RESULTS.md").read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(md) if l.startswith("## Leaderboard"))
    except StopIteration:
        print("  !! leaderboard header not found in RESULTS.md; skipping")
        return
    rows = []
    for l in md[start:]:
        if l.startswith("## ") and not l.startswith("## Leaderboard"):
            break
        if not l.strip().startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        try:
            net = num(cells[3])
        except ValueError:
            continue  # header / separator row
        ci = re.findall(r"[+\-−]?\d*\.?\d+", cells[4].replace("−", "-"))
        ci_lo, ci_hi = (float(ci[0]), float(ci[1])) if len(ci) >= 2 else ("", "")
        system = cells[1].replace("*", "").strip()
        rows.append([cells[0].replace("—", "-").strip(), system, cells[2],
                     net, ci_lo, ci_hi, num(cells[6]) if cells[6].strip() else "",
                     cells[7]])
    write_csv("leaderboard.csv",
              ["rank", "system", "type", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p", "own_r2"],
              rows)


# ---- ablation (5 columns on one common mask) + paired deltas ---------------
def build_ablation():
    sv = load("seed_variance")["part2_five_columns"]
    m = sv["marginals"]
    rows = []
    for col in ["unseeded", "seed0", "seed1", "zeroother", "cfg0"]:
        d = m[col]
        c = d["ci"]
        rows.append([col, d["g_a_m"], d["g_shift"], d["net"], c["lo"], c["hi"],
                     c["p"], d["own_r2"], sv["n_stems"], sv["n_identities"]])
    write_csv("ablation_marginals.csv",
              ["column", "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4", "ci_lo_x1e4",
               "ci_hi_x1e4", "p", "own_r2", "n_stems", "n_identities"], rows)

    pd = load("common4way")["paired_deltas"]
    rows = [[k, v["delta"], v["lo"], v["hi"], v["p"], v["sig"], v.get("ci_width", "")]
            for k, v in pd.items()]
    write_csv("ablation_paired_deltas.csv",
              ["pair", "delta_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p", "sig", "ci_width_x1e4"], rows)


# ---- seed variance (3 stock noise realizations) ----------------------------
def build_seed_variance():
    p1 = load("seed_variance")["part1_three_stock_seeds"]
    m = p1["marginals"]
    rows = []
    for col in ["unseeded", "seed0", "seed1"]:
        d = m[col]
        c = d["ci"]
        rows.append([col, d["g_a_m"], d["g_shift"], d["net"], c["lo"], c["hi"], c["p"], d["own_r2"]])
    write_csv("seed_variance.csv",
              ["stock_run", "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4", "ci_lo_x1e4",
               "ci_hi_x1e4", "p", "own_r2"], rows)
    s = p1["seed_spread"]
    write_csv("seed_variance_summary.csv",
              ["mean_x1e4", "sd_sample_x1e4", "sd_pop_x1e4", "min_x1e4", "max_x1e4", "range_x1e4"],
              [[s["mean"], s["sd_sample"], s["sd_pop"], s["min"], s["max"], s["range"]]])
    pd = p1["paired_deltas"]
    write_csv("seed_variance_paired_deltas.csv",
              ["pair", "delta_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p", "sig"],
              [[k, v["delta"], v["lo"], v["hi"], v["p"], v["sig"]] for k, v in pd.items()])


# ---- dose curve (raw points for error bars) --------------------------------
def build_dose_curve():
    d = load("dose_curve")
    write_csv("dose_curve_sampled.csv",
              ["k", "frac", "mean_x1e4", "p50_x1e4", "lo_x1e4", "hi_x1e4", "n_subsets"],
              [[r["k"], r["frac"], r["mean"], r.get("p50", ""), r["lo"], r["hi"], r["n_subsets"]]
               for r in d["sampled_curve"]])
    cf = d["closed_form_curve"]
    write_csv("dose_curve_closedform.csv", ["k", "net_x1e4"],
              list(zip(cf["k"], cf["net"])))
    write_csv("dose_curve_identity_ci.csv", ["k", "point_x1e4", "ci_lo_x1e4", "ci_hi_x1e4"],
              [[int(k), v["point"], v["lo"], v["hi"]] for k, v in d["identity_ci"].items()])
    write_csv("dose_curve_additivity.csv", ["k", "additive_x1e4", "exact_x1e4", "abs_err_x1e4"],
              [[r["k"], r["additive"], r["exact"], r["abs_err"]] for r in d["additivity_validation"]])


# ---- metric-internal seed sweep (source-level + net) -----------------------
def build_metric_seed_sweep():
    d = load("metric_seed_sweep")
    # knobs 1 & 2: PCA basis / B operator source metrics + per-setting nets
    rows = []
    for knob in ["knob1_pca_stem_seed", "knob2_pca_row_seed"]:
        for s in d[knob]["settings"]:
            a, mo, b, n = (s["audio_pca_vs_default"], s["motion_pca_vs_default"],
                           s["B_operator_rel_frobenius_vs_default"], s["nets"])
            rows.append([knob.split("_")[0], s["label"],
                         a["max_principal_angle_deg"], a["mean_cos2_overlap"], a["rel_proj_fixed_features"],
                         mo["max_principal_angle_deg"], mo["mean_cos2_overlap"], mo["rel_proj_fixed_features"],
                         b["audio"], b["motion"], n["GT"], n["seeded"], n["zero"]])
    write_csv("metric_seed_sweep_pca.csv",
              ["knob", "setting", "aud_max_angle_deg", "aud_overlap", "aud_rel_proj",
               "mot_max_angle_deg", "mot_overlap", "mot_rel_proj",
               "B_rel_frob_audio", "B_rel_frob_motion",
               "net_GT_x1e4", "net_seeded_x1e4", "net_zero_x1e4"], rows)

    # knob 4: timeshift-floor base
    pb = d["knob4_timeshift_base"]["per_base"]
    rows = []
    for base, v in pb.items():
        fl, se, n = v["floor"], v["floor_se_within_draw"], v["nets"]
        rows.append([int(base), fl["GT"], fl["seeded"], fl["zero"],
                     se["GT"], se["seeded"], se["zero"], n["GT"], n["seeded"], n["zero"]])
    write_csv("metric_seed_sweep_timeshift.csv",
              ["base", "floor_GT_x1e4", "floor_seeded_x1e4", "floor_zero_x1e4",
               "floor_se_GT_x1e4", "floor_se_seeded_x1e4", "floor_se_zero_x1e4",
               "net_GT_x1e4", "net_seeded_x1e4", "net_zero_x1e4"], rows)

    # summary: metric-seed net SD per knob + combined vs generation seed
    rows = []
    for knob in ["knob1_pca_stem_seed", "knob2_pca_row_seed"]:
        sd = d[knob]["net_point_SD_incl_default"]
        rows.append([knob.split("_")[0], sd["GT"], sd["seeded"], sd["zero"]])
    ts = d["knob4_timeshift_base"]["floor_SD_across_bases"]
    rows.append(["knob4_floor_SD", ts["GT"], ts["seeded"], ts["zero"]])
    comb = d["combined_metric_seed_net_SD"]
    rows.append(["combined_metric_seed", comb["GT"], comb["seeded"], comb["zero"]])
    g = d["generation_seed_SD_for_comparison"]
    rows.append(["generation_seed (ref)", g, g, g])
    write_csv("metric_seed_sweep_summary.csv",
              ["source", "net_SD_GT_x1e4", "net_SD_seeded_x1e4", "net_SD_zero_x1e4"], rows)


def main():
    print("copying authoritative JSON -> results/metrics/")
    copy_json()
    print("writing tables -> results/tables/")
    build_leaderboard()
    build_ablation()
    build_seed_variance()
    build_dose_curve()
    build_metric_seed_sweep()
    print("done.")


if __name__ == "__main__":
    main()
