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

JSON = ["common4way", "seed_variance", "dose_curve", "dose_curve_cfg0", "cfgother0_rdgg",
        "metric_seed_sweep", "count_sweep", "pca_size_sweep", "shift_ablation",
        "operator_ablation", "motion_cond_ablation",
        "lmax_ablation__L50", "lmax_ablation__L150",
        "hubert_ablation__l6", "hubert_ablation__l7", "hubert_ablation__l8",
        "hubert_ablation__l9", "hubert_ablation__l10", "hubert_ablation__l11",
        "hubert_ablation__l12",
        "pca_dim_ablation__verify64", "pca_dim_ablation__ka32", "pca_dim_ablation__ka96",
        "pca_dim_ablation__kmv16", "pca_dim_ablation__kmv64", "pca_dim_ablation__l12_ka96",
        # night validation suite (2026-11)
        "main_contrast", "power_analysis", "seed_variance_dyadic", "model_x_other",
        "repr_ablation__emoca", "repr_ablation__lp",
        "dose_curve_seedavg__zeroother", "dose_curve_seedavg__cfg0", "latency_curve"]


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


# ---- shift-correction ablation (every model x every seed + GT controls) -----
def build_shift_ablation(name="shift_ablation"):
    """Shift-correction ablation on the COMMON mask (identical frames across columns)."""
    d = load(name)
    cols = d["columns"]
    rows = []
    for lbl in d["order"]:
        c = cols[lbl]
        rows.append([lbl, c["model"], c["group"], c["seed"], c["n_stems"], c["n_identities"],
                     c["own_r2"], c["g_a_m"], c["g_shift"], c["net"], c["floor_sd"],
                     c["ci_net"]["lo"], c["ci_net"]["hi"], c["ci_net"]["p"],
                     c["ci_raw"]["lo"], c["ci_raw"]["hi"], c["ci_raw"]["p"]])
    write_csv(f"{name}.csv",
              ["column", "model", "group", "seed", "n_stems", "n_identities", "own_r2",
               "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4", "floor_sd_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p",
               "raw_ci_lo_x1e4", "raw_ci_hi_x1e4", "raw_p"], rows)
    # seed-averaged dyadic: per model, average the per-stem energies over the 3
    # seeds, then the EXACT identity bootstrap -> seed-averaged net + g_a_m, each
    # with an identity CI. per_seed_*_sd is the spread of the 3 single-seed points.
    sa = d["seed_averaged_dyadic"]
    rows = []
    for m, v in sa.items():
        n, g = v["net"], v["g_a_m"]
        rows.append([m, v["n_seeds"],
                     n["point"], n["lo"], n["hi"], n["p"], v["per_seed_net_sd"],
                     g["point"], g["lo"], g["hi"], g["p"], v["per_seed_g_a_m_sd"]])
    write_csv(f"{name}_seed_avg.csv",
              ["model", "n_seeds",
               "net_x1e4", "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p", "net_per_seed_sd_x1e4",
               "g_a_m_x1e4", "g_a_m_ci_lo_x1e4", "g_a_m_ci_hi_x1e4", "g_a_m_p",
               "g_a_m_per_seed_sd_x1e4"], rows)


# ---- operator ablation: GT reference operator (B_GT) vs the column's own ----
def build_operator_ablation(name="operator_ablation"):
    """Reference-operator ablation: R-DGG net with B_GT (reference) vs B_self (own)."""
    d = load(name)
    cols = d["columns"]
    rows = []
    for lbl in d["order"]:
        c = cols[lbl]
        for oper in ("reference", "self"):
            o = c.get(oper)
            if o is None:          # self operator could not be fit (fold too small)
                rows.append([lbl, c["model"], c["group"], c["seed"], oper,
                             "", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue
            rows.append([lbl, c["model"], c["group"], c["seed"], oper,
                         o["n_stems"], o["n_identities"], o["own_r2"],
                         o["g_a_m"], o["g_shift"], o["net"], o["floor_sd"],
                         o["ci_net"]["lo"], o["ci_net"]["hi"], o["ci_net"]["p"],
                         o["ci_raw"]["lo"], o["ci_raw"]["hi"], o["ci_raw"]["p"]])
    write_csv(f"{name}.csv",
              ["column", "model", "group", "seed", "operator",
               "n_stems", "n_identities", "own_r2",
               "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4", "floor_sd_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p",
               "raw_ci_lo_x1e4", "raw_ci_hi_x1e4", "raw_p"], rows)
    # seed-averaged dyadic: seed-averaged net + identity CI for BOTH operators.
    sa = d["seed_averaged_dyadic"]
    rows = []
    for m, v in sa.items():
        for oper in ("reference", "self"):
            o = v.get(oper)
            if o is None:
                continue
            rows.append([m, oper, v["n_seeds"],
                         o["net"], o["lo"], o["hi"], o["p"], o["per_seed_sd"]])
    write_csv(f"{name}_seed_avg.csv",
              ["model", "operator", "n_seeds",
               "net_x1e4", "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p",
               "net_per_seed_sd_x1e4"], rows)


# ---- motion-conditioning ablation: conditional aud|mot vs marginal aud -----
MOTION_COND_CHANNELS = ("conditional", "marginal", "motion", "motion_given_audio")


def build_motion_cond_ablation(name="motion_cond_ablation"):
    """Motion-conditioning ablation: four channels per column on one common mask —
    conditional aud|mot (R-DGG), marginal aud (motion not removed), motion mot (the
    visible-motion channel), and mot|aud — each with identity CI."""
    d = load(name)
    cols = d["columns"]
    rows = []
    for lbl in d["order"]:
        c = cols[lbl]
        for chan in MOTION_COND_CHANNELS:
            o = c[chan]
            rows.append([lbl, c["model"], c["group"], c["seed"], chan,
                         o["n_stems"], o["n_identities"], o["own_r2"],
                         o["g_a_m"], o["g_shift"], o["net"], o["floor_sd"],
                         o["ci_net"]["lo"], o["ci_net"]["hi"], o["ci_net"]["p"],
                         o["ci_raw"]["lo"], o["ci_raw"]["hi"], o["ci_raw"]["p"]])
    write_csv(f"{name}.csv",
              ["column", "model", "group", "seed", "channel",
               "n_stems", "n_identities", "own_r2",
               "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4", "floor_sd_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p",
               "raw_ci_lo_x1e4", "raw_ci_hi_x1e4", "raw_p"], rows)
    # seed-averaged dyadic: seed-averaged net + identity CI for every channel.
    sa = d["seed_averaged_dyadic"]
    rows = []
    for m, v in sa.items():
        for chan in MOTION_COND_CHANNELS:
            o = v[chan]
            rows.append([m, chan, v["n_seeds"],
                         o["point"], o["lo"], o["hi"], o["p"], o["per_seed_sd"]])
    write_csv(f"{name}_seed_avg.csv",
              ["model", "channel", "n_seeds",
               "net_x1e4", "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p",
               "net_per_seed_sd_x1e4"], rows)


# ---- LMAX (lag-window) ablation: net at 3 windows, same common mask ---------
# L100 is the production window (the committed shift_ablation common-mask run); L48 and
# L150 come from scratchpad/run_lmax_ablation.py (it edits LMAX in _config on a branch and
# re-runs the same shift_ablation harness). "L50" realizes max lag 48 (nearest grid node).
LMAX_WINDOWS = [("L48", "lmax_ablation__L50", 1.92), ("L100", "shift_ablation", 4.0),
                ("L150", "lmax_ablation__L150", 6.0)]


def build_lmax_ablation(name="lmax_ablation"):
    """Lag-window robustness: R-DGG net per column at windows 48/100/150 frames
    (1.9/4/6 s), each on the common mask, with identity CI. One row per column×window."""
    D = {tag: load(src) for tag, src, _ in LMAX_WINDOWS}
    order = D["L100"]["order"]
    rows = []
    for lbl in order:
        base = D["L100"]["columns"][lbl]
        for tag, _src, sec in LMAX_WINDOWS:
            c = D[tag]["columns"][lbl]
            rows.append([lbl, base["model"], base["group"], base["seed"], tag, sec,
                         c["n_stems"], c["n_identities"], c["g_a_m"], c["g_shift"], c["net"],
                         c["ci_net"]["lo"], c["ci_net"]["hi"], c["ci_net"]["p"]])
    write_csv(f"{name}.csv",
              ["column", "model", "group", "seed", "window", "window_sec",
               "n_stems", "n_identities", "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p"], rows)
    # seed-averaged dyadic net per window
    rows = []
    for m in D["L100"]["seed_averaged_dyadic"]:
        for tag, _src, sec in LMAX_WINDOWS:
            v = D[tag]["seed_averaged_dyadic"][m]["net"]
            rows.append([m, tag, sec, v["point"], v["lo"], v["hi"], v["p"]])
    write_csv(f"{name}_seed_avg.csv",
              ["model", "window", "window_sec", "net_x1e4", "net_ci_lo_x1e4",
               "net_ci_hi_x1e4", "net_p"], rows)


# ---- HuBERT-layer ablation: net at layers 6 / 9 / 12, same common mask ------
# All three from the one re-extracted GT cache data/GT/LS_AL_hubert_l6912 (GT audio is
# sufficient: R-DGG always reads GT audio). l9 is the production layer; its fresh-cache run
# is cross-checked against the production shift_ablation by scratchpad/run_hubert_ablation.py.
HUBERT_LAYERS = [(f"l{L}", f"hubert_ablation__l{L}", L) for L in (6, 7, 8, 9, 10, 11, 12)]
# Participation ratio (effective dimensionality of the 768-dim features) per layer, computed
# offline on the per-dim-standardised GT HuBERT features (as the metric standardises before PCA):
# PR = (sum lambda)^2 / sum(lambda^2) over the correlation-matrix eigenvalues. ~187k frames / 15 stems.
HUBERT_EFFDIM = {6: 79.8, 7: 67.2, 8: 68.0, 9: 71.3, 10: 67.4, 11: 52.6, 12: 31.0}


def build_hubert_ablation(name="hubert_ablation"):
    """HuBERT-layer sweep 6..12 (9 is the default), each on the common mask, with identity
    CI, plus the per-layer participation ratio (effective dimensionality)."""
    D = {tag: load(src) for tag, src, _ in HUBERT_LAYERS}
    order = D["l9"]["order"]
    rows = []
    for lbl in order:
        base = D["l9"]["columns"][lbl]
        for tag, _src, layer in HUBERT_LAYERS:
            c = D[tag]["columns"][lbl]
            rows.append([lbl, base["model"], base["group"], base["seed"], layer,
                         c["n_stems"], c["n_identities"], c["g_a_m"], c["g_shift"], c["net"],
                         c["ci_net"]["lo"], c["ci_net"]["hi"], c["ci_net"]["p"]])
    write_csv(f"{name}.csv",
              ["column", "model", "group", "seed", "hubert_layer",
               "n_stems", "n_identities", "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p"], rows)
    rows = []
    for m in D["l9"]["seed_averaged_dyadic"]:
        for tag, _src, layer in HUBERT_LAYERS:
            v = D[tag]["seed_averaged_dyadic"][m]["net"]
            rows.append([m, layer, v["point"], v["lo"], v["hi"], v["p"]])
    write_csv(f"{name}_seed_avg.csv",
              ["model", "hubert_layer", "net_x1e4", "net_ci_lo_x1e4",
               "net_ci_hi_x1e4", "net_p"], rows)
    # per-layer effective dimensionality (participation ratio) alongside the GT anchor net,
    # for the "net vs eff-dim" mechanism read: the late decline (10->11->12) tracks the collapse.
    rows = []
    for tag, _src, layer in HUBERT_LAYERS:
        gt = D[tag]["columns"]["GT"]
        rows.append([layer, HUBERT_EFFDIM[layer], gt["net"], gt["ci_net"]["p"]])
    write_csv(f"{name}_effdim.csv",
              ["hubert_layer", "eff_dim_participation_ratio", "GT_net_x1e4", "GT_net_p"], rows)


# ---- PCA output-dim ablation: audio K_A and motion KMV sweeps at layer 9 -----
# All configs share GRAM_CHUNK=4096 + the in-place scoring (transient memory fixes for the
# wide K_A=96 point); verify64 is the chunk-matched K_A=64/KMV=32 baseline, so the sweep is
# internally consistent. L12_ka96 is the layer-12 PCA-budget-vs-content diagnostic.
PCA_DIM_AUDIO = [("ka32", 32), ("verify64", 64), ("ka96", 96)]     # KMV fixed at 32
PCA_DIM_MOTION = [("kmv16", 16), ("verify64", 32), ("kmv64", 64)]  # K_A fixed at 64
PCA_DIM_SWEEPS = [("audio_K_A", PCA_DIM_AUDIO), ("motion_KMV", PCA_DIM_MOTION)]


def build_pca_dim_ablation(name="pca_dim_ablation"):
    """PCA output-dimension sweep on layer 9 (common mask): audio K_A {32,64,96} and
    motion KMV {16,32,64}, each with identity CI; plus the L12 g_a_m recovery diagnostic."""
    rows = []
    for knob, sweep in PCA_DIM_SWEEPS:
        D = {tag: load(f"{name}__{tag}") for tag, _ in sweep}
        base_tag = sweep[1][0]
        for lbl in D[base_tag]["order"]:
            b = D[base_tag]["columns"][lbl]
            for tag, dim in sweep:
                c = D[tag]["columns"][lbl]
                rows.append([knob, dim, lbl, b["model"], b["group"], b["seed"],
                             c["g_a_m"], c["g_shift"], c["net"],
                             c["ci_net"]["lo"], c["ci_net"]["hi"], c["ci_net"]["p"]])
    write_csv(f"{name}.csv",
              ["knob", "dim", "column", "model", "group", "seed",
               "g_a_m_x1e4", "g_shift_x1e4", "net_x1e4",
               "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p"], rows)
    rows = []
    for knob, sweep in PCA_DIM_SWEEPS:
        D = {tag: load(f"{name}__{tag}") for tag, _ in sweep}
        for m in D[sweep[1][0]]["seed_averaged_dyadic"]:
            for tag, dim in sweep:
                v = D[tag]["seed_averaged_dyadic"][m]["net"]
                rows.append([knob, dim, m, v["point"], v["lo"], v["hi"], v["p"]])
    write_csv(f"{name}_seed_avg.csv",
              ["knob", "dim", "model", "net_x1e4", "net_ci_lo_x1e4", "net_ci_hi_x1e4", "net_p"], rows)
    # L12 diagnostic: GT aligned g_a_m -- does a wider audio PCA recover layer 12?
    l9_64 = load(f"{name}__verify64")["columns"]["GT"]["g_a_m"]
    l12_64 = load("hubert_ablation__l12")["columns"]["GT"]["g_a_m"]
    l12_96 = load(f"{name}__l12_ka96")["columns"]["GT"]["g_a_m"]
    recov = round(100 * (l12_96 - l12_64) / (l9_64 - l12_64), 1)
    write_csv(f"{name}_l12_diagnostic.csv",
              ["point", "g_a_m_x1e4", "note"],
              [["l9_K64", l9_64, "layer-9 target (default)"],
               ["l12_K64", l12_64, "layer-12 at default dim"],
               ["l12_K96", l12_96, "layer-12 at wider audio PCA"],
               ["recovery_pct", recov, "percent of the l12->l9 g_a_m gap closed by K_A 64->96 (low=content degradation)"]])


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
def build_dose_curve(name="dose_curve", tag=""):
    d = load(name)
    write_csv(f"dose_curve{tag}_sampled.csv",
              ["k", "frac", "mean_x1e4", "p50_x1e4", "lo_x1e4", "hi_x1e4", "n_subsets"],
              [[r["k"], r["frac"], r["mean"], r.get("p50", ""), r["lo"], r["hi"], r["n_subsets"]]
               for r in d["sampled_curve"]])
    cf = d["closed_form_curve"]
    write_csv(f"dose_curve{tag}_closedform.csv", ["k", "net_x1e4"],
              list(zip(cf["k"], cf["net"])))
    write_csv(f"dose_curve{tag}_identity_ci.csv", ["k", "point_x1e4", "ci_lo_x1e4", "ci_hi_x1e4"],
              [[int(k), v["point"], v["lo"], v["hi"]] for k, v in d["identity_ci"].items()])
    write_csv(f"dose_curve{tag}_additivity.csv", ["k", "additive_x1e4", "exact_x1e4", "abs_err_x1e4"],
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


def build_pca_size_sweep():
    d = load("pca_size_sweep")
    for dim, key, sizelab in [("dimA_stem_subsample_size", "stem", "n_stems"),
                              ("dimB_row_subsample_size", "row", "n_rows")]:
        rows = []
        for r in d[dim]["per_size"]:
            s = r["net_SD"]
            rows.append([r["size"], r["K"], s["GT"], s["seeded"], s["zero"],
                         r["aud_overlap_mean"], r["mot_overlap_mean"],
                         r["aud_max_angle_mean"], r["mot_max_angle_mean"]])
        write_csv(f"pca_size_{key}.csv",
                  [sizelab, "K", "net_SD_GT_x1e4", "net_SD_seeded_x1e4", "net_SD_zero_x1e4",
                   "aud_overlap", "mot_overlap", "aud_max_angle_deg", "mot_max_angle_deg"], rows)


def build_count_sweep():
    """REWRITTEN count_sweep: per-column dim1 (timeshift N) + dim2 (bootstrap B) + slopes."""
    d = load("count_sweep")
    rows = []
    for col, c in d["columns"].items():
        for r in c["dim1_timeshift"]["per_N"]:
            rows.append([col, r["N"], r["sd_floor_net_empirical"], r["sd_theory_1_over_sqrtN"],
                         r["sd_theory_finite_pool"], c["dim1_timeshift"]["loglog_slope_Nle100"]])
    write_csv("count_sweep_timeshift.csv",
              ["column", "n_nulls", "sd_net_empirical_x1e4", "sd_theory_1_over_sqrtN_x1e4",
               "sd_theory_finite_pool_x1e4", "loglog_slope_Nle100"], rows)
    rows = []
    for col, c in d["columns"].items():
        for r in c["dim2_bootstrap"]["per_B"]:
            rows.append([col, r["B"], r["sd_ci_lo"], r["sd_ci_hi"], r.get("point_sd", 0),
                         c["dim2_bootstrap"]["loglog_slope_lo"], c["dim2_bootstrap"]["loglog_slope_hi"]])
    write_csv("count_sweep_bootstrap.csv",
              ["column", "n_bootstrap", "sd_ci_lo_x1e4", "sd_ci_hi_x1e4", "point_sd_x1e4",
               "loglog_slope_lo", "loglog_slope_hi"], rows)


# ---- night validation suite (2026-11) -------------------------------------
def build_main_contrast():
    d = load("main_contrast")
    c = d["contrast"]
    write_csv("main_contrast.csv",
              ["quantity", "value_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p_one_sided", "n_identities"],
              [["dyadic_minus_talkinghead", c["point"], c["lo"], c["hi"], c["p_one_sided"], d["n_identities"]],
               ["mean_dyadic", d["mean_dyadic_net"], "", "", "", ""],
               ["mean_talking_head", d["mean_talking_head_net"], "", "", "", ""]])


def build_power():
    d = load("power_analysis")
    write_csv("power_analysis.csv",
              ["k_identities", "contrast_ci_width_x1e4", "contrast_ci_width_sd"]
              + [f"pair_ci_width__{k}" for k in d["per_k"][0]["pair_ci_width_mean"]],
              [[r["k"], r["contrast_ci_width_mean"], r.get("contrast_ci_width_sd", "")]
               + [r["pair_ci_width_mean"][k] for k in r["pair_ci_width_mean"]] for r in d["per_k"]])
    th = d["identities_for_contrast_width_THEORY_1oversqrtk"]
    write_csv("power_extrapolation.csv", ["target_ci_width", "identities_needed_1oversqrtk"],
              [[t, th[t]] for t in th])


def build_seed_variance_dyadic():
    d = load("seed_variance_dyadic")
    rows = []
    for m, v in d["per_model"].items():
        ps = v["per_seed_net"]
        rows.append([m, ps.get("unseeded", ""), ps.get("seed0", ""), ps.get("seed1", ""),
                     v["mean_net"], v["sd_net_n3"], v["range_net"]])
    write_csv("seed_variance_dyadic.csv",
              ["model", "net_unseeded_x1e4", "net_seed0_x1e4", "net_seed1_x1e4",
               "mean_net_x1e4", "sd_net_n3_x1e4", "range_net_x1e4"], rows)
    rows = []
    for m, v in d["per_model"].items():
        for p in v["pairwise_seed_deltas"]:
            rows.append([m, p["pair"], p["delta_net"], p["lo"], p["hi"], p["p_two_sided"]])
    write_csv("seed_variance_dyadic_deltas.csv",
              ["model", "pair", "delta_net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p_two_sided"], rows)


def _column_rows(d):
    """(label, group, net, lo, hi, p) rows from a columns-dict harness output."""
    out = []
    for lbl, c in d["columns"].items():
        ci = c["ci_net"]
        out.append([lbl, c.get("group", ""), c["net"], ci["lo"], ci["hi"], ci["p"]])
    return out


def build_model_x_other():
    d = load("model_x_other")
    write_csv("model_x_other.csv",
              ["column", "group", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p"], _column_rows(d))
    write_csv("model_x_other_seed_avg.csv",
              ["model", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p"],
              [[m, v["net"]["point"], v["net"]["lo"], v["net"]["hi"], v["net"]["p"]]
               for m, v in d["seed_averaged_dyadic"].items()])


def build_repr_ablation():
    for rep in ["emoca", "lp"]:
        d = load(f"repr_ablation__{rep}")
        write_csv(f"repr_ablation_{rep}.csv",
                  ["column", "group", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p"], _column_rows(d))
        write_csv(f"repr_ablation_{rep}_seed_avg.csv",
                  ["model", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p"],
                  [[m, v["net"]["point"], v["net"]["lo"], v["net"]["hi"], v["net"]["p"]]
                   for m, v in d["seed_averaged_dyadic"].items()])


def build_dose_curve_seedavg():
    for abl in ["zeroother", "cfg0"]:
        d = load(f"dose_curve_seedavg__{abl}")
        cf = d["closed_form_curve"]
        write_csv(f"dose_curve_seedavg_{abl}_closedform.csv", ["k", "net_x1e4"],
                  list(zip(cf["k"], cf["net"])))
        write_csv(f"dose_curve_seedavg_{abl}_identity_ci.csv",
                  ["k", "point_x1e4", "ci_lo_x1e4", "ci_hi_x1e4"],
                  [[int(k), v["point"], v["lo"], v["hi"]] for k, v in d["identity_ci"].items()])


def build_latency():
    d = load("latency_curve")
    # per-column curve (long form)
    rows = []
    for lbl, cur in d["curves"].items():
        if cur is None:
            continue
        for ms, net in cur:
            rows.append([lbl, ms, net])
    write_csv("latency_curve.csv", ["column", "offset_ms", "net_x1e4"], rows)
    # peaks
    write_csv("latency_peaks.csv", ["column", "peak_offset_ms", "net_at_peak_x1e4", "net_at_0_x1e4"],
              [[lbl, p["offset_ms"], p["net_at_peak"], p["net_at_0"]] for lbl, p in d["peaks"].items()])
    # seed-avg dyadic peaks with CI
    write_csv("latency_seed_avg_peaks.csv",
              ["model", "peak_offset_ms", "net_x1e4", "ci_lo_x1e4", "ci_hi_x1e4", "p"],
              [[m, v["peak"]["offset_ms"], v["peak"]["point"], v["peak"]["lo"], v["peak"]["hi"], v["peak"]["p"]]
               for m, v in d["seed_averaged_dyadic"].items()])


def build_classic_on_rdgg():
    d = json.load(open(SRC / "classic_metrics_si184_table.json"))
    shutil.copy2(SRC / "classic_metrics_si184_table.json", METRICS / "classic_metrics_si184_table.json")
    rows = []
    for src, variants in d.items():
        for variant, models in variants.items():
            for model, mr in models.items():
                for key, val in mr.items():
                    if val is None:
                        continue
                    metric, _, region = key.partition("_")
                    rows.append([src, variant, model, metric, region, val])
    write_csv("classic_on_rdgg_rows.csv",
              ["feature_source", "variant", "model", "metric", "region", "value"], rows)


def main():
    print("copying authoritative JSON -> results/metrics/")
    copy_json()
    print("writing tables -> results/tables/")
    build_leaderboard()
    build_ablation()
    build_shift_ablation()
    build_operator_ablation()
    build_motion_cond_ablation()
    build_lmax_ablation()
    build_hubert_ablation()
    build_pca_dim_ablation()
    build_seed_variance()
    build_dose_curve()
    build_dose_curve("dose_curve_cfg0", "_cfg0")
    build_metric_seed_sweep()
    build_count_sweep()
    build_pca_size_sweep()
    # night validation suite (2026-11)
    build_main_contrast()
    build_power()
    build_seed_variance_dyadic()
    build_model_x_other()
    build_repr_ablation()
    build_dose_curve_seedavg()
    build_latency()
    build_classic_on_rdgg()
    print("done.")


if __name__ == "__main__":
    main()
