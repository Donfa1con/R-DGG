"""Build results/ablation_report.html — a paper-style report of all 15 R-DGG ablations/validations.

One self-contained HTML (Chart.js via CDN). Reads results/tables/*.csv; summaries are written here.
Each of the 15 items gets: hypothesis, a chart, a table, and a one-line verdict.

    python3 results/make_report.py    ->    results/ablation_report.html
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "tables")


def rd(f):
    p = os.path.join(T, f)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(x):
    try:
        return round(float(x), 3)
    except (ValueError, TypeError):
        return None


OK = {"blue": "#0173B2", "orange": "#DE8F05", "green": "#029E73", "red": "#D55E00",
      "purple": "#CC78BC", "grey": "#949494"}

def extract_curves():
    """Curve points for the line charts, straight from the committed CSVs (self-contained)."""
    cv = {}
    for abl in ["zeroother", "cfg0"]:
        cv[f"dose_{abl}"] = [[int(r["k"]), float(r["point_x1e4"]), float(r["ci_lo_x1e4"]), float(r["ci_hi_x1e4"])]
                             for r in rd(f"dose_curve_seedavg_{abl}_identity_ci.csv")]
    lc = {}
    for r in rd("latency_curve.csv"):
        lc.setdefault(r["column"], []).append([int(float(r["offset_ms"])), float(r["net_x1e4"])])
    want = ["GT", "GT×other", "AVTR-1 [seed-avg]", "DyStream [seed-avg]", "AvatarForcing [seed-avg]"]
    cv["latency"] = {k: lc[k][::2] for k in want if k in lc}
    cv["count_n"] = [[int(r["n_nulls"]), float(r["sd_net_empirical_x1e4"])]
                     for r in rd("count_sweep_timeshift.csv") if r["column"] == "GT"]
    cv["count_b"] = [[int(r["n_bootstrap"]), float(r["sd_ci_lo_x1e4"])]
                     for r in rd("count_sweep_bootstrap.csv") if r["column"] == "GT"]
    cv["power"] = [[int(r["k_identities"]), float(r["contrast_ci_width_x1e4"])] for r in rd("power_analysis.csv")]
    cv["hubert"] = [[int(r["hubert_layer"]), float(r["GT_net_x1e4"]), float(r["eff_dim_participation_ratio"])]
                    for r in rd("hubert_ablation_effdim.csv")]
    return cv


CURVE = extract_curves()


# ---------------------------------------------------------------- per-item builders
def item1():  # g-shift
    sa = rd("shift_ablation_seed_avg.csv")
    rows = [["GT", "+3.94 (indic.)", "", "+0.60", "0.003"]]
    for r in sa:
        rows.append([r["model"], r["g_a_m_x1e4"], "", r["net_x1e4"], r["net_p"]])
    labels = ["GT"] + [r["model"] for r in sa]
    gam = [3.94] + [f(r["g_a_m_x1e4"]) for r in sa]
    net = [0.60] + [f(r["net_x1e4"]) for r in sa]
    return {
        "hyp": "H4 — the time-shift floor correction (net = g_a|m − g_shift) is needed.",
        "summary": "Without the shift correction every column's raw aligned gain g_a|m is ~+3.6–4.0 "
                   "(a large common offset, the floor). Subtracting the mean of 100 time-shift nulls "
                   "leaves the net on which GT and the dyadic models are positive and the controls read ~0.",
        "verdict": "Correction kept — it is what makes the negative controls sit at zero.",
        "table": {"headers": ["column", "g_a|m (raw ×1e4)", "", "net = R-DGG ×1e4", "p"], "rows": rows},
        "chart": {"type": "bar", "labels": labels,
                  "datasets": [{"label": "g_a|m (uncorrected)", "data": gam, "backgroundColor": OK["grey"]},
                               {"label": "net (R-DGG, shift-corrected)", "data": net, "backgroundColor": OK["blue"]}],
                  "yLabel": "×1e4"}}


def item2():  # reference vs self operator
    d = rd("operator_ablation_seed_avg.csv")
    by = {}
    for r in d:
        by.setdefault(r["model"], {})[r["operator"]] = r
    models = list(by)
    rows = [[m, by[m]["reference"]["net_x1e4"], by[m]["reference"]["net_p"],
             by[m]["self"]["net_x1e4"], by[m]["self"]["net_p"]] for m in models]
    return {
        "hyp": "H4 — the GT reference projection B_GT is needed (vs each column's own self-fit operator).",
        "summary": "A self-fitted operator inflates the model columns (DyStream most), because it can "
                   "absorb column-specific structure. The shared GT reference operator keeps the scores "
                   "comparable across systems.",
        "verdict": "Reference operator kept — the self operator inflates model columns.",
        "table": {"headers": ["model", "reference net", "p", "self net", "p"], "rows": rows},
        "chart": {"type": "bar", "labels": models,
                  "datasets": [{"label": "reference B_GT (R-DGG)", "data": [f(by[m]["reference"]["net_x1e4"]) for m in models], "backgroundColor": OK["blue"]},
                               {"label": "self operator", "data": [f(by[m]["self"]["net_x1e4"]) for m in models], "backgroundColor": OK["orange"]}],
                  "yLabel": "net ×1e4"}}


def item3():  # motion conditioning
    d = rd("motion_cond_ablation_seed_avg.csv")
    by = {}
    for r in d:
        by.setdefault(r["model"], {})[r["channel"]] = r
    models = list(by)
    chans = ["conditional", "marginal", "motion", "motion_given_audio"]
    rows = [[m] + [by[m][c]["net_x1e4"] for c in chans] for m in models]
    cols = [OK["blue"], OK["green"], OK["orange"], OK["purple"]]
    return {
        "hyp": "H3/H4 — is partialling out the speaker's visible motion (aud|mot) needed?",
        "summary": "The MOTION channel is positive for every column — controls included — so visible "
                   "motion magnitude does not discriminate reactive from non-reactive. Only the AUDIO "
                   "channel separates. aud|mot ≈ marginal aud (conditioning is a no-op for the number on "
                   "SI-184), so R-DGG is the audio gain net of motion — robust, not fragile.",
        "verdict": "Keep aud|mot: it is the audio-specific definition; conditioning is free insurance.",
        "table": {"headers": ["model", "aud|mot (R-DGG)", "aud", "mot", "mot|aud"], "rows": rows},
        "chart": {"type": "bar", "labels": models,
                  "datasets": [{"label": c, "data": [f(by[m][c]["net_x1e4"]) for m in models], "backgroundColor": cols[i]}
                               for i, c in enumerate(chans)],
                  "yLabel": "net ×1e4"}}


def item4():  # lmax
    d = rd("lmax_ablation_seed_avg.csv")
    by = {}
    for r in d:
        by.setdefault(r["model"], {})[r["window"]] = r
    models = list(by)
    wins = ["L48", "L100", "L150"]
    rows = [[m] + [by[m][w]["net_x1e4"] for w in wins] for m in models]
    cols = [OK["orange"], OK["blue"], OK["green"]]
    return {
        "hyp": "H3 — robustness to the lag window (1.9 / 4 / 6 s).",
        "summary": "All three windows pass: GT stays significant, controls ~0, dyadic positive. GT net "
                   "declines mildly as the window widens (most coupling is at short lags). No window "
                   "strictly dominates, so the 4 s default is kept.",
        "verdict": "Robust to the window; keep 4 s (L100).",
        "table": {"headers": ["model", "1.9 s (L48)", "4 s (L100)", "6 s (L150)"], "rows": rows},
        "chart": {"type": "bar", "labels": models,
                  "datasets": [{"label": ["1.9 s", "4 s (default)", "6 s"][i], "data": [f(by[m][w]["net_x1e4"]) for m in models], "backgroundColor": cols[i]}
                               for i, w in enumerate(wins)],
                  "yLabel": "net ×1e4"}}


def item5():  # hubert
    h = CURVE.get("hubert", [])
    rows = [[str(l), f"{net:+.2f}", f"{ed:.0f}"] for l, net, ed in h]
    return {
        "hyp": "H3 — which HuBERT layer; is layer 9 a knife-edge?",
        "summary": "GT anchor is significant across layers 7–11 (a robust plateau) and fails only at the "
                   "ends: layer 6 is too acoustic (lights the controls) and layer 12 collapses to low "
                   "effective rank (participation ratio 31 vs ~67–71). Layer 9 sits comfortably inside the band.",
        "verdict": "Keep layer 9 — central in a robust 7–11 plateau, not a knife-edge.",
        "table": {"headers": ["layer", "GT net ×1e4", "eff. dim (PR)"], "rows": rows},
        "chart": {"type": "line", "labels": [str(l) for l, _, _ in h], "xLabel": "HuBERT layer",
                  "datasets": [{"label": "GT net ×1e4", "data": [net for _, net, _ in h], "borderColor": OK["blue"], "yAxisID": "y"},
                               {"label": "effective dim", "data": [ed for _, _, ed in h], "borderColor": OK["orange"], "yAxisID": "y2"}],
                  "dualAxis": True, "yLabel": "GT net ×1e4", "y2Label": "effective dim"}}


def item6():  # pca dim
    d = [r for r in rd("pca_dim_ablation.csv") if r["knob"] == "audio_K_A" and r["column"] == "GT"]
    dims = [r["dim"] for r in d]
    rows = [[r["dim"], r["net_x1e4"], r["net_p"]] for r in d]
    return {
        "hyp": "H3 — PCA output dimension (audio K_A); is 64 arbitrary?",
        "summary": "K_A = 64 is an optimum: at 32 the controls creep up (narrow PCA loses specificity), at "
                   "96 GT drops and a talking-head control turns significantly positive. Motion K_MV is a "
                   "flat no-op. The default 64 holds the controls closest to 0 with a significant anchor.",
        "verdict": "Keep K_A = 64 (a sweet spot), K_MV = 32 (immaterial).",
        "table": {"headers": ["audio K_A", "GT net ×1e4", "p"], "rows": rows},
        "chart": {"type": "bar", "labels": dims,
                  "datasets": [{"label": "GT net ×1e4", "data": [f(r["net_x1e4"]) for r in d], "backgroundColor": OK["blue"]}],
                  "yLabel": "GT net ×1e4"}}


def item7():  # count sweep
    cn, cb = CURVE.get("count_n", []), CURVE.get("count_b", [])
    rows = [["timeshift N=100 (default)", f"{[s for n,s in cn if n==100][0]:.4f}"],
            ["bootstrap B=20000 (default)", f"{[s for b,s in cb if b==20000][0]:.4f}"]]
    return {
        "hyp": "H8 — are the default Monte-Carlo counts (N=100 nulls, B=20000 bootstrap) adequate?",
        "summary": "Both components converge ∝ 1/√count (log-log slopes ≈ −0.5 across all 18 columns). At "
                   "the defaults the noise is tiny: SD(net) from the floor ≈ 0.013, SD(CI bound) from the "
                   "bootstrap ≈ 0.004. The even log-grid fixed the earlier uneven spacing.",
        "verdict": "Defaults adequate — the metric nuisance is not the null/bootstrap counts.",
        "table": {"headers": ["component (at default)", "SD ×1e4"], "rows": rows},
        "chart": {"type": "line", "labels": [str(n) for n, _ in cn],
                  "xLabel": "count (log)", "logX": True,
                  "datasets": [{"label": "SD(net) vs N nulls", "data": [{"x": n, "y": s} for n, s in cn], "borderColor": OK["blue"]},
                               {"label": "SD(CI) vs B bootstrap", "data": [{"x": b, "y": s} for b, s in cb], "borderColor": OK["orange"]}],
                  "yLabel": "SD ×1e4"}}


def item8():  # seed variance
    d = rd("seed_variance_dyadic.csv")
    rows = [[r["model"], r["net_unseeded_x1e4"], r["net_seed0_x1e4"], r["net_seed1_x1e4"],
             r["sd_net_n3_x1e4"]] for r in d]
    models = [r["model"] for r in d]
    seeds = ["net_unseeded_x1e4", "net_seed0_x1e4", "net_seed1_x1e4"]
    cols = [OK["blue"], OK["orange"], OK["green"]]
    return {
        "hyp": "H2 — how much does R-DGG move from one generation seed to another?",
        "summary": "Three generation seeds per dyadic model. Pooled per-run SD ≈ 0.164 ×1e4 — on the order "
                   "of the ablation effects. Every pairwise seed delta is non-significant (p 0.18–0.79): the "
                   "seed noise sits inside the identity CI, so seed-averaging centres the estimate without "
                   "widening the CI.",
        "verdict": "Average over 3 seeds (done throughout); do not add a seed term to the CI.",
        "table": {"headers": ["model", "unseeded", "seed0", "seed1", "SD(n=3)"], "rows": rows},
        "chart": {"type": "bar", "labels": models,
                  "datasets": [{"label": ["unseeded", "seed0", "seed1"][i], "data": [f(r[s]) for r in d], "backgroundColor": cols[i]}
                               for i, s in enumerate(seeds)],
                  "yLabel": "net ×1e4"}}


def item9():  # repr
    sa = {rep: rd(f"repr_ablation_{rep}_seed_avg.csv") for rep in ["emoca", "lp"]}
    col = {rep: rd(f"repr_ablation_{rep}.csv") for rep in ["emoca", "lp"]}
    def gt(rep):
        return next(r for r in col[rep] if r["column"] == "GT")
    def contrast(rep):
        dy = [f(r["net_x1e4"]) for r in sa[rep]]
        th = [f(r["net_x1e4"]) for r in col[rep] if r["group"] == "non-dyad"]
        return round(sum(dy) / len(dy) - sum(th) / len(th), 3)
    rows = [["EMOCA-only", gt("emoca")["net_x1e4"], gt("emoca")["p"], f"{contrast('emoca'):+.3f}"],
            ["LP-only", gt("lp")["net_x1e4"], gt("lp")["p"], f"{contrast('lp'):+.3f}"]]
    models = [r["model"] for r in sa["emoca"]]
    return {
        "hyp": "H3 — does the result survive on a single motion representation (not the combo)?",
        "summary": "The whole pipeline rebuilt on EMOCA-only and on LivePortrait-only motion. GT stays "
                   "significant and the dyadic−talking-head contrast stays positive in BOTH. EMOCA is "
                   "stronger for GT/DyStream, LP for AVTR-1 — the combo default is the compromise, not a "
                   "cherry-pick.",
        "verdict": "Holds on each representation; combo is not an overfit choice.",
        "table": {"headers": ["representation", "GT net", "p", "contrast"], "rows": rows},
        "chart": {"type": "bar", "labels": ["GT"] + models + ["contrast"],
                  "datasets": [
                      {"label": "EMOCA-only", "backgroundColor": OK["blue"],
                       "data": [f(gt("emoca")["net_x1e4"])] + [f(r["net_x1e4"]) for r in sa["emoca"]] + [contrast("emoca")]},
                      {"label": "LP-only", "backgroundColor": OK["orange"],
                       "data": [f(gt("lp")["net_x1e4"])] + [f(r["net_x1e4"]) for r in sa["lp"]] + [contrast("lp")]}],
                  "yLabel": "net ×1e4"}}


def item10():  # model x other
    al = {r["model"]: r for r in rd("shift_ablation_seed_avg.csv")}
    sc = {r["model"]: r for r in rd("model_x_other_seed_avg.csv")}
    models = [m for m in al if m in sc]
    rows = [[m, al[m]["net_x1e4"], sc[m]["net_x1e4"], sc[m]["p"]] for m in models]
    return {
        "hyp": "H1/H2 — does a model's gain vanish when its listener is paired with the WRONG partner?",
        "summary": "Each model's listener motion is scrambled onto another conversation (same model), scored "
                   "vs the real speaker audio. All dyadic CIs include 0 (specificity holds): AVTR-1 "
                   "+0.53→+0.09, AvatarForcing +0.23→−0.19. DyStream +0.66→+0.48 (p=0.094) drops only weakly "
                   "— its gain is partly not specific to the true partner (candidate: generic turn-taking).",
        "verdict": "Specific for AVTR-1 / AvatarForcing; DyStream only weakly — flagged for discussion.",
        "table": {"headers": ["model", "aligned net", "× other net", "p"], "rows": rows},
        "chart": {"type": "bar", "labels": models,
                  "datasets": [{"label": "aligned (true partner)", "data": [f(al[m]["net_x1e4"]) for m in models], "backgroundColor": OK["blue"]},
                               {"label": "× other (scrambled)", "data": [f(sc[m]["net_x1e4"]) for m in models], "backgroundColor": OK["red"]}],
                  "yLabel": "net ×1e4"}}


def item11():  # dose seed-avg
    z, c = CURVE.get("dose_zeroother", []), CURVE.get("dose_cfg0", [])
    rows = [["k=0 (seed-avg stock)", f"{z[0][1]:+.3f}", f"[{z[0][2]:+.2f}, {z[0][3]:+.2f}]"],
            ["k=N zeroother", f"{z[-1][1]:+.3f}", f"[{z[-1][2]:+.2f}, {z[-1][3]:+.2f}]"],
            ["k=N cfg0", f"{c[-1][1]:+.3f}", f"[{c[-1][2]:+.2f}, {c[-1][3]:+.2f}]"]]
    return {
        "hyp": "H1 — dose-response as videos become non-reactive, with a seed-averaged baseline.",
        "summary": "Baseline (k=0) is now the seed-averaged stock (3 AVTR-1 seeds, +0.47, CI [0.15,0.77]) "
                   "rather than a single-seed lottery. zeroother stays flat (+0.47→+0.52 — silence is a weak "
                   "ablation) while cfg_other=0 falls (+0.47→+0.15 — it removes the speaker-audio pathway).",
        "verdict": "Silence is a weak ablation; cfg_other=0 is the one that removes reactivity.",
        "table": {"headers": ["point", "net ×1e4", "95% identity CI"], "rows": rows},
        "chart": {"type": "line", "labels": [str(k) for k, *_ in z], "xLabel": "k = ablated videos",
                  "datasets": [{"label": "stock → zeroother", "data": [p for _, p, _, _ in z], "borderColor": OK["blue"]},
                               {"label": "stock → cfg_other=0", "data": [p for _, p, _, _ in c], "borderColor": OK["red"]}],
                  "yLabel": "net ×1e4"}}


def item12():  # main contrast
    d = {r["quantity"]: r for r in rd("main_contrast.csv")}
    c = d["dyadic_minus_talkinghead"]
    rows = [["mean dyadic", d["mean_dyadic"]["value_x1e4"], ""],
            ["mean talking-head", d["mean_talking_head"]["value_x1e4"], ""],
            ["contrast (dyadic − TH)", c["value_x1e4"], f"[{c['ci_lo_x1e4']}, {c['ci_hi_x1e4']}], p={c['p_one_sided']}"]]
    return {
        "hyp": "H1/H8 — the single pre-registered test: mean dyadic − mean talking-head > 0.",
        "summary": "One identity-cluster bootstrap, all systems sharing the resampled clusters, no "
                   "multiplicity correction. The contrast is +0.459 with CI [0.033, 0.833] (p=0.017) — the "
                   "interval EXCLUDES 0, so R-DGG separates systems-with-speaker-access from those without, "
                   "even though single systems are underpowered.",
        "verdict": "PASS — the pre-registered contrast is significant (p=0.017).",
        "table": {"headers": ["quantity", "value ×1e4", "95% CI / p"], "rows": rows},
        "chart": {"type": "bar", "labels": ["mean dyadic", "mean TH", "contrast"],
                  "datasets": [{"label": "net ×1e4",
                                "data": [f(d["mean_dyadic"]["value_x1e4"]), f(d["mean_talking_head"]["value_x1e4"]), f(c["value_x1e4"])],
                                "backgroundColor": [OK["blue"], OK["grey"], OK["green"]]}],
                  "yLabel": "net ×1e4", "errorbar": [None, None, [f(c["ci_lo_x1e4"]), f(c["ci_hi_x1e4"])]]}}


def item13():  # classic on R-DGG rows
    d = [r for r in rd("classic_on_rdgg_rows.csv")
         if r["feature_source"] == "emoca" and r["variant"] == "rdgg" and r["region"] == "pose"]
    want = {"GT": "GT", "AVTR-1": "AVTR-1", "dystream": "DyStream", "AvatarForcing": "AvatarForcing",
            "FLOAT": "FLOAT", "SoulX_FlashHead_Pro": "SoulX Pro", "ditto": "Ditto"}
    def val(model, metric):
        for r in d:
            if r["model"] == model and r["metric"] == metric.lower():
                return f(r["value"])
        return None
    rows = []
    for mk, mv in want.items():
        rows.append([mv, val(mk, "rpcc"), val(mk, "pfd")])
    labels = [v for v in want.values()]
    rpcc = [val(k, "rpcc") for k in want]
    pfd = [val(k, "pfd") for k in want]
    return {
        "hyp": "H3 — on the EXACT R-DGG rows, do the classic metrics separate dyadic from talking-head?",
        "summary": "On identical frames, the classic metrics still rank talking-heads at or above the dyadic "
                   "systems: by rPCC the best is SoulX Pro (talking-head), by PFD the best is FLOAT "
                   "(talking-head) — both below every dyadic. The classic metrics do not measure reactivity; "
                   "R-DGG does. This is the honest motivation for the new metric.",
        "verdict": "Classic metrics do NOT separate dyadic from talking-head; R-DGG does.",
        "table": {"headers": ["model", "rPCC (↓ better)", "PFD (↓ better)"], "rows": rows},
        "chart": {"type": "bar", "labels": labels,
                  "datasets": [{"label": "rPCC (pose)", "data": rpcc, "backgroundColor": OK["blue"]},
                               {"label": "PFD (pose)", "data": pfd, "backgroundColor": OK["orange"]}],
                  "yLabel": "distance from GT (lower = closer)"}}


def item14():  # power
    p = CURVE.get("power", [])
    ex = rd("power_extrapolation.csv")
    rows = [[r["target_ci_width"], r["identities_needed_1oversqrtk"]] for r in ex]
    return {
        "hyp": "H8 — how many identities are needed to tighten the main-contrast CI?",
        "summary": "Subsampling k=4..13 identity clusters, the contrast CI width falls 1.03→0.80. The in-range "
                   "slope is shallow (finite-population ceiling at k→13), so extrapolate by the asymptotic "
                   "1/√k anchored at 13: ~34 identities for a CI width of 0.5, ~52 for 0.4, ~93 for 0.3.",
        "verdict": "The 13-identity CI (±0.4) is the binding limit; ~34 identities would halve-ish it.",
        "table": {"headers": ["target CI width", "identities needed (1/√k)"], "rows": rows},
        "chart": {"type": "line", "labels": [str(k) for k, _ in p], "xLabel": "number of identity clusters k",
                  "datasets": [{"label": "contrast 95% CI width", "data": [w for _, w in p], "borderColor": OK["blue"]}],
                  "yLabel": "CI width ×1e4"}}


def item15():  # latency
    lat = CURVE.get("latency", {})
    pk = rd("latency_seed_avg_peaks.csv")
    rows = [[r["model"], f"{int(float(r['peak_offset_ms']))} ms", r["net_x1e4"], r["p"]] for r in pk]
    rows.append(["GT", "−480 ms", "+0.78", "—"])
    colmap = {"GT": "#000000", "GT×other": OK["grey"], "AVTR-1 [seed-avg]": OK["blue"],
              "DyStream [seed-avg]": OK["green"], "AvatarForcing [seed-avg]": OK["orange"]}
    ds = []
    for name, pts in lat.items():
        ds.append({"label": name.replace(" [seed-avg]", ""), "data": [{"x": ms, "y": n} for ms, n in pts],
                   "borderColor": colmap.get(name, OK["purple"]), "borderWidth": 3 if name == "GT" else 1.8})
    return {
        "hyp": "H5 — the latency curve: R-DGG net vs the speaker-driver offset over ±1 s.",
        "summary": "Seed-averaged dyadic peaks are all significant and at NEGATIVE offsets: AVTR-1 −160 ms, "
                   "DyStream −320 ms, AvatarForcing −600 ms. Single seeds scatter widely (−160…−960 ms), so "
                   "seed-averaging is load-bearing. NOTE: GT also peaks negative (−480 ms), not within the "
                   "pre-registered ±80 ms — the curve has a systematic negative bias to resolve.",
        "verdict": "Clear dyadic peaks (seed-avg); GT −480 ms peak is an open question (discuss).",
        "table": {"headers": ["model", "peak offset", "net ×1e4", "p"], "rows": rows},
        "chart": {"type": "line", "labels": None, "xLabel": "speaker-driver offset (ms)", "scatterLine": True,
                  "datasets": ds, "yLabel": "net ×1e4"}}


ITEMS = [
    ("g-shift (time-shift floor correction)", item1),
    ("Reference vs self operator", item2),
    ("Motion conditioning (aud|mot)", item3),
    ("Lag window LMAX (1.9 / 4 / 6 s)", item4),
    ("HuBERT layer (6–12)", item5),
    ("PCA output dimension (K_A, K_MV)", item6),
    ("Monte-Carlo count sweep (N, B)", item7),
    ("Generation-seed variance", item8),
    ("EMOCA-only / LivePortrait-only", item9),
    ("model × other (specificity)", item10),
    ("Dose curve, seed-avg baseline", item11),
    ("Main pre-registered contrast", item12),
    ("Classic metrics on R-DGG rows", item13),
    ("Power analysis (identities)", item14),
    ("Latency curve (±1 s)", item15),
]


def build():
    items = []
    for i, (title, fn) in enumerate(ITEMS, 1):
        d = fn()
        d["n"] = i
        d["title"] = title
        items.append(d)
    html = TEMPLATE.replace("/*DATA*/", json.dumps(items))
    out = os.path.join(HERE, "ablation_report.html")
    open(out, "w").write(html)
    print(f"wrote {out} ({len(items)} items)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-DGG — Ablation &amp; Validation Suite (SI-184)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--ink:#1a1a1a;--mute:#6b7280;--line:#e5e7eb;--bg:#fafafa;--accent:#0173B2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:28px 32px}
h1{margin:0 0 4px;font-size:24px;font-weight:700}
.sub{color:var(--mute);font-size:14px}
.wrap{max-width:980px;margin:0 auto;padding:24px 32px}
.toc{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}
.toc a{font-size:12.5px;color:var(--accent);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:3px 10px;background:#fff}
.toc a:hover{background:#eef6fb}
section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px 24px;margin:18px 0;scroll-margin-top:16px}
.num{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:7px;background:var(--accent);color:#fff;font-weight:700;font-size:14px;margin-right:10px}
h2{font-size:18px;margin:0;display:flex;align-items:center}
.hyp{color:var(--mute);font-size:13px;font-style:italic;margin:8px 0 14px}
.grid{display:grid;grid-template-columns:1.15fr 1fr;gap:22px;align-items:start}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.chartbox{position:relative;height:260px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mute);font-weight:600;border-bottom:1.5px solid var(--line)}
.summary{font-size:13.5px;color:#333;margin:14px 0 10px}
.verdict{font-size:13.5px;font-weight:600;background:#f0f7ff;border-left:3px solid var(--accent);padding:8px 12px;border-radius:0 6px 6px 0}
footer{color:var(--mute);font-size:12px;text-align:center;padding:30px}
</style></head><body>
<header><div class="wrap" style="padding:0">
<h1>R-DGG — Ablation &amp; Validation Suite</h1>
<div class="sub">SI-184 (184 dyadic listener stems, 13 identities). Metric operating point fixed:
HuBERT layer 9, LMAX=100 (4 s), K_A=64, K_MV=32, aud|mot. Values are net R-DGG ×10⁻⁴ unless noted.
15 items — hypothesis, chart, table, verdict.</div>
<div class="toc" id="toc"></div>
</div></header>
<div class="wrap" id="app"></div>
<footer>Generated from <code>results/tables/*.csv</code> by <code>results/make_report.py</code>.
Numbers are also in <code>results/metrics/*.json</code> and <code>CVPR_zeroother_design.md</code> §13/§15/§16.</footer>
<script>
const ITEMS = /*DATA*/;
const toc = document.getElementById('toc'), app = document.getElementById('app');
ITEMS.forEach(d=>{
  const a=document.createElement('a');a.href='#s'+d.n;a.textContent=d.n+'. '+d.title;toc.appendChild(a);
  const s=document.createElement('section');s.id='s'+d.n;
  const th=d.table.headers.map(h=>`<th>${h}</th>`).join('');
  const tr=d.table.rows.map(r=>'<tr>'+r.map(c=>`<td>${c===null||c===undefined?'—':c}</td>`).join('')+'</tr>').join('');
  s.innerHTML=`<h2><span class="num">${d.n}</span>${d.title}</h2>
    <div class="hyp">${d.hyp}</div>
    <div class="grid">
      <div class="chartbox"><canvas id="c${d.n}"></canvas></div>
      <div><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>
    </div>
    <div class="summary">${d.summary}</div>
    <div class="verdict">▸ ${d.verdict}</div>`;
  app.appendChild(s);
});
ITEMS.forEach(d=>{
  const ch=d.chart, ctx=document.getElementById('c'+d.n);
  const scales={y:{title:{display:!!ch.yLabel,text:ch.yLabel||''},grid:{color:'#f0f0f0'}}};
  if(ch.dualAxis)scales.y2={position:'right',title:{display:true,text:ch.y2Label||''},grid:{drawOnChartArea:false}};
  if(ch.xLabel)scales.x={title:{display:true,text:ch.xLabel}};
  if(ch.logX){scales.x={type:'logarithmic',title:{display:true,text:ch.xLabel}}}
  if(ch.scatterLine||ch.logX){scales.x=Object.assign(scales.x||{},{type:ch.logX?'logarithmic':'linear'})}
  const datasets=ch.datasets.map(ds=>Object.assign({tension:.25,pointRadius:ch.type==='line'?2:0,
     borderWidth:ds.borderWidth||2,fill:false},ds));
  new Chart(ctx,{type:ch.type,
    data:ch.labels?{labels:ch.labels,datasets}:{datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{boxWidth:12,font:{size:11}}}},
      scales}});
});
</script></body></html>"""


if __name__ == "__main__":
    build()
