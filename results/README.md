# results/ — paper-ready numbers and figures (R-DGG ablation study)

Durable, version-controlled results of the AVTR-1 ablation study, so the paper can
quote numbers and (re)build figures without a GPU or the corpus. Full method and
narrative: `../CVPR_zeroother_design.md`. All R-DGG values are the net directed
gain (aud|mot) ×1e4.

**PCA basis.** The metric default is the **full-data deterministic PCA basis** (all
184 stems, all rows). Every number here is on that basis. The PCA-seed nuisance the
default removes was measured by `metric_seed_sweep.py` / `pca_size_sweep.py`; those
measurements stay as **provenance** (the `metric_seed_sweep_*` and `pca_size_*` CSVs,
the harnesses, and the `fig_metric_seed` / `fig_pca_size` builders in
`make_figures.py`), but the two figures are **no longer rendered into the paper set**
— the seed does not affect the deterministic full-data basis, so the plots are not
used in the paper.

## Layout

| path | what |
|------|------|
| `metrics/*.json` | authoritative computed outputs, copied from the gitignored `data/GT/_metrics/` |
| `tables/*.csv` | tidy — one row per data point; CIs as `ci_lo`/`ci_hi`; units in column names (`_x1e4`, `_deg`) |
| `make_tables.py` | rebuilds `metrics/` + `tables/` from `../data/GT/_metrics/*.json` and `../RESULTS.md` |
| `make_figures.py` | rebuilds `figures/*.pdf` from `tables/*.csv` (matplotlib; edit to restyle) |
| `figures/*.pdf` | rendered figures — regenerable from the CSVs |

The large / regenerable artifacts stay out of git (video + feature dirs under
`data/`, and the `*.npz` under `data/GT/_metrics/`).

## Tables

| csv | content |
|-----|---------|
| `leaderboard.csv` | 8 systems + GT anchor + GT×other surrogate: net, CI, p, own_r2 (from `../RESULTS.md`) |
| `ablation_marginals.csv` | 5 AVTR-1 columns on one common mask: unseeded / seed0 / seed1 / zeroother / cfg0 — g_a_m, g_shift, net, CI, p, own_r2 |
| `ablation_paired_deltas.csv` | identity-clustered paired Δ among those columns, with CI, p, sig |
| `shift_ablation.csv` | shift-correction ablation on ONE shared COMMON mask (identical frames = GT-reactive & every model-seed column's face_ok, so every column has the same n≈177): every model × every seed + GT & surrogate controls — `g_a_m` (uncorrected), `g_shift`, `net` (R-DGG), and identity CI on BOTH `net` and raw `g_a_m` |
| `shift_ablation_seed_avg.csv` | per dyadic model, seed-averaged `net` and `g_a_m` with identity CI (average the per-stem energies over the 3 seeds, then the exact identity bootstrap) + `*_per_seed_sd` (spread of the 3 single-seed points) |
| `operator_ablation.csv` | reference-operator ablation on the SAME common mask: every column × 2 operators — `reference` (GT operator `B_GT`, = R-DGG) and `self` (fit on the column's own reaction, leave-one-identity-fold cross-fit) — `g_a_m`, `g_shift`, `net`, CI on `net` and raw, `own_r2` |
| `operator_ablation_seed_avg.csv` | per dyadic model × operator, seed-averaged `net` with identity CI (same energy-average method) + `net_per_seed_sd` |
| `motion_cond_ablation.csv` | motion-conditioning ablation on the SAME common mask: every column × 4 channels — `conditional` (aud\|mot = audio net of motion, = R-DGG), `marginal` (aud, motion NOT removed), `motion` (mot, the visible-motion channel), `motion_given_audio` (mot\|aud) — `g_a_m`, `g_shift`, `net`, CI on `net` and raw, `own_r2` |
| `motion_cond_ablation_seed_avg.csv` | per dyadic model × channel (4), seed-averaged `net` with identity CI (same energy-average method) + `net_per_seed_sd` |
| `lmax_ablation.csv` | lag-window robustness on the SAME common mask: every column × 3 windows — `L48`/`L100`/`L150` (max lag 48/100/150 frames = 1.9/4/6 s; L100 is the default, from `shift_ablation`) — `g_a_m`, `g_shift`, `net`, identity CI, p |
| `lmax_ablation_seed_avg.csv` | per dyadic model × window, seed-averaged `net` with identity CI |
| `hubert_ablation.csv` | HuBERT-layer ablation on the SAME common mask: every column × 3 layers — `6`/`9`/`12` (9 is the default; GT audio re-extracted at all three into one cache) — `g_a_m`, `g_shift`, `net`, identity CI, p |
| `hubert_ablation_seed_avg.csv` | per dyadic model × HuBERT layer, seed-averaged `net` with identity CI |
| `pca_dim_ablation.csv` | PCA output-dim sweep at layer 9 on the common mask: audio `K_A` {32,64,96} and motion `K_MV` {16,32,64} × every column — `g_a_m`, `g_shift`, `net`, identity CI (verify64 = chunk-matched K_A=64/KMV=32 baseline) |
| `pca_dim_ablation_seed_avg.csv` | per dyadic model × knob × dim, seed-averaged `net` with identity CI |
| `pca_dim_ablation_l12_diagnostic.csv` | GT aligned `g_a_m` at l9/K64, l12/K64, l12/K96 + the % of the l12→l9 gap a wider audio PCA recovers (14% = content degradation, not PCA budget) |

> **DyStream seed provenance.** The two seeded DyStream dirs (`dystream_seeded`,
> `dystream_seeded_b1`) were re-extracted at face-detection threshold
> `--det_thresh 0.1` (default is 0.5). The generated videos are fine, but SCRFD
> scores their large frontal faces below 0.5, so the default pass detected a face on
> only ~37% of frames and dropped most stems below the 4 s minimum (n fell to ~81,
> with a 4× inflated time-shift floor). At 0.1 face_ok is ~1.00 and n is ~178, in
> line with the other models. The unseeded DyStream dir and all other models detect
> fine at 0.5 and are unchanged.
| `seed_variance.csv` | three stock noise seeds (unseeded / base0 / base1): net, CI, own_r2 |
| `seed_variance_summary.csv` | their mean, sample/pop SD, min, max, range |
| `seed_variance_paired_deltas.csv` | paired Δ among the three seeds |
| `dose_curve_sampled.csv` | dose curve stock→zeroother k=0..178: sampled mean + band |
| `dose_curve_closedform.csv` | the closed-form mean curve (179 points) |
| `dose_curve_identity_ci.csv` | identity-CI points at k ∈ {0,44,89,133,178} |
| `dose_curve_additivity.csv` | additive-estimate vs exact-capture validation |
| `dose_curve_cfg0_*.csv` | same dose curve but stock→cfg0 (declines 0.57→0.26, vs the flat stock→zeroother) |
| `metric_seed_sweep_pca.csv` | per PCA-seed setting: subspace max angle, overlap, proj-delta, ‖ΔB‖, per-column net |
| `metric_seed_sweep_timeshift.csv` | per timeshift base: floor, within-draw SE, net |
| `metric_seed_sweep_summary.csv` | metric-seed net SD per knob + combined, vs the generation-seed SD |
| `count_sweep_timeshift.csv` | SD(net) vs number of timeshift nulls N (empirical + 1/√N theory) |
| `count_sweep_bootstrap.csv` | SD(CI bound) vs number of bootstrap resamples B |
| `pca_size_stem.csv` / `pca_size_row.csv` | net SD + basis overlap vs PCA subsample SIZE (stems / rows) |

## Figures (`make_figures.py`)

Paper set (rendered by `make_figures.py` `main()`): `fig_leaderboard` (bar + CI,
coloured by system type), `fig_ablation` (5 columns + CI vs GT anchor / floor),
`fig_shift_ablation` (two panels, every model × every seed + GT controls on ONE
shared common mask = identical frames for every column: uncorrected `g_a_m` left vs
corrected `net` = R-DGG right — shows the correction is needed to read the controls
as ~0), `fig_operator_ablation` (two panels, every model × every seed + GT controls on
the SAME common mask: `net` with the GT reference operator `B_GT` = R-DGG left vs `net`
with each column's own cross-fit operator `B_self` right — shows the self operator
inflates the model columns, DyStream most, so the GT reference projection is needed),
`fig_motion_cond_ablation` (two panels. LEFT = AUDIO channel: `aud|mot` = R-DGG filled vs
marginal `aud` hollow — dyadic/GT sit >0, controls at ~0, and the two markers coincide
(conditioning is a no-op for the number). RIGHT = MOTION channel `mot` — positive for
EVERY column, controls included, so the visible-motion coupling does not discriminate;
this is why R-DGG reads the audio channel, net of motion),
`fig_lmax_ablation` (R-DGG net per column at 3 lag windows 1.9/4/6 s, coloured by window —
the three nearly coincide for every column, GT stays significant and controls stay ~0, so
the metric is robust to the window; GT net declines mildly as the window widens),
`fig_hubert_ablation` (R-DGG net per column at HuBERT layers 6 / 9 / 12, coloured by layer —
UNLIKE the window, the layer is load-bearing: layer 9 is the only one where the GT anchor
is significant; layers 6 and 12 both lose it, and layer 6 also turns two talking-head
controls significantly positive),
`fig_pca_dim_ablation` (two panels: audio `K_A` {32,64,96} has an optimum at 64 — narrow
inflates the controls, wide drops GT and turns a talking-head control significant; motion
`K_MV` {16,32,64} is a flat no-op),
`fig_seed_variance` (3 stock seeds + mean±SD),
`fig_dose_curve` (stock→zeroother, k=0..178), `fig_dose_curve_cfg0` (stock→cfg0,
declines), `fig_count_sweep` (Monte-Carlo noise vs count, ∝ 1/√count).

Provenance only (builders kept, CSVs kept, NOT rendered into the paper set):
`fig_metric_seed` (metric-seed net SD per knob vs the generation-seed SD) and
`fig_pca_size` (net effect + basis overlap vs PCA subsample size) — they measured
the OLD subsample nuisance that the full-data default removes.

## How to regenerate

- **Re-plot only** (no GPU): `python3 results/make_figures.py` (needs matplotlib; system python3 here has it).
- **Rebuild the CSVs** (no GPU): `python3 results/make_tables.py` — reads `../data/GT/_metrics/*.json` + `../RESULTS.md`.
- **Recompute the numbers** (needs features/GPU): run the matching harness in `rdgg/`
  (e.g. `pixi run --manifest-path rdgg/pixi.toml python rdgg/common4way.py --corpus data --split LS_AL`),
  then re-run `make_tables.py` and `make_figures.py`.

## Headline numbers

- **Metric is valid at the broad level:** GT +0.70 (p<0.001); dyadic models
  significant; non-dyadic (lip-sync) baselines ~0; surrogate ~0. (`leaderboard.csv`)
- **No AVTR-1 ablation is a significant drop** at n=13; the speaker-audio-pathway
  contribution is not established. (`ablation_*`, and `../CVPR_zeroother_design.md` §11)
- **The time-shift correction is necessary:** uncorrected, every column sits on a
  common ≈+3.3…+4.0×1e4 floor and reads p≈0 (GT +3.95, surrogate +3.54, FLOAT +3.37,
  Ditto +3.38 — indistinguishable); subtracting `g_shift` collapses every control to
  ~0 (surrogate +0.04 p=0.43; talking-heads −0.12…+0.14, all n.s.) and leaves only GT
  and the dyadic models positive. (`shift_ablation.csv`, `fig_shift_ablation`)
- **The GT reference operator is necessary:** with each column's OWN cross-fit operator
  `B_self`, the model columns inflate — DyStream most (self net ≈ +5.0 vs reference
  +0.66, seed-averaged), AVTR-1 up (+0.77 vs +0.53) — because the self operator fits the
  model's own motion artifact as "reaction". The GT reference operator `B_GT` holds every
  column on one scale, and the controls (surrogate, talking-heads) stay ~0 under both
  operators. The seed-averaged reference net matches the shift ablation net to ±0.001 (a
  cross-check: both are the same R-DGG net under `B_GT`). (`operator_ablation*`,
  `fig_operator_ablation`)
- **Reactivity must be read off the AUDIO channel, net of motion — that is the metric's
  definition, and the MOTION channel shows why.** The speaker's visible motion is NOT a
  usable reactivity signal, two ways. By point estimate it is large and positive for EVERY
  column — surrogate mot +0.34, talking-heads +0.35…+0.78, GT +0.92, dyadic +0.37…+1.19 —
  and the ordering is scrambled: talking-heads score as high as or above genuine dyadic
  seeds (SoulX Pro +0.78 > AVTR-1 seed1 +0.40), so motion magnitude does not rank real
  listeners above non-reactive ones. By significance the motion identity CIs are wide and
  NOTHING is significant, GT included (p=0.056; surrogate p=0.12; talking-heads p≥0.08).
  So reading reactivity off visible motion would both mis-rank and certify nothing. Only
  the AUDIO channel separates: GT (p=0.003) and the dyadic models >0, surrogate and
  talking-heads ~0 and n.s. R-DGG reports `aud|mot` = the audio gain net of motion, which
  IS the audio-specific definition, not a bolt-on. **Second, on SI-184 conditioning is a
  no-op for the number** — `aud|mot` equals marginal `aud` to |Δ|≤0.03 (GT +0.599 vs
  +0.590, surrogate +0.043 vs +0.045), because audio and motion are near-orthogonal in the
  reaction (`V(both)≈V(aud)+V(mot)`, seen in `aud|mot≈aud` AND `mot|aud≈mot`), with the
  same seed-spread and CI in both channels. So the audio-specificity is robust here (not
  fragile), and conditioning is free insurance where audio and motion ARE collinear. The
  pre-registered prediction (marginal inflates a control) was NOT observed. We keep
  `aud|mot` (rubric rule 4). (`motion_cond_ablation*`, `fig_motion_cond_ablation`, `../CVPR_zeroother_design.md` §13.1)
- **R-DGG is robust to the lag window (1.9–6 s).** Across windows of max lag 48 / 100 / 150
  frames (1.9 / 4 / 6 s), GT stays significant and positive (+0.691 / +0.599 / +0.492,
  p≤0.01), every control stays ~0 and non-significant (surrogate +0.09 / +0.04 / +0.06,
  talking-heads all n.s.), and the dyadic models stay positive and significant — so no
  conclusion depends on the window. GT net declines mildly as the window widens, i.e. most
  of the audio→reaction coupling sits at short lags (<2 s); the longest window slightly
  dilutes it and costs AvatarForcing its p<0.05 (p=0.08 at 6 s). No window strictly
  dominates (the short window raises GT but nudges the controls up too), so we keep the 4 s
  default (rubric rule 4; switching re-computes the whole leaderboard).
  (`lmax_ablation*`, `fig_lmax_ablation`, `../CVPR_zeroother_design.md` §13.2)
- **The HuBERT layer is load-bearing, and layer 9 is necessary — not a robustness no-op.**
  Re-extracting GT audio at layers 6 / 9 / 12 (GT audio is sufficient; R-DGG always reads GT
  audio), layer 9 is the ONLY one of the three where the GT anchor is significant (+0.60,
  p<0.01). Both neighbours lose the anchor: layer 6 GT +0.17 (p=0.21), layer 12 GT +0.16
  (p=0.21). Layer 6 is the worst — it also turns two talking-head controls significantly
  positive (SoulX Lite +0.37 p=0.04, Ditto +0.26 p=0.02), so the control nulls break too.
  The dyadic models mostly stay positive at all three layers (AVTR-1, DyStream robust), but
  without a significant GT the metric is not validated — construct validity holds only at
  layer 9. This confirms the pre-registered layer choice and mirrors the checkpoint
  precedent (`hubert-large` was rejected for losing the held-out anchor; see
  `extractors/hubert/README.md`). The fresh cache is faithful: layer 9 re-extracted
  reproduces the production baseline to max |Δnet| = 0.005. (`hubert_ablation*`,
  `fig_hubert_ablation`, `../CVPR_zeroother_design.md` §13.3)
- **The PCA output dim: audio `K_A`=64 is an optimum, motion `K_MV` is immaterial.** At
  layer 9, the audio PCA dim has a sweet spot: narrow `K_A`=32 inflates the controls
  (surrogate +0.04→+0.17, SoulX Lite −0.12→+0.34, still n.s.) and the dyadic nets (loses
  specificity); wide `K_A`=96 drops GT to +0.35 and turns a talking-head control
  significant (SoulX Pro +0.45, p=0.01). The default 64 holds the controls closest to 0
  with a significant anchor. The motion dim `K_MV` {16,32,64} changes nothing (|Δ|<0.02) —
  the conditioning removes motion, so its dim is immaterial. **Why layer 12 fails
  (mechanism):** not an extraction/LayerNorm/normalisation artifact (extraction is
  identical per layer; post-norm checkpoint so `hidden_states[12]==last_hidden_state`;
  features are z-scored before PCA). The final HuBERT layer is specialised to the
  pretraining objective and collapses to a low effective rank — participation ratio
  79.8/71.3/31.0 for layers 6/9/12 — squeezing out the reaction signal. A wider PCA at
  layer 12 recovers only 14% of the gap: content degradation, not a PCA budget limit.
  (`pca_dim_ablation*`, `fig_pca_dim_ablation`, `../CVPR_zeroother_design.md` §13.4)
- **Generation seed moves R-DGG** by SD ≈ 0.13 over 3 realizations (0.45 / 0.35 /
  0.60 for the same stock model+audio). (`seed_variance*`)
- **Metric-internal randomness** moves the net less — combined SD ≈ 0.05–0.08 per
  column (biggest mover: the PCA stem-subsample seed); the timeshift floor and the
  bootstrap are effectively deterministic (SD ≈ 0.01–0.02 and ≈ 0). (`metric_seed_sweep_*`)
- **Together**, generation + metric nuisance SD ≈ 0.14 for the stock column —
  comparable to the ablation effects (0.07–0.30), which is why single-seed deltas
  are not reliable at this n.
- **The metric's Monte-Carlo components are well-converged at the defaults:** the
  timeshift floor and the bootstrap CI both shrink ∝ 1/√count (log-log slope ≈ −0.5),
  and at N=100 nulls / B=20000 the noise is small (SD(net) ≈ 0.014 / SD(CI) ≈ 0.004).
  So they are NOT the source of the ~0.05–0.08 metric nuisance — the PCA subsample
  is. (`count_sweep_*`)
- **The PCA subsample is the mover, and only the FULL set removes it.** As the PCA
  stem/row subsample grows, the basis overlap converges smoothly to 1, but the net
  effect on R-DGG stays ~0.04–0.08 across intermediate sizes and drops to 0 only at
  the full set (deterministic basis). Partial increases barely help the metric; use
  all stems / all rows to make the PCA nuisance vanish. (`pca_size_*`)

### Caveat on the PCA source metrics

In `metric_seed_sweep_pca.csv`, `B_rel_frob_*` and `*_rel_proj` are NOT invariant
to in-subspace rotation of the near-degenerate PCA tail directions, so they
OVERSTATE functional change. Anchor the "is the basis stable" claim on the
rotation-invariant `*_overlap` (mean cos² of principal angles, ≈0.95) and on the
downstream net SD, not on ‖ΔB‖.
