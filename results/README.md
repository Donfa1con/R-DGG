# results/ — paper-ready numbers and figures (R-DGG ablation study)

Durable, version-controlled results of the AVTR-1 ablation study, so the paper can
quote numbers and (re)build figures without a GPU or the corpus. Full method and
narrative: `../CVPR_zeroother_design.md`. All R-DGG values are the net directed
gain (aud|mot) ×1e4.

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

## Figures (`make_figures.py`)

`fig_leaderboard` (bar + CI, coloured by system type), `fig_ablation` (5 columns +
CI vs GT anchor / floor), `fig_seed_variance` (3 stock seeds + mean±SD),
`fig_dose_curve` (stock→zeroother, k=0..178), `fig_dose_curve_cfg0` (stock→cfg0,
declines), `fig_metric_seed` (metric-seed net SD per knob vs the generation-seed
SD), `fig_count_sweep` (Monte-Carlo noise vs count, ∝ 1/√count).

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
- **Generation seed moves R-DGG** by SD ≈ 0.11 over 3 realizations (0.40 / 0.57 /
  0.62 for the same stock model+audio). (`seed_variance*`)
- **Metric-internal randomness** moves the net less — combined SD ≈ 0.05–0.08 per
  column (biggest mover: the PCA stem-subsample seed); the timeshift floor and the
  bootstrap are effectively deterministic (SD ≈ 0.01–0.02 and ≈ 0). (`metric_seed_sweep_*`)
- **Together**, generation + metric nuisance SD ≈ 0.13 for the stock column —
  comparable to the ablation effects (0.07–0.31), which is why single-seed deltas
  are not reliable at this n.
- **The metric's Monte-Carlo components are well-converged at the defaults:** the
  timeshift floor and the bootstrap CI both shrink ∝ 1/√count (log-log slope ≈ −0.5),
  and at N=100 nulls / B=20000 the noise is small (SD(net) ≈ 0.014 / SD(CI) ≈ 0.004).
  So they are NOT the source of the ~0.05–0.08 metric nuisance — the PCA subsample
  is. (`count_sweep_*`)

### Caveat on the PCA source metrics

In `metric_seed_sweep_pca.csv`, `B_rel_frob_*` and `*_rel_proj` are NOT invariant
to in-subspace rotation of the near-degenerate PCA tail directions, so they
OVERSTATE functional change. Anchor the "is the basis stable" claim on the
rotation-invariant `*_overlap` (mean cos² of principal angles, ≈0.95) and on the
downstream net SD, not on ‖ΔB‖.
