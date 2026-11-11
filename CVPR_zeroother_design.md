# CVPR zeroother — design and implementation notes

This file records the design for the AVTR-1 **zeroother** control and the
R-DGG dose-response curve built on it. It is the durable memory for this work
(see CLAUDE.md, principle M). Code is regenerable; this rationale is not.

Status tags: **[done]** = already implemented on this branch; **[proposed]** =
designed, waits for an explicit command before any run or commit.

---

## 1. Goal

Show that R-DGG measures **directed reactivity** (speaker audio → listener
motion), not motion energy or artifacts. The method generates a listener that,
by design, cannot react, then measures how R-DGG responds.

The control condition is **zeroother**: generate AVTR-1 with the speaker
(listen) audio set to zero, so the listener never hears the speaker.

---

## 2. AVTR-1 zeroother generation [done]

The driver `models/generate/avtr1.py` gained three optional flags. The flags are
off by default, so the stock run stays byte-identical.

| flag | effect |
|------|--------|
| `--zero_other_audio` | Zero the `audio_listen` track (speaker audio). The listener is driven by its own speech only. The track keeps its length, so the frame count matches the stock run. |
| `--seed_per_video` | Reseed the torch RNG per stem, so each video has a distinct, reproducible seed (SHA-256 of the stem). The seed prints to the log. |
| `--seed_base N` | Shift every per-stem seed; change it for a whole fresh random set. |

Audio semantics (from the model contract, `avtr-1/src/.../types.py`):
`audio_speech` = the listener's own wav (drives talking motion); `audio_listen`
= the speaker's wav (drives listening / reactive motion). zeroother zeroes
`audio_listen` only.

Run command and output column:

```bash
cd models/avtr-1
pixi run -e renderer python ../generate/avtr1.py \
    --wav_dir ../../data/GT/LS_AL_wav \
    --out_dir ../../data/AVTR-1_zeroother/LS_AL \
    --zero_other_audio --seed_per_video
```

Result: `data/AVTR-1_zeroother/LS_AL/<stem>.mp4`, 184 videos, 25 fps, CFG 3/3/3.
The stock column `data/AVTR-1/LS_AL/` stays intact.

---

## 3. Two prerequisite fixes [done]

Both were real bugs found while running this task.

### 3.1 SI-184 download mapping

`data/download_si184.py` derived the SI file_id by stripping the trailing role
letter (`...P1273A` -> `...P1273`). That id is absent from the release, so the
download could not fetch any stem. The mapping is the **identity**: each stem is
already the file_id, trailing letter included. All 184 stems match verbatim in
the official `seamless_interaction/assets/filelist.csv` (label=improvised,
split=test). `derive_file_id` now returns the whole stem.

### 3.2 insightface ran on CPU

`extractors/insightface/pixi.toml` pinned `cudnn = "8.*"`, but `onnxruntime-gpu
== 1.19.2` needs cuDNN 9. The CUDA provider failed to load (`libcudnn.so.9`
missing) and ONNX Runtime fell back to CPU silently (~1000-6000 s per video).
The fix sets `cudnn = "9.*"` and re-solves the lock. A plain `pixi run` now uses
`CUDAExecutionProvider`. Note: this bug was always present, so the GT and stock
AVTR-1 insightface features were computed on CPU too. Those values are valid;
only the speed was wrong.

---

## 4. Scoring rule — the metric always reads GT audio

**At scoring, R-DGG always reads the real GT speaker audio, never the zeroed
audio.** The zeroed audio is a generation-time tool only. It manufactures the
non-reactive motion; it must not enter the metric.

Reasons:

1. R-DGG reads `GT/LS_AL_hubert` and `GT/LS_AL_vad` by construction
   (`rdgg/_config.py`) and fits the reference operator on GT. No path consumes
   per-model or zeroed audio.
2. Zeroed audio is degenerate. The VAD mask (`reactive_segments_mask`) finds no
   speaker speech, so `collect_design` drops the stem silently when
   `mask.sum() < P + 10` (`rdgg/_features.py:329`). That is a broken mask, not an
   honest floor. (This is NOT the `no-speaker-audio` branch, which fires only
   when the HuBERT speaker cache is empty; silence still yields HuBERT features.)
3. One audio axis keeps the design clean. Only the **motion** reactivity varies
   with the dose; the scoring audio stays fixed.

Clean separation:
- **Generation time:** `audio_other = 0` produces non-reactive motion.
- **Scoring time:** real GT audio vs the motion. For zeroother motion the gain
  lands at the floor. That floor is the signal.

---

## 5. How R-DGG combines gain and floor (for reference)

From `rdgg/_granger.py`:

- `gain(resid) = log( Σ energy_own / Σ row_energy(resid) )` — a **pooled**
  log-energy-ratio over frames and listeners, not a per-video mean.
- `g_a_m = gain(both) − gain(motion_only) = log( Σ resid_motion / Σ resid_both )`
  — the audio gain **beyond the listener's own past motion**. The conditioning
  on past motion makes it robust to pure motion energy.
- `g_shift` = the mean of `g_a_m` over `N_TIMESHIFT = 100` time-shift nulls.
- **R-DGG = `g_a_m(aligned) − g_shift`** = gain above the time-shift floor.

The uncertainty is identity-clustered: a bootstrap over the **13 participant
identities** (`_ident_key(stem) = stem.split("_")[3]`). The resampling unit is
the **person**, not the video.

The cross-fit folds are a different unit. `load_corpus` union-finds sessions that
share an identity into **components**, and sets `NFOLD = len(components)`
(`compute.py:86-112`). The 12 sessions form **4 components**, so there are **4
folds**, not 13. `NFOLD_MAX = 50` (`_config.py:71`) is only an upper guard and
never binds. Persons inside one component are not independent (they share a
session), so an interaction-clustered CI (`_intr_key`) runs alongside the
identity one.

---

## 6. Dose-response curve [done]

### 6.1 Idea

Trace R-DGG as a controlled amount of the data becomes non-reactive by design.
The curve must fall from the stock value toward the floor. A graded curve is
stronger evidence of construct validity than a single null point.

This null is also stronger than the time-shift null in one way. The time-shift
null breaks alignment at scoring, but the model did react. zeroother removes the
stimulus at generation, so no true link exists in the data.

### 6.2 Dose by zeroed videos, no new generation

Use the two existing columns (stock and zeroother). The dose `k` = how many of
the 184 videos have their audio zeroed — use the zeroother motion for k videos
and the stock motion for the rest. The speaker audio stays the real GT audio for
every video.

- At dose `k`, draw random `k`-video subsets, score each, and average for the
  mean R-DGG(k); the spread over subsets gives the band. The mean also has a
  cheap closed form from the per-video energy sums (section 7).
- Report the identity-clustered bootstrap CI (13 persons) that the metric already
  runs, alongside the curve.
- No new generation is needed. All features already exist (stock + zeroother).

X-axis = **number of zeroed videos** (0..184), the controlled quantity. Fraction
of frames is a secondary diagnostic only, used if subset scatter stays visible.

### 6.3 What this curve cannot show

This dosing (whole videos) answers "how does R-DGG scale with the number of
non-reactive videos". It does **not** show within-conversation partial
reactivity. That second question needs temporal zeroing at generation (zero a
fraction of the timeline), which costs extra generation passes. It stays out of
scope unless requested.

### 6.4 Expected endpoint

- `R-DGG(all-zero) ≈ 0`: the k = 184 end of the curve sits at the floor. With no
  true link, the aligned gain equals the shift floor (`g_a_m(all-zero) ≈
  g_shift(all-zero)`). This is the curve's bottom, nothing more.

The separate question "does `g_shift(stock)` equal `g_a_m(all-zero)`?" is a
different analysis, NOT part of this curve. See section 10.

### 6.5 Confounds to report

1. **Motion energy / OOD.** Silence-fed motion may be more static. `g_a_m`
   conditions on past motion, so it resists this, but report the motion energy
   of zeroother vs stock to close the question.
2. **Self-audio turn-taking leak.** In zeroother the listener's own audio is
   real. Real speaker and listener audio correlate by turn-taking, so aligned
   speaker audio can keep a small residual gain that the shift removes. The
   reactive-segment mask (listener silent) suppresses most of it. Comparing
   `g_a_m(zero)` with `g_shift(zero)` tests whether it leaks past the mask.
3. **Common capture frames.** Stock and zeroother have different `face_ok`
   (per-video face tracking, not audio). Capture both on ONE shared mask (GT
   reactive ∩ stock face_ok ∩ zero face_ok) so a video swap exchanges identical
   rows. Confirm the per-stem row counts match; if zero `face_ok` is full, the
   masks coincide and this is moot.
4. **X-axis = number of zeroed videos.** Plot the controlled quantity (0..184).
   The gain pools frames, so videos differ in weight, but averaging over subsets
   at each k absorbs it. Use fraction of frames only as a secondary diagnostic.
5. **Band is not a CI.** The subset spread is a finite-population spread. Overlay
   the identity-bootstrap CI; do not read the band as statistical uncertainty.

(The noise-seed difference between the two columns is accepted and not treated as
a confound, by decision.)

### 6.6 Term: "expected-over-schedules" (how the implemented dose CI works)

A **schedule** is one choice of WHICH k of the N videos are switched to the
ablation at dose level k. There are C(N,k) schedules. **Expected-over-schedules**
means the curve value at k is the EXPECTED net over a random schedule of size k,
i.e. the mean net over many random size-k schedules. The point is this expectation
(so the choice of which videos is averaged into the POINT, not left in the tails).

`rdgg/dose_curve.py` `dose_ci()` reports this: the net is averaged over `n_sched`
random schedules (averaging the realized log-ratio, so no Jensen bias), then the
identity clusters are bootstrapped on that schedule-averaged net. So the CI is pure
population (identity) uncertainty, its width is comparable across k, and it does NOT
balloon mid-curve from the combinatorics of picking k videos. At k=0 and k=N there
is one schedule, so `dose_ci` reduces to the plain identity bootstrap. This
supersedes the earlier "band is not a CI" plan (point 5 above): the schedule spread
is integrated into the point, and the plotted interval is a valid identity CI.

---

## 7. Harness plan [done]

**Hard constraint: do not edit existing code.** The harness is a NEW standalone
script. It only IMPORTS the functions already written in `rdgg/` and writes new
code on top. It never modifies `compute.py` or the metric modules, and it does
NOT refactor them. If a needed step is only inline in `score_source` and is not
importable, do not refactor it — reproduce the composition from importable
sub-calls, or ask first.

Importable building blocks to reuse: `set_paths`, `load_corpus`,
`load_all_motion`, `collect_design`, `gridded_face_ok`, `fit_bgt_fold`,
`capture_energies`, `gnet`, `gnet_boot_ci`.

**Fit once (column-independent) — confirmed subset-independent:**
- The GT reference operator `BGT` over the 4 folds: `fit_bgt_fold(pg, fold, ctx)`
  reads only the GT design and GT speaker features (`compute.py:274-277`,
  `_granger.py:211-237`).
- The driver / `s_hat` (`ctx.driver`, GT speaker HuBERT + GT motion) and
  `target_scale_inv` (`compute.py:223-273`).

**Capture once per pure column (column-dependent):**
- `Y` (listener motion) and the own-block motion half come from the scored
  column (`collect_design`, `_features.py:303-348`), so `ControlFit` differs
  between stock and zero. Capture `E_stock` and `E_zero` once each with
  `capture_energies(per, ops, ctx)` (`_granger.py:351-388`). This is the only
  GPU-heavy step.
- Use ONE shared capture mask `em[ls] = mask_gtonly ∩ stock_face_ok ∩
  zero_face_ok` via `gridded_face_ok` (`_features.py:353-367`), like the paired
  8x8 `GT&A&B` mask (`_output.py:42-48`). Here `mask_gtonly` carries the shared
  GT reactive/audio part; the two `face_ok` terms are the per-video terms.

**Dose sweep (cheap numpy), x = number of zeroed videos k = 0..184:**
- At each `k`, draw random `k`-video subsets `S`: `E_mix[stem] = E_zero[stem] if
  stem ∈ S else E_stock[stem]`.
- Compute the point as the metric does: `raw = log(Σ MR / Σ BR)`, `floor =
  mean_k log(Σ MN_k / Σ BN_k)`, `R-DGG = (raw − floor) * 1e4` — reuse
  `_bootstrap.gnet` (`_bootstrap.py:11-26,49-59`).
- Average over the sampled subsets for R-DGG(k); percentiles for the band. The
  mean also has a cheap closed form: the per-video sums are additive, so
  `E[Σ] = Σ_stock + (k/184)(Σ_zero − Σ_stock)` per energy term (note Jensen: this
  is `R-DGG` at the mean sums, close but not equal to the mean of `R-DGG`).
- Run `gnet_boot_ci(E_mix, stems, _ident_key, BOOTSTRAP)` for the
  identity-clustered CI at selected k.

**Validate the additivity (exact):** at k=0, k=184, and 2-3 mid subsets, build a
true mixed design `per_mix[ls] = per_zero[ls] if ls ∈ S else per_stock[ls]` and
call `capture_energies(per_mix, ops, ctx)` directly (it refits the pooled
`ControlFit`). Compare to the additive estimate to bound the error. k=0 must also
match the leaderboard AVTR-1 value.

**Outputs:** write `CORPUS/GT/_metrics/dose_curve__LS_AL.{json,npz}` with
`(k, R-DGG_mean, band_lo, band_hi, ci_lo, ci_hi)`, k = number of zeroed videos.

**Do NOT add `AVTR-1_zeroother` to `_config.MODELS`.** That would force
`preflight` to require its dirs and intersect its `face_ok` into the leaderboard
COMMON mask, re-scoring every model (`compute.py:196-210`, `_features.py:216-231`).
Load zeroother by path only.

---

## 8. Status [done]

Setup complete: avtr1.py flags + zeroother generation (184/184, 0 fail); download-mapping and
insightface cuDNN fixes; all zeroother video features extracted (insightface/emoca/liveportrait,
184/184). GT hubert/vad validated against the re-downloaded wavs (median diff 1.2e-4, Pearson
r = 1.000000 — content-identical). The dose-response harness (§6/§7) is implemented; its
pre-coding risks (energy-additivity error, the shared GT∩stock∩zero capture mask, 4 B-fit folds
vs 13 identity clusters, band = subset spread paired with an identity CI) were resolved and the
additivity check validated — see `results/` (`dose_curve*`, incl. `dose_curve_additivity.csv`).

---

## 10. Separate analysis — g_shift vs g_a_m (NOT part of the dose curve)

This is a distinct question about the floor's calibration. Keep it separate from
the dose curve; do not fold it into that task or its harness.

Question: is the stock time-shift floor equal to the zeroother aligned gain,
`g_shift(stock) ≈ g_a_m(all-zero)`? It is a two-link chain:

1. **No-link identity:** `g_a_m(all-zero) ≈ g_shift(all-zero)` — on non-reactive
   motion, aligning the real audio equals shifting it (both are the floor).
2. **Floor-invariance:** `g_shift(all-zero) ≈ g_shift(stock)` — the time-shift
   floor does not depend on the motion distribution.

A gap in either link breaks the equality; a floor-invariance gap flags a
motion-OOD effect. This explains why the metric's built-in null is the time-shift
(motion-matched), not zeroing.

If pursued, report the four numbers — `g_a_m(all-zero)`, `g_shift(all-zero)`,
`g_shift(stock)`, `g_a_m(stock)` — as a small standalone table. It needs no dose
sweep.

---

## 11. Results so far (zeroother + cfg0 ablations)

> **PCA basis note.** The metric default is now the **full-data deterministic PCA
> basis** (all 184 stems, all rows — no random subsample). Every number in §11 and
> in `RESULTS.md` is recomputed on this basis. Releases before this change used a
> 60-stem / 40k-row random subsample, which added a ~0.05–0.08 (×1e4) nuisance per
> net; the qualitative story (silence is a weak ablation, cfg0 the stronger one,
> and no drop is significant at n=13) is unchanged. See §11.2 and the gallery's
> `fig_metric_seed` / `fig_pca_size` for the motivation.

Three AVTR-1 columns, all scored against real GT audio, on one shared mask
(GT reactive ∩ stock ∩ zero ∩ cfg0 face_ok; 178 stems, 13 identities). Harness:
`rdgg/dose_curve.py` + `rdgg/cfg0_compare.py` (new files; existing code untouched).

| column | how the speaker audio was ablated | net R-DGG ×1e4 | identity CI | p |
|--------|-----------------------------------|----------------|-------------|---|
| stock | none (real audio, cfg_other=3) | +0.448 | [0.06, 0.86] | 0.010 |
| zeroother | waveform zeroed, cfg_other=3 | +0.520 | [0.08, 0.95] | 0.010 |
| cfg0 | real audio, cfg_other=0 | +0.152 | [−0.28, 0.58] | 0.254 |

Reference marginals (RESULTS.md): GT +0.70 (p<0.001); non-dyadic baselines ~0
(FLOAT −0.01, Ditto −0.05, SoulX-L −0.24, all n.s.). Ordering:
`non-dyad ~0 < cfg0 +0.15 < stock +0.45 < zeroother +0.52 < GT +0.70`.

Findings:

1. **The metric is not compromised.** Non-dyadic (own-audio-driven, no speaker
   input) models score ~0, and GT scores +0.70. So GT-audio scoring does NOT
   cause a systemic self-audio leak; the metric separates reactive from
   non-reactive. The earlier "self-audio leak" reading of zeroother is wrong:
   own-audio-driven lip-sync models would then also sit at ~0.5, but they are ~0.
2. **Silence is a weak ablation; cfg_other=0 is the proper one.** cfg_other=0
   removes ~0.30 of reactivity vs stock; zeroing the waveform removes nothing
   (zeroother +0.52 ≈ stock +0.45, marginally higher, within seed noise). A dyadic
   model fed silence still steers toward reactive behaviour (cfg=3 on silence), so
   it stays reactive.
3. **AVTR-1's reactivity is mixed.** cfg0 (~0.15) sits well below stock (~0.45),
   so part flows through the speaker-audio guidance pathway. But cfg0 stays above
   the non-dyad floor (~0), so a substantial share is prior-driven (own audio +
   listener prior). Roughly half-half.
4. **Not statistically significant.** Every paired drop spans 0
   (Δ(stock−cfg0)=+0.30 p=0.08; Δ(zero−cfg0)=+0.37 p=0.21). The limit is n=13
   identities. The split is directional evidence, not a firm verdict.

Outputs: `data/GT/_metrics/{dose_curve,cfgother0_rdgg}__LS_AL.{json,npz}`.

### 11.1 Seed-matched 4-way (final, supersedes the significance read above)

A seed-matched stock column (`data/AVTR-1_seeded/`, real audio + cfg_other=3 +
per-video seed = same seeds as zeroother and cfg0) was added so all three
ablations share one noise realization. Harness: `rdgg/seeded_4way.py` (new file;
existing code untouched). Shared mask 178 stems, 13 identities.

Marginals: seeded-stock **+0.351** [−0.04, 0.72] p=0.041; zeroother **+0.520**
[0.08, 0.95] p=0.010; cfg0 **+0.152** [−0.28, 0.58] p=0.254; unseeded-stock
cross-check **+0.448**. (New-basis numbers come from `rdgg/common4way.py`, the
seed-matched four-column common-mask harness.)

Seed-matched paired deltas (identical frames AND noise): Δ(stock−cfg0)=**+0.199**
[−0.15, 0.57] p=0.269; Δ(stock−zero)=−0.169 p=0.508; Δ(zero−cfg0)=+0.368 p=0.212 —
**none significant**.

Two findings:

1. **R-DGG carries substantial seed noise (3 realizations).** The same model,
   audio, and cfg_other=3, run with three different noise seeds (unseeded,
   SEED_BASE=0, SEED_BASE=1), gives net = **+0.445 / +0.350 / +0.602** on one
   common mask (`rdgg/seed_variance.py`, 178 stems). Spread: mean 0.47,
   **SD(n=3) ≈ 0.13, range 0.25**. No seed-only pairwise delta is significant
   (p=0.36–0.59 at n=13), but the realized point estimate moves ~0.25 across seeds —
   the seed SD (~0.13) EXCEEDS the stock↔zeroother gap (0.07) and is ~0.4× the
   stock↔cfg0 gap (0.30). So a single-run R-DGG is seed-noisy ON THE ORDER OF the
   ablation effects; seed0 was a low draw, which made the seed0-based ablation
   deltas look smaller. The identity bootstrap does NOT capture this (one video per
   person). Reliable ablation/ranking claims need seed-averaging (and/or more
   identities). Final 5-column table (one common mask):
   `cfg0 0.15 < seed0 0.35 < unseeded 0.45 < zeroother 0.52 < seed1 0.60`; the
   three stock seeds are the SAME condition, so 0.35–0.60 is pure seed noise.
2. **The speaker-audio-pathway contribution is not established.** Once noise is
   matched, Δ(stock−cfg0) shrinks from +0.30 to +0.20 and is non-significant; the
   three ablations (seeded 0.35 / zeroother 0.52 / cfg0 0.15) are mutually
   indistinguishable. Much of the earlier apparent gap was noise, not the audio
   pathway.

**Final verdict.** On SI-184 (13 identities) we cannot establish that AVTR-1's
R-DGG depends on the speaker-audio content — not via silence, not via cfg_other=0,
not seed-matched. The power limit is DOUBLE: n=13 identities (identity CI ±0.4–0.6)
AND single-run seed noise (SD ≈ 0.13), both on the order of or larger than the
ablation effects (0.07–0.30). AVTR-1's reactivity is statistically consistent with
a listener prior; we can prove neither that it uses nor that it ignores the speaker
audio. The metric itself still holds at the broad level (GT +0.70 p<0.001; dyadic
significant; non-dyad ~0). An independent audit confirmed the generation and
scoring are correct. For any reliable fine-grained (close-model or within-model
ablation) claim, average R-DGG over several seeds and/or add identities; a
single-run value carries ~0.13 of seed noise the reported CI does not include.

Outputs: `data/GT/_metrics/seeded_4way__LS_AL.json`.

### 11.2 Metric-internal seed sweep (`rdgg/metric_seed_sweep.py`)

Does the metric's OWN randomness move the result? Measured at the SOURCE (the
fitted object), not only the final net, on GT / seeded-stock / zeroother.

- **PCA projections are NOT bitwise-deterministic.** The dominant subspace is
  shared (mean cos² overlap ≈ 0.92–0.97) but the low-variance tail of the k=64/32
  components rotates/sign-flips between subsamples, so the worst principal angle
  reaches ~88°. `‖ΔB‖_F` and the projection delta read >100%, but those are NOT
  rotation-invariant (they live in the wobbling coordinate frame) and overstate
  functional change — anchor the claim on the overlap and the net SD.
- **Timeshift floor and bootstrap ARE effectively deterministic** (floor SD across
  bases ≈ 0.01–0.02 ≈ its own within-draw SE; bootstrap point SD = 0, CI-bound SD
  ≈ 0.003–0.005).
- **Net wobble from metric randomness:** combined SD ≈ 0.05–0.08 per column
  (biggest mover: the PCA stem-subsample seed). Smaller than the generation-seed SD
  (0.11) but the same order.
- **Combined nuisance** (generation + metric) ≈ √(0.11² + 0.08²) ≈ 0.13 for the
  stock column — comparable to the ablation effects (0.07–0.31). This independently
  reinforces the underpower finding: the point estimate itself carries seed/metric
  noise on the order of the effects.

- **Monte-Carlo components are well-converged at the defaults** (`rdgg/count_sweep.py`,
  variance-vs-count): the timeshift floor and the bootstrap CI both shrink ∝ 1/√count
  (log-log slope ≈ −0.5, matching theory), and at N=100 / B=20000 the noise is small
  (SD(net) ≈ 0.014, SD(CI) ≈ 0.004). So the default counts are adequate; the metric
  nuisance comes from the PCA subsample, not the null/bootstrap counts.
- **PCA subsample SIZE sweep** (`rdgg/pca_size_sweep.py`): as the stem/row subsample
  grows, the basis overlap converges smoothly to 1, but the NET effect on R-DGG stays
  ~0.04–0.08 across intermediate sizes and reaches 0 only at the FULL set
  (deterministic basis). So the basis wobble damps to a roughly-constant net nuisance;
  partial increases barely help, and only using all stems / all rows removes it. This
  directly measures "how the basis variance affects the final metric" — the answer is
  a near-constant ~0.05–0.08 until full determinism. (K=5 → the intermediate SD values
  are noisy; the smooth overlap→1 and the net→0-only-at-full are the robust reads.)

Validated: an independent audit confirmed the metric_seed_sweep code (faithful seeded
variants, correct knob isolation, rotation-robust subspace measure, no existing-module
edits). The count_sweep reuses those validated blocks and its empirical SD matches the
1/√count theory (slopes −0.49 to −0.56), which checks the implementation.

### 11.3 Paper-ready results

`results/` (committed) holds the durable paper assets: `tables/*.csv` (tidy, CIs as
lo/hi), `metrics/*.json` (authoritative copies), `make_tables.py` / `make_figures.py`
(rebuild the CSVs / the `figures/*.pdf` with no GPU), and a `README.md`. Quote the
CSVs; rebuild or restyle figures from them.

`fig_metric_seed` and `fig_pca_size` measured the OLD 60-stem / 40k-row subsample
nuisance (their harnesses monkeypatch the fitters to the old random subsample) and
motivated the full-data PCA default. The measurement is **kept as provenance** (the
`metric_seed_sweep_*` / `pca_size_*` CSVs, the harnesses, and the figure builders in
`make_figures.py`), but the two figures are **not rendered into the paper set** and
`make_figures.py` `main()` does not build them: the PCA seed no longer affects the
deterministic full-data basis, so the plots are not used in the paper.

Open options (await direction): within-video temporal dosing for within-subject
power; fold the ~0.13 nuisance SD into the reported CI; or report the honest
"underpowered, no significant directed-audio dependence" framing as is.
New files: `rdgg/{dose_curve,cfg0_compare,seeded_4way,common4way,seed_variance,metric_seed_sweep}.py`,
`results/*`; no existing metric code edited.

---

## 12. Seed-averaged leaderboard [done]

**Status: DONE** (commit `f0e71fe`). Implemented as the `seed_averaged_dyadic` block every
ablation harness writes: 3 seeds per stochastic dyadic model → average each stem's energies over
seeds → one exact identity-cluster bootstrap; used throughout §13, with `sigma_seed` reported as a
diagnostic. The reasoning below is the record of why it is done this way.

### 12.1 How we want to compute the leaderboard
Report each model's R-DGG as the **mean over 3 generation seeds**, with an
identity-clustered CI computed on the **per-identity seed-averaged values**:

1. Generate each stochastic model with 3 seeds (same seed set across models).
2. For each identity, average its R-DGG contribution over the 3 seeds.
3. Run the identity-cluster bootstrap ONCE on those seed-averaged per-identity
   values. That interval is the reported CI.

**Do NOT add a separate seed-variance term to the CI.**

### 12.2 Why (the reasoning we must not forget)
The seeds are **per video** (`seed = sha256("{base}:{stem}")`), so the generation
seed noise sits INSIDE each identity's value. The identity bootstrap resamples
identities from their observed values, so it ALREADY contains the per-video seed
noise in the between-identity spread. Adding a `sqrt(identity^2 + seed^2)` term on
top **double-counts**. Averaging seeds does two clean things: it **centres** the
point estimate (a single run is a seed lottery: SD ~0.13; the mean of 3 moves the
lottery to 0.13/sqrt(3) ~0.075) and it slightly **tightens** the CI by de-noising
the per-identity values. It never widens the CI.

`sigma_seed` (the per-run SD, ~0.13 x1e4 on the full-data basis) is therefore only
a **reported diagnostic** and the reason **n=3 seeds is enough** (the knee: single
run centre-SD 0.13, n=3 -> 0.075, n=5 -> 0.058). It is NOT part of the CI formula.
Pool `sigma_seed` across the stochastic models to estimate it well (do not run 20
seeds per model).

**One caveat to check:** a run-level COMMON-MODE seed shift (all videos move
together with the base) would NOT show in the between-identity spread, and would
then need a `sigma_common/sqrt(n)` term. With independent per-video seeds this is
expected to be negligible. Verify it: compare the run-to-run SD of the corpus point
(~0.13) against what one run's identity bootstrap predicts; if they agree, there is
no hidden common mode and the average-then-bootstrap recipe is complete.

### 12.3 Why NOT model-vs-model significance
The binding limit is **13 identities**, not seeds. The significant-pair threshold is
Delta ~0.35-0.5; adjacent models (e.g. DyStream +0.50 vs AVTR-1 +0.46) differ by
far less. More seeds cannot make a close pair significant; only more identities can.
Seed averaging is for a **reproducible leaderboard and an honest CI**, not for
separating the reactive models from each other.

### 12.4 How it was done
All three stochastic diffusion dyadic models now have 3 seeds + motion features (AVTR-1, DyStream,
AvatarForcing); per-stem reseed hooks (`--seed_per_video`/`--seed_base`, `seed = sha256("{base}:{stem}")`)
are in each generator. Audio is shared across seeds (the seed changes only the generated listener
motion; the speaker audio is the real GT input), so only emoca + liveportrait were re-extracted per
seed — vad + hubert reused. The averaging + exact identity bootstrap lives in the ablation harnesses
(`seed_averaged_dyadic`). See §14 for the general ablation workflow.

## 13. Operating-point ablations — pre-registered decision rubric [done]

We ablate three fixed knobs: the motion conditioning, the lag window `LMAX`, and
the HuBERT layer. We declare the decision rule HERE, before any ablation number
is seen. We judge a variant by **construct validity**, not by a larger `net`.
Selecting the variant with the biggest `net` is selection on the outcome; we do
not do it. This rubric repeats the criterion the repo already used to reject
`hubert-large` ("loses the held-out anchor") and waveform normalisation ("moves a
control from covering zero to excluding it").

### 13.0 The rubric (ordered; applied on the ONE common mask)

A variant must hold ALL of these, in priority order:

1. **Positive anchor holds.** GT stays significant and positive. Reference now:
   common-mask net ≈ +0.60, p ≈ 0.003 (full basis +0.70, p < 0.001). A variant
   that drives GT to non-significance (p ≥ 0.05), or roughly halves its net,
   is disqualified.
2. **Negative controls stay at zero.** The surrogate (GT×other) and every
   talking-head (FLOAT, Ditto, SoulX) keep a CI that covers 0. Reference now:
   surrogate +0.04 (p = 0.43); talking-heads −0.12…+0.14 (all n.s.). A variant
   that turns any control significantly positive (one-sided p < 0.05) is
   disqualified — a positive control means the metric credits non-dyadic coupling.
3. **Separation holds.** The dyadic models (AVTR-1, DyStream, AvatarForcing) stay
   positive and significant, and the dyadic−control margin does not collapse.
4. **Minimalism tie-break.** When two settings pass 1–3 equally, keep the fixed
   operating point. Switching the operating point re-computes the committed
   leaderboard and every table, so the bar to switch is high.

**Decision.** The expected and acceptable outcome is that the baseline (layer 9,
`LMAX = 100`, conditional `aud|mot`) passes 1–3 at least as well as each variant.
Then the ablation is a **robustness check**: the result is not an artifact of the
knob. We switch the operating point ONLY when a variant **strictly dominates** on
1–3 with a mechanistic reason, never on a bare `net` gain.

### 13.1 Motion conditioning — `aud|mot` vs marginal `aud`

- **Change.** Add one residual, `e_aud_only = residual_with([audio])`, beside the
  existing `both` and `motion_only` in `capture_energies` (`rdgg/_granger.py`).
  Report marginal `aud` net = log(e_own / e_aud_only) next to `aud|mot`. Code
  only; no re-extraction.
- **Pre-registered prediction (before the run).** The current target conditions
  out the speaker's visible motion, because speaker audio and speaker motion are
  co-speech collinear (the `SCHUR_RIDGE` note, `rdgg/_config.py`). We predicted:
  marginal `aud` moves a control (surrogate or a talking-head) significantly
  positive (fails rubric rule 2), because motion mimicry leaks into the audio block.

- **Result (what actually happened). The prediction was FALSIFIED, and the real
  finding is stronger, in two parts.** We report four channels (each the same net,
  relabelled): `aud|mot` (R-DGG), marginal `aud`, marginal `mot` (motion only),
  `mot|aud`.
  1. **The MOTION channel does NOT discriminate, two ways.** By point estimate it
     is large and positive for every column: surrogate `mot` +0.34, talking-heads
     +0.35…+0.78, GT +0.92, dyadic +0.37…+1.19 (all ×1e4), and the ordering is
     scrambled — talking-heads score as high as or above genuine dyadic seeds
     (SoulX Pro +0.78 > AVTR-1 seed1 +0.40), so motion magnitude does not rank real
     listeners above non-reactive ones. By significance the motion identity CIs are
     wide and NOTHING is significant, GT included (p=0.056; surrogate p=0.12;
     talking-heads p≥0.08). So reading reactivity off visible motion would both
     mis-rank and certify nothing. Only the AUDIO channel separates (GT p=0.003 and
     dyadic >0, surrogate and talking-heads ~0 and n.s.). So `aud|mot` — the audio
     gain net of motion — IS the audio-specific definition of the metric, not a
     bolt-on. This is the paper-relevant result of the ablation.
  2. **Conditioning is a no-op for the NUMBER on SI-184.** `aud|mot` equals marginal
     `aud` to |Δ|≤0.03 (GT +0.599 vs +0.590; surrogate +0.043 vs +0.045; dyadic
     seed-averaged within +0.02; no control crosses into significance; same
     seed-spread 0.14–0.20 and same single-seed CI width in both channels). Audio
     and motion are near-orthogonal in the reaction: `V(both)≈V(aud)+V(mot)`, seen
     in the symmetry `aud|mot≈aud` AND `mot|aud≈mot`. So the feared co-speech
     collinearity does not bite the floor-corrected net HERE; the audio-specificity
     is robust, not fragile, and the conditioning is free insurance on corpora or
     against models where audio and motion ARE collinear in the reaction.

- **Decision.** Keep the conditional `aud|mot` (rubric rule 4 — it is the
  construct-valid definition and costs nothing here). The honest write-up leads
  with result 1 (why the metric is audio-specific), then reports result 2 (the
  conditioning is a no-op on this corpus — a robustness statement, not a necessity
  one). See `results/` (`motion_cond_ablation*`, `fig_motion_cond_ablation`).

### 13.2 Lag window — `LMAX` 50 / 100 / 150

- **Change.** `make_lag_grid` `base` stops at 100, so `LMAX = 150` is a no-op as
  written. `scratchpad/run_lmax_ablation.py` extends `base` with 125, 150 for the
  6 s window, runs the same `shift_ablation.py` harness at each window in a fresh
  process (editing `LMAX` in `_config.py` on the branch, then restoring it), and
  tags the output `lmax_ablation__L{N}`. L100 is the production window (the committed
  `shift_ablation` common-mask run); only L50 and L150 are run. "L50" realizes max
  lag 48 (nearest grid node ≤50; base jumps 48→60), so the windows are 48/100/150
  frames = 1.9/4/6 s. Code only; no re-extraction.
- **Result (robustness confirmed; baseline kept).** All three windows pass the
  rubric, so no conclusion depends on the window:
  1. **Anchor (rule 1):** GT significant and positive at every window — +0.691 /
     +0.599 / +0.492 (p = 0.001 / 0.003 / 0.007). GT net declines monotonically as
     the window widens: most of the audio→reaction coupling sits at short lags
     (<2 s), and the longer window dilutes it.
  2. **Controls (rule 2):** surrogate ~0 and n.s. at every window (+0.09 / +0.04 /
     +0.06), all talking-heads n.s. (p ≥ 0.12). None crosses into significance.
  3. **Separation (rule 3):** dyadic seed-averaged positive at every window (AVTR-1
     ≈ +0.48…+0.53 p ≤ 0.002; DyStream ≈ +0.65…+0.76 p ≤ 0.01; AvatarForcing
     ≈ +0.20…+0.22). AvatarForcing is the weakest and loses p < 0.05 only at the 6 s
     window (p = 0.08).
  4. **Decision (rule 4):** no window strictly dominates — the short 1.9 s window
     raises the GT anchor but also nudges the controls up, and the 6 s window shrinks
     every column and costs AvatarForcing its significance. So keep the 4 s default
     (switching re-computes the whole leaderboard). The mild trend argues, if
     anything, AGAINST lengthening the window, consistent with the fixed 4 s. Report
     as a robustness check. See `results/` (`lmax_ablation*`, `fig_lmax_ablation`).

### 13.3 HuBERT layer — 6 / 9 / 12

- **Change.** Re-extract **GT only** at layers 6, 9, 12 into one fresh cache
  `data/GT/LS_AL_hubert_l6912` (`extractors/hubert/extract.py --layers 6 9 12`). GT
  is sufficient because R-DGG always reads GT audio for the speaker driver AND the
  listener own-audio block; no path reads per-model audio (§4; `rdgg/compute.py:197`).
  `scratchpad/run_hubert_ablation.py` then runs the `shift_ablation.py` harness at
  each layer in a fresh process (editing `HUB_LAYER` and the `HUB` dir in `_config.py`
  on the branch, then restoring), tagging the output `hubert_ablation__l{N}`. Layer 9
  is re-run from the fresh cache as a cross-check against the production
  `shift_ablation`.
- **Cross-check (cache is faithful).** Layer 9 from the fresh cache reproduces the
  production layer-9 baseline to max |Δnet| = 0.005 ×1e4 over the 15 columns (fp16
  rounding), so the layer-6 and layer-12 differences are real, not an extraction
  artifact.
- **Result (FULL sweep 6..12 — layer 9 sits in a robust working plateau 7–11; only
  the two extreme layers fail, by two distinct mechanisms).** A first pass sampled
  only 6/9/12 and concluded "only 9 works, both neighbours fail"; that was a
  3-point artifact (6 and 12 are the failing extremes). Extracting the four missing
  layers (7,8,10,11, GT only) gives the real shape:
  1. **Anchor across layers.** GT net / p: l6 +0.17/0.21 (n.s.), l7 +0.49/0.04,
     l8 +0.65/0.0003, l9 +0.60/0.003, l10 +0.67/0.002, l11 +0.38/0.017, l12
     +0.15/0.21 (n.s.). So the GT anchor is **significant at 7,8,9,10,11** and fails
     only at the two ends (6 and 12). Layer 9 is comfortably inside the plateau, not
     a knife-edge — the choice is robust to ±2 layers.
  2. **Controls.** Only **layer 6** turns controls significantly positive (SoulX
     Lite +0.36 p=0.04, Ditto +0.26 p=0.02). Layers 7–12 keep every control ~0 and
     n.s.
  3. **Dyadic.** AVTR-1 and DyStream are **robustly positive and significant at every
     layer** (p<0.01) — they are strong signals and do not discriminate the layer.
     AvatarForcing (the weakest) is n.s. at 6/10/11, significant at 7/8/9/12. The
     discriminating criteria are the anchor and the control-nullness.
  4. **Two failure mechanisms at the ends.** Layer **6** fails despite the HIGHEST
     effective rank (participation ratio 80) — it is too early/acoustic, so it reads
     generic coupling that lights up the talking-head controls, without the specific
     speech-reaction signal (anchor n.s.). Layer **12** fails the other way: the
     final layer specialises to the pretraining objective and collapses to low rank
     (participation ratio 31 vs ~67–71 for 7–10), losing the subtle signal (controls
     stay clean, anchor lost). The GT net decline 10→11→12 (+0.67→+0.38→+0.15) tracks
     the rank collapse (67→53→31). See the mechanism note above and
     `results/hubert_ablation_effdim.csv`.
  5. **Decision.** Keep layer 9 — it is central in the working band 7–11. The honest
     framing is NOT "only 9 works" but "9 is a safe, central choice in a robust
     plateau; the metric breaks only at the extreme early (too acoustic) and final
     (rank-collapsed) layers." See `results/` (`hubert_ablation*`,
     `fig_hubert_ablation`).

- **Mechanism (why layer 12 fails — verified, not a hunch).** It is NOT an extraction
  or normalisation artifact. (a) Extraction is identical for every layer: `extract.py`
  takes `hidden_states[N]`, same code/index/dtype. (b) LayerNorm is symmetric: the
  checkpoint is post-norm (`do_stable_layer_norm=False`), each transformer layer ends
  with its own `final_layer_norm`, and the one `encoder.layer_norm` is applied at the
  START (verified in `modeling_hubert.py:966-968`); layer 12 skips no norm and 6/9 get
  no extra one, and `hidden_states[12] == last_hidden_state` empirically. (c) The metric
  DOES normalise before PCA (per-dim z-score in `fit_pca`, then the lagged design is
  standardised again). The real cause is the LEARNED representation of the final layer:
  it is specialised toward HuBERT's pretraining objective and collapses to a much lower
  effective rank. Measured on the extracted features (per-dim standardised, as the
  metric does): participation ratio (effective dim of 768) is 79.8 / 71.3 / **31.0** for
  layers 6 / 9 / 12, and the per-frame norm grows 9.0 → 9.0 → 12.6. So the subtle
  "reacts to what was said" signal that layer 9 holds across ~71 directions is squeezed
  out at layer 12. The §13.4 L12 diagnostic (a wider PCA recovers only 14% of the gap)
  confirms this is content degradation, not a PCA-budget effect.

### 13.4 PCA output dimension — audio `K_A_TARGET`, motion `KMV` (with the L12 diagnostic)

This is the PCA OUTPUT dimension, not the PCA subsample SIZE (that is the separate
`pca_size_sweep`, which the full-data default already settled). The audio driver is
HuBERT reduced to `K_A_TARGET = 64` PCA dims; the motion driver is EMOCA reduced to
`KMV = 32`.

- **Change.** `scratchpad/run_pca_dim_ablation.py` edits `K_A_TARGET` / `KMV` in
  `_config.py` on the branch (unique anchors, verified), runs the `shift_ablation.py`
  harness in a fresh process per config, tags the output `pca_dim_ablation__{tag}`,
  and restores. Code only; no re-extraction. Five new configs on the common mask
  (baseline K_A=64/KMV=32/layer 9 is the committed `shift_ablation`):
  1. audio sweep (layer 9): K_A ∈ {32, 128}
  2. motion sweep (layer 9): KMV ∈ {16, 64}
  3. **L12 diagnostic:** (layer 12, K_A=128) — reuses the `LS_AL_hubert_l6912` cache,
     so it is the one extra run that folds the layer-12 question into this batch.
- **Rubric expectation (§13.0).** The dim is an operating-point knob, so judge it by
  the rubric: does a different dim keep GT significant, the controls ~0, and the
  dyadic models positive? Expected: a robustness result around 64/32 (the metric is
  not a knife-edge on the dim), report it as such unless a dim strictly dominates.
- **Folded-in L12 diagnostic (mechanistic, not rubric-gated).** §13.3 showed layer 12
  loses the GT anchor via a lower aligned `g_a_m` (the floor `g_shift` and `own_r2`
  are layer-invariant), consistent with either (a) content degradation in the final
  HuBERT layer, or (b) a PCA-budget effect — layer 12 is anisotropic, so the top-64
  PCA directions may be spent on nuisance (speaker identity, "rogue" dims) and capture
  less reaction-relevant content. The test: compare `g_a_m(l12, K_A=128)` against
  `g_a_m(l12, K_A=64)` and `g_a_m(l9, K_A=64)`. If the wider audio PCA lifts the
  layer-12 aligned gain toward the layer-9 level, the failure was the PCA budget; if
  it stays low, it is content degradation and PCA is not the cause. (The ridge is
  already ruled out: `floor_sd` is stable across layers, no coefficient blow-up.)

- **Memory note.** K_A=128 does not fit in 22 GiB (the audio design doubles; OOM in
  the fit AND the scoring). `scratchpad/run_pca_dim_ablation.py` makes the wide point
  **K_A=96** fit with two transient, numerically-verified-safe `_granger.py` edits
  (restored in `finally`): lower `GRAM_CHUNK` (fp64 Gram block size), and rewrite
  `apply_lagged_standardisation` in-place (its one caller passes a throwaway gather
  copy). A `verify64` run (K_A=64 under the edits) reproduces the committed baseline to
  max |Δnet| = 0.003 (the GRAM fp32-reassociation; the in-place edit is bit-identical),
  so the edits are harmless; the whole sweep runs at one `GRAM_CHUNK`, with `verify64`
  as the chunk-matched K_A=64 baseline.

- **Result.**
  1. **Audio `K_A`: 64 is an optimum, both neighbours are worse (keep 64).** GT stays
     significant at all three (K_A 32/64/96 → +0.70/+0.60/+0.35, p ≤ 0.04), but at
     **K_A=32** the controls creep up (surrogate +0.04→+0.17, SoulX Lite −0.12→+0.34;
     still n.s. but the margin shrinks) and the dyadic nets inflate — the narrow audio
     PCA loses specificity. At **K_A=96** GT drops to +0.35 AND a talking-head control
     turns **significantly positive** (SoulX Pro +0.45, p=0.01) — a rule-2 failure. The
     default K_A=64 holds the controls closest to 0 with a significant anchor. So 64 is
     not arbitrary; it sits at the sweet spot, and the ablation reports that.
  2. **Motion `KMV`: immaterial (pure no-op).** KMV 16/32/64 changes nothing — GT
     +0.584/+0.599/+0.597, every control and dyadic net within |Δ|<0.02. Expected: the
     motion contribution is near-orthogonal and the `aud|mot` conditioning removes it,
     so how many motion PCA dims there are does not matter.
  3. **L12 diagnostic → content degradation, not PCA budget.** Widening the audio PCA
     at layer 12 (K_A 64→96) lifts GT's aligned `g_a_m` only from +3.51 to +3.57 —
     **14%** of the +3.51→+3.95 gap to layer 9 — and GT stays non-significant
     (p=0.15). So the layer-12 failure is NOT the PCA budget discarding content into
     dims 65-96; it is content degradation. This confirms the anisotropy reading
     (§13.3): layer 12 already holds 0.68 of its variance in the top-64, so there is
     little to recover by widening. (Caveat: K_A=96 is only 1.5×; a much wider PCA was
     not testable in 22 GiB, but +14% over +50% dims is a weak trend consistent with
     content, not budget.) See `results/` (`pca_dim_ablation*`, `fig_pca_dim_ablation`).

## 14. Running a new ablation — the established workflow (repeatable after a session reset)

Every ablation in §13 followed ONE recipe. Repeat it for any new knob. The drivers live in
`scratchpad/` (gitignored, but they persist on disk across a session reset); copy the closest
one as a template: `run_lmax_ablation.py` (one fixed constant), `run_hubert_ablation.py`
(constant + a re-extracted GT cache, with `--layers`/`--cache-suffix`), `run_pca_dim_ablation.py`
(two constants + transient memory fixes + a `verify64` safety gate).

**The core mechanism (all drivers share it).** The metric's operating point is fixed in
`rdgg/_config.py` (a guard even blocks `GRANGER_*` env knobs); the charter says to try an
alternative by editing the constant on a branch. So a driver:
1. backs up `rdgg/_config.py` (and the production `data/GT/_metrics/shift_ablation__LS_AL.{json,npz}`);
2. for each config: transiently edits the fixed constant(s) in `_config.py` (unique string
   anchors, each asserted present), runs `rdgg/shift_ablation.py` in a FRESH
   `pixi run --manifest-path rdgg/pixi.toml python rdgg/shift_ablation.py` subprocess (so it
   re-imports the edited constants), then moves the output to `<ablation>__<tag>__LS_AL.{json,npz}`;
3. restores `_config.py` and the production output in a `finally` (so a crash can't leave the
   metric edited; confirm `git diff --quiet rdgg/_config.py` after).
One `shift_ablation.py` run scores the common mask = 15 columns (GT, GT×other surrogate, 3 dyadic
× 3 seeds with seed-averaging built in, 4 talking-heads), ~25-30 min on the 22 GiB A10 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Reuse committed baselines (e.g. the production
`shift_ablation` = the default operating point) instead of re-running them.

**The steps.**
1. **Pre-register** the decision rule in §13.0 terms BEFORE running (construct validity, not
   `net` maximization): a variant must keep the GT anchor significant, the controls (surrogate +
   talking-heads) ~0 and n.s., and the dyadic models positive; keep the baseline unless a variant
   strictly dominates with a mechanistic reason. Add a `### 13.x` stub stating the knob + expectation.
2. **Features.** If the knob needs new features (e.g. a HuBERT layer), re-extract **GT only** —
   R-DGG always reads GT audio for both the speaker driver and the listener own-block (§4); no
   path reads per-model audio. Existing GT HuBERT caches on disk: `LS_AL_hubert_l6912` (layers
   6/9/12), `LS_AL_hubert_l7811` (7/8/10/11).
3. **Write the driver** (copy a template). Put the per-config list at the top; support a `--tags`
   subset filter so the most informative config can run first.
4. **Validate the driver with a subagent BEFORE running** (standing rule). Check: guaranteed
   `finally` restore; no cumulative patching (each config patches from the pristine backup);
   anchor strings unique; output naming matches `make_tables` (`__LS_AL` suffix); and, for ANY
   edit to metric code, numerical identity.
5. **Memory (only for configs that OOM, e.g. a wider PCA).** Two transient, numerically-checked
   `_granger.py` edits make a wide config fit 22 GiB: lower `GRAM_CHUNK` (fp64 Gram block size),
   and rewrite `apply_lagged_standardisation` in-place (its one caller passes a throwaway gather
   copy). Prove harmlessness with a `verify64`-style run (the default config under the edits must
   reproduce the committed baseline to ~fp-epsilon, max |Δnet| ≈ 0.003), and use that run as the
   chunk-matched baseline so the edit's tiny effect cancels within the sweep.
6. **Downstream** (all clone-sufficient, no GPU): add a builder to `results/make_tables.py`
   (and the config tags to its `JSON` list) → `results/tables/*.csv`; add a figure to
   `results/make_figures.py`; rebuild both; **revert the timestamp-only figure PDFs**
   (`git checkout -- results/figures/<unchanged>.pdf` for every figure whose data did not
   change — `make_figures` rewrites all of them). Write the `### 13.x` Result and a
   `results/README.md` headline bullet + table rows. `make_figures` reads ONLY
   `results/tables/*.csv`, so the committed CSVs rebuild every figure without a GPU or the corpus.
7. **Commit ONLY on an explicit user command.** Stage the specific files; EXCLUDE `pixi.lock`
   (×2), the `models/*` submodule pointers, `models/generate/avtr1.py`, `scratchpad/`, `data/`,
   `RESUME.md`, `R-DGG_CVPR_plan.xlsx`, `seamless_interaction_fs.log`. Author
   `-c user.name="Dmitry Ziganshin" -c user.email="dziganshin@in3d.io"`; end the message with
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**Supporting analyses (offline, no GPU).** The per-stem energies are stashed in the output
`.npz`, so re-scoring (extra channels, seed-averaging, the exact identity bootstrap) runs offline
(see `scratchpad/rebuild_motion_cond_4ch.py`). Feature-level diagnostics (e.g. the participation
ratio in §13.3) read the extracted caches directly.

---

Commits to the `cvpr` branch happen only on an explicit command.
