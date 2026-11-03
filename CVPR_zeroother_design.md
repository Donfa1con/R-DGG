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

## 6. Dose-response curve [proposed]

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

---

## 7. Harness plan [proposed]

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

## 8. Status

- [done] avtr1.py flags; zeroother generation (184/184, 0 failed).
- [done] download mapping fix; insightface cuDNN fix.
- [done] design validated against the code by an opus review; sections 4, 5, 6,
  7 corrected from its findings.
- [done] zeroother video features: insightface, emoca, liveportrait all 184/184,
  0 fail, on GPU. The harness is unblocked.
- [done] GT hubert/vad staleness check: a recomputed stem matches the stored
  feature to the float16 ULP (median diff 1.2e-4, Pearson r = 1.000000), so the
  re-downloaded wavs are content-identical and the GT audio features are valid.
- [proposed] dose-response harness (section 7); awaits command.

## 9. Highest-risk items to resolve before coding

1. **Energy-additivity approximation.** The cheap sweep holds per-stem energies
   fixed, but a true mixed column refits the pooled `ControlFit`
   (`_granger.py:294-339,363`). Quantify the error against exact `capture_energies`
   runs (section 7, Option A) before trusting the mid-curve.
2. **Common capture mask.** Capture stock and zero on the shared
   GT ∩ stock ∩ zero mask; confirm per-stem row counts match.
3. **Features not ready.** zeroother EMOCA and LivePortrait do not exist yet; the
   extraction is running. Do not disturb the GPU job; do not run the harness yet.
4. **Fold count = 4, not 13.** Keep plots and text from conflating the 13
   identity bootstrap clusters with the 4 B-fit folds.
5. **Band semantics.** Pair the subset spread with an identity-bootstrap CI; plot
   the number of zeroed videos as the x-axis.

(The noise-seed difference is accepted by decision and is not a risk item.)

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

Three AVTR-1 columns, all scored against real GT audio, on one shared mask
(GT reactive ∩ stock ∩ zero ∩ cfg0 face_ok; 178 stems, 13 identities). Harness:
`rdgg/dose_curve.py` + `rdgg/cfg0_compare.py` (new files; existing code untouched).

| column | how the speaker audio was ablated | net R-DGG ×1e4 | identity CI | p |
|--------|-----------------------------------|----------------|-------------|---|
| stock | none (real audio, cfg_other=3) | +0.564 | [0.16, 0.99] | 0.003 |
| zeroother | waveform zeroed, cfg_other=3 | +0.498 | [−0.01, 0.94] | 0.027 |
| cfg0 | real audio, cfg_other=0 | +0.258 | [−0.20, 0.73] | 0.14 |

Reference marginals (RESULTS.md): GT +0.70 (p<0.001); non-dyadic baselines ~0
(FLOAT +0.03, Ditto 0.00, SoulX-L −0.21, all n.s.). Ordering:
`non-dyad ~0 < cfg0 +0.26 < zeroother +0.50 < stock +0.56 < GT +0.70`.

Findings:

1. **The metric is not compromised.** Non-dyadic (own-audio-driven, no speaker
   input) models score ~0, and GT scores +0.70. So GT-audio scoring does NOT
   cause a systemic self-audio leak; the metric separates reactive from
   non-reactive. The earlier "self-audio leak" reading of zeroother is wrong:
   own-audio-driven lip-sync models would then also sit at ~0.5, but they are ~0.
2. **Silence is a weak ablation; cfg_other=0 is the proper one.** cfg_other=0
   removes ~0.31 of reactivity vs stock; zeroing the waveform removes only ~0.07.
   A dyadic model fed silence still steers toward reactive behaviour (cfg=3 on
   silence), so it stays reactive.
3. **AVTR-1's reactivity is mixed.** Part flows through the speaker-audio pathway
   (cfg0 < zeroother < stock). But cfg0 stays ~0.26, above the non-dyad floor, so
   a substantial share is prior-driven (own audio + listener prior). Roughly
   half-half.
4. **Not statistically significant.** Every paired drop spans 0
   (Δ(stock−cfg0)=+0.31 p=0.16; Δ(zero−cfg0)=+0.24 p=0.43). The limit is n=13
   identities. The split is directional evidence, not a firm verdict.

Outputs: `data/GT/_metrics/{dose_curve,cfgother0_rdgg}__LS_AL.{json,npz}`.

### 11.1 Seed-matched 4-way (final, supersedes the significance read above)

A seed-matched stock column (`data/AVTR-1_seeded/`, real audio + cfg_other=3 +
per-video seed = same seeds as zeroother and cfg0) was added so all three
ablations share one noise realization. Harness: `rdgg/seeded_4way.py` (new file;
existing code untouched). Shared mask 178 stems, 13 identities.

Marginals: seeded-stock **+0.403** [0.05, 0.72] p=0.013; zeroother **+0.497**
[−0.01, 0.94] p=0.027; cfg0 **+0.259** [−0.20, 0.73] p=0.14; unseeded-stock
cross-check **+0.564**.

Seed-matched paired deltas (identical frames AND noise): Δ(stock−cfg0)=**+0.144**
[−0.28, 0.56] p=0.49; Δ(stock−zero)=−0.094 p=0.70; Δ(zero−cfg0)=+0.238 p=0.44 —
**none significant**.

Two findings:

1. **R-DGG carries substantial seed noise (3 realizations).** The same model,
   audio, and cfg_other=3, run with three different noise seeds (unseeded,
   SEED_BASE=0, SEED_BASE=1), gives net = **+0.566 / +0.404 / +0.618** on one
   common mask (`rdgg/seed_variance.py`, 178 stems). Spread: mean 0.53,
   **SD(n=3) ≈ 0.11, range 0.21**. No seed-only pairwise delta is significant
   (p=0.33–0.87 at n=13), but the realized point estimate moves ~0.2 across seeds —
   the seed SD (~0.11) EXCEEDS the stock↔zeroother gap (0.07) and is ~1/3 of the
   stock↔cfg0 gap (0.31). So a single-run R-DGG is seed-noisy ON THE ORDER OF the
   ablation effects; seed0 was a low draw, which made the seed0-based ablation
   deltas look smaller. The identity bootstrap does NOT capture this (one video per
   person). Reliable ablation/ranking claims need seed-averaging (and/or more
   identities). Final 5-column table (one common mask):
   `cfg0 0.26 < seed0 0.40 < zeroother 0.50 < unseeded 0.57 < seed1 0.62`; the
   three stock seeds are the SAME condition, so 0.40–0.62 is pure seed noise.
2. **The speaker-audio-pathway contribution is not established.** Once noise is
   matched, Δ(stock−cfg0) shrinks from +0.31 to +0.14 and is non-significant; the
   three ablations (0.40 / 0.50 / 0.26) are mutually indistinguishable. Much of
   the earlier apparent gap was noise, not the audio pathway.

**Final verdict.** On SI-184 (13 identities) we cannot establish that AVTR-1's
R-DGG depends on the speaker-audio content — not via silence, not via cfg_other=0,
not seed-matched. The power limit is DOUBLE: n=13 identities (identity CI ±0.4–0.6)
AND single-run seed noise (SD ≈ 0.11), both on the order of or larger than the
ablation effects (0.07–0.31). AVTR-1's reactivity is statistically consistent with
a listener prior; we can prove neither that it uses nor that it ignores the speaker
audio. The metric itself still holds at the broad level (GT +0.70 p<0.001; dyadic
significant; non-dyad ~0). An independent audit confirmed the generation and
scoring are correct. For any reliable fine-grained (close-model or within-model
ablation) claim, average R-DGG over several seeds and/or add identities; a
single-run value carries ~0.11 of seed noise the reported CI does not include.

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

Open options (await direction): within-video temporal dosing for within-subject
power; fold the ~0.13 nuisance SD into the reported CI; or report the honest
"underpowered, no significant directed-audio dependence" framing as is.
New files: `rdgg/{dose_curve,cfg0_compare,seeded_4way,common4way,seed_variance,metric_seed_sweep}.py`,
`results/*`; no existing metric code edited.

---

Commits to the `cvpr` branch happen only on an explicit command.
