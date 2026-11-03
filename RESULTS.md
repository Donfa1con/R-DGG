# SI-184 reference results

**Corpus = SI-184** — 184 improvised `ipc_conversation` listener stems (the 2
`grounded_gesture` stems, one pair `V00_S2050_I00001256_P1307A/P1308A`, are
excluded as non-conversational; `data/pairs184.txt` holds exactly these 184 in its
first column). **Speaker = GT** — the real interlocutor each model was
conditioned on.

These are the **reference numbers a reproduction should match**:

- **R-DGG** — match each system's value within its reported **95% CI**.
- **Classic metrics** — match within seed / rounding.

The **R-DGG** numbers are **recomputed in this repo** on the full-data
deterministic PCA basis (see the R-DGG provenance below). The **classic-metric**
numbers are **copied verbatim** from the locked reference tables (not recomputed
here). Visual-quality metrics (FID / FVD / IQA / VQA / CSIM / Blinks) are **out of
scope** for this repo and are not reported here.

---

# R-DGG (Reference-Based Directed Granger Gain)

R-DGG is a **corpus-level** statistic: a single GT reference reaction operator is
fit across all identities, and the leaderboard / pairwise numbers are reported
over the whole SI-184 set. It has no per-video value (see `tests/golden/README.md`).

**Statistic:** `aud|mot` = directed Granger gain of the speaker's AUDIO on the
listener's motion, conditioned on the speaker's own co-speech MOTION (visible
co-speech regressed out). Each system is tested vs its OWN timeshift-null;
significance is an identity-clustered bootstrap.

**Provenance / config (as run):** 13 identity clusters (folds NFOLD=4, sizes
48/46/46/44); the target dims are balanced by per-dim GT-target variance before the
fit, and the residual energy is unwhitened; `RIDGE=1e-4`, `SCHUR_RIDGE=1e-1`,
`N_TIMESHIFT=100`, `PAIRED_NULLS=40` (8×8), `BOOTSTRAP=20000`, K_A=64, KMV=32,
LMAX=100, minseg=4s.

**PCA basis (default):** the audio/motion PCA basis is now fit on the **full data**
(all 184 stems, all rows), so the fit is **deterministic** — no random stem or row
subsample. The R-DGG numbers below are **recomputed in this repo on this full-data
basis** (not the earlier 60-stem / 40k-row subsample). Earlier releases used the
subsample basis, which added a ~0.05–0.08 (×1e4) nuisance to each net; removing it
shifts points slightly and tightens the pairwise CIs (see the figure gallery's
`fig_metric_seed` / `fig_pca_size` for the motivation).

## Leaderboard (aud|mot net ×1e4, common mask, 100 nulls)

| # | model | kind | net | 95% CI | p50 | p | ownR² |
|---|---|---|---|---|---|---|---|
| 1 | **GT** (anchor) | GT | **+0.73** | [+0.31, +1.15] | +0.72 | **0.000** | 0.962 |
| 2 | **DyStream** | dyadic | **+0.50** | [+0.07, +1.01] | +0.50 | **0.009** | 0.947 |
| 3 | **AVTR-1** | dyadic | **+0.46** | [+0.09, +0.86] | +0.46 | **0.006** | 0.943 |
| 4 | **AvatarForcing** | dyadic | **+0.32** | [+0.05, +0.54] | +0.31 | **0.010** | 0.952 |
| 5 | SoulX Pro | non-dyad | +0.10 | [−0.35, +0.50] | +0.09 | 0.338 | 0.937 |
| — | GT×other (surrogate) | control | +0.04 | [−0.37, +0.53] | +0.05 | 0.417 | 0.910 |
| 6 | Ditto | non-dyad | −0.02 | [−0.27, +0.22] | −0.01 | 0.546 | 0.933 |
| 7 | FLOAT | non-dyad | −0.02 | [−0.33, +0.33] | −0.02 | 0.558 | 0.952 |
| 8 | SoulX Lite | non-dyad | −0.26 | [−0.54, +0.11] | −0.27 | 0.924 | 0.937 |

**Validation anchors (all hold):** GT anchor #1, p<0.001 · all 3 dyadic
significant (p≤0.010) and above every non-dyadic · all 4 non-dyadic
NON-significant (FLOAT ≈0) · surrogate ≈0 (+0.04, p=0.42) · rank-stability top5
identical under mean/median null. The dyadic #2/#3 pair (DyStream, AVTR-1) swaps
vs earlier releases; the two were never a significant ordering (Δ=−0.03, n.s.), so
the swap is inside the removed PCA nuisance.

**Headline claim (correct framing):** GT shows a significant listener R-DGG
(+0.73, p<0.001); non-dyadic baselines incl. FLOAT show NO significant R-DGG
(FLOAT −0.02, p=0.56; all non-dyadic n.s.). Frame as "GT reacts, FLOAT does not"
(each vs the no-reaction null).

## 8×8 pairwise

### Marginals (own GT&model mask, n varies, 100 nulls)

```
GT         +0.704  [+0.282, +1.115]  n=178
DyStream   +0.500  [+0.073, +1.004]  n=177
AVTR-1     +0.447  [+0.064, +0.852]  n=178
AvatarF    +0.301  [+0.053, +0.516]  n=178
SoulX-P    +0.070  [−0.362, +0.467]  n=178
FLOAT      −0.010  [−0.329, +0.361]  n=178
Ditto      −0.053  [−0.272, +0.155]  n=178
SoulX-L    −0.238  [−0.564, +0.186]  n=178
```

### Pairwise Δ = G_row − G_col (×1e4), GT&A&B frame-exact paired, 40 nulls; * = 95% CI excludes 0

| row \ col | GT | DySt | AVTR | AvFo | SxP | FLOAT | Ditto | SxL |
|---|---|---|---|---|---|---|---|---|
| **GT** | — | +0.23 | +0.24 | +0.38 | +0.61\* | +0.72\* | +0.73\* | +0.94\* |
| **DyStream** | −0.23 | — | +0.03 | +0.17 | +0.39 | +0.53 | +0.50\* | +0.76\* |
| **AVTR-1** | −0.24 | −0.03 | — | +0.14 | +0.37 | +0.47 | +0.49\* | +0.69\* |
| **AvatarF** | −0.38 | −0.17 | −0.14 | — | +0.23 | +0.34 | +0.35\* | +0.55\* |
| **SoulX-P** | −0.61\* | −0.39 | −0.37 | −0.23 | — | +0.10 | +0.12 | +0.32 |
| **FLOAT** | −0.72\* | −0.53 | −0.47 | −0.34 | −0.10 | — | +0.01 | +0.22 |
| **Ditto** | −0.73\* | −0.50\* | −0.49\* | −0.35\* | −0.12 | −0.01 | — | +0.20 |
| **SoulX-L** | −0.94\* | −0.76\* | −0.69\* | −0.55\* | −0.32 | −0.22 | −0.20 | — |

### Significant pairs (10; CI excludes 0), stronger > weaker

- GT > SoulX Lite Δ=+0.94 · DyStream > SoulX Lite +0.76 · AVTR-1 > SoulX Lite +0.69 · AvatarF > SoulX Lite +0.55
- GT > Ditto +0.73 · DyStream > Ditto +0.50 · AVTR-1 > Ditto +0.49 · AvatarF > Ditto +0.35
- GT > FLOAT +0.72 · GT > SoulX Pro +0.61

**Pairwise power is capped by 13 identity clusters (signal-limited, NOT
null-limited).** Only MORE identities would add significant pairs. On the full-data
basis the pairwise CIs are tighter than the earlier subsample basis, so **10** pairs
clear 0 (up from 7): GT now separates from FLOAT and SoulX Pro, and DyStream from
Ditto. Every dyadic system still significantly beats both near-zero baselines
(Ditto, SoulX Lite); no two dyadic systems separate from each other, and no
non-dyadic system separates from the surrogate.

---

# Classic dyadic-listener metrics

**Speaker = GT.** **target_fps = 25** (30 fps GT resampled to the models' 25 fps —
recreates the originals' uniform-fps condition). **No normalization.**

**Definitions = the published originals**, verified verbatim from source:
rPCC / P-FD / SID / Var from DIM (Boese0601/Dyadic-Interaction-Modeling), TLCC from
react2024 (reactmultimodalchallenge/baseline_react2024). Region split: EMOCA
{pose[0:6], exp[6:56]}, LP {rot[0:3], exp = brow⊕eyes⊕mouth [3:42]}.

- **rPCC** = mean over videos of `|corr(GT_listener, speaker) − corr(pred_listener, speaker)|`. Lower = closer to GT (0 = identical). GT anchor = 0 by construction.
- **P-FD** = mean over videos of the paired Fréchet distance between concat(GT_speaker, GT_listener) and concat(GT_speaker, pred_listener). Lower = closer to GT joint dynamics (0 = identical). GT anchor = 0.
- **TLCC\|off\|** = mean over stems of `|lag offset|` (frames) of the peak speaker→listener cross-correlation (±49 @25fps). **TLCC peak** (auxiliary, shown separately) = the mean *signed* peak cross-correlation.
- **SID** = Shannon entropy (bits) of pred-listener assignments to GT-fit k-means clusters (pose/rot k=20, exp k=40); pooled. Higher = more diverse.
- **Var** = pooled per-region variance of the listener motion (flattened over frames×dims); pooled over the corpus. Diversity readout.

Two variants: **full** (all frames) and **listening** (VAD listener-silent mask).
Per-column ×10ⁿ scale factors are folded into the headers so every cell is a plain
fixed-decimal number.

## EMOCA — full

| Model | rPCC_pose | rPCC_exp | TLCC\|off\|_pose | TLCC\|off\|_exp | PFD_pose (×10⁻²) | PFD_exp | SID_pose | SID_exp | Var_pose (×10⁻²) | Var_exp |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GT | 0.000 | 0.000 | 33.46 | 28.37 | 0.000 | 0.00 | 4.116 | 5.251 | 1.700 | 1.552 |
| AVTR-1 | 0.136 | 0.063 | 34.22 | 34.40 | 5.924 | 19.85 | 3.478 | 5.096 | 0.900 | 1.432 |
| AvatarForcing | 0.145 | 0.080 | 37.72 | 36.42 | 5.891 | 27.78 | 3.917 | 4.887 | 1.800 | 1.337 |
| FLOAT | 0.128 | 0.077 | 33.60 | 35.45 | 3.953 | 23.71 | 3.220 | 4.816 | 0.811 | 1.206 |
| SoulX Lite | 0.148 | 0.061 | 34.34 | 35.36 | 5.988 | 19.94 | 3.419 | 5.065 | 0.975 | 1.411 |
| SoulX Pro | 0.125 | 0.056 | 33.05 | 35.70 | 4.751 | 18.75 | 3.499 | 5.103 | 0.982 | 1.441 |
| Ditto | 0.180 | 0.074 | 30.62 | 32.24 | 6.645 | 28.82 | 3.459 | 4.953 | 1.341 | 1.671 |
| DyStream | 0.147 | 0.101 | 34.47 | 34.23 | 7.106 | 33.32 | 3.840 | 4.731 | 1.673 | 1.300 |

## EMOCA — listening

| Model | rPCC_pose | rPCC_exp | TLCC\|off\|_pose | TLCC\|off\|_exp | PFD_pose (×10⁻²) | PFD_exp | SID_pose | SID_exp | Var_pose (×10⁻²) | Var_exp |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GT | 0.000 | 0.000 | 29.91 | 25.40 | 0.000 | 0.00 | 4.001 | 5.231 | 1.789 | 1.463 |
| AVTR-1 | 0.140 | 0.083 | 34.14 | 29.62 | 6.417 | 25.98 | 3.254 | 4.970 | 0.930 | 1.313 |
| AvatarForcing | 0.163 | 0.109 | 35.70 | 31.82 | 7.560 | 37.51 | 3.848 | 4.749 | 2.099 | 1.213 |
| FLOAT | 0.132 | 0.117 | 31.65 | 32.07 | 4.753 | 34.09 | 2.961 | 4.322 | 0.843 | 1.007 |
| SoulX Lite | 0.149 | 0.077 | 32.38 | 31.15 | 6.501 | 25.67 | 3.309 | 4.935 | 1.013 | 1.335 |
| SoulX Pro | 0.129 | 0.071 | 31.12 | 31.19 | 5.418 | 24.26 | 3.354 | 5.060 | 1.001 | 1.386 |
| Ditto | 0.168 | 0.092 | 29.61 | 31.51 | 6.790 | 35.02 | 3.253 | 5.006 | 1.386 | 1.604 |
| DyStream | 0.141 | 0.128 | 32.75 | 29.11 | 7.598 | 40.87 | 3.606 | 4.497 | 1.731 | 1.164 |

## LP — full

| Model | rPCC_rot | rPCC_exp | TLCC\|off\|_rot | TLCC\|off\|_exp | PFD_rot (×10⁻²) | PFD_exp (×10⁻³) | SID_rot | SID_exp | Var_rot (×10⁻²) | Var_exp (×10⁻⁵) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GT | 0.000 | 0.000 | 32.16 | 30.03 | 0.000 | 0.000 | 3.862 | 5.219 | 2.027 | 4.000 |
| AVTR-1 | 0.165 | 0.110 | 32.84 | 33.93 | 3.824 | 0.567 | 3.191 | 5.043 | 0.922 | 3.200 |
| AvatarForcing | 0.152 | 0.143 | 36.82 | 33.68 | 4.653 | 1.043 | 3.830 | 4.854 | 2.947 | 4.700 |
| FLOAT | 0.144 | 0.129 | 31.69 | 33.99 | 2.764 | 0.686 | 3.252 | 4.830 | 0.865 | 3.800 |
| SoulX Lite | 0.179 | 0.092 | 32.60 | 35.17 | 3.867 | 0.554 | 3.133 | 5.021 | 0.854 | 3.400 |
| SoulX Pro | 0.154 | 0.091 | 32.86 | 34.31 | 3.089 | 0.521 | 3.309 | 5.070 | 1.044 | 3.500 |
| Ditto | 0.177 | 0.123 | 32.33 | 32.08 | 4.300 | 0.836 | 3.027 | 5.012 | 1.004 | 3.700 |
| DyStream | 0.172 | 0.168 | 31.74 | 35.27 | 4.349 | 1.155 | 3.458 | 4.544 | 1.212 | 3.900 |

## LP — listening

| Model | rPCC_rot | rPCC_exp | TLCC\|off\|_rot | TLCC\|off\|_exp | PFD_rot (×10⁻²) | PFD_exp (×10⁻³) | SID_rot | SID_exp | Var_rot (×10⁻²) | Var_exp (×10⁻⁵) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GT | 0.000 | 0.000 | 28.40 | 24.58 | 0.000 | 0.000 | 3.745 | 5.171 | 1.900 | 4.000 |
| AVTR-1 | 0.169 | 0.112 | 30.58 | 32.10 | 3.967 | 0.671 | 3.152 | 4.760 | 0.920 | 3.300 |
| AvatarForcing | 0.175 | 0.138 | 35.78 | 30.86 | 5.784 | 1.242 | 3.813 | 4.802 | 3.352 | 5.200 |
| FLOAT | 0.152 | 0.127 | 31.67 | 30.88 | 3.240 | 0.878 | 3.061 | 4.488 | 0.852 | 4.300 |
| SoulX Lite | 0.188 | 0.098 | 32.27 | 34.18 | 4.018 | 0.685 | 3.140 | 4.914 | 0.857 | 3.600 |
| SoulX Pro | 0.170 | 0.094 | 32.94 | 32.75 | 3.332 | 0.633 | 3.271 | 4.997 | 1.013 | 3.700 |
| Ditto | 0.174 | 0.109 | 28.74 | 29.98 | 4.106 | 0.882 | 3.010 | 4.930 | 1.002 | 3.800 |
| DyStream | 0.178 | 0.151 | 31.79 | 30.66 | 4.392 | 1.335 | 3.436 | 4.421 | 1.239 | 4.300 |

## TLCC peak (auxiliary — signed cross-correlation, ×10⁻²)

The offset (`TLCC|off|`, above) overlaps heavily across systems; the signed peak
separates GT (positive speaker→listener coupling) from the models (≈0 / negative),
in both encoders and strongest in the `full` variant.

### EMOCA — peak

| Model | pose·full | exp·full | all·full | pose·listen | exp·listen | all·listen |
|:--|--:|--:|--:|--:|--:|--:|
| GT | 1.61 | 1.70 | 1.50 | 4.82 | 3.69 | 3.48 |
| AVTR-1 | 0.52 | −0.09 | −0.20 | 3.57 | 2.34 | 2.17 |
| AvatarForcing | 0.09 | 0.17 | 0.01 | 2.51 | 2.25 | 2.02 |
| FLOAT | 0.43 | −1.09 | −1.08 | 3.33 | 2.01 | 1.89 |
| SoulX Lite | 1.00 | −0.34 | −0.37 | 3.49 | 2.07 | 1.92 |
| SoulX Pro | 0.76 | 0.12 | 0.02 | 3.02 | 2.12 | 1.93 |
| Ditto | −0.38 | 0.29 | 0.06 | 1.99 | 2.12 | 1.85 |
| DyStream | 0.43 | −0.81 | −0.89 | 3.61 | 2.01 | 1.87 |

### LP — peak

| Model | rot·full | exp·full | all·full | rot·listen | exp·listen | all·listen |
|:--|--:|--:|--:|--:|--:|--:|
| GT | 6.50 | 1.86 | 1.96 | 9.15 | 3.34 | 3.43 |
| AVTR-1 | 3.45 | 0.13 | 0.18 | 5.23 | 1.90 | 1.84 |
| AvatarForcing | 1.34 | 0.72 | 0.64 | 2.86 | 1.71 | 1.57 |
| FLOAT | 3.75 | −0.10 | 0.02 | 5.42 | 1.87 | 1.82 |
| SoulX Lite | 3.38 | −0.21 | −0.13 | 4.91 | 1.50 | 1.46 |
| SoulX Pro | 2.61 | −0.01 | 0.02 | 4.42 | 1.70 | 1.63 |
| Ditto | 1.60 | 0.15 | 0.10 | 2.70 | 1.45 | 1.30 |
| DyStream | 3.76 | −0.07 | 0.04 | 5.92 | 1.79 | 1.80 |

---

_Sources: R-DGG tables recomputed in this repo (`rdgg/compute.py`, full-data
deterministic PCA basis) → `data/GT/_metrics/reactivity_matrix__LS_AL.json`;
classic-metric tables copied verbatim from the classic-metrics reference._
