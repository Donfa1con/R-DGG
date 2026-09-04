# pfd — Paired Fréchet Distance for synchrony

Captures *joint* listener-speaker dynamics that single-side rPCC
misses. For each frame `t` of every paired (speaker, listener) video,
build the concatenated vector

    paired(t) = [ listener_features(t) ; speaker_features(t) ]

Pool across all matched pairs in a corpus, fit a multivariate Gaussian,
then compute Fréchet distance between GT pair distribution and `<MODEL>`
pair distribution.

```
P-FD² = ||μ_gt − μ_pred||² + tr(Σ_gt + Σ_pred − 2·sqrt(Σ_gt·Σ_pred))
```

Lower = better. GT-vs-GT (same speaker + listener dirs) ⇒ `0`.

Per-group columns mirror the rest of the toolkit (overall ⊕ overall,
posecode ⊕ posecode, etc.). Dimension doubles vs single-side
(e.g. EMOCA overall = 56·2 = 112-d).

## Pairing modes

Same as rPCC / TLCC: `exact` | `flip01` (paired-split corpora, not shipped
here) | `from_file:<pairs.txt>` (SI-184: `data/pairs184.txt`).

## Environment

The six metrics share one pixi env — the manifest is `rdgg/pixi.toml`. Install
it once with `cd ../../rdgg && pixi install`. Every `pixi run` example below
runs through the shared manifest.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --gt_speaker     $ROOT/GT/LS_AL_emoca \
    --gt_listener    $ROOT/GT/LS_AL_emoca \
    --pred_speaker   $ROOT/GT/LS_AL_emoca \
    --pred_listener  $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca \
    --pairing        from_file:$ROOT/pairs184.txt \
    [--label         <MODEL>] [--target_fps 25] \
    [--out           $ROOT/<MODEL>/_metrics/pfd_emoca__LS_AL.csv]
```

Corpus-level CSV — one row per `<MODEL>` × `<split>`. Columns:
`source, n_videos, pfd_<group>...`.

CSV append is idempotent — re-running with the same `--label` is a
no-op (use `--overwrite` to recompute).

## Reactive-regime PFD (`--vad_dir` + `--mask_speaker` / `--mask_listener`)

On SI/LS_AL each mp4 contains both dyad roles, so the joint pool
across all frames mixes 4 behavioural regimes:

| speaker | listener | regime |
|---|---|---|
| speech  | silence  | **reactive** (listener actually listening) |
| silence | speech   | inverted-role |
| speech  | speech   | co-speech overlap |
| silence | silence  | dead air |

Response-aware PFD evaluates on the reactive regime alone.
Two independent VAD masks (one per side, intersected) isolate it:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --gt_speaker     $ROOT/GT/LS_AL_emoca \
    --gt_listener    $ROOT/GT/LS_AL_emoca \
    --pred_speaker   $ROOT/GT/LS_AL_emoca \
    --pred_listener  $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca --pairing from_file:$ROOT/pairs184.txt \
    --label          <MODEL> \
    --vad_dir        $ROOT/GT/LS_AL_vad \
    --mask_speaker   speech --mask_listener silence \
    --variant_suffix reactive
# → pfd_emoca_reactive__LS_AL.csv
```

`--mask_speaker` and `--mask_listener` each accept `both | speech |
silence`. `--variant_suffix` controls the output filename suffix
(when omitted, auto-derived as `<spk_mode>_spk_<ls_mode>_ls`).

This repo ships the SI-184 (`LS_AL`) corpus, where both dyad roles share
an mp4; `pipeline/5_classic.sh` computes the listener-silent ("listening")
variant via `--mask_listener silence`.
