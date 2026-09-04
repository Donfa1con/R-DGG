# variance — feature-trajectory dynamism

Pooled flattened variance of the motion-feature time-series over the
WHOLE corpus (the original DIM definition). Pool every included clip's
per-group features, concatenate across clips, then take one flattened
variance per group:

```
var_<group> = var( concat(all_clips)[group].reshape(-1) )
```

This is ONE scalar per region over the whole corpus, not a per-stem
value. It captures between-clip diversity (the mode-collapse signal)
and within-clip motion together: a model that outputs the same pose in
every clip has low pooled variance even if each clip moves a little.

Higher = more dynamic / more diverse. Mode-collapsed model outputs
trend toward `0`. Single-source — does **not** require a paired
GT/MODEL.

Two feature backends (dual-source convention):

| flag                        | reads                       | groups (`var_overall` plus...) |
|---|---|---|
| `--feature_source emoca`    | `<split>_emoca/<stem>.npz`  | `pose (6), exp (50)` |
| `--feature_source lp`       | `<split>_liveportrait/...`  | `rot (3), exp (39 = brow⊕eyes⊕mouth)` |

LP groups follow the LivePortrait motion-vector layout: so3 rotation
plus the 39 `LIPSYNC_COORDS` indices into the 21×3 expression.

## Environment

The six metrics share one pixi env — the manifest is `rdgg/pixi.toml`. Install
it once with `cd ../../rdgg && pixi install`. Every `pixi run` example below
runs through the shared manifest.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --features_dir   $ROOT/GT/LS_AL_emoca \
    --feature_source emoca \
    [--stems         $ROOT/pairs184.txt] [--target_fps 25] \
    [--out           $ROOT/GT/_metrics/variance_emoca__LS_AL.csv] \
    [--overwrite]
```

Every source frame is `face_ok`-filtered, resampled to `--target_fps`
(default 25), and — when a VAD mask is requested — restricted to the
kept frames before pooling.

Idempotent (label-based): a `source` label already in the CSV is
skipped on re-runs. Pass `--overwrite` to recompute.

## CSV columns

One pooled row per `<source>` × `<split>`:

```
EMOCA: source, n_videos, n_frames, var_overall, var_pose, var_exp
LP:    source, n_videos, n_frames, var_overall, var_rot,  var_exp
```

`n_videos` is how many clips contributed frames; `n_frames` is the
total pooled frame count. `overall` is dimensionally imbalanced (see
below); the balanced read is `pose`/`exp` (EMOCA) and `rot`/`exp` (LP).

## `--vad_dir` / `--vad_mode` — speech/silence-restricted variance

On dyadic data (SI/LS_AL: both roles in one mp4) the unfiltered
variance pools speech and listening regimes into one number. The
listening side has its own dynamics — micro-expressions, blinks, head
nods — that speech-time jaw articulation drowns out. Restrict to one
regime by passing a VAD cache from `extractors/vad`:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --features_dir   $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca \
    --vad_dir        $ROOT/GT/LS_AL_vad \
    --vad_mode       silence \
    --out            $ROOT/<MODEL>/_metrics/variance_emoca_silence__LS_AL.csv
```

`--vad_mode silence` keeps listener-silent frames (= they are
listening); `speech` keeps the inverse; `none` (default) keeps every
frame. Apply only when the speaker and listener share one split
(SI-184: `LS_AL`). On a paired-split corpus (separate LS/AL dirs, not
shipped here) an in-split VAD filter would just drop a few self-speech
frames with no diagnostic gain.
