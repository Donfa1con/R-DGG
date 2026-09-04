# sid — Semantic ID / cluster diversity

Adapted from `baseline_react2024/metric/calcuate_sid` (REACT2024).
Quantifies how well a model's feature distribution covers the modes of
the GT distribution:

```
1. Pool GT features → fit k-means with k clusters
2. Predict cluster labels for all model features
3. Histogram of predicted cluster assignments → Shannon entropy (bits)
```

High entropy ≈ `log2(k)` ⇒ model spreads predictions evenly across
GT modes (good coverage). Low entropy ⇒ predictions concentrated in
a few clusters (mode collapse).

Per-group, dual-source. Default `k`:

| group         | k  | rationale |
|---|---|---|
| EMOCA `posecode`     | 20 | 6-d, smaller feature space |
| EMOCA `expcode`      | 40 | 50-d expression PCA |
| EMOCA `overall`      | 40 | 56-d concat |
| LP `rot`             | 20 | 3-d so3 |
| LP `exp`             | 40 | 39-d (brow⊕eyes⊕mouth) |
| LP `overall`         | 40 | 42-d full motion vector |

## Environment

The six metrics share one pixi env — the manifest is `rdgg/pixi.toml`. Install
it once with `cd ../../rdgg && pixi install`. Every `pixi run` example below
runs through the shared manifest.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --gt_features    $ROOT/GT/LS_AL_emoca \
    --pred_features  $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca \
    [--label         <MODEL>] [--k_pose 20] [--k_exp 40] [--target_fps 25] \
    [--out           $ROOT/<MODEL>/_metrics/sid_emoca__LS_AL.csv]
```

Output: corpus-level — one row per `<MODEL>` × `<split>`. Columns:
`source, n_gt_frames, n_pred_frames, sid_<group>... , k_<group>...`.

## Notes

* k-means uses `random_state=0` for repeatability; the `n_init='auto'`
  default in scikit-learn picks 1 for "lloyd" (k-means++ init), which
  is sufficient given large GT.
* GT-vs-GT entropy is close to `log2(k)` but not exactly — k-means
  clusters can be slightly imbalanced even for the data they were fit
  on (cluster sizes vary).
* The CSV append is **idempotent** — re-running with the same `--label`
  is a no-op (the row already exists is skipped). Use `--overwrite`
  to recompute.

## `--vad_dir` / `--vad_mode` — speech/silence-restricted SID

On SI/LS_AL each mp4 contains both dyad roles, so the pool mixes
speaking-mode and listening-mode features. The k-means clusters
straddle both regimes and the entropy averages a mode-collapsed
listener with an active speaker. Two variants isolate each regime:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --gt_features    $ROOT/GT/LS_AL_emoca \
    --pred_features  $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca --label <MODEL> \
    --vad_dir        $ROOT/GT/LS_AL_vad \
    --vad_mode       silence
# → sid_emoca_silence__LS_AL.csv  ("listening diversity")
```

`silence` keeps listener-silent frames, `speech` the inverse. Apply
only when speaker and listener share a split (SI-184: `LS_AL`). On a
paired-split corpus (separate LS/AL dirs, not shipped here) an in-split
VAD filter adds no signal.
