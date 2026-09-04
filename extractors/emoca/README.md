# EMOCA extractor

Batched, GPU-resident EMOCA inference. Writes per-video FLAME codes to
`<stem>.npz`. The pipeline keeps every heavy stage on the GPU (face crop,
224x224 affine warp, EMOCA encoder); the only CPU work is video decoding via
`decord` workers.

## Setup (one-time)

Prerequisites:
* Linux x86-64, CUDA-capable GPU (tested on L40S, sm_89).
* CUDA toolkit (any 12.x — only used while building pytorch3d).
* `INFERNO` vendored at `extractors/emoca/_vendor/inferno` — this is a real,
  editable dependency of `pixi.toml`, so it must be present before `pixi install`.
  Clone it and pull submodules:
  ```bash
  git clone --recurse-submodules https://github.com/radekd91/inferno \
      extractors/emoca/_vendor/inferno
  ```
  Then download its assets (FLAME + EMOCA_v2 + DECA + FaceRecognition, ~3 GB, license-gated)
  into `_vendor/inferno/assets/{EMOCA,FLAME,DECA,FaceRecognition}` via that repo's
  `inferno_apps/EMOCA/demos/download_assets.sh`. `--path_to_models` then defaults to
  `_vendor/inferno/assets/EMOCA/models`.
* `pixi >= 0.50`.

```bash
pixi install --frozen    # installs from the lock (pytorch 2.1.1 + cu121, mediapipe, ...); needs _vendor/inferno present
pixi run --frozen setup   # builds pytorch3d v0.7.6 + patches deprecated np.int aliases
```

`--frozen` is required for the emoca env. It installs from `pixi.lock` and skips
the solve. A plain solve re-resolves and fails on `onnxruntime-gpu==1.16.2`.

If `INFERNO` lives elsewhere, edit the `inferno = { path = ... }` line in
`pixi.toml` (the path is relative to this directory, or absolute), and point
`--path_to_models` / `$EMOCA_PATH_TO_MODELS` at its `assets/EMOCA/models`.

## Usage

```bash
pixi run --frozen python extract.py \
    --videos_dir /path/to/videos \
    --out_dir    /path/to/videos_emoca \
    --batch_size 32 --num_workers 4
```

Multiple input/output pairs in one model load — useful for the
`<corpus>/{GT,<model>}/<split>` layout used in this repo:

```bash
CORPUS=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --frozen python extract.py \
    --pairs \
        $CORPUS/GT/LS:$CORPUS/GT/LS_emoca \
        $CORPUS/GT/AL:$CORPUS/GT/AL_emoca \
        $CORPUS/<MODEL>/LS:$CORPUS/<MODEL>/LS_emoca \
        $CORPUS/<MODEL>/AL:$CORPUS/<MODEL>/AL_emoca \
    --batch_size 32 --num_workers 4
```

The script is idempotent: an existing `<stem>.npz` is skipped unless
`--overwrite` is passed.

### Useful flags

| flag                 | default                                      | notes |
|---|---|---|
| `--batch_size`       | 32                                           | drop to 16 for ≥1080p |
| `--num_workers`      | 4                                            | decord workers; 0 keeps decode in main thread |
| `--mode`             | `detail`                                     | EMOCA stage |
| `--model_name`       | `EMOCA_v2_lr_mse_20`                         | checkpoint dir under `--path_to_models` |
| `--path_to_models`   | `_vendor/inferno/assets/EMOCA/models`        | or `$EMOCA_PATH_TO_MODELS` |
| `--bbox_source`      | `insightface`                                | default reads lm106 from the matching insightface npz; `sfd` runs the per-frame SFD detector instead |
| `--insightface_dir`  | `<videos_dir>_insightface`                   | only used with `--bbox_source insightface` |
| `--limit N`          | 0                                            | run only the first N videos (smoke test) |
| `--overwrite`        | off                                          | redo videos even if `.npz` exists |

### `--bbox_source insightface`

Skip SFD detection and reuse the lm106 landmarks already cached by
`extractors/insightface/` for the visual metrics. The pipeline applies a
calibrated `lm106 → (cx, cy, size)` mapping (constants at the top of
`extract.py`) that reproduces the SFD-derived crop region. The crop is
not bit-identical (~13 px per-video median bbox shift), but
expcode / posecode time-series stay highly correlated with the SFD-based
extraction (per-video mean Pearson p50 ≈ 0.91–0.93 across the
calibration set). For correlation-based downstream metrics like rPCC this is
within the noise; if you need exact code parity, stay on `sfd`. See the
top-level README for figures and per-corpus calibration parameters.

### Stubborn videos

Some MP4s trip decord's default `DECORD_EOF_RETRY_MAX=10240`. Re-run with the
variable raised and `--num_workers 0` to keep the exception in the main
thread:

```bash
DECORD_EOF_RETRY_MAX=40960 pixi run --frozen python extract.py \
    --videos_dir /path/to/videos --out_dir /path/to/videos_emoca \
    --num_workers 0
```

## Output schema

Each `<stem>.npz` contains:

| key          | shape         | dtype   | meaning |
|---|---|---|---|
| `expcode`    | `(T, 50)`     | float32 | FLAME expression |
| `posecode`   | `(T, 6)`      | float32 | head rot (0–2) + jaw pose (3–5), axis-angle |
| `shapecode`  | `(T, 100)`    | float32 | FLAME shape |
| `texcode`    | `(T, 50)`     | float32 | FLAME texture |
| `cam`        | `(T, 3)`      | float32 | weak-perspective camera (s, tx, ty) |
| `lightcode`  | `(T, 9, 3)`   | float32 | SH lighting |
| `detailcode` | `(T, 128)`    | float32 | EMOCA detail latent |
| `bbox_xywh`  | `(T, 4)`      | float32 | crop bbox used for the 224² warp |
| `face_ok`    | `(T,)`        | uint8   | 1 if a face was detected, 0 if a fallback was used |
| `fps`        | scalar        | float32 | |
| `n_frames`   | scalar        | int32   | |
| `h`, `w`     | scalar        | int32   | original video resolution |
