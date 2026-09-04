# insightface extractor

Shared face-detection cache for the rest of the toolkit. Runs insightface
`buffalo_l` (RetinaFace + 106-pt landmarks + ArcFace) on every frame of
every input video and saves a single `<stem>.npz` with bounding boxes,
5- and 106-point landmarks, a temporally-smoothed square head+shoulders
crop bbox, the ArcFace embedding, and a per-frame `face_ok` flag.

Consumers (in this repo):

* `extractors/emoca` and `extractors/liveportrait` consume the cached `lm106`
  via their calibrated `lm106 → (cx, cy, size)` crop mapping (no second face
  detector run);
* the cached head+shoulders crop and ArcFace `embedding` are also the inputs to
  identity / image-quality visual metrics — those metrics are not part of this
  repo.

## Setup

```bash
pixi install
```

`buffalo_l` weights are auto-downloaded by insightface to
`~/.insightface/models/buffalo_l/` on first use; if you already have them
on the host they're picked up from there.

The env uses `onnxruntime-gpu==1.19.2`, which ships CUDA 12 wheels
(1.16 only has CUDA 11 wheels).

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run python extract.py \
    --videos_dir $ROOT/GT/LS_AL \
    --out_dir    $ROOT/GT/LS_AL_insightface \
    --num_workers 4
```

Idempotent: existing `.npz` files are skipped unless `--overwrite` is set.

### Recovering missed frames (model outputs with artefacts)

When evaluating generated videos, SCRFD's default `det_thresh=0.5` can drop
frames where a face is visibly present but distorted by model artefacts.
Three switches address this, all running in the same process as the primary
detection pass:

* `--auto_retry_thresh 0.2` — after the primary pass, re-run detection at
  the lower threshold *only on frames with `face_ok=0`*. Accepted frames'
  bbox / lm5 / lm106 / embedding replace the empty slots.
* `--interp_gaps [--max_gap N]` — after detection (and retry, if enabled),
  fill *interior* `face_ok=0` runs (i.e. surrounded by valid detections on
  both sides) by linear interpolation of bbox / lm5 / lm106. `head_bbox_xywh`
  is rebuilt from the interpolated `lm106` and the same temporal-median
  smoothing. The ArcFace recognition head is then **re-run on the crop
  derived from the interpolated `lm5`** so the embedding reflects what is
  actually visible on that frame (no nearest-neighbour copy).
  Leading / trailing runs without a two-sided anchor are left untouched.
* `--enhance_only` — skip primary detection on stems whose npz already
  exists and apply only the enhancement post-pass (i.e. retry +
  interpolation) to them. Use to retrofit already-extracted caches without
  redetecting them from scratch.

Fresh run with recovery enabled:

```bash
pixi run python extract.py \
    --videos_dir $ROOT/dystream/AL \
    --out_dir    $ROOT/dystream/AL_insightface \
    --auto_retry_thresh 0.2 --interp_gaps
```

Retrofit existing caches (no primary re-detection):

```bash
pixi run python extract.py \
    --videos_dir $ROOT/dystream/AL \
    --out_dir    $ROOT/dystream/AL_insightface \
    --enhance_only --auto_retry_thresh 0.2 --interp_gaps
```

Whenever enhancement modifies an existing npz, the pre-enhancement
`face_ok` mask is preserved under the key `face_ok_orig`.

## Output schema

Per `<stem>.mp4` → `<stem>.npz`:

| key              | shape          | dtype   | meaning |
|---|---|---|---|
| `bbox`           | `(T, 4)`       | float32 | xyxy, raw RetinaFace bbox |
| `lm5`            | `(T, 5, 2)`    | float32 | 5-point landmarks (ArcFace anchors) |
| `lm106`          | `(T, 106, 2)`  | float32 | dense 106-point landmarks |
| `head_bbox_xywh` | `(T, 4)`       | float32 | square head+shoulders crop, **smoothed** |
| `embedding`      | `(T, 512)`     | float16 | ArcFace embedding (un-normalised, l2-normalise before cos-sim) |
| `face_ok`        | `(T,)`         | uint8   | 1 if a face was detected on that frame (or interpolated, see below) |
| `face_ok_orig`   | `(T,)`         | uint8   | *Only present if `--auto_retry_thresh` / `--interp_gaps` modified the npz.* Mask from the primary pass before any recovery. |
| `fps`, `n_frames`, `h`, `w` | scalar | various | source video meta |

Storage cost of the embedding: ≈ 1 KB / frame in fp16. ArcFace's head is
already l2-normalised, so fp16 round-off is well below the
cosine-similarity sensitivity threshold.

## Square head+shoulders crop

Anchored on `lm106` (not on the rectangular face bbox, which would distort
the aspect ratio when squared):

```python
top   = min y of lm106 − 0.6 * face_h
chin  = max y of lm106
left  = min x of lm106 − 0.5 * face_w
right = max x of lm106 + 0.5 * face_w
side  = max(right − left, chin − top)
cx    = (left + right) / 2
cy    = (top + chin) / 2 + 0.1 * side    # bias slightly down for shoulders
```

Then `(cx − side/2, cy − side/2, side, side)` is written per frame and
finally passed through `scipy.ndimage.median_filter(window=7)` along the
time axis to kill jitter.

Frames where insightface didn't find a face get the previous successful
bbox (carry-forward); `face_ok==0` for those rows. Metrics that need the
crop pixels can use the bbox; metrics that aggregate over frames should
mask by `face_ok`.
