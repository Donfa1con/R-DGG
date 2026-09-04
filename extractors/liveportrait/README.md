# LivePortrait motion extractor

Per-frame `kp / exp / R / t / scale / yaw / pitch / roll` from
KwaiVGI/LivePortrait's `MotionExtractor` (ConvNeXtV2-tiny backbone),
consuming the cached **insightface lm106 landmarks** for the affine
crop. No second face-detector run, no 203-pt landmark refiner, no
temporal smoothing — raw model output saved as-is so downstream metrics
get the unfiltered signal.

## Setup

```bash
pixi install                # torch 2.1.1 + cu121 + cv2 + roma
pixi run download-weights   # ~107 MB motion_extractor.pth from KwaiVGI HF
```

The LivePortrait source is vendored under `_vendor/` (a shallow clone of
`KwaiVGI/LivePortrait`) so we can import their crop helpers and
`MotionExtractor` module without polluting other envs. `_vendor/` is
gitignored, so clone it once during setup — extract.py does
`sys.path.insert(0, _vendor)` then `from src.modules... import ...`, so the
repo (with its top-level `src/`) must land directly in `_vendor/`:

```bash
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait \
    extractors/liveportrait/_vendor
```

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run python extract.py \
    --videos_dir       $ROOT/GT/LS_AL \
    --insightface_dir  $ROOT/GT/LS_AL_insightface \
    --out_dir          $ROOT/GT/LS_AL_liveportrait \
    --batch_size 64 --num_workers 4 [--overwrite]
```

`batch_size 64 / num_workers 4` were chosen via `_bench.py` on 5 SI
1080p videos (the heavy-corpus realistic workload). Speedup vs single
process: ~1.25× (decode parallelisation hits diminishing returns
because GPU forward dominates after CPU decode is split). Throughput
≈ 43 sec/video on SI 1080p, ≈ 5 sec/video on HDTF/ViCo.

`--insightface_dir` provides the `lm106` field used to reproduce LP's
similarity affine warp (the one in their `crop.py:crop_image`):

```
crop = warpAffine(frame, affine_from(lm106, scale=2.2, vy_ratio=-0.1, dsize=512))
crop = resize(crop, 256×256)
```

(`scale=2.2` and `vy_ratio=-0.1` are the LP "driving video" config from
`_vendor/src/config/crop_config.py`.)

Idempotent — existing `<stem>.npz` is skipped unless `--overwrite`.

## Output schema

Per `<stem>.npz`:

| key       | shape         | dtype   | meaning |
|---|---|---|---|
| `kp`      | `(T, 21, 3)`  | float32 | canonical 3D keypoints |
| `exp`     | `(T, 21, 3)`  | float32 | expression deformation |
| `R`       | `(T, 3, 3)`   | float32 | rotation matrix (from yaw/pitch/roll) |
| `t`       | `(T, 3)`      | float32 | translation |
| `scale`   | `(T,)`        | float32 | size |
| `yaw`     | `(T,)`        | float32 | degrees (softmax-weighted over 66 bins) |
| `pitch`   | `(T,)`        | float32 | degrees |
| `roll`    | `(T,)`        | float32 | degrees |
| `face_ok` | `(T,)`        | uint8   | inherited from insightface npz |
| `fps`, `n_frames`, `h`, `w` | scalar | various | source video meta |

## Convention used downstream

Motion-feature metrics consume LP as a 42-d motion vector: the so3
rotation followed by the 39 expression dims that `LIPSYNC_COORDS`
selects from the 21×3 keypoint expression:

```
motion_vec (T, 42) = concat(
    so3 (T, 3)  = rotmat_to_rotvec(R),
    exp_lipsync (T, 39)  = exp.reshape(T, -1)[:, LIPSYNC_COORDS]
)
groups:
    rot = [0:3]    head rotation, axis-angle
    exp = [3:42]   brow (kp 1,2,6) ⊕ eyes (kp 11..16) ⊕ mouth (kp 17..20)
```

The metrics group the vector as `rot` and `exp`. Each metric's local
`_motion.py` helper implements this layout.

## Notes

* `torch.load` on the published `motion_extractor.pth` returns a flat
  `state_dict` (no `model` wrapper); we bypass `MotionExtractor.load_pretrained`
  and call `load_state_dict(state, strict=True)` directly.
* Carry-forward: when `face_ok==0` for a frame, the most recent good
  `lm106` is reused for the affine warp. The model still produces
  outputs for every frame, but face_ok flags the unreliable rows for
  downstream filtering.
