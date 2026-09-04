"""LivePortrait motion extractor — per-frame kp / exp / R / t / scale.

Reuses the cached lm106 from `extractors/insightface/` for the affine
crop (no second face detector run). Runs the official KwaiVGI/LivePortrait
`MotionExtractor` (vendored under `_vendor/`) on the canonical 256×256
face crop and saves raw outputs only — no temporal smoothing, no 203-pt
landmark refiner.

Per video `<stem>.mp4` → `<stem>.npz` with:
    kp     (T, 21, 3)  float32  canonical 3D keypoints
    exp    (T, 21, 3)  float32  expression deformation
    R      (T, 3, 3)   float32  rotation matrix
    t      (T, 3)      float32  translation
    scale  (T,)        float32  size
    yaw, pitch, roll   (T,)     float32  Euler angles in degrees
    face_ok (T,)       uint8    1 if insightface had a face on that frame
    fps, n_frames, h, w

Usage:
  python extract.py \\
      --videos_dir       <corpus>/<src>/<split> \\
      --insightface_dir  <corpus>/<src>/<split>_insightface \\
      [--out_dir         <corpus>/<src>/<split>_liveportrait] \\
      [--batch_size 32]  [--num_workers 4]  [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import cv2

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
WEIGHTS = THIS_DIR / "_weights" / "motion_extractor.pth"
sys.path.insert(0, str(THIS_DIR / "_vendor"))

from src.modules.motion_extractor import MotionExtractor                # noqa: E402
from src.utils.crop import (                                              # noqa: E402
    _estimate_similar_transform_from_pts,
    _transform_img,
)
from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix  # noqa: E402


# Driving-video config from `_vendor/src/config/crop_config.py`.
CROP_DSIZE  = 512
CROP_SCALE  = 2.2
CROP_VY     = -0.1
INPUT_SIZE  = 256   # MotionExtractor input (matches LP inference_config)


# ─── crop pipeline (replicates LP's crop.crop_image) ───────────────────────

def affine_warp_lm106(frame_bgr: np.ndarray, lm106: np.ndarray,
                      dsize: int = CROP_DSIZE,
                      scale: float = CROP_SCALE,
                      vy_ratio: float = CROP_VY) -> np.ndarray:
    """Replicates `_vendor/src/utils/crop.py:crop_image` for video driving
    config. Returns (dsize, dsize, 3) uint8 BGR."""
    M_INV, _ = _estimate_similar_transform_from_pts(
        lm106, dsize=dsize, scale=scale, vy_ratio=vy_ratio,
        flag_do_rot=True, use_lip=True,
    )
    return _transform_img(frame_bgr, M_INV, dsize)


# ─── motion-extractor batch forward ────────────────────────────────────────

@torch.no_grad()
def encode_batch(model, crops_uint8_bhwc: np.ndarray, device: torch.device
                 ) -> dict[str, torch.Tensor]:
    """uint8 (B, H, W, 3) BGR → MotionExtractor output dict.

    Post-processing:
      * raw pitch/yaw/roll logits (B, 66) → softmax-weighted degrees (B,)
      * exp / kp reshaped from (B, 63) to (B, 21, 3)
      * R derived from Euler angles via LP's get_rotation_matrix → (B, 3, 3)
    """
    # LP MotionExtractor was trained on RGB float [0,1] in NCHW format.
    crops_rgb = crops_uint8_bhwc[..., ::-1]  # BGR → RGB
    x = (torch.from_numpy(crops_rgb.copy())
         .to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0)
    raw = model(x)
    pitch_d = headpose_pred_to_degree(raw["pitch"])  # (B,)
    yaw_d   = headpose_pred_to_degree(raw["yaw"])
    roll_d  = headpose_pred_to_degree(raw["roll"])
    R = get_rotation_matrix(pitch_d, yaw_d, roll_d)  # (B, 3, 3)
    B = x.shape[0]
    return {
        "kp":    raw["kp"].view(B, 21, 3),
        "exp":   raw["exp"].view(B, 21, 3),
        "R":     R,
        "t":     raw["t"],
        "scale": raw["scale"].view(B),
        "yaw":   yaw_d,
        "pitch": pitch_d,
        "roll":  roll_d,
    }


def _build_model(device: torch.device) -> torch.nn.Module:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"weights missing at {WEIGHTS}. Run `pixi run download-weights`.")
    m = MotionExtractor(num_kp=21, backbone="convnextv2_tiny")
    # The published checkpoint is a flat state_dict (no 'model' wrapper),
    # so bypass MotionExtractor.load_pretrained which assumes the wrapper.
    state = torch.load(str(WEIGHTS), map_location="cpu")
    if "model" in state:
        state = state["model"]
    m.load_state_dict(state, strict=True)
    m = m.to(device).eval()
    return m


# ─── per-video driver ──────────────────────────────────────────────────────

def process_video(video_path: Path, npz_path: Path, out_path: Path,
                  model, device: torch.device, batch_size: int = 32,
                  resize_to: int = INPUT_SIZE) -> str:
    if out_path.is_file():
        return "skip"
    if not npz_path.is_file():
        return "no_npz"

    d = np.load(npz_path)
    lm106 = d["lm106"]
    face_ok = d["face_ok"].astype(bool)
    fps = float(d["fps"])
    T_meta = int(d["n_frames"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "empty"
    T = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), T_meta)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if T == 0:
        cap.release()
        return "empty"

    out = {
        "kp":    np.zeros((T, 21, 3), dtype=np.float32),
        "exp":   np.zeros((T, 21, 3), dtype=np.float32),
        "R":     np.zeros((T, 3, 3),  dtype=np.float32),
        "t":     np.zeros((T, 3),     dtype=np.float32),
        "scale": np.zeros((T,),       dtype=np.float32),
        "yaw":   np.zeros((T,),       dtype=np.float32),
        "pitch": np.zeros((T,),       dtype=np.float32),
        "roll":  np.zeros((T,),       dtype=np.float32),
    }

    last_lm106 = None
    pending_idxs: list[int] = []
    pending_crops: list[np.ndarray] = []

    def _flush():
        if not pending_idxs:
            return
        crops = np.stack(pending_crops, axis=0)
        kpv = encode_batch(model, crops, device)
        for k in out:
            arr = kpv[k].cpu().numpy()
            for j, i in enumerate(pending_idxs):
                if k in ("kp", "exp"):
                    out[k][i] = arr[j].reshape(21, 3)
                elif k == "R":
                    out[k][i] = arr[j]
                elif k in ("yaw", "pitch", "roll", "scale"):
                    out[k][i] = float(arr[j].squeeze())
                else:  # t
                    out[k][i] = arr[j]
        pending_idxs.clear()
        pending_crops.clear()

    actual_T = 0
    for i in range(T):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        actual_T = i + 1
        if face_ok[i]:
            last_lm106 = lm106[i]
        if last_lm106 is None:
            # no face detected anywhere up to here — skip; output zeros for now
            continue
        crop = affine_warp_lm106(frame_bgr, last_lm106)
        if resize_to != crop.shape[0]:
            crop = cv2.resize(crop, (resize_to, resize_to),
                              interpolation=cv2.INTER_AREA)
        pending_idxs.append(i)
        pending_crops.append(crop)
        if len(pending_idxs) >= batch_size:
            _flush()
    _flush()
    cap.release()

    # Trim if cv2 reported a smaller actual T.
    if actual_T < T:
        T = actual_T
        for k, v in list(out.items()):
            out[k] = v[:T]
        face_ok = face_ok[:T]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        face_ok=face_ok.astype(np.uint8),
        fps=np.float32(fps),
        n_frames=np.int32(T),
        h=np.int32(H), w=np.int32(W),
        **out,
    )
    return "ok"


# ─── multiprocessing plumbing ───────────────────────────────────────────────

_W_MODEL = None
_W_DEVICE = None


def _init_worker(weights_path: str):
    global _W_MODEL, _W_DEVICE
    _W_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _W_DEVICE.type == "cuda":
        torch.cuda.set_device(0)
    _W_MODEL = _build_model(_W_DEVICE)


def _worker_run(arg):
    video, npz, out, batch_size = arg
    try:
        r = process_video(video, npz, out, _W_MODEL, _W_DEVICE,
                          batch_size=batch_size)
        if _W_DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {"stem": video.stem, "result": r}
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"stem": video.stem, "result": "fail",
                "_error": f"{type(e).__name__}: {e}"}


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir",      required=True, type=Path)
    p.add_argument("--insightface_dir", required=True, type=Path)
    p.add_argument("--out_dir",         type=Path, default=None)
    p.add_argument("--ext",             default="mp4")
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--num_workers",     type=int, default=1)
    p.add_argument("--overwrite",       action="store_true")
    p.add_argument("--limit",           type=int, default=0)
    return p.parse_args()


def default_out_dir(videos_dir: Path) -> Path:
    return videos_dir.parent / f"{videos_dir.name}_liveportrait"


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    out_dir = args.out_dir or default_out_dir(args.videos_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.videos_dir.glob(f"*.{args.ext}"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"no *.{args.ext} under {args.videos_dir}")
        return

    todo = []
    for v in videos:
        out = out_dir / f"{v.stem}.npz"
        if out.is_file() and not args.overwrite:
            continue
        if args.overwrite and out.is_file():
            out.unlink()
        todo.append((v, args.insightface_dir / f"{v.stem}.npz", out, args.batch_size))
    if not todo:
        print(f"nothing to do (skipped={len(videos)})")
        return

    stats = {"ok": 0, "skip": 0, "no_npz": 0, "empty": 0, "fail": 0}

    if args.num_workers <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(device)
        for v, npz, out, bs in tqdm(todo, ncols=80, desc=args.videos_dir.name):
            try:
                r = process_video(v, npz, out, model, device, batch_size=bs)
            except Exception as e:
                print(f"[fail] {v.name}: {type(e).__name__}: {e}", file=sys.stderr)
                r = "fail"
            stats[r] = stats.get(r, 0) + 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers, initializer=_init_worker,
                      initargs=(str(WEIGHTS),)) as pool:
            for r in tqdm(pool.imap_unordered(_worker_run, todo, chunksize=1),
                          total=len(todo), ncols=80, desc=args.videos_dir.name):
                if "_error" in r:
                    print(f"[fail] {r['stem']}: {r['_error']}", file=sys.stderr)
                stats[r["result"]] = stats.get(r["result"], 0) + 1

    print(f"done — {stats}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
