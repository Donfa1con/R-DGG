"""
Batched EMOCA extractor for the R-DGG evaluation. GPU-resident pipeline.

Per video:
  workers (CPU) – decord frame decode → uint8 RGB tensor
  main (GPU)    – face crop: cached insightface lm106 (default) or batched SFD detection
                – batched 224x224 affine crop via grid_sample
                – emoca.encode (encoder only -- no decoder, no rendering)
  main (CPU)    – save expression / pose / shape / cam / detail to one .npz

Usage:
  python extract.py --videos_dir <in> --out_dir <out>
                    [--batch_size 32] [--num_workers 4] [--overwrite]

Output (per <stem>.mp4 -> <stem>.npz):
  expcode (T,50), posecode (T,6), shapecode (T,100), texcode (T,50),
  cam (T,3), lightcode (T,9,3), detailcode (T,128),
  bbox_xywh (T,4), face_ok (T,), fps, n_frames, h, w
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# Decord defaults to 10240 EOF retry attempts; some MP4s in the wild
# take more before the demuxer admits EOF.
# Bump well above the default — no cost on healthy videos because the
# retry path is only entered at end-of-stream.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "81920")

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from inferno.datasets.ImageDatasetHelpers import bbox2point  # type: ignore
from inferno_apps.EMOCA.utils.load import load_model  # type: ignore


# Calibrated coefficients for the lm106→(cx,cy,size) mapping that reproduces
# the SFD-derived EMOCA crop. See extractors/insightface/_calibrate_sfd_crop.py
# for the fit (a 131-video calibration set).
EMOCA_LM106_PARAMS = {"beta_x": -0.0019, "beta_y": -0.0442, "gamma": 1.3955}


# ────────────────────────────────────────────────────────────────────────────
# Worker: decord → uint8 HWC tensor (CPU, parallel via DataLoader)
# ────────────────────────────────────────────────────────────────────────────

class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        from decord import VideoReader, cpu
        self._VR_cls = VideoReader
        self._cpu = cpu
        vr = VideoReader(self.video_path, ctx=cpu(0))
        self.n_frames = len(vr)
        self.fps = float(vr.get_avg_fps())
        f0 = vr[0].asnumpy()
        self.h, self.w = f0.shape[:2]
        del vr
        self._reader = None

    def _ensure(self):
        if self._reader is None:
            self._reader = self._VR_cls(self.video_path, ctx=self._cpu(0))

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        self._ensure()
        frame = self._reader[idx].asnumpy()  # uint8 HWC RGB
        return idx, torch.from_numpy(frame)


# ────────────────────────────────────────────────────────────────────────────
# GPU square crop of `size` centred on `center` → out×out, normalised to [0,1]
# ────────────────────────────────────────────────────────────────────────────

def gpu_crop_to_224(images_chw_float: torch.Tensor, centers: torch.Tensor,
                    sizes: torch.Tensor, out: int = 224) -> torch.Tensor:
    """images: (B, 3, H, W) float in [0,255]. Returns (B, 3, out, out) in [0,1]."""
    B, _, H, W = images_chw_float.shape
    cx, cy = centers[:, 0], centers[:, 1]
    s_x = sizes / W
    s_y = sizes / H
    tx = 2.0 * cx / W - 1.0
    ty = 2.0 * cy / H - 1.0
    M = torch.zeros(B, 2, 3, device=centers.device, dtype=torch.float32)
    M[:, 0, 0] = s_x
    M[:, 1, 1] = s_y
    M[:, 0, 2] = tx
    M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(B, 3, out, out), align_corners=False)
    crop = F.grid_sample(images_chw_float / 255.0, grid,
                         mode="bilinear", padding_mode="zeros",
                         align_corners=False)
    return crop


def detect_faces_gpu(sfd, x_full: torch.Tensor, det_max_side: int = 384):
    """Detect faces in (B, 3, H, W) float[0,255] tensor.
    SFD activations balloon at full resolution, so we run detection on a
    downsampled copy and rescale the bbox back. Returns python list of [B][N][5]
    with bbox coords in *original* image pixels."""
    B, _, H, W = x_full.shape
    long_side = max(H, W)
    if long_side <= det_max_side:
        return sfd.detect_from_batch(x_full)
    scale = det_max_side / float(long_side)
    h2 = int(round(H * scale))
    w2 = int(round(W * scale))
    x_small = F.interpolate(x_full, size=(h2, w2), mode="bilinear",
                            align_corners=False, antialias=False)
    bbl = sfd.detect_from_batch(x_small)
    inv = 1.0 / scale
    out = []
    for boxes in bbl:
        rescaled = []
        for b in boxes:
            b = list(b)
            for i in range(4):
                b[i] = float(b[i]) * inv
            rescaled.append(b)
        out.append(rescaled)
    return out


# ────────────────────────────────────────────────────────────────────────────
# lm106-based crop computation (drop-in for SFD when an insightface npz exists)
# ────────────────────────────────────────────────────────────────────────────

def emoca_crop_from_lm106(lm106: np.ndarray, face_ok: np.ndarray,
                          params: dict = EMOCA_LM106_PARAMS
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the SFD-derived EMOCA crop region from lm106 only.

    For frames with face_ok=0 we carry forward the most recent good crop
    (matches SFD-pipeline behaviour); leading misses are back-filled from
    the first good frame.
    """
    T = lm106.shape[0]
    xs = lm106[..., 0]; ys = lm106[..., 1]
    x_min, x_max = xs.min(axis=-1), xs.max(axis=-1)
    y_min, y_max = ys.min(axis=-1), ys.max(axis=-1)
    face_w = x_max - x_min
    face_h = y_max - y_min
    ext = np.maximum(face_w, face_h)
    cx = (x_max + x_min) / 2 + params["beta_x"] * face_w
    cy = (y_max + y_min) / 2 + params["beta_y"] * face_h
    size = params["gamma"] * ext

    centres = np.stack([cx, cy], axis=1).astype(np.float32)
    sizes = size.astype(np.float32)
    oks = face_ok.astype(np.uint8)

    # carry-forward: replace bad rows with most recent good
    bad = oks == 0
    if bad.any() and (~bad).any():
        good_idx = np.flatnonzero(~bad)
        # for each bad i, find the largest good j <= i; if none, use the first good
        prev_good = np.searchsorted(good_idx, np.arange(T), side="right") - 1
        prev_good = np.where(prev_good < 0, 0, prev_good)
        src = good_idx[prev_good]
        centres[bad] = centres[src][bad]
        sizes[bad] = sizes[src][bad]
    return centres, sizes, oks


# ────────────────────────────────────────────────────────────────────────────
# bbox-list → centres / sizes (with previous-frame fall-back)
# ────────────────────────────────────────────────────────────────────────────

def select_centers(bboxes_per_frame, prev=None):
    centers, sizes, oks = [], [], []
    for bbl in bboxes_per_frame:
        if len(bbl) == 0:
            if prev is None:
                centers.append(np.zeros(2, np.float32))
                sizes.append(0.0)
                oks.append(0)
            else:
                centers.append(prev[0])
                sizes.append(prev[1])
                oks.append(0)
        else:
            best = max(bbl, key=lambda b: b[4] if len(b) >= 5 else 0)
            l, t, r, b = best[:4]
            old_size, c = bbox2point(l, r, t, b, type="bbox")
            size = float(old_size) * 1.25
            c = np.asarray(c, dtype=np.float32)
            centers.append(c); sizes.append(size); oks.append(1)
            prev = (c, size)
    return (np.stack(centers).astype(np.float32),
            np.asarray(sizes, np.float32),
            np.asarray(oks, np.uint8),
            prev)


# ────────────────────────────────────────────────────────────────────────────
# EMOCA encoder pass
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(emoca, imgs_b3hw: torch.Tensor) -> dict:
    out = emoca.encode({"image": imgs_b3hw}, training=False)
    keep = ("shapecode", "expcode", "posecode", "texcode", "cam", "lightcode", "detailcode")
    return {k: out[k].detach() for k in keep if k in out and torch.is_tensor(out[k])}


# ────────────────────────────────────────────────────────────────────────────
# Per-video driver
# ────────────────────────────────────────────────────────────────────────────

def process_video(video_path: Path, out_path: Path, emoca, sfd,
                  batch_size: int, num_workers: int, device: str = "cuda",
                  overwrite: bool = False,
                  bbox_source: str = "sfd",
                  insightface_npz=None):
    if out_path.exists() and not overwrite:
        return "skip"
    ds = VideoFrameDataset(str(video_path))
    if ds.n_frames == 0:
        return "empty"

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    H, W = ds.h, ds.w

    # ─ pre-compute (centres, sizes, oks) per video if using lm106 source ─
    centres_all = sizes_all = oks_all = None
    if bbox_source == "insightface":
        if insightface_npz is None or not insightface_npz.is_file():
            return "no_npz"
        d = np.load(insightface_npz)
        T_npz = int(d["n_frames"])
        T = min(ds.n_frames, T_npz)
        if T == 0:
            return "empty"
        centres_all, sizes_all, oks_all = emoca_crop_from_lm106(
            d["lm106"][:T], d["face_ok"][:T])

    codes: dict = {}
    bboxes_xywh = np.zeros((ds.n_frames, 4), dtype=np.float32)
    ok_arr = np.zeros(ds.n_frames, dtype=np.uint8)
    prev = None
    pbar = tqdm(dl, desc=video_path.name, leave=False, ncols=80)
    for idxs, frames in pbar:
        idxs = idxs.numpy()
        # uint8 HWC → float CHW on GPU (still in [0,255])
        x = frames.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float()  # (B,3,H,W)

        if bbox_source == "insightface":
            valid = idxs < (oks_all.size if centres_all is None else centres_all.shape[0])
            centers = np.zeros((len(idxs), 2), dtype=np.float32)
            sizes = np.zeros(len(idxs), dtype=np.float32)
            oks = np.zeros(len(idxs), dtype=np.uint8)
            centers[valid] = centres_all[idxs[valid]]
            sizes[valid] = sizes_all[idxs[valid]]
            oks[valid] = oks_all[idxs[valid]]
        else:
            # SFD on a downsampled copy → keeps memory bounded for 1080p+
            bbl = detect_faces_gpu(sfd, x, det_max_side=384)
            centers, sizes, oks, prev = select_centers(bbl, prev)
        if oks.sum() == 0:
            cx, cy = W / 2.0, H / 2.0
            centers = np.tile(np.array([cx, cy], dtype=np.float32), (len(idxs), 1))
            sizes = np.full(len(idxs), float(min(H, W)) * 0.6, dtype=np.float32)
        ctr = torch.from_numpy(centers).to(device)
        szs = torch.from_numpy(sizes).to(device)
        crop = gpu_crop_to_224(x, ctr, szs)  # (B, 3, 224, 224) in [0,1]
        out = encode_batch(emoca, crop)
        if not codes:
            for k, v in out.items():
                codes[k] = np.zeros((ds.n_frames,) + tuple(v.shape[1:]), dtype=np.float32)
        for k, v in out.items():
            codes[k][idxs] = v.cpu().numpy()
        bboxes_xywh[idxs, 0] = centers[:, 0] - sizes / 2
        bboxes_xywh[idxs, 1] = centers[:, 1] - sizes / 2
        bboxes_xywh[idxs, 2] = sizes
        bboxes_xywh[idxs, 3] = sizes
        ok_arr[idxs] = oks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fps=np.float32(ds.fps),
        n_frames=np.int32(ds.n_frames),
        h=np.int32(H), w=np.int32(W),
        bbox_xywh=bboxes_xywh,
        face_ok=ok_arr,
        **codes,
    )
    return "ok"


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=None,
                   help="space-separated list of <videos_dir>:<out_dir> pairs; "
                        "e.g. /a/in:/a/out /b/in:/b/out")
    p.add_argument("--videos_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_name", default="EMOCA_v2_lr_mse_20")
    p.add_argument("--path_to_models",
                   default=os.environ.get(
                       "EMOCA_PATH_TO_MODELS",
                       str(Path(__file__).resolve().parent / "_vendor" / "inferno" / "assets" / "EMOCA" / "models")),
                   help="EMOCA/DECA model dir; default: vendored _vendor/inferno assets (or $EMOCA_PATH_TO_MODELS)")
    p.add_argument("--mode", default="detail", choices=["coarse", "detail"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ext", default="mp4")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bbox_source", choices=("sfd", "insightface"),
                   default="insightface",
                   help="`insightface` (default): read lm106 from the matching "
                        "npz under `<videos_dir>_insightface/<stem>.npz` and "
                        "apply the calibrated lm106→(cx,cy,size) mapping. "
                        "`sfd`: fall back to the per-frame SFD detector.")
    p.add_argument("--insightface_dir", default=None,
                   help="override the default `<videos_dir>_insightface` lookup")
    return p.parse_args()


def main():
    args = parse_args()
    pairs = []
    if args.pairs:
        for s in args.pairs:
            i, o = s.split(":", 1)
            pairs.append((Path(i), Path(o)))
    if args.videos_dir and args.out_dir:
        pairs.append((Path(args.videos_dir), Path(args.out_dir)))
    if not pairs:
        print("No --pairs and no --videos_dir/--out_dir given", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(args.path_to_models):
        sys.exit(
            f"EMOCA models dir not found: {args.path_to_models}\n"
            "Pass --path_to_models (or set $EMOCA_PATH_TO_MODELS). See README.md: clone inferno into\n"
            "extractors/emoca/_vendor/inferno and download the EMOCA/FLAME/DECA/FaceRecognition assets."
        )

    print(f"Loading EMOCA ({args.model_name}, mode={args.mode}) …")
    emoca, _conf = load_model(args.path_to_models, args.model_name, args.mode)
    emoca.cuda().eval()

    sfd = None
    if args.bbox_source == "sfd":
        print("Loading SFD detector on GPU …")
        from face_alignment.detection.sfd.sfd_detector import SFDDetector  # type: ignore
        sfd = SFDDetector(device="cuda", filter_threshold=0.5)
    else:
        print(f"Using insightface lm106 → calibrated EMOCA crop "
              f"(β_x={EMOCA_LM106_PARAMS['beta_x']:+.4f}, "
              f"β_y={EMOCA_LM106_PARAMS['beta_y']:+.4f}, "
              f"γ={EMOCA_LM106_PARAMS['gamma']:.4f})")

    grand_t0 = time.time()
    grand_stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    for videos_dir, out_dir in pairs:
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = sorted(videos_dir.glob(f"*.{args.ext}"))
        if args.limit > 0:
            videos = videos[: args.limit]
        if not videos:
            print(f"No *.{args.ext} found in {videos_dir}")
            continue
        print(f"\n=== {videos_dir} → {out_dir}  ({len(videos)} videos) ===")
        stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
        t0 = time.time()
        if args.bbox_source == "insightface":
            insf_dir = (Path(args.insightface_dir) if args.insightface_dir
                        else videos_dir.parent / f"{videos_dir.name}_insightface")
            print(f"  insightface npzs: {insf_dir}")
        else:
            insf_dir = None
        for v in tqdm(videos, desc=videos_dir.name, ncols=80):
            out_path = out_dir / (v.stem + ".npz")
            insf_npz = (insf_dir / f"{v.stem}.npz") if insf_dir is not None else None
            try:
                r = process_video(v, out_path, emoca, sfd,
                                  args.batch_size, args.num_workers,
                                  overwrite=args.overwrite,
                                  bbox_source=args.bbox_source,
                                  insightface_npz=insf_npz)
            except Exception as e:
                print(f"[FAIL] {v.name}: {e}", file=sys.stderr)
                r = "fail"
            stats[r] = stats.get(r, 0) + 1
            grand_stats[r] = grand_stats.get(r, 0) + 1
        dt = time.time() - t0
        print(f"  {videos_dir.name}: done in {dt/60:.1f} min – {stats}")
    print(f"\nALL: {(time.time()-grand_t0)/60:.1f} min – {grand_stats}")


if __name__ == "__main__":
    main()
