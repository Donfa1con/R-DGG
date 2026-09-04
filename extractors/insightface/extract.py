"""
Shared insightface detection cache for the R-DGG evaluation's visual metrics.

For each <stem>.mp4 we run insightface buffalo_l on every frame and write
<stem>.npz with:

  bbox            (T, 4)     float32   xyxy from RetinaFace
  lm5             (T, 5, 2)  float32   5-point landmarks (ArcFace anchors)
  lm106           (T, 106, 2) float32  dense 106-point landmarks
  head_bbox_xywh  (T, 4)     float32   smoothed square head+shoulders crop
                                       (temporal median filter, window 7)
  embedding       (T, 512)   float16   ArcFace embedding (consumed by csim)
  face_ok         (T,)       uint8     1 if a face was detected on that frame
  fps             scalar     float32
  n_frames        scalar     int32
  h, w            scalar     int32

`embedding` is fp16 to keep storage modest (≈1 KB/frame). buffalo_l's
ArcFace head is l2-normalised at the source, so fp16 round-off is well
below the cosine-similarity sensitivity threshold.

Usage:
  python extract.py --videos_dir <in> --out_dir <out>
                    [--num_workers N] [--ext mp4] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")

DET_SIZE = (640, 640)        # insightface internal det resolution
DET_THRESH_DEFAULT = 0.5     # SCRFD score threshold; lower → more permissive
SMOOTH_WIN = 7               # temporal median window for head_bbox_xywh
SHOULDER_X_MARGIN = 0.5      # × face_w added on each side (left+right)
FOREHEAD_MARGIN = 0.6        # × face_h added above lm106 top
SHOULDER_Y_BIAS = 0.1        # square offset down to grab shoulders


# ─── insightface -------------------------------------------------------------

def _build_app(device: str = "cuda", det_thresh: float = DET_THRESH_DEFAULT):
    from insightface.app import FaceAnalysis
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=["detection", "landmark_2d_106",
                                        "recognition"])
    app.prepare(ctx_id=0 if device == "cuda" else -1,
                det_size=DET_SIZE, det_thresh=det_thresh)
    return app


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ─── square head+shoulders bbox from lm106 ----------------------------------

def head_bbox_from_lm106(lm106: np.ndarray, face_bbox: np.ndarray) -> np.ndarray:
    """Square xywh that brackets head + shoulders, anchored on the 106 landmarks."""
    face_w = float(face_bbox[2] - face_bbox[0])
    face_h = float(face_bbox[3] - face_bbox[1])
    xs = lm106[:, 0]
    ys = lm106[:, 1]
    top = float(ys.min()) - FOREHEAD_MARGIN * face_h
    chin = float(ys.max())
    left = float(xs.min()) - SHOULDER_X_MARGIN * face_w
    right = float(xs.max()) + SHOULDER_X_MARGIN * face_w
    side = float(max(right - left, chin - top))
    cx = (left + right) / 2.0
    cy = (top + chin) / 2.0 + SHOULDER_Y_BIAS * side
    x = cx - side / 2.0
    y = cy - side / 2.0
    return np.array([x, y, side, side], dtype=np.float32)


def smooth_head_bbox(head_bbox_xywh: np.ndarray, window: int = SMOOTH_WIN) -> np.ndarray:
    """Temporal median over each of (x, y, side, side)."""
    out = np.zeros_like(head_bbox_xywh)
    for i in range(head_bbox_xywh.shape[1]):
        out[:, i] = scipy.ndimage.median_filter(head_bbox_xywh[:, i],
                                                size=window, mode="nearest")
    return out


# ─── gap fill helpers (used by --auto_retry / --interp_gaps) -----------------

def interior_gap_indices(face_ok: np.ndarray,
                         max_gap: int | None = None) -> np.ndarray:
    """Indices of `face_ok=0` frames sandwiched between `face_ok=1` anchors.

    Optionally limited to gaps of length ≤ `max_gap`. Leading / trailing
    runs (no anchor on one side) are excluded — interpolation only.
    """
    n = len(face_ok)
    out: list[int] = []
    i = 0
    while i < n:
        if face_ok[i]:
            i += 1; continue
        start = i
        while i < n and not face_ok[i]:
            i += 1
        end = i  # exclusive
        if start > 0 and end < n and face_ok[start - 1] and face_ok[end]:
            if max_gap is None or (end - start) <= max_gap:
                out.extend(range(start, end))
    return np.array(out, dtype=np.int64)


def interp_axis0(arr: np.ndarray, valid_idx: np.ndarray,
                 fill_idx: np.ndarray) -> np.ndarray:
    """Linearly interpolate `arr` along axis 0 at `fill_idx`, anchored at
    `valid_idx`. Returns a new array of the same dtype as `arr`.
    """
    flat = arr.reshape(len(arr), -1).astype(np.float64)
    out = flat.copy()
    for j in range(flat.shape[1]):
        out[fill_idx, j] = np.interp(fill_idx, valid_idx, flat[valid_idx, j])
    return out.reshape(arr.shape).astype(arr.dtype)


def rebuild_head_bbox(lm106: np.ndarray, bbox: np.ndarray,
                      face_ok: np.ndarray, h: int, w: int) -> np.ndarray:
    """Recompute the smoothed head bbox after lm106/bbox have changed.

    Mirrors the path used by `process_video`: per-frame
    `head_bbox_from_lm106` for face_ok frames, carry-forward for gaps,
    backfill leading zero rows, then temporal median smoothing.
    """
    n = len(face_ok)
    head_raw = np.zeros((n, 4), dtype=np.float32)
    last_good = None
    for i in range(n):
        if face_ok[i]:
            head_raw[i] = head_bbox_from_lm106(lm106[i], bbox[i])
            last_good = head_raw[i]
        elif last_good is not None:
            head_raw[i] = last_good
    if face_ok.sum() > 0:
        first = int(np.argmax(face_ok > 0))
        if first > 0:
            head_raw[:first] = head_raw[first]
    else:
        side = float(min(h, w))
        head_raw[:] = (w / 2 - side / 2, h / 2 - side / 2, side, side)
    return smooth_head_bbox(head_raw, SMOOTH_WIN)


def re_embed_frames(video_path: Path, frame_idxs: np.ndarray, lm5: np.ndarray,
                    rec_model) -> dict[int, np.ndarray]:
    """For each requested frame: read it, ArcFace-align via `lm5[idx]`, run
    the recognition head. Returns `{frame_idx: 512-d float32 vector}`.

    Frames that fail to decode are silently dropped.
    """
    import cv2
    from insightface.utils import face_align
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    out: dict[int, np.ndarray] = {}
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        aligned = face_align.norm_crop(frame_bgr, landmark=lm5[int(fi)],
                                       image_size=112)
        feat = rec_model.get_feat(aligned)
        out[int(fi)] = np.asarray(feat, dtype=np.float32).flatten()
    cap.release()
    return out


# ─── per-video driver --------------------------------------------------------

def process_video(video_path: Path, out_path: Path, app,
                  overwrite: bool = False) -> str:
    if out_path.exists() and not overwrite:
        return "skip"

    # Use opencv VideoCapture (purely sequential, no internal frame cache)
    # instead of decord. decord's get_batch caches decoded GOPs and can grow
    # one worker to ~32 GB RSS on a long 1080p video; opencv reads
    # one frame at a time and stays bounded.
    import cv2
    import gc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "empty"
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if T == 0 or W == 0 or H == 0:
        cap.release()
        return "empty"

    bbox = np.zeros((T, 4), dtype=np.float32)
    lm5 = np.zeros((T, 5, 2), dtype=np.float32)
    lm106 = np.zeros((T, 106, 2), dtype=np.float32)
    head_raw = np.zeros((T, 4), dtype=np.float32)
    embedding = np.zeros((T, 512), dtype=np.float16)
    face_ok = np.zeros(T, dtype=np.uint8)
    last_good_head = None

    actual_T = 0
    for i in range(T):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        actual_T = i + 1
        faces = app.get(frame_bgr)
        best = _largest_face(faces)
        if best is None:
            if last_good_head is not None:
                head_raw[i] = last_good_head
            continue
        face_ok[i] = 1
        bbox[i] = best.bbox.astype(np.float32)
        lm5[i] = best.kps.astype(np.float32)
        if getattr(best, "embedding", None) is not None:
            embedding[i] = best.embedding.astype(np.float16)
        if hasattr(best, "landmark_2d_106") and best.landmark_2d_106 is not None:
            lm106[i] = best.landmark_2d_106.astype(np.float32)
            hb = head_bbox_from_lm106(lm106[i], bbox[i])
        else:
            fb = bbox[i]
            fw, fh = float(fb[2] - fb[0]), float(fb[3] - fb[1])
            cx = (fb[0] + fb[2]) / 2.0
            cy = (fb[1] + fb[3]) / 2.0
            side = max(fw * (1 + 2 * SHOULDER_X_MARGIN),
                       fh * (1 + FOREHEAD_MARGIN + 0.5))
            hb = np.array([cx - side / 2, cy - side / 2 + 0.1 * side,
                           side, side], dtype=np.float32)
        head_raw[i] = hb
        last_good_head = hb
    cap.release()
    del cap
    gc.collect()

    # Trim arrays if cv2 reported a different frame count than actually decoded.
    if actual_T < T:
        bbox = bbox[:actual_T]; lm5 = lm5[:actual_T]; lm106 = lm106[:actual_T]
        head_raw = head_raw[:actual_T]; face_ok = face_ok[:actual_T]
        embedding = embedding[:actual_T]
        T = actual_T

    # Backfill leading zero-bbox frames (no detection ever before the first hit)
    if face_ok.sum() > 0:
        first = int(np.argmax(face_ok > 0))
        if first > 0:
            head_raw[:first] = head_raw[first]
        # Carry-forward any remaining all-zero rows
        zero_rows = (head_raw[:, 2] <= 0)
        if zero_rows.any():
            # carry the previous non-zero row forward (first good row for a leading gap)
            for i in range(T):
                if zero_rows[i]:
                    head_raw[i] = head_raw[i - 1] if i > 0 else head_raw[first]
    else:
        # No face anywhere — full-frame square fallback
        side = float(min(H, W))
        cx, cy = W / 2.0, H / 2.0
        head_raw[:] = (cx - side / 2.0, cy - side / 2.0, side, side)

    # Temporal median smooth
    head_smooth = smooth_head_bbox(head_raw, SMOOTH_WIN)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        bbox=bbox,
        lm5=lm5,
        lm106=lm106,
        head_bbox_xywh=head_smooth,
        embedding=embedding,
        face_ok=face_ok,
        fps=np.float32(fps),
        n_frames=np.int32(T),
        h=np.int32(H),
        w=np.int32(W),
    )
    return "ok"


# ─── retry / interpolation post-pass ----------------------------------------

def enhance_npz(out_path: Path, video_path: Path, app,
                primary_thresh: float,
                retry_thresh: float | None,
                interp_gaps: bool,
                max_gap: int | None) -> dict:
    """In-place enhancement of a freshly written `<stem>.npz`:

    1. If `retry_thresh` is set, re-run detection at the lower threshold on
       the frames where the primary pass found no face (`face_ok=0`),
       overwriting their bbox / lm5 / lm106 / embedding if a face is now
       found.
    2. If `interp_gaps`, fill interior `face_ok=0` runs by linear
       interpolation of bbox / lm5 / lm106, rebuild `head_bbox_xywh` from
       the interpolated landmarks, and re-run ArcFace on the interpolated
       crops so the embedding reflects what is actually visible on those
       frames.

    The original mask is preserved as `face_ok_orig` the first time we
    modify the npz. Returns a stats dict.
    """
    stats = {"retry_recovered": 0, "interp_filled": 0, "interp_embedded": 0}
    if retry_thresh is None and not interp_gaps:
        return stats
    if not out_path.is_file():
        return stats

    z = dict(np.load(out_path))
    n = int(z["n_frames"])
    if n == 0:
        return stats
    face_ok = z["face_ok"][:n].astype(np.uint8).copy()
    bbox = z["bbox"][:n].copy()
    lm5  = z["lm5"][:n].copy()
    lm106 = z["lm106"][:n].copy()
    embedding = z["embedding"][:n].astype(np.float32).copy()
    H = int(z["h"]); W = int(z["w"])

    touched = False

    # 1) Retry pass on previously failed frames at the lower threshold.
    if retry_thresh is not None and face_ok.sum() < n:
        import cv2
        failed = np.where(face_ok == 0)[0]
        orig_thresh = app.det_model.det_thresh
        app.det_model.det_thresh = retry_thresh
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                for fi in failed:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        continue
                    best = _largest_face(app.get(frame_bgr))
                    if best is None:
                        continue
                    face_ok[fi] = 1
                    bbox[fi] = best.bbox.astype(np.float32)
                    lm5[fi]  = best.kps.astype(np.float32)
                    if getattr(best, "embedding", None) is not None:
                        embedding[fi] = best.embedding.astype(np.float32)
                    if (hasattr(best, "landmark_2d_106")
                            and best.landmark_2d_106 is not None):
                        lm106[fi] = best.landmark_2d_106.astype(np.float32)
                    stats["retry_recovered"] += 1
                cap.release()
        finally:
            app.det_model.det_thresh = orig_thresh
        if stats["retry_recovered"] > 0:
            touched = True

    # 2) Interpolation across interior gaps + ArcFace re-embed.
    if interp_gaps and 0 < face_ok.sum() < n:
        idxs_fill = interior_gap_indices(face_ok.astype(bool), max_gap)
        if len(idxs_fill) > 0:
            valid_idx = np.where(face_ok)[0]
            bbox  = interp_axis0(bbox,  valid_idx, idxs_fill)
            lm5   = interp_axis0(lm5,   valid_idx, idxs_fill)
            lm106 = interp_axis0(lm106, valid_idx, idxs_fill)
            face_ok[idxs_fill] = 1
            embs = re_embed_frames(video_path, idxs_fill, lm5,
                                   app.models["recognition"])
            for fi, v in embs.items():
                embedding[fi] = v
            stats["interp_filled"] = int(len(idxs_fill))
            stats["interp_embedded"] = int(len(embs))
            touched = True

    if not touched:
        return stats

    # Rebuild head_bbox from the (possibly updated) lm106 / bbox / face_ok.
    head_smooth = rebuild_head_bbox(lm106, bbox, face_ok.astype(bool), H, W)

    if "face_ok_orig" not in z:
        z["face_ok_orig"] = z["face_ok"][:n].astype(np.uint8)
    z["bbox"]           = bbox
    z["lm5"]            = lm5
    z["lm106"]          = lm106
    z["head_bbox_xywh"] = head_smooth
    z["embedding"]      = embedding.astype(np.float16)
    z["face_ok"]        = face_ok
    np.savez_compressed(out_path, **z)
    return stats


# ─── multiprocessing plumbing -----------------------------------------------

_W_APP = None
_W_DET_THRESH = DET_THRESH_DEFAULT
_W_RETRY_THRESH: float | None = None
_W_INTERP_GAPS = False
_W_MAX_GAP: int | None = None


def _init_worker(det_thresh: float = DET_THRESH_DEFAULT,
                 retry_thresh: float | None = None,
                 interp_gaps: bool = False,
                 max_gap: int | None = None) -> None:
    """Spawn-pool initializer. Must take args explicitly — `mp.spawn`
    re-imports the module in each child so any module-level state mutated
    in the parent is not visible here.
    """
    global _W_APP, _W_DET_THRESH, _W_RETRY_THRESH, _W_INTERP_GAPS, _W_MAX_GAP
    _W_DET_THRESH = det_thresh
    _W_RETRY_THRESH = retry_thresh
    _W_INTERP_GAPS = interp_gaps
    _W_MAX_GAP = max_gap
    _W_APP = _build_app(device="cuda", det_thresh=det_thresh)


def _worker_run(arg):
    video, out, overwrite = arg
    try:
        r = process_video(video, out, _W_APP, overwrite=overwrite)
        if r == "ok":
            enhance_npz(out, video, _W_APP,
                        primary_thresh=_W_DET_THRESH,
                        retry_thresh=_W_RETRY_THRESH,
                        interp_gaps=_W_INTERP_GAPS,
                        max_gap=_W_MAX_GAP)
        return {"stem": video.stem, "result": r}
    except Exception as e:
        return {"stem": video.stem,
                "result": "fail",
                "_error": f"{type(e).__name__}: {e}"}


def _worker_enhance_only(arg):
    video, out = arg
    try:
        enhance_npz(out, video, _W_APP,
                    primary_thresh=_W_DET_THRESH,
                    retry_thresh=_W_RETRY_THRESH,
                    interp_gaps=_W_INTERP_GAPS,
                    max_gap=_W_MAX_GAP)
        return {"stem": video.stem, "result": "ok"}
    except Exception as e:
        return {"stem": video.stem,
                "result": "fail",
                "_error": f"{type(e).__name__}: {e}"}


# ─── CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--ext", default="mp4")
    p.add_argument("--num_workers", type=int, default=1,
                   help=">1 uses multiprocessing.Pool with spawn")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--det_thresh", type=float, default=DET_THRESH_DEFAULT,
                   help=f"SCRFD detection score threshold "
                        f"(default {DET_THRESH_DEFAULT}; lower → more permissive). "
                        f"Use a lower value (e.g. 0.2) when re-running on videos "
                        f"with model artefacts the default missed.")
    p.add_argument("--auto_retry_thresh", type=float, default=None,
                   help="If set, after the primary pass automatically retry "
                        "frames with face_ok=0 at this lower threshold "
                        "(e.g. 0.2). Done in the same run, no separate script.")
    p.add_argument("--interp_gaps", action="store_true",
                   help="After detection (and retry, if enabled), fill "
                        "interior face_ok=0 runs by linear interpolation of "
                        "bbox/lm5/lm106 and re-run ArcFace on the resulting "
                        "crops so the embedding reflects the actual frame.")
    p.add_argument("--max_gap", type=int, default=None,
                   help="Max interior gap length in frames for --interp_gaps "
                        "(default: unlimited).")
    p.add_argument("--enhance_only", action="store_true",
                   help="Skip primary detection on stems whose npz already "
                        "exists and run only the enhancement post-pass "
                        "(--auto_retry_thresh / --interp_gaps) over them. "
                        "Use to retrofit existing caches without redetecting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.videos_dir.glob(f"*.{args.ext}"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"no *.{args.ext} under {args.videos_dir}")
        return

    todo = []
    enhance_existing: list[tuple[Path, Path]] = []
    for v in videos:
        out = args.out_dir / (v.stem + ".npz")
        if out.exists() and not args.overwrite:
            if args.enhance_only:
                enhance_existing.append((v, out))
            continue
        todo.append((v, out, args.overwrite))
    if not todo and not enhance_existing:
        print(f"nothing to do (skipped={len(videos)})")
        return

    stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}

    if args.num_workers <= 1:
        app = _build_app(device="cuda", det_thresh=args.det_thresh)
        for v, out, ow in tqdm(todo, ncols=80, desc=args.videos_dir.name):
            try:
                r = process_video(v, out, app, overwrite=ow)
                if r == "ok":
                    enhance_npz(out, v, app,
                                primary_thresh=args.det_thresh,
                                retry_thresh=args.auto_retry_thresh,
                                interp_gaps=args.interp_gaps,
                                max_gap=args.max_gap)
            except Exception as e:
                print(f"[fail] {v.name}: {e}", file=sys.stderr)
                r = "fail"
            stats[r] = stats.get(r, 0) + 1

        if args.enhance_only and enhance_existing:
            for v, out in tqdm(enhance_existing, ncols=80,
                               desc=f"enhance:{args.videos_dir.name}"):
                try:
                    enhance_npz(out, v, app,
                                primary_thresh=args.det_thresh,
                                retry_thresh=args.auto_retry_thresh,
                                interp_gaps=args.interp_gaps,
                                max_gap=args.max_gap)
                except Exception as e:
                    print(f"[enhance-fail] {v.name}: {e}", file=sys.stderr)
                    stats["fail"] = stats.get("fail", 0) + 1
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers,
                      initializer=_init_worker,
                      initargs=(args.det_thresh, args.auto_retry_thresh,
                                args.interp_gaps, args.max_gap)) as pool:
            for r in tqdm(pool.imap_unordered(_worker_run, todo, chunksize=1),
                          total=len(todo), ncols=80, desc=args.videos_dir.name):
                if "_error" in r:
                    print(f"[fail] {r['stem']}: {r['_error']}", file=sys.stderr)
                stats[r["result"]] = stats.get(r["result"], 0) + 1

            if args.enhance_only and enhance_existing:
                enhance_jobs = [(v, out) for v, out in enhance_existing]
                for r in tqdm(pool.imap_unordered(_worker_enhance_only,
                                                  enhance_jobs, chunksize=1),
                              total=len(enhance_jobs), ncols=80,
                              desc=f"enhance:{args.videos_dir.name}"):
                    if "_error" in r:
                        print(f"[enhance-fail] {r['stem']}: {r['_error']}",
                              file=sys.stderr)
                        stats["fail"] = stats.get("fail", 0) + 1

    print(f"done — {stats}")


if __name__ == "__main__":
    main()
