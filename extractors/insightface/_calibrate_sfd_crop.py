"""
Calibrate an lm106→(cx, cy, size) mapping that reproduces the SFD-derived
EMOCA / SyncNet crop on the evaluation-corpus frames.

Inputs (already cached):
    data/GT/<split>/<stem>.mp4
    data/GT/<split>_insightface/<stem>.npz   bbox, lm106, head_bbox_xywh, face_ok
    data/GT/<split>_emoca/<stem>.npz         bbox_xywh   (SFD → bbox2point → 1.25×)

We do NOT need the SyncNet npz: given the EMOCA crop bbox, the SyncNet
crop bbox is closed-form (both come from the same SFD xyxy with different
formulas):
    sync_cx   = emoca_cx
    sync_cy   = emoca_cy + 0.064 · emoca_w
    sync_size = 1.12 · emoca_w

Calibration model (separately for emoca and syncnet targets):
    let xs, ys = lm106[:, 0], lm106[:, 1]
        mid_x  = (xs.max() + xs.min()) / 2
        mid_y  = (ys.max() + ys.min()) / 2
        face_w = xs.max() - xs.min()
        face_h = ys.max() - ys.min()
        ext    = max(face_w, face_h)

    cx_pred   = mid_x + β_x · face_w
    cy_pred   = mid_y + β_y · face_h
    size_pred = γ · ext

We pick {β_x, β_y, γ} by minimising mean-squared residual over a sample of
faces with face_ok=1, then dump:
  - the calibrated coefficients (one set per target)
  - per-component residuals (px) at p50, p90, p99 of the corpus
  - a side-by-side overlay PNG: SFD-derived box vs lm106-fit box vs lm106 dots.

Usage:
  pixi run python _calibrate_sfd_crop.py [--n_frames 4000] [--root <SI/GT>]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ─── analytic SFD→syncnet bbox (closed form from EMOCA bbox) ──────────────

def sync_from_emoca(emoca_xywh: np.ndarray) -> np.ndarray:
    """Convert EMOCA crop bbox_xywh (square) to SyncNet crop bbox_xywh.

    Both are derived from the same SFD xyxy via different formulas, so the
    mapping is exact:
        old_size  = emoca_w / 1.25
        bs        = old_size / 2 = 0.4 · emoca_w
        sync_size = 2.8 · bs   = 1.12 · emoca_w
        sync_cy   = emoca_cy + 0.064 · emoca_w
        sync_cx   = emoca_cx
    """
    out = np.zeros_like(emoca_xywh)
    w = emoca_xywh[..., 2]
    sync_w = 1.12 * w
    cx = emoca_xywh[..., 0] + w / 2
    cy = emoca_xywh[..., 1] + w / 2 + 0.064 * w
    out[..., 0] = cx - sync_w / 2
    out[..., 1] = cy - sync_w / 2
    out[..., 2] = sync_w
    out[..., 3] = sync_w
    return out


# ─── lm106 features ───────────────────────────────────────────────────────

def lm106_features(lm106: np.ndarray) -> dict[str, np.ndarray]:
    """lm106 shape: (T, 106, 2) → dict of (T,)-shaped feature arrays."""
    xs = lm106[..., 0]
    ys = lm106[..., 1]
    x_min, x_max = xs.min(axis=-1), xs.max(axis=-1)
    y_min, y_max = ys.min(axis=-1), ys.max(axis=-1)
    face_w = x_max - x_min
    face_h = y_max - y_min
    return {
        "mid_x":  (x_max + x_min) / 2,
        "mid_y":  (y_max + y_min) / 2,
        "face_w": face_w,
        "face_h": face_h,
        "ext":    np.maximum(face_w, face_h),
    }


# ─── fitting --------------------------------------------------------------

def fit_one_axis(target: np.ndarray, anchor: np.ndarray, scale: np.ndarray
                 ) -> tuple[float, np.ndarray]:
    """Fit β so that anchor + β·scale ≈ target. Returns (β, residuals)."""
    diff = target - anchor
    # Least squares: minimise Σ(diff − β·scale)² → β = Σ(diff·scale)/Σ(scale²)
    beta = float(np.sum(diff * scale) / np.sum(scale * scale))
    resid = (anchor + beta * scale) - target
    return beta, resid


def fit_size(target: np.ndarray, scale: np.ndarray) -> tuple[float, np.ndarray]:
    gamma = float(np.sum(target * scale) / np.sum(scale * scale))
    resid = gamma * scale - target
    return gamma, resid


def calibrate(lm106: np.ndarray, target_xywh: np.ndarray
              ) -> tuple[dict, dict]:
    """Returns (params, residuals_summary)."""
    f = lm106_features(lm106)
    cx_t = target_xywh[..., 0] + target_xywh[..., 2] / 2
    cy_t = target_xywh[..., 1] + target_xywh[..., 3] / 2
    size_t = target_xywh[..., 2]

    bx, rx = fit_one_axis(cx_t, f["mid_x"], f["face_w"])
    by, ry = fit_one_axis(cy_t, f["mid_y"], f["face_h"])
    g,  rs = fit_size(size_t, f["ext"])

    def _summary(r, denom=None):
        ar = np.abs(r) if denom is None else np.abs(r) / denom
        return {"p50": float(np.percentile(ar, 50)),
                "p90": float(np.percentile(ar, 90)),
                "p99": float(np.percentile(ar, 99)),
                "max": float(ar.max())}

    # px residuals + relative-to-target-size residuals (so we can compare
    # corpora with different face sizes meaningfully).
    return ({"beta_x": bx, "beta_y": by, "gamma": g},
            {"cx_px":   _summary(rx),
             "cy_px":   _summary(ry),
             "size_px": _summary(rs),
             "cx_rel":   _summary(rx, size_t),
             "cy_rel":   _summary(ry, size_t),
             "size_rel": _summary(rs, size_t)})


def predict(lm106: np.ndarray, params: dict) -> np.ndarray:
    f = lm106_features(lm106)
    cx = f["mid_x"] + params["beta_x"] * f["face_w"]
    cy = f["mid_y"] + params["beta_y"] * f["face_h"]
    size = params["gamma"] * f["ext"]
    out = np.stack([cx - size / 2, cy - size / 2, size, size], axis=-1)
    return out


# ─── corpus aggregation ---------------------------------------------------

def gather(root: Path, split: str, n_frames: int,
           per_video_max: int = 50, seed: int = 0
           ) -> tuple[np.ndarray, np.ndarray, list]:
    """Returns (lm106_stack, emoca_xywh_stack, list of (video_stem, frame_idx))."""
    rng = random.Random(seed)
    insf_dir = root / f"{split}_insightface"
    emoc_dir = root / f"{split}_emoca"
    videos = sorted(p.stem for p in (root / split).glob("*.mp4")
                    if (insf_dir / f"{p.stem}.npz").exists()
                    and (emoc_dir / f"{p.stem}.npz").exists())
    if not videos:
        raise RuntimeError(f"no matched npzs under {root}")
    # round-robin pick frames until we hit n_frames
    rng.shuffle(videos)
    lms, embs, prov = [], [], []
    for stem in videos:
        i = np.load(insf_dir / f"{stem}.npz")
        e = np.load(emoc_dir / f"{stem}.npz")
        T = min(int(i["n_frames"]), int(e["n_frames"]))
        ok = (i["face_ok"][:T].astype(bool) & e["face_ok"][:T].astype(bool))
        valid = np.flatnonzero(ok)
        if valid.size == 0:
            continue
        n_taken = sum(a.shape[0] for a in lms)
        budget = n_frames - n_taken
        if budget <= 0:
            break
        # cap per-video so a few corpora don't dominate
        take = min(budget, valid.size, per_video_max)
        sel = rng.sample(list(valid), take)
        lms.append(i["lm106"][sel])
        embs.append(e["bbox_xywh"][sel])
        prov.extend((stem, int(s)) for s in sel)
    return (np.concatenate(lms, axis=0),
            np.concatenate(embs, axis=0),
            prov)


# ─── visualisation --------------------------------------------------------

def draw_box(im: np.ndarray, xywh: np.ndarray, color: tuple[int, int, int],
             label: str = "", thick: int = 2):
    x, y, w, h = [int(round(v)) for v in xywh]
    cv2.rectangle(im, (x, y), (x + w, y + h), color, thick)
    if label:
        cv2.putText(im, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"can't read frame {frame_idx} of {video}")
    return frame


def _tight_crop(frame: np.ndarray, boxes: list[np.ndarray], pad: int = 60
                ) -> tuple[np.ndarray, int, int]:
    """Return (panel, x0, y0) — a tight panel around the union of `boxes`."""
    H, W = frame.shape[:2]
    xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y1 = min(H, int(max(ys)) + pad)
    return frame[y0:y1, x0:x1].copy(), x0, y0


def _shift_box(b: np.ndarray, x0: int, y0: int) -> np.ndarray:
    return np.array([b[0] - x0, b[1] - y0, b[2], b[3]], dtype=np.float32)


def render_pair_panel(frame: np.ndarray, lm106: np.ndarray,
                      target_xywh: np.ndarray, pred_xywh: np.ndarray,
                      title: str) -> np.ndarray:
    panel, x0, y0 = _tight_crop(frame, [target_xywh, pred_xywh], pad=40)
    lm = lm106 - np.array([x0, y0])
    for (x, y) in lm.astype(int):
        if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
            cv2.circle(panel, (int(x), int(y)), 1, (255, 255, 0), -1)
    draw_box(panel, _shift_box(target_xywh, x0, y0), (0, 0, 255), "SFD",    2)
    draw_box(panel, _shift_box(pred_xywh, x0, y0),    (0, 255, 0), "lm106", 2)
    cv2.putText(panel, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def make_overlay(video: Path, frame_idx: int, lm106: np.ndarray,
                 emoca_xywh: np.ndarray, params_emoca: dict,
                 params_sync: dict, out_path: Path):
    frame = _read_frame(video, frame_idx)
    sync_xywh = sync_from_emoca(emoca_xywh)
    pred_e = predict(lm106[None], params_emoca)[0]
    pred_s = predict(lm106[None], params_sync)[0]
    panel_e = render_pair_panel(frame.copy(), lm106, emoca_xywh, pred_e, "EMOCA")
    panel_s = render_pair_panel(frame.copy(), lm106, sync_xywh, pred_s, "SyncNet")
    # equalise panel heights via padding so cv2 hconcat works
    h = max(panel_e.shape[0], panel_s.shape[0])
    def _pad(p):
        if p.shape[0] == h:
            return p
        pad = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=p.dtype)
        return np.vstack([p, pad])
    out = np.hstack([_pad(panel_e), _pad(panel_s)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def _crop_to_square(frame: np.ndarray, xywh: np.ndarray, out: int = 224
                    ) -> np.ndarray:
    """Pad-then-crop frame to xywh, resize to (out, out). uint8 BGR."""
    H, W = frame.shape[:2]
    x, y, w, h = [float(v) for v in xywh]
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((out, out, 3), dtype=np.uint8)
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def _label_strip(text: str, w: int, h: int = 28,
                 color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    strip = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return strip


def make_what_the_model_sees(video: Path, frame_idx: int, lm106: np.ndarray,
                             emoca_xywh: np.ndarray, params_emoca: dict,
                             params_sync: dict, out_path: Path,
                             out_size: int = 224):
    """Render 4 actual 224x224 crops side-by-side: SFD-emoca, lm106-emoca,
    SFD-sync, lm106-sync. This is what the model literally consumes."""
    frame = _read_frame(video, frame_idx)
    sync_xywh = sync_from_emoca(emoca_xywh)
    pred_e = predict(lm106[None], params_emoca)[0]
    pred_s = predict(lm106[None], params_sync)[0]

    crops = [
        ("EMOCA  | SFD",   _crop_to_square(frame, emoca_xywh, out_size)),
        ("EMOCA  | lm106", _crop_to_square(frame, pred_e,    out_size)),
        ("SyncNet| SFD",   _crop_to_square(frame, sync_xywh, out_size)),
        ("SyncNet| lm106", _crop_to_square(frame, pred_s,    out_size)),
    ]
    cols = []
    for label, c in crops:
        cols.append(np.vstack([_label_strip(label, c.shape[1]), c]))
    out = np.hstack(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def make_mosaic(prov: list, lm106: np.ndarray, emoca: np.ndarray,
                params_emoca: dict, params_sync: dict,
                video_dir: Path, out_path: Path,
                rng: random.Random, n: int = 6, target: str = "emoca"):
    """`target` ∈ {"emoca","sync"} — pick a fixed target for all tiles."""
    pred = predict(lm106, params_emoca if target == "emoca" else params_sync)
    target_xywh = emoca if target == "emoca" else sync_from_emoca(emoca)
    err = np.linalg.norm(pred[:, :2] - target_xywh[:, :2], axis=1) + \
          np.abs(pred[:, 2] - target_xywh[:, 2])
    # pick a spread: median, p75, p90, p95 etc. interleaved with random
    n_total = len(prov)
    order = np.argsort(err)
    picks = sorted({order[int(n_total * q)] for q in
                    (0.10, 0.30, 0.50, 0.75, 0.90, 0.95)})[:n]
    while len(picks) < n:
        picks.append(rng.randrange(n_total))

    tiles = []
    for k in picks:
        stem, fidx = prov[k]
        v = video_dir / f"{stem}.mp4"
        try:
            frame = _read_frame(v, fidx)
        except Exception:
            continue
        title = f"{target} err={err[k]:5.1f}px"
        panel = render_pair_panel(frame, lm106[k], target_xywh[k], pred[k], title)
        # uniform tile size
        panel = cv2.resize(panel, (320, 320), interpolation=cv2.INTER_AREA)
        tiles.append(panel)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * 320, cols * 320, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320] = t
    cv2.imwrite(str(out_path), grid)


# ─── main -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[2] / "data" / "GT",
                   help="corpus GT dir holding <split>/, <split>_insightface/, <split>_emoca/")
    p.add_argument("--split", default="LS_AL",
                   help="subdir under --root: <split>/, <split>_insightface/, <split>_emoca/")
    p.add_argument("--tag", default=None,
                   help="prefix for output filenames (default: derived from --split)")
    p.add_argument("--n_frames", type=int, default=4000)
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).resolve().parent,
                   help="where to write the calibration overlay/mosaic PNGs (default: this script's dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or args.split.lower()
    out_overlay = args.out_dir / f"_{tag}_crop_calibration.png"
    out_mosaic_emoca = args.out_dir / f"_{tag}_crop_emoca_mosaic.png"
    out_mosaic_sync = args.out_dir / f"_{tag}_crop_syncnet_mosaic.png"
    params_out = args.out_dir / f"_{tag}_crop_params.json"

    print(f"gathering up to {args.n_frames} frames from {args.root}/{args.split} ...")
    lm106, emoca, prov = gather(args.root, args.split, args.n_frames)
    sync = sync_from_emoca(emoca)
    print(f"corpus: {len(prov)} frames from "
          f"{len({p[0] for p in prov})} videos\n")

    p_emoca, r_emoca = calibrate(lm106, emoca)
    p_sync, r_sync = calibrate(lm106, sync)

    def _print(name, params, resid):
        print(f"=== {name} target ===")
        print(f"  params: β_x={params['beta_x']:+.4f}  "
              f"β_y={params['beta_y']:+.4f}  γ={params['gamma']:.4f}")
        print(f"  residuals (absolute px):")
        for k in ("cx_px", "cy_px", "size_px"):
            s = resid[k]
            print(f"    {k:>7}: p50={s['p50']:6.2f}  p90={s['p90']:6.2f}  "
                  f"p99={s['p99']:6.2f}  max={s['max']:7.2f}")
        print(f"  residuals (relative to crop size, %):")
        for k in ("cx_rel", "cy_rel", "size_rel"):
            s = resid[k]
            print(f"    {k:>7}: p50={s['p50']*100:5.2f}%  p90={s['p90']*100:5.2f}%  "
                  f"p99={s['p99']*100:5.2f}%  max={s['max']*100:6.2f}%")

    _print("EMOCA",   p_emoca, r_emoca)
    print()
    _print("SyncNet", p_sync,  r_sync)

    params_out.write_text(json.dumps(
        {"split": args.split,
         "emoca": p_emoca, "syncnet": p_sync,
         "residuals_emoca_px": r_emoca, "residuals_syncnet_px": r_sync,
         "n_frames": len(prov)}, indent=2))
    print(f"\nwrote params → {params_out}")

    rng = random.Random(0)
    video_dir = args.root / args.split

    # Median-residual overlay (one source frame, EMOCA + SyncNet panels side by side)
    pred_e = predict(lm106, p_emoca)
    err = np.linalg.norm(pred_e[:, :2] - emoca[:, :2], axis=1)
    med_idx = int(np.argsort(err)[len(err) // 2])
    stem, fidx = prov[med_idx]
    video = video_dir / f"{stem}.mp4"
    print(f"\noverlay frame: {stem}.mp4  idx={fidx}  cx_err={err[med_idx]:.2f}px")
    make_overlay(video, fidx, lm106[med_idx], emoca[med_idx],
                 p_emoca, p_sync, out_overlay)
    print(f"wrote overlay → {out_overlay}")
    out_crops = args.out_dir / f"_{tag}_what_the_model_sees.png"
    make_what_the_model_sees(video, fidx, lm106[med_idx], emoca[med_idx],
                             p_emoca, p_sync, out_crops)
    print(f"wrote model-input crops → {out_crops}")

    # Multi-frame mosaics (median through p95 residuals) per metric
    make_mosaic(prov, lm106, emoca, p_emoca, p_sync,
                video_dir, out_mosaic_emoca, rng, n=6, target="emoca")
    make_mosaic(prov, lm106, emoca, p_emoca, p_sync,
                video_dir, out_mosaic_sync, rng, n=6, target="sync")
    print(f"wrote emoca mosaic → {out_mosaic_emoca}")
    print(f"wrote sync mosaic  → {out_mosaic_sync}")


if __name__ == "__main__":
    main()
