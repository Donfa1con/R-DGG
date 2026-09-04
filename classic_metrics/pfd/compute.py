"""P-FD — Paired Fréchet Distance for listener-speaker synchrony (DIM).

Captures *joint* listener-speaker dynamics that single-side rPCC misses.
Original DIM definition — computed **PER VIDEO**, then averaged over
videos. For each paired (speaker, listener) video and region group `g`:

    real_g = concat([ speaker_g , GT_listener_g   ], axis=-1)   # (T, 2·Dg)
    fake_g = concat([ speaker_g , pred_listener_g ], axis=-1)   # (T, 2·Dg)
    fd_g   = frechet( N(μ,Σ) of real_g , N(μ,Σ) of fake_g )
    P-FD_g = mean_over_videos( fd_g )

where `frechet² = ||μ1−μ2||² + tr(Σ1+Σ2−2·sqrt(Σ1·Σ2))`.

The SPEAKER is GT in BOTH real and fake — the identical array — so P-FD
measures only how the model listener changes the *joint* speaker↔listener
distribution relative to the GT listener. (`--pred_speaker` is accepted
for CLI compatibility but IGNORED: the speaker side is always GT, which is
the original definition and prevents an accidental non-original variant.)

Same dual-source convention: EMOCA pairs and LP pairs are
computed independently, two CSVs. Per-group columns are the paper's two
region columns plus overall:

    EMOCA: pfd_overall (overall⊕overall, dim=112),
           pfd_pose (12 = 6 spk ⊕ 6 lis), pfd_exp (100 = 50 ⊕ 50)
    LP:    pfd_overall (84), pfd_rot (6 = 3⊕3), pfd_exp (78 = 39⊕39)

Lower = better. NO normalization by default.

Output: corpus-level CSV — one row per `<MODEL>` × `<split>` holding the
mean-over-videos P-FD per group.

Usage:
  python compute.py \\
      --gt_speaker     <corpus>/GT/LS_emoca \\
      --gt_listener    <corpus>/GT/AL_emoca \\
      --pred_speaker   <corpus>/<MODEL>/LS_emoca \\   # accepted, ignored
      --pred_listener  <corpus>/<MODEL>/AL_emoca \\
      --feature_source emoca \\
      --pairing        flip01 \\
      [--out           <corpus>/<MODEL>/_metrics/pfd_emoca__<split>.csv]
"""

from __future__ import annotations

import argparse
import csv
import os.path as _osp
import sys
from pathlib import Path

import numpy as np
from scipy import linalg
from tqdm import tqdm

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..")))
from _motion import load_motion_groups  # noqa: E402

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..", "..", "extractors", "vad")))
from vad_mask import per_frame_keep_mask, reactive_segments_mask  # noqa: E402


GROUPS = {
    "emoca": ["overall", "pose", "exp"],
    "lp":    ["overall", "rot", "exp"],
}


# --- pairing -----------------------------------------------------------------

def _flip01(stem: str) -> str | None:
    if stem.endswith(".0"):
        return stem[:-2] + ".1"
    if stem.endswith(".1"):
        return stem[:-2] + ".0"
    return None


def _read_pairs_file(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line: continue
        toks = line.split()
        if len(toks) != 2: raise ValueError(raw)
        mapping[Path(toks[0]).stem] = Path(toks[1]).stem
    return mapping


def speaker_for(listener_stem: str, mode: str, pairs: dict | None) -> str | None:
    if mode == "exact":  return listener_stem
    if mode == "flip01": return _flip01(listener_stem)
    if mode == "from_file":
        return pairs.get(listener_stem) if pairs else None
    raise ValueError(mode)


def parse_pairing_arg(s: str) -> tuple[str, dict | None]:
    if s == "exact" or s == "flip01":
        return s, None
    if s.startswith("from_file:"):
        path = Path(s[len("from_file:"):])
        if not path.is_file():
            raise FileNotFoundError(path)
        return "from_file", _read_pairs_file(path)
    raise ValueError(s)


# --- math --------------------------------------------------------------------

def _load_vad_mask(vad_dir: "Path | None", stem: str, mode: str,
                   target_len: int) -> "np.ndarray | None":
    """Thin wrapper around shared `per_frame_keep_mask`. Returns None
    when vad_dir is None or mode is the identity ('both')."""
    if vad_dir is None:
        return None
    return per_frame_keep_mask(vad_dir / f"{stem}.npz", mode, target_len)


def pfd_per_video(gt_speaker_dir: Path, gt_listener_dir: Path,
                  pred_listener_dir: Path, source: str,
                  pairing_mode: str, pairs: dict | None,
                  vad_dir: "Path | None" = None,
                  mask_speaker: str = "both",
                  mask_listener: str = "both",
                  reactive_segments: bool = False,
                  target_fps: "float | None" = None,
                  min_frames: int = 2,
                  ) -> "tuple[dict[str, list], int, int, int]":
    """Per-video Fréchet distance (original DIM). For every listener stem
    present in BOTH the GT and pred listener dirs with a resolvable GT
    speaker, build

        real_g = concat([ GT_speaker_g , GT_listener_g   ], axis=-1)
        fake_g = concat([ GT_speaker_g , pred_listener_g ], axis=-1)

    and compute the per-video Fréchet distance per group. The SPEAKER is
    GT on BOTH sides (identical array), so the diff isolates the listener.

    VAD masking (optional) — intersection of speaker & listener masks, or
    reactive-segment selection — is applied identically to real and fake
    (same GT speaker, same kept frames).

    Returns ({group: [fd_per_video, ...]}, n_pairs, n_skipped, n_kept_frames).
    """
    listener_npzs = sorted(pred_listener_dir.glob("*.npz"))
    fds: dict[str, list] = {g: [] for g in GROUPS[source]}
    n_pairs = 0
    n_skipped = 0
    n_kept_frames = 0
    if reactive_segments and vad_dir is None:
        raise ValueError("--reactive_segments requires --vad_dir")
    use_vad = vad_dir is not None and (
        reactive_segments
        or not (mask_speaker == "both" and mask_listener == "both")
    )
    for p in tqdm(listener_npzs, ncols=80, desc=f"pfd({pred_listener_dir.name})"):
        stem = p.stem
        gt_lis_path = gt_listener_dir / f"{stem}.npz"
        if not gt_lis_path.is_file():
            n_skipped += 1; continue
        sp_stem = speaker_for(stem, pairing_mode, pairs)
        if sp_stem is None:
            n_skipped += 1; continue
        sp_path = gt_speaker_dir / f"{sp_stem}.npz"      # speaker = GT
        if not sp_path.is_file():
            n_skipped += 1; continue
        try:
            sp_g  = load_motion_groups(sp_path,     source, target_fps=target_fps)
            gtl_g = load_motion_groups(gt_lis_path, source, target_fps=target_fps)
            prl_g = load_motion_groups(p,           source, target_fps=target_fps)
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {e}")
            n_skipped += 1; continue
        T = min(sp_g["overall"].shape[0], gtl_g["overall"].shape[0],
                prl_g["overall"].shape[0])
        if T < min_frames:
            n_skipped += 1; continue

        keep = None
        if use_vad:
            if reactive_segments:
                keep = reactive_segments_mask(
                    vad_dir / f"{stem}.npz",
                    vad_dir / f"{sp_stem}.npz", T)
                if keep is None:
                    n_skipped += 1; continue
            else:
                sp_mask = _load_vad_mask(vad_dir, sp_stem, mask_speaker, T)
                ls_mask = _load_vad_mask(vad_dir, stem,    mask_listener, T)
                # If either side requested a real mask and we couldn't
                # load it, drop the pair (don't silently use the unmasked
                # side).
                if mask_speaker != "both" and sp_mask is None:
                    n_skipped += 1; continue
                if mask_listener != "both" and ls_mask is None:
                    n_skipped += 1; continue
                keep = np.ones(T, dtype=bool)
                if sp_mask is not None: keep &= sp_mask
                if ls_mask is not None: keep &= ls_mask
            if int(keep.sum()) < min_frames:
                n_skipped += 1; continue

        used = False
        for g in GROUPS[source]:
            spk = sp_g[g][:T]; gtl = gtl_g[g][:T]; prl = prl_g[g][:T]
            real = np.concatenate([spk, gtl], axis=1)
            fake = np.concatenate([spk, prl], axis=1)
            if keep is not None:
                real = real[keep]; fake = fake[keep]
            if real.shape[0] < 2:            # need ≥2 frames for np.cov
                continue
            mu_r, cov_r = _fit_gaussian(real)
            mu_f, cov_f = _fit_gaussian(fake)
            fd = frechet_distance(mu_r, cov_r, mu_f, cov_f)
            if np.isfinite(fd):
                fds[g].append(fd)
                used = True
        if used:
            n_pairs += 1
            n_kept_frames += int(keep.sum()) if keep is not None else T
        else:
            n_skipped += 1
    print(f"  pairs: matched={n_pairs}, skipped={n_skipped}, "
          f"kept frames={n_kept_frames}")
    return fds, n_pairs, n_skipped, n_kept_frames


def _fit_gaussian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=0), np.cov(x, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd2 = (diff @ diff
           + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return float(max(fd2, 0.0))


# --- I/O ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt_speaker",     required=True, type=Path,
                   help="GT speaker dir — the speaker in BOTH real and fake.")
    p.add_argument("--gt_listener",    required=True, type=Path)
    p.add_argument("--pred_speaker",   required=True, type=Path,
                   help="Accepted for CLI compatibility but IGNORED — the "
                        "speaker side is always GT (original DIM definition).")
    p.add_argument("--pred_listener",  required=True, type=Path)
    p.add_argument("--feature_source", choices=("emoca", "lp"), required=True)
    p.add_argument("--pairing", default="exact",
                   help="exact | flip01 | from_file:<path-to-pairs.txt>")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Resample every source to this fps before pairing. "
                        "DEFAULT 25 — re-creates the original's uniform-fps "
                        "condition (the models are 25fps, GT is 30fps; "
                        "matching to 25 reproduces the original metric, not "
                        "a deviation). No-op on already-25fps sources. Pass "
                        "0 to disable.")
    p.add_argument("--min_frames", type=int, default=2,
                   help="Skip a video whose usable frame count is below "
                        "this (need ≥2 to estimate a covariance). Guards "
                        "degenerate short clips without crashing.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vad_dir", type=Path, default=None,
                   help="VAD npz cache shared between speaker and "
                        "listener sides (audio is per-stem, same for "
                        "GT and model).")
    p.add_argument("--mask_speaker",  choices=("both", "speech", "silence"),
                   default="both")
    p.add_argument("--mask_listener", choices=("both", "speech", "silence"),
                   default="both")
    p.add_argument("--reactive_segments", action="store_true",
                   help="Use shared reactive-segment selection instead "
                        "of per-frame mask intersection: keeps frames "
                        "inside maximal listener-silence intervals that "
                        "(a) contain ≥1 speaker-speech frame, (b) ≥3s "
                        "long, with 200ms inward padding on listener "
                        "silence. Implies --mask_*=both (those are "
                        "ignored). Requires --vad_dir.")
    p.add_argument("--variant_suffix", default=None,
                   help="Optional name suffix for the output CSV "
                        "(e.g. 'reactive'). If not set and any VAD mask "
                        "is non-default, defaults to "
                        "'<spk_mode>_spk_<ls_mode>_ls'.")
    return p.parse_args()


def default_out(pred_listener_dir: Path, source: str,
                variant_suffix: "str | None" = None) -> Path:
    name = pred_listener_dir.name
    for sfx in ("_emoca", "_liveportrait"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]; break
    vs = f"_{variant_suffix}" if variant_suffix else ""
    return pred_listener_dir.parent / "_metrics" / f"pfd_{source}{vs}__{name}.csv"


def main() -> None:
    args = parse_args()
    pairing_mode, pairs = parse_pairing_arg(args.pairing)

    # Resolve output suffix from VAD flags if not given explicitly.
    variant_suffix = args.variant_suffix
    if variant_suffix is None:
        if args.reactive_segments:
            variant_suffix = "reactive_segments"
        elif args.mask_speaker != "both" or args.mask_listener != "both":
            variant_suffix = f"{args.mask_speaker}_spk_{args.mask_listener}_ls"

    out_path = args.out or default_out(args.pred_listener,
                                       args.feature_source,
                                       variant_suffix=variant_suffix)
    label = args.label or args.pred_listener.parent.name
    if args.overwrite and out_path.is_file():
        out_path.unlink()

    # --pred_speaker is accepted but ignored: the speaker is GT in both
    # real and fake (original DIM definition). Warn if a different dir was
    # passed so nobody thinks the model speaker is being used.
    if args.pred_speaker is not None and \
            Path(args.pred_speaker) != Path(args.gt_speaker):
        print(f"note: --pred_speaker ({args.pred_speaker}) is IGNORED; the "
              f"speaker is GT ({args.gt_speaker}) in BOTH real and fake.")

    print(f"=== per-video P-FD: {label} (speaker = GT in real & fake) ===")
    fds, n_pairs, n_skipped, n_kept = pfd_per_video(
        args.gt_speaker, args.gt_listener, args.pred_listener,
        args.feature_source, pairing_mode, pairs,
        vad_dir=args.vad_dir,
        mask_speaker=args.mask_speaker,
        mask_listener=args.mask_listener,
        reactive_segments=args.reactive_segments,
        target_fps=args.target_fps,
        min_frames=args.min_frames)

    print(f"\nmatched videos: {n_pairs}, skipped: {n_skipped}")

    header = ["source", "n_videos"] + [
        f"pfd_{g}" for g in GROUPS[args.feature_source]
    ]
    row: dict = {"source": label, "n_videos": n_pairs}
    for g in GROUPS[args.feature_source]:
        vals = np.asarray(fds[g], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            row[f"pfd_{g}"] = ""; continue
        v = float(vals.mean())                      # mean over videos
        row[f"pfd_{g}"] = f"{v:.6f}"
        print(f"  {g:>9}  P-FD = {v:.4f}  (mean over {len(vals)} videos)")

    # Idempotency: skip append if a row with this label already exists.
    if out_path.is_file():
        with out_path.open() as f:
            existing = {r.get("source") for r in csv.DictReader(f)}
        if label in existing:
            print(f"out: {out_path}  (skip — row '{label}' already present)")
            return

    new = not out_path.is_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow(row)
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
