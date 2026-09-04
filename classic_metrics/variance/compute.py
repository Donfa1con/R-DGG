"""Variance metric — feature trajectory dynamism, POOLED over the corpus.

Original DIM definition — ONE scalar per region over the WHOLE corpus
(NOT per-stem). Pool every included clip's per-group features, concat
across clips, then take the flattened variance:

    var_<group> = np.var( concat(all_clips)[group].reshape(-1) )

i.e. the variance of ALL frames×dims of ALL clips flattened into one
scalar. This captures between-clip diversity (the mode-collapse signal)
as well as within-clip motion — a model that outputs the same pose in
every clip has low pooled variance even if each clip moves a little.
It is NOT sum(var, axis=0) and NOT mean(var, axis=0), and NOT a per-stem
mean.

`features` are face_ok-masked motion features from EMOCA or LivePortrait
(see `_motion.py`), resampled to `target_fps` (default 25) and, when a
VAD mask is requested, restricted to the kept (e.g. listener-silent)
frames before pooling.

Output: corpus-level CSV — ONE row per `<source>` × `<split>` with
columns `source, n_videos, n_frames, var_{overall,pose,exp}` (EMOCA) /
`var_{overall,rot,exp}` (LP). `overall` is dimensionally imbalanced
(kept as aux); the paper reports pose/exp (EMOCA) & rot/exp (LP).

Usage:
  python compute.py \\
      --features_dir <corpus>/<source>/<split>_emoca \\
      --feature_source emoca \\
      [--stems <184-stem list>] \\
      [--out          <corpus>/<source>/_metrics/variance_emoca__<split>.csv] \\
      [--label <source>] [--overwrite]

  python compute.py \\
      --features_dir <corpus>/<source>/<split>_liveportrait \\
      --feature_source lp ...

Single-source — no GT pairing required. Runs on every corpus / source
dir independently.
"""

from __future__ import annotations

import argparse
import csv
import os.path as _osp
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..")))
from _motion import load_motion_groups  # noqa: E402

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..", "..", "extractors", "vad")))
from vad_mask import per_frame_keep_mask  # noqa: E402


GROUPS = {
    "emoca": ["overall", "pose", "exp"],
    "lp":    ["overall", "rot", "exp"],
}


def _var_flat(arr: np.ndarray) -> float:
    """Flattened variance of a pooled (N, D) region array: one scalar
    over ALL frames×dims (`np.var(arr.reshape(-1))`)."""
    return float(np.var(arr.reshape(-1)))


def load_vad_mask(vad_dir: "Path | None", stem: str, mode: str,
                  target_len: int) -> "np.ndarray | None":
    """Thin wrapper around shared `per_frame_keep_mask`. Returns None
    when vad_dir is None or mode is the identity ('none')."""
    if vad_dir is None:
        return None
    return per_frame_keep_mask(vad_dir / f"{stem}.npz", mode, target_len)


def _read_stems(path: "Path | None") -> "set[str] | None":
    """Read data/pairs184.txt ("<listener>.wav <speaker>.wav" per line). The stem is the first whitespace
    token of each non-blank non-`#`-comment line with a trailing `.wav` stripped. Returns None when `path`
    is None (⇒ pool every npz in the dir)."""
    if path is None:
        return None
    stems: set[str] = set()
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()[0]
        stems.add(tok[:-4] if tok.endswith(".wav") else tok)
    return stems


def csv_header(source: str) -> list[str]:
    base = ["source", "n_videos", "n_frames"]
    if source == "emoca":
        return base + ["var_overall", "var_pose", "var_exp"]
    if source == "lp":
        return base + ["var_overall", "var_rot", "var_exp"]
    raise ValueError(source)


def default_out(features_dir: Path, source: str, vad_mode: str = "none") -> Path:
    """`<split>_<source>` → `<split>` (strip suffix), then
    `<corpus>/<src>/_metrics/variance_<source>[_<vmode>]__<split>.csv`."""
    split = features_dir.name
    for sfx in ("_emoca", "_liveportrait"):
        if split.endswith(sfx):
            split = split[: -len(sfx)]
            break
    suffix = "" if vad_mode == "none" else f"_{vad_mode}"
    return features_dir.parent / "_metrics" / f"variance_{source}{suffix}__{split}.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True, type=Path,
                   help="<split>_emoca or <split>_liveportrait directory")
    p.add_argument("--feature_source", choices=("emoca", "lp"), required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--label", default=None,
                   help="Row label (the `source` column). Default: the "
                        "parent dir name of --features_dir (the model name).")
    p.add_argument("--stems", type=Path, default=None,
                   help="Optional stem-list file (one stem per line; `#` "
                        "comments ignored). Restricts the pooled stem set — "
                        "e.g. the paper's 184 conversational SI stems. "
                        "Default None = every npz in the dir.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--vad_dir", type=Path, default=None)
    p.add_argument("--vad_mode", choices=("none", "silence", "speech"),
                   default="none")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Resample every source to this fps before computing "
                        "variance. DEFAULT 25 — re-creates the original's "
                        "uniform-fps condition (the models are 25fps, GT is "
                        "30fps; matching to 25 reproduces the original "
                        "metric, not a deviation). No-op on already-25fps "
                        "sources. Pass 0 to disable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or default_out(args.features_dir,
                                       args.feature_source,
                                       vad_mode=args.vad_mode)
    label = args.label or args.features_dir.parent.name
    if args.overwrite and out_path.is_file():
        out_path.unlink()

    # Idempotency (label-based, like pfd — the output is now one pooled row).
    if out_path.is_file():
        with out_path.open() as f:
            existing = {r.get("source") for r in csv.DictReader(f)}
        if label in existing:
            print(f"out: {out_path}  (skip — row '{label}' already present)")
            return

    stems = _read_stems(args.stems)
    npzs = sorted(args.features_dir.glob("*.npz"))
    if stems is not None:
        npzs = [p for p in npzs if p.stem in stems]
        print(f"--stems: restricting pool to {len(stems)} stems "
              f"({len(npzs)} present) from {args.stems}")
    if args.limit:
        npzs = npzs[: args.limit]
    if not npzs:
        print(f"no *.npz under {args.features_dir}"
              + (" matching --stems" if stems is not None else ""))
        return

    # Pool every included stem's per-group features (after face_ok filter +
    # target_fps resample + optional VAD mask), then take one flattened
    # variance per group over the whole concatenated corpus.
    pooled: dict[str, list[np.ndarray]] = {g: [] for g in GROUPS[args.feature_source]}
    n_videos = 0
    n_dropped_vad = 0
    failed = 0
    use_vad = args.vad_dir is not None and args.vad_mode != "none"
    for p in tqdm(npzs, ncols=80, desc=args.features_dir.name):
        try:
            groups = load_motion_groups(p, args.feature_source,
                                        target_fps=args.target_fps)
        except Exception as e:
            print(f"[fail] {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        T = groups["overall"].shape[0]
        mask = None
        if use_vad:
            mask = load_vad_mask(args.vad_dir, p.stem, args.vad_mode,
                                 target_len=T)
            if mask is None:
                # VAD requested but unavailable → drop the stem rather than
                # pool its unmasked frames (would bias the pool).
                n_dropped_vad += 1
                continue
        used = False
        for g in GROUPS[args.feature_source]:
            arr = groups[g]
            if mask is not None:
                arr = arr[mask[:arr.shape[0]]]
            if arr.shape[0] > 0:
                pooled[g].append(arr)
                used = True
        if used:
            n_videos += 1

    concat = {g: (np.concatenate(a, axis=0) if a else np.zeros((0, 0)))
              for g, a in pooled.items()}
    n_frames = concat["overall"].shape[0]
    if n_frames < 2:
        print(f"[abort] only {n_frames} pooled frames "
              f"(videos={n_videos}, dropped_vad={n_dropped_vad}, failed={failed})")
        return

    row: dict = {"source": label, "n_videos": n_videos, "n_frames": n_frames}
    for g in GROUPS[args.feature_source]:
        v = _var_flat(concat[g])
        row[f"var_{g}"] = f"{v:.6f}"
        print(f"  {g:>9}  var = {v:.6f}  (pooled over {n_frames} frames)")

    header = csv_header(args.feature_source)
    new = not out_path.is_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"videos pooled: {n_videos}  frames: {n_frames}  "
          f"dropped_vad: {n_dropped_vad}  failed: {failed}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
