"""SID — Semantic ID / cluster diversity (REACT2024 baseline).

Quantifies how well the model's feature distribution covers the modes
of the GT distribution:

    1. Pool all GT (per-frame) features, fit k-means with k clusters.
    2. Predict cluster labels for all model features.
    3. Histogram of predicted cluster assignments → Shannon entropy.

High entropy = model spreads predictions across the GT modes (covers
the distribution); low = mode collapse to a few clusters.

Pairwise — needs `<MODEL>` features. Operates **per-group** on the
shared motion-feature layout from `_motion.py`:

    EMOCA groups: overall (k=40), pose (k=20), exp (k=40)
    LP    groups: overall (k=40), rot (k=20), exp (k=40)

(k=20 for pose-like blocks, k=40 for expression-like, matching the
REACT2024 reference; configurable.)

Output: corpus-level CSV — one row per `<MODEL>`. Lower than max
entropy (`log2(k)`) ⇒ less coverage / mode collapse.

Usage:
  python compute.py \\
      --gt_features    <corpus>/GT/<split>_emoca \\
      --pred_features  <corpus>/<MODEL>/<split>_emoca \\
      --feature_source emoca \\
      [--out           <corpus>/<MODEL>/_metrics/sid_emoca__<split>.csv] \\
      [--label         <MODEL>] [--k_pose 20] [--k_exp 40]
"""

from __future__ import annotations

import argparse
import csv
import os.path as _osp
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..")))
from _motion import load_motion_groups  # noqa: E402

sys.path.insert(0, _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..", "..", "extractors", "vad")))
from vad_mask import per_frame_keep_mask  # noqa: E402


# Per-group default k (REACT2024 reference): pose-like blocks k=20,
# expression-like k=40.
K_PER_GROUP = {
    "emoca": {"overall": 40, "pose": 20, "exp": 40},
    "lp":    {"overall": 40, "rot": 20, "exp": 40},
}

GROUPS = {
    "emoca": ["overall", "pose", "exp"],
    "lp":    ["overall", "rot", "exp"],
}


def _load_vad_mask(vad_dir: Path, stem: str, mode: str, target_len: int
                   ) -> "np.ndarray | None":
    """Thin wrapper around shared `per_frame_keep_mask`. Returns None
    when vad_dir is None or mode is the identity ('none')."""
    if vad_dir is None:
        return None
    return per_frame_keep_mask(vad_dir / f"{stem}.npz", mode, target_len)


def _read_stems(path: "Path | None") -> "set[str] | None":
    """Read data/pairs184.txt ("<listener>.wav <speaker>.wav" per line). The stem is the first whitespace
    token of each non-blank non-`#`-comment line with a trailing `.wav` stripped. Returns None when `path`
    is None (⇒ use every npz in the dir)."""
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


def _pool_group(features_dir: Path, source: str, group: str,
                vad_dir: "Path | None" = None, vad_mode: str = "none",
                target_fps: "float | None" = None,
                stems: "set[str] | None" = None,
                ) -> np.ndarray:
    """Concatenate `<group>` slice across every npz in the dir → (N, D).

    If `stems` is given, only npzs whose stem is in that set are pooled
    (restricts pooling to the 184-stem conversational subset).

    If `vad_dir/vad_mode` are given, applies a per-stem mask resampled
    to the per-stem feature length before concatenation. Same mask
    semantics as rpcc/tlcc.
    """
    paths = sorted(features_dir.glob("*.npz"))
    if stems is not None:
        paths = [p for p in paths if p.stem in stems]
    if not paths:
        raise FileNotFoundError(f"no *.npz under {features_dir}"
                                + (" matching --stems" if stems is not None else ""))
    arrs: list[np.ndarray] = []
    n_kept_total = 0; n_dropped = 0
    for p in paths:
        try:
            g = load_motion_groups(p, source, target_fps=target_fps)
            arr = g[group]
        except Exception as e:
            print(f"  [skip] {p.name}: {type(e).__name__}: {e}")
            continue
        if vad_dir is not None and vad_mode != "none":
            mask = _load_vad_mask(vad_dir, p.stem, vad_mode, target_len=len(arr))
            if mask is None:
                # Stem has no VAD cache → drop entire stem to avoid biasing
                # the pool with un-masked frames.
                n_dropped += 1
                continue
            arr = arr[mask]
        if len(arr) > 0:
            arrs.append(arr)
            n_kept_total += len(arr)
    if not arrs:
        raise RuntimeError(f"no frames survived in {features_dir} "
                           f"(vad_mode={vad_mode}); n_dropped={n_dropped}")
    if vad_dir is not None and vad_mode != "none":
        print(f"  vad={vad_mode}: kept {n_kept_total} frames, "
              f"dropped {n_dropped} stems without VAD")
    return np.concatenate(arrs, axis=0)


def _entropy(labels: np.ndarray, k: int, eps: float = 1e-6) -> float:
    """Shannon entropy of label histogram (in bits)."""
    cnt = np.bincount(labels, minlength=k).astype(np.float64)
    p = cnt / max(cnt.sum(), 1)
    h = -np.sum(p * np.log2(p + eps))
    return float(h)


def sid(gt_features: np.ndarray, pred_features: np.ndarray, k: int,
        random_state: int = 0) -> tuple[float, float]:
    """Returns (sid_entropy, max_entropy=log2(k))."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(gt_features)
    pred_labels = km.predict(pred_features)
    return _entropy(pred_labels, k), float(np.log2(k))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt_features",   required=True, type=Path)
    p.add_argument("--pred_features", required=True, type=Path)
    p.add_argument("--feature_source", choices=("emoca", "lp"), required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--k_pose", type=int, default=20)
    p.add_argument("--k_exp",  type=int, default=40)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vad_dir", type=Path, default=None,
                   help="VAD npz cache (audio is shared between GT and "
                        "model, so the same dir applies to both sides)")
    p.add_argument("--vad_mode", choices=("none", "silence", "speech"),
                   default="none")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Resample every source to this fps before pooling. "
                        "DEFAULT 25 — re-creates the original's uniform-fps "
                        "condition (the models are 25fps, GT is 30fps; "
                        "matching to 25 reproduces the original metric, not "
                        "a deviation). No-op on already-25fps sources. Pass "
                        "0 to disable.")
    p.add_argument("--stems", type=Path, default=None,
                   help="Optional stem-list file (one stem per line; `#` "
                        "comments ignored). Restricts BOTH the GT pool and "
                        "the pred pool to these stems — e.g. the paper's "
                        "184 conversational SI stems. Default None = every "
                        "npz in each dir.")
    return p.parse_args()


def default_out(pred_features_dir: Path, source: str,
                vad_mode: str = "none") -> Path:
    name = pred_features_dir.name
    for sfx in ("_emoca", "_liveportrait"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
            break
    suffix = "" if vad_mode == "none" else f"_{vad_mode}"
    return pred_features_dir.parent / "_metrics" / f"sid_{source}{suffix}__{name}.csv"


def main() -> None:
    args = parse_args()
    out_path = args.out or default_out(args.pred_features, args.feature_source,
                                       vad_mode=args.vad_mode)
    label = args.label or args.pred_features.parent.name
    if args.overwrite and out_path.is_file():
        out_path.unlink()

    # Resolve per-group k (override defaults from CLI when applicable).
    k_per_group = dict(K_PER_GROUP[args.feature_source])
    # Apply user overrides for pose-like / expression-like groups.
    if args.feature_source == "emoca":
        k_per_group["pose"] = args.k_pose
        k_per_group["exp"]  = args.k_exp
    else:  # lp
        k_per_group["rot"] = args.k_pose      # pose-like ⇒ k_pose
        k_per_group["exp"] = args.k_exp
        k_per_group["overall"] = args.k_exp

    header = ["source", "n_gt_frames", "n_pred_frames"] + [
        f"sid_{g}" for g in GROUPS[args.feature_source]
    ] + [f"k_{g}" for g in GROUPS[args.feature_source]]

    print(f"loading GT/pred groups ({args.feature_source}, "
          f"vad_mode={args.vad_mode}) ...")
    stems = _read_stems(args.stems)
    if stems is not None:
        print(f"--stems: restricting GT and pred pools to {len(stems)} stems "
              f"from {args.stems}")
    gt_groups = {g: _pool_group(args.gt_features, args.feature_source, g,
                                vad_dir=args.vad_dir, vad_mode=args.vad_mode,
                                target_fps=args.target_fps, stems=stems)
                 for g in GROUPS[args.feature_source]}
    pred_groups = {g: _pool_group(args.pred_features, args.feature_source, g,
                                  vad_dir=args.vad_dir, vad_mode=args.vad_mode,
                                  target_fps=args.target_fps, stems=stems)
                   for g in GROUPS[args.feature_source]}
    n_gt = gt_groups["overall"].shape[0]
    n_pr = pred_groups["overall"].shape[0]
    print(f"  GT  : {n_gt} frames")
    print(f"  pred: {n_pr} frames")

    row: dict = {"source": label, "n_gt_frames": n_gt, "n_pred_frames": n_pr}
    for g in tqdm(GROUPS[args.feature_source], ncols=80, desc="kmeans"):
        k = k_per_group[g]
        h, h_max = sid(gt_groups[g], pred_groups[g], k=k)
        row[f"sid_{g}"] = f"{h:.6f}"
        row[f"k_{g}"]   = int(k)
        print(f"  {g:>9}  k={k:2d}  H={h:.4f}  (max={h_max:.4f}, ratio={h/h_max:.3f})")

    # Idempotency: if a row with this same label already exists, skip
    # the append (unless --overwrite cleared the file above).
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
