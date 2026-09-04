"""rPCC (reactive Pearson correlation) for paired listener evaluation.

For each (speaker, gt_listener, gen_listener) triplet, per **group** g
of features:

    pcc_gt_g  = pearson( speaker_g.flatten(), gt_listener_g.flatten() )
    pcc_gen_g = pearson( speaker_g.flatten(), gen_listener_g.flatten() )
    rPCC_g    = | pcc_gt_g - pcc_gen_g |

Lower = better (`0` ⇒ generated listener correlates with the speaker
identically to the GT listener).

Two feature sources are supported (dual-source convention):

  --feature_source emoca   reads <split>_emoca/<stem>.npz
                           groups: overall, pose (6d), exp (50d)

  --feature_source lp      reads <split>_liveportrait/<stem>.npz
                           groups: overall, rot (3d), exp (39d = brow⊕eyes⊕mouth)
                           (LivePortrait motion vector: so3 ⊕ 39 LIPSYNC_COORDS dims)

Output: per-stem CSV under
    `<gen_parent>/_metrics/rpcc_<source>__<split>.csv`
columns `stem, n_frames, rPCC_overall, rPCC_<group1>, ...`.

Pairing modes (resolves listener stem → speaker stem):
  --pairing exact           gen and gt share the speaker stem
  --pairing flip01          ViCo: speaker stem flips trailing `.0` ↔ `.1`
  --pairing from_file:PATH  SI: PATH lines `<listener>.wav <speaker>.wav`

Convention (verified against `RLD_data.xlsx`):
  ViCo: LS = speaker, AL = listener
  SI: both roles in `LS_AL`; `pairs.txt` provides the listener→speaker map.

Usage:
  python compute.py \\
      --speaker        <corpus>/GT/LS_emoca \\
      --gt_listener    <corpus>/GT/AL_emoca \\
      --gen_listener   <corpus>/<MODEL>/AL_emoca \\
      --feature_source emoca \\
      --pairing        flip01 \\
      [--out           <corpus>/<MODEL>/_metrics/rpcc_emoca__AL.csv] \\
      [--overwrite]
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


# --- math --------------------------------------------------------------------

def _flat_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rpcc_per_group(speaker_groups: dict[str, np.ndarray],
                   gt_groups: dict[str, np.ndarray],
                   gen_groups: dict[str, np.ndarray],
                   keep_mask: np.ndarray | None = None,
                   ) -> dict[str, dict[str, float]]:
    """Returns {group: {'rPCC': |gt-gen|, 'pcc_gt': pcc(sp,gt),
                        'pcc_gen': pcc(sp,gen)}} per group.

    pcc_gt / pcc_gen are kept as trace columns so a downstream
    consumer can derive the signed direction (proactivity =
    pcc_gen − pcc_gt) without recomputing.
    """
    out: dict[str, dict[str, float]] = {}
    for g in speaker_groups:
        if g not in gt_groups or g not in gen_groups:
            out[g] = {"rPCC": float("nan"), "pcc_gt": float("nan"),
                      "pcc_gen": float("nan")}
            continue
        sp = speaker_groups[g]; gt = gt_groups[g]; gn = gen_groups[g]
        T = min(len(sp), len(gt), len(gn))
        sp, gt, gn = sp[:T], gt[:T], gn[:T]
        if keep_mask is not None:
            m = np.asarray(keep_mask[:T]).astype(bool)
            sp, gt, gn = sp[m], gt[m], gn[m]
            if len(sp) < 5:
                out[g] = {"rPCC": float("nan"), "pcc_gt": float("nan"),
                          "pcc_gen": float("nan")}
                continue
        pcc_gt = _flat_pearson(sp, gt)
        pcc_gen = _flat_pearson(sp, gn)
        if np.isnan(pcc_gt) or np.isnan(pcc_gen):
            out[g] = {"rPCC": float("nan"), "pcc_gt": pcc_gt, "pcc_gen": pcc_gen}
        else:
            out[g] = {"rPCC": abs(pcc_gt - pcc_gen),
                      "pcc_gt": pcc_gt, "pcc_gen": pcc_gen}
    return out


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
        if not line:
            continue
        toks = line.split()
        if len(toks) != 2:
            raise ValueError(f"bad pairs line in {path}: {raw!r}")
        listener_stem = Path(toks[0]).stem
        speaker_stem = Path(toks[1]).stem
        mapping[listener_stem] = speaker_stem
    return mapping


def speaker_for(listener_stem: str, mode: str, pairs: dict[str, str] | None
                ) -> str | None:
    if mode == "exact":
        return listener_stem
    if mode == "flip01":
        return _flip01(listener_stem)
    if mode == "from_file":
        return pairs.get(listener_stem) if pairs is not None else None
    raise ValueError(f"unknown pairing mode: {mode!r}")


def parse_pairing_arg(s: str) -> tuple[str, dict[str, str] | None]:
    if s == "exact" or s == "flip01":
        return s, None
    if s.startswith("from_file:"):
        path = Path(s[len("from_file:"):])
        if not path.is_file():
            raise FileNotFoundError(f"pairs file not found: {path}")
        return "from_file", _read_pairs_file(path)
    raise ValueError(f"--pairing: 'exact' | 'flip01' | 'from_file:<path>'; got {s!r}")


# --- I/O ---------------------------------------------------------------------

GROUPS = {
    "emoca": ["overall", "pose", "exp"],
    "lp":    ["overall", "rot", "exp"],
}


def csv_header(source: str) -> list[str]:
    base = ["stem", "n_frames", "n_kept"]
    cols = []
    for g in GROUPS[source]:
        cols += [f"rPCC_{g}", f"pcc_gt_{g}", f"pcc_gen_{g}"]
    return base + cols


def read_existing_stems(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    with csv_path.open("r", newline="") as f:
        return {r["stem"] for r in csv.DictReader(f) if r.get("stem")}


def append_rows(csv_path: Path, header: list[str], rows: list[dict]) -> None:
    write_header = not csv_path.is_file()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def default_out_path(gen_listener_dir: Path, source: str,
                     vad_mode: str = "none") -> Path:
    """`<split>_<source>` → drop suffix, write to `<src>/_metrics/rpcc_<source>[_<vmode>]__<split>.csv`."""
    name = gen_listener_dir.name
    for sfx in ("_emoca", "_liveportrait"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
            break
    suffix = "" if vad_mode == "none" else f"_{vad_mode}"
    return gen_listener_dir.parent / "_metrics" / f"rpcc_{source}{suffix}__{name}.csv"


# --- driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--speaker", required=True, type=Path,
                   help="<split>_emoca or <split>_liveportrait dir")
    p.add_argument("--gt_listener", required=True, type=Path)
    p.add_argument("--gen_listener", required=True, type=Path)
    p.add_argument("--feature_source", choices=("emoca", "lp"), required=True)
    p.add_argument("--pairing", default="exact",
                   help="exact | flip01 | from_file:<path-to-pairs.txt>")
    p.add_argument("--vad_dir", type=Path, default=None,
                   help="Optional VAD npz dir (see extractors/vad). Used "
                        "together with --vad_mode to restrict the metric "
                        "to listener silence / speech frames.")
    p.add_argument("--vad_mode", choices=("none", "silence", "speech"),
                   default="none",
                   help="Which listener frames to keep when computing pcc: "
                        "`silence` keeps frames where the listener is silent "
                        "(= they're actually listening); `speech` keeps frames "
                        "where the listener is speaking. Requires --vad_dir.")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV; default: <gen_parent>/_metrics/rpcc_<source>__<split>.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Resample every source to this fps before "
                        "correlation. DEFAULT 25 — this re-creates the "
                        "original's uniform-fps condition (the models are "
                        "25fps, GT is 30fps; matching them to 25 is part of "
                        "reproducing the original metric, not a deviation). "
                        "No-op on already-25fps sources. Pass 0 to disable.")
    return p.parse_args()


def _load_vad_mask(vad_dir: Path | None, stem: str, mode: str,
                   target_len: int) -> "np.ndarray | None":
    """Thin wrapper around shared `per_frame_keep_mask`. Returns None
    when vad_dir is None or mode is the identity ('none')."""
    if vad_dir is None:
        return None
    return per_frame_keep_mask(vad_dir / f"{stem}.npz", mode, target_len)


def main() -> None:
    args = parse_args()
    pairing_mode, pairs = parse_pairing_arg(args.pairing)
    out_path = args.out or default_out_path(args.gen_listener,
                                            args.feature_source,
                                            vad_mode=args.vad_mode)
    header = csv_header(args.feature_source)
    if args.overwrite and out_path.is_file():
        out_path.unlink()
    seen = read_existing_stems(out_path)

    listener_stems_gt = {p.stem for p in args.gt_listener.glob("*.npz")}
    listener_stems_gen = {p.stem for p in args.gen_listener.glob("*.npz")}
    common_listener = sorted(listener_stems_gt & listener_stems_gen)

    rows: list[dict] = []
    skipped_no_speaker = 0
    skipped_already = 0
    failed = 0
    for stem in tqdm(common_listener, ncols=80, desc=f"rpcc_{args.feature_source}"):
        if stem in seen:
            skipped_already += 1; continue
        sp_stem = speaker_for(stem, pairing_mode, pairs)
        if sp_stem is None:
            skipped_no_speaker += 1; continue
        sp_path = args.speaker / f"{sp_stem}.npz"
        if not sp_path.is_file():
            skipped_no_speaker += 1; continue
        try:
            sp_g = load_motion_groups(sp_path,                                  args.feature_source, target_fps=args.target_fps)
            gt_g = load_motion_groups(args.gt_listener / f"{stem}.npz",         args.feature_source, target_fps=args.target_fps)
            gn_g = load_motion_groups(args.gen_listener / f"{stem}.npz",        args.feature_source, target_fps=args.target_fps)
        except Exception as e:
            print(f"[fail] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1; continue
        T = int(min(len(sp_g["overall"]), len(gt_g["overall"]),
                    len(gn_g["overall"])))
        keep = _load_vad_mask(args.vad_dir, stem, args.vad_mode,
                              target_len=T)
        n_kept = int(keep.sum()) if keep is not None else T
        row = {"stem": stem, "n_frames": T, "n_kept": n_kept}
        per_g = rpcc_per_group(sp_g, gt_g, gn_g, keep_mask=keep)
        for g in GROUPS[args.feature_source]:
            d = per_g.get(g, {"rPCC": float("nan"),
                              "pcc_gt": float("nan"),
                              "pcc_gen": float("nan")})
            row[f"rPCC_{g}"]    = "" if np.isnan(d["rPCC"])    else f"{d['rPCC']:.6f}"
            row[f"pcc_gt_{g}"]  = "" if np.isnan(d["pcc_gt"])  else f"{d['pcc_gt']:.6f}"
            row[f"pcc_gen_{g}"] = "" if np.isnan(d["pcc_gen"]) else f"{d['pcc_gen']:.6f}"
        rows.append(row)

    if rows:
        append_rows(out_path, header, rows)

    print(f"matched listener stems (gt ∩ gen): {len(common_listener)}")
    print(f"  written:  {len(rows)}")
    print(f"  skipped (already in csv): {skipped_already}")
    print(f"  skipped (no speaker found): {skipped_no_speaker}")
    print(f"  failed: {failed}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
