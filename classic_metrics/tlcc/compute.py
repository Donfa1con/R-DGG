"""TLCC — Time-Lagged Cross-Correlation between speaker and listener.

Adapted from `baseline_react2024/metric/TLCC.py`. For each video pair
(speaker, listener) and lag `l` over `range(-(2·fps-1), 2·fps)` (i.e.
±(2·fps−1) frames — ±49 @ 25fps, the react2024 2-second window),
compute the mean per-dim Pearson correlation between speaker[t, :] and
listener[t+l, :] (equal weight over the region's dims). Report:

    tlcc_offset  = center − argmax   (in frames; the react2024 headline —
                   the reported number is mean|offset| over stems)
    tlcc_peak    = max over lags of mean correlation  (auxiliary; discarded
                   in the react2024 original but retained here)
    tlcc_center  = correlation at lag 0                (auxiliary)

`center` is the lag-0 index `len(rs)//2`, so `offset = center − argmax`
matches react2024 exactly.

We compute all of the above per **group** — the paper's two region
columns plus overall (overall + pose/exp for EMOCA; overall + rot/exp
for LP), as required by the repo's dual-source convention. The
original has no region split; the pose/exp & rot/exp breakdown is an
explicit user deviation.

The script handles ONE direction (speaker → listener); for symmetric
analysis run again with --speaker / --listener swapped.

Usage:
  python compute.py \\
      --speaker         <corpus>/<src>/LS_emoca \\
      --listener        <corpus>/<src>/AL_emoca \\
      --feature_source  emoca \\
      --pairing         flip01 \\
      --fps             30 \\
      [--max_lag_sec 2] \\
      [--out <corpus>/<src>/_metrics/tlcc_emoca__<split>.csv] \\
      [--overwrite]

Output: per-stem CSV with columns
    stem, n_frames, fps,
    tlcc_peak_overall, tlcc_center_overall, tlcc_offset_overall,
    tlcc_peak_<group>, tlcc_center_<group>, tlcc_offset_<group>, ...
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

def _shift(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return x[lag:], y[:-lag]
    if lag < 0:
        return x[:lag], y[-lag:]
    return x, y


def _crosscorr_one_lag(speaker: np.ndarray, listener: np.ndarray,
                       lag: int) -> float:
    """Mean per-dim Pearson at the given lag. (T, D) inputs.

    Vectorised across dims. Same dim-validity gate:
    `std < 1e-12` ⟺ ssq < n·1e-24.
    """
    sp_l, ls_l = _shift(speaker, listener, lag)
    n = sp_l.shape[0]
    if n < 2:
        return float("nan")
    sp_c = sp_l - sp_l.mean(0)            # (n, D)
    ls_c = ls_l - ls_l.mean(0)
    num    = (sp_c * ls_c).sum(0)         # (D,)
    sp_ssq = (sp_c * sp_c).sum(0)
    ls_ssq = (ls_c * ls_c).sum(0)
    valid  = (sp_ssq > n * 1e-24) & (ls_ssq > n * 1e-24)
    if not valid.any():
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / np.sqrt(sp_ssq * ls_ssq)
    return float(np.mean(r[valid]))


def effective_max_lag(T: int, max_lag: int, adaptive: bool) -> int:
    """Largest lag we can search given T frames.

    Cross-correlation at lag L needs at least T - |L| ≥ 4 paired samples
    on each side of the shift, so |L| ≤ (T - 4) / 2.

    - adaptive=False: returns `max_lag` and the caller is expected to
      bail out on too-short videos (the strict react2024 behaviour).
    - adaptive=True:  returns `min(max_lag, (T - 4) // 2)`, never larger
      than what fits. Returns 0 if even lag=0 wouldn't have enough samples.
    """
    if not adaptive:
        return max_lag
    fit = (T - 4) // 2
    if fit < 1:
        return 0
    return min(max_lag, fit)


def tlcc_one_group(speaker: np.ndarray, listener: np.ndarray,
                   max_lag: int, adaptive: bool = False,
                   keep_mask: np.ndarray | None = None,
                   ) -> tuple[float, float, int, int]:
    """Compute (peak, center, offset, eff_max_lag) over lags ∈ [-L, +L]
    where L = effective_max_lag(T, max_lag, adaptive).

    With `adaptive=True` short videos are NOT dropped — the search window
    shrinks to what their length allows. `eff_max_lag` is returned so the
    consumer can tell which lag was actually used and treat offsets in
    short clips with appropriate skepticism.

    If `keep_mask` is provided (1-D bool/uint8), only frames where the
    mask is 1 are used (after truncation to common length). Use for
    silence-only / speech-only restriction.
    """
    T = min(len(speaker), len(listener))
    if keep_mask is not None:
        m = np.asarray(keep_mask[:T]).astype(bool)
        speaker = speaker[:T][m]
        listener = listener[:T][m]
        T = len(speaker)
    L = effective_max_lag(T, max_lag, adaptive)
    if L <= 0 or (not adaptive and T < 2 * max_lag + 4):
        return float("nan"), float("nan"), 0, L
    sp = speaker[:T]; ls = listener[:T]
    rs = [_crosscorr_one_lag(sp, ls, lag) for lag in range(-L, L + 1)]
    rs_arr = np.array(rs, dtype=np.float64)
    if np.all(np.isnan(rs_arr)):
        return float("nan"), float("nan"), 0, L
    peak = float(np.nanmax(rs_arr))
    center = float(rs_arr[L]) if not np.isnan(rs_arr[L]) else float("nan")
    argmax = int(np.nanargmax(rs_arr))
    offset = L - argmax  # positive ⇒ listener lags speaker
    return peak, center, offset, L


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
    raise ValueError(mode)


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
    cols = ["stem", "n_frames", "n_kept", "fps", "max_lag", "eff_max_lag"]
    for g in GROUPS[source]:
        cols += [f"tlcc_peak_{g}", f"tlcc_center_{g}", f"tlcc_offset_{g}"]
    return cols


def read_existing_stems(p: Path) -> set[str]:
    if not p.is_file(): return set()
    with p.open("r", newline="") as f:
        return {r["stem"] for r in csv.DictReader(f) if r.get("stem")}


def append_rows(p: Path, header: list[str], rows: list[dict]) -> None:
    new = not p.is_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow(r)


def default_out(listener_dir: Path, source: str) -> Path:
    name = listener_dir.name
    for sfx in ("_emoca", "_liveportrait"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]; break
    return listener_dir.parent / "_metrics" / f"tlcc_{source}__{name}.csv"


# --- driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--speaker",  required=True, type=Path)
    p.add_argument("--listener", required=True, type=Path)
    p.add_argument("--feature_source", choices=("emoca", "lp"), required=True)
    p.add_argument("--pairing", default="exact",
                   help="exact | flip01 | from_file:<path-to-pairs.txt>")
    p.add_argument("--fps", type=float, default=None,
                   help="frame rate for the lag window; if omitted, read "
                        "from npz `fps` field of first stem (overridden by "
                        "--target_fps when that is set)")
    p.add_argument("--target_fps", type=float, default=25.0,
                   help="Resample every source to this fps before "
                        "cross-correlation AND use it as the lag-window fps. "
                        "DEFAULT 25 — this re-creates the original's "
                        "uniform-fps condition (the models are 25fps, GT is "
                        "30fps; matching them to 25 reproduces the original "
                        "metric, it is not a deviation). No-op on already-"
                        "25fps sources. Pass 0 to disable (native npz fps).")
    p.add_argument("--max_lag_sec", type=float, default=2.0,
                   help="lag window in seconds; react2024 default 2 ⇒ lags "
                        "range(-(2·fps-1), 2·fps) = ±49 frames @ 25fps")
    p.add_argument("--adaptive_max_lag", action="store_true",
                   help="If set, shrink the search window per-video when "
                        "T < 2*max_lag+4 instead of returning NaN. The "
                        "actual lag used is recorded in the `eff_max_lag` "
                        "CSV column. Lets short clips contribute a "
                        "narrower-window measurement instead of being "
                        "dropped entirely.")
    p.add_argument("--vad_dir", type=Path, default=None,
                   help="Optional VAD npz dir (see extractors/vad). Used "
                        "with --vad_mode to mask features before TLCC.")
    p.add_argument("--vad_mode", choices=("none", "silence", "speech"),
                   default="none",
                   help="`silence` keeps frames where the LISTENER is "
                        "silent (= they're actually listening — the right "
                        "regime for measuring listener responsiveness). "
                        "`speech` keeps frames where the listener is "
                        "speaking. Requires --vad_dir.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairing_mode, pairs = parse_pairing_arg(args.pairing)
    out_path = args.out or default_out(args.listener, args.feature_source)
    header = csv_header(args.feature_source)
    if args.overwrite and out_path.is_file():
        out_path.unlink()
    seen = read_existing_stems(out_path)

    listener_npzs = sorted(args.listener.glob("*.npz"))
    if args.limit:
        listener_npzs = listener_npzs[: args.limit]
    if not listener_npzs:
        print(f"no *.npz in {args.listener}"); return

    # Determine the fps used for the lag window. --target_fps (opt-in)
    # wins and also triggers resampling of the motion series below.
    if args.target_fps:
        fps = float(args.target_fps)
    elif args.fps is not None:
        fps = args.fps
    else:
        fps = float(np.load(listener_npzs[0])["fps"])
    # react2024 window: lags range(-(2·fps-1), 2·fps) ⇒ magnitude 2·fps-1
    # (±49 @ 25fps, sec=2). Our loop uses range(-L, L+1) with L=max_lag,
    # so L must be (2·fps-1).
    max_lag = int(round(args.max_lag_sec * fps)) - 1
    if max_lag < 1:
        max_lag = 1

    rows: list[dict] = []
    skipped_no_speaker = 0
    skipped_already = 0
    failed = 0
    for p in tqdm(listener_npzs, ncols=80, desc=f"tlcc_{args.feature_source}"):
        stem = p.stem
        if stem in seen:
            skipped_already += 1; continue
        sp_stem = speaker_for(stem, pairing_mode, pairs)
        if sp_stem is None:
            skipped_no_speaker += 1; continue
        sp_path = args.speaker / f"{sp_stem}.npz"
        if not sp_path.is_file():
            skipped_no_speaker += 1; continue
        try:
            sp_g = load_motion_groups(sp_path, args.feature_source,
                                      target_fps=args.target_fps)
            ls_g = load_motion_groups(p,        args.feature_source,
                                      target_fps=args.target_fps)
        except Exception as e:
            print(f"[fail] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1; continue
        T = min(len(sp_g["overall"]), len(ls_g["overall"]))
        keep = None
        if args.vad_mode != "none" and args.vad_dir is not None:
            keep = per_frame_keep_mask(args.vad_dir / f"{stem}.npz",
                                       args.vad_mode, T)
        n_kept = int(np.asarray(keep[:T]).sum()) if keep is not None else T
        eff_L = effective_max_lag(n_kept, max_lag, args.adaptive_max_lag)
        row = {"stem": stem, "n_frames": int(T), "n_kept": int(n_kept),
               "fps": f"{fps:.4f}",
               "max_lag": int(max_lag), "eff_max_lag": int(eff_L)}
        for g in GROUPS[args.feature_source]:
            peak, center, offset, _ = tlcc_one_group(
                sp_g[g], ls_g[g], max_lag,
                adaptive=args.adaptive_max_lag, keep_mask=keep)
            row[f"tlcc_peak_{g}"]   = "" if np.isnan(peak)   else f"{peak:.6f}"
            row[f"tlcc_center_{g}"] = "" if np.isnan(center) else f"{center:.6f}"
            row[f"tlcc_offset_{g}"] = int(offset)
        rows.append(row)

    if rows:
        append_rows(out_path, header, rows)

    print(f"matched listener stems: {len(listener_npzs)}")
    print(f"  written:  {len(rows)}")
    print(f"  skipped (already in csv): {skipped_already}")
    print(f"  skipped (no speaker found): {skipped_no_speaker}")
    print(f"  failed: {failed}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
