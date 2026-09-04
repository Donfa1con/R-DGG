#!/usr/bin/env python3
"""Reproduction smoke check on the single shortest SI-184 clip.

Golden stem (listener): V00_S2035_I00000998_P1293A
Its speaker (dyadic partner): V00_S2035_I00000998_P1292A

After the pipeline extracts the features, this recomputes a small FINGERPRINT for
the golden stem. It compares the fingerprint to a committed golden JSON within
tolerances. The fingerprint is the shape, mean, and std of every array in the GT
emoca / liveportrait / hubert / vad npz for the golden stem.

This check verifies the EXTRACTORS reproduce on the shortest clip. It does NOT
check per-video rPCC or TLCC. Upstream landmark detection is GPU-nondeterministic,
so the difference-of-correlations (rPCC) and the argmax lag (TLCC) are not
bit-reproducible per video. The R-DGG metric's reproducibility is CORPUS-level:
its golden is the committed SI-184 per-model numbers, verified by a full-corpus
run. See tests/golden/README.md.

Usage:
  python tests/check_reproduction.py                 # compare to tests/golden/P1293A.json
  python tests/check_reproduction.py --write         # (re)generate the golden from this run
  python tests/check_reproduction.py --rtol 1e-2 --atol 1e-4

Needs numpy (run under the shared metrics env, from the repo root:
`pixi run --manifest-path rdgg/pixi.toml python tests/check_reproduction.py`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))               # tests/
_REPO = os.path.dirname(_HERE)
DEFAULT_CORPUS = os.path.join(_REPO, "data")
DEFAULT_GOLDEN = os.path.join(_HERE, "golden", "P1293A.json")

GOLDEN_STEM = "V00_S2035_I00000998_P1293A"
SPEAKER_STEM = "V00_S2035_I00000998_P1292A"
SPLIT = "LS_AL"

# feature name -> corpus dir suffix (all read from GT)
FEATURE_DIRS = {
    "emoca":        f"{SPLIT}_emoca",
    "liveportrait": f"{SPLIT}_liveportrait",
    "hubert":       f"{SPLIT}_hubert",
    "vad":          f"{SPLIT}_vad",
}


def _num(x):
    """np scalar / python -> float, or None (JSON-safe) for nan / empty."""
    if x is None:
        return None
    v = float(x)
    return None if np.isnan(v) else v


def fingerprint_features(corpus: str, stem: str) -> dict:
    """shape + mean + std of every array in each GT feature npz for `stem`."""
    out: dict = {}
    for name, sfx in FEATURE_DIRS.items():
        path = os.path.join(corpus, "GT", sfx, f"{stem}.npz")
        if not os.path.isfile(path):
            sys.exit(f"missing feature file: {path}\n  run pipeline/3_extract.sh first.")
        with np.load(path, allow_pickle=False) as z:
            per = {}
            for key in sorted(z.files):
                arr = np.asarray(z[key])
                if arr.dtype == bool or np.issubdtype(arr.dtype, np.number):
                    a = arr.astype(np.float64)
                    per[key] = {
                        "shape": list(arr.shape),
                        "mean": _num(np.nanmean(a)) if a.size else None,
                        "std":  _num(np.nanstd(a)) if a.size else None,
                    }
                else:
                    per[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            out[name] = per
    return out


def build_fingerprint(corpus: str, stem: str) -> dict:
    return {
        "stem": stem,
        "speaker": SPEAKER_STEM,
        "features": fingerprint_features(corpus, stem),
    }


# ---- comparison -------------------------------------------------------------
def _compare(exp, got, path, rtol, atol, diffs: list):
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            diffs.append(f"{path}: type dict vs {type(got).__name__}")
            return
        for k in exp:
            if k not in got:
                diffs.append(f"{path}/{k}: missing in current run")
            else:
                _compare(exp[k], got[k], f"{path}/{k}", rtol, atol, diffs)
        return
    if isinstance(exp, list):  # shapes / dtype-string lists
        if list(got) != list(exp):
            diffs.append(f"{path}: {got} != {exp}")
        return
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        if not np.isclose(got, exp, rtol=rtol, atol=atol):
            diffs.append(f"{path}: {got:.6g} != {exp:.6g} (rtol={rtol}, atol={atol})")
        return
    if exp != got:  # None / strings
        diffs.append(f"{path}: {got!r} != {exp!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus root (default: <repo>/data)")
    ap.add_argument("--stem", default=GOLDEN_STEM)
    ap.add_argument("--golden", default=DEFAULT_GOLDEN, help="committed golden JSON path")
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--write", action="store_true",
                    help="(re)generate the golden JSON from this run instead of comparing")
    args = ap.parse_args()

    fp = build_fingerprint(args.corpus, args.stem)

    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.golden)), exist_ok=True)
        with open(args.golden, "w") as fh:
            json.dump(fp, fh, indent=2, allow_nan=False)
        print(f"[write] golden -> {args.golden}")
        return

    if not os.path.isfile(args.golden):
        print(f"SKIP: golden not yet populated ({args.golden}).\n"
              "      run with --write once the pipeline has produced the golden-stem outputs.")
        return  # known-absent state (owner populates separately); not a failure

    with open(args.golden) as fh:
        golden = json.load(fh)

    diffs: list = []
    _compare(golden, fp, "", args.rtol, args.atol, diffs)
    if diffs:
        print(f"MISMATCH ({len(diffs)}) vs {args.golden}  (rtol={args.rtol}, atol={args.atol}):")
        for d in diffs:
            print(f"  - {d.lstrip('/')}")
        sys.exit(1)
    print(f"OK: golden-stem reproduction matches {os.path.basename(args.golden)} "
          f"within rtol={args.rtol}, atol={args.atol}")


if __name__ == "__main__":
    main()
