"""Shared loader for motion features from EMOCA / LivePortrait caches.

Returns a dict of `(group_name → (T, D) np.ndarray)` so downstream metrics
can compute on overall + per-region columns uniformly.

Region groups match the original DIM / react2024 listener metrics:

EMOCA layout (`<split>_emoca/<stem>.npz`), 56-d:
    pose   (T, 6)   = posecode  (first 3 head-rotation axis-angle, last 3 jaw)
    exp    (T, 50)  = expcode   (FLAME expression)
    overall = concat(posecode, expcode)  → (T, 56)

LivePortrait layout (`<split>_liveportrait/<stem>.npz`), 42-d — so3
rotation plus the 39 `LIPSYNC_COORDS` indices into the 21×3 expression:
    rot   (T, 3)    so3 = rotmat_to_rotvec(R)
    exp   (T, 39)   exp.flatten()[LIPSYNC_COORDS]  (brow⊕eyes⊕mouth, kp 1..20)
    overall = concat(rot, exp)  → (T, 42)

So each source exposes the paper's two region columns plus `overall`:
    EMOCA → {overall, pose, exp}    LP → {overall, rot, exp}

`face_ok` is honoured — frames where the upstream extractor failed are
dropped from the returned arrays (so downstream stats see only valid frames).

`target_fps` (optional): resample the returned per-frame groups from the
source video fps (npz `fps` field) to a common target fps. This corrects
the SI 30fps-GT vs 25fps-model mismatch: correlation metrics (rPCC,
TLCC, ...) time-align a GT series against a model series frame-by-frame,
which is biased when the two are sampled at different rates. Passing the
SAME target_fps to every source puts them on one timeline before
truncate-to-min. No-op when the source is already at target_fps (or `fps`
is missing). The metrics default this to 25 (the models are 25fps, GT is
30fps) so that no-flags == the original metric on matched-fps data;
`load_motion_groups` itself leaves it None unless the caller passes it.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


# LIPSYNC_COORDS — 39 indices into exp.flatten() (21*3=63 dims).
LIPSYNC_COORDS = np.array([
    3, 4, 5,         # kp[1]
    6, 7, 8,         # kp[2]
    18, 19, 20,      # kp[6]
    33, 34, 35,      # kp[11]
    36, 37, 38,      # kp[12]
    39, 40, 41,      # kp[13]
    42, 43, 44,      # kp[14]
    45, 46, 47,      # kp[15]
    48, 49, 50,      # kp[16]
    51, 52, 53,      # kp[17]
    54, 55, 56,      # kp[18]
    57, 58, 59,      # kp[19]
    60, 61, 62,      # kp[20]
], dtype=np.int64)


def _so3_from_rotmat(R: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotation matrix → (T, 3) axis-angle. scipy is in env."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec().astype(np.float64)


def _resample_time(arr: np.ndarray, n_dst: int) -> np.ndarray:
    """Linear-interp resample a (T, D) array along time to n_dst frames."""
    n_src = arr.shape[0]
    if n_src == n_dst or n_src < 2 or n_dst < 1:
        return arr
    src_idx = np.arange(n_src)
    dst_idx = np.linspace(0, n_src - 1, n_dst)
    out = np.empty((n_dst, arr.shape[1]), dtype=arr.dtype)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(dst_idx, src_idx, arr[:, j])
    return out


def _maybe_resample(groups: dict, src_fps, target_fps) -> dict:
    """Resample every group to the target-fps frame count (based on the
    face_ok-filtered length). No-op when fps already matches / unknown."""
    if not target_fps or not src_fps or abs(src_fps - target_fps) < 1e-3:
        return groups
    n_src = len(groups["overall"])
    n_dst = max(1, int(round(n_src * target_fps / src_fps)))
    if n_dst == n_src:
        return groups
    return {k: _resample_time(v, n_dst) for k, v in groups.items()}


def load_motion_groups(npz_path: Path, source: str,
                       target_fps: "float | None" = None) -> dict[str, np.ndarray]:
    """Returns {group_name: (T_face_ok, D) np.ndarray}.

    Source is `"emoca"` or `"lp"`. Always includes an `"overall"` group.
    If `target_fps` is given, groups are resampled from the npz `fps` to it.
    """
    d = np.load(npz_path)
    src_fps = float(d["fps"]) if "fps" in d.files else None

    if source == "emoca":
        T = int(d["n_frames"])
        face_ok = d["face_ok"][:T].astype(bool)
        posecode = d["posecode"][:T].astype(np.float64)
        expcode = d["expcode"][:T].astype(np.float64)
        overall = np.concatenate([posecode, expcode], axis=1)
        groups = {
            "overall": overall[face_ok],
            "pose":    posecode[face_ok],
            "exp":     expcode[face_ok],
        }
        return _maybe_resample(groups, src_fps, target_fps)

    if source == "lp":
        T = int(d["n_frames"])
        face_ok = d["face_ok"][:T].astype(bool)
        # Filter to face_ok rows BEFORE the axis-angle conversion: rows
        # where the extractor never wrote (face_ok=0 throughout) have
        # all-zero R which trips scipy's rotmat_to_rotvec ("non-positive
        # determinant"). Filtering first keeps only valid rotation matrices.
        if not face_ok.any():
            return {"overall": np.zeros((0, 42), dtype=np.float64),
                    "rot":     np.zeros((0, 3),  dtype=np.float64),
                    "exp":     np.zeros((0, 39), dtype=np.float64)}
        R = d["R"][:T][face_ok].astype(np.float64)
        exp = d["exp"][:T][face_ok].astype(np.float64)
        so3 = _so3_from_rotmat(R)                    # (N_ok, 3)
        N = R.shape[0]
        exp_lipsync = exp.reshape(N, -1)[:, LIPSYNC_COORDS]  # (N_ok, 39)
        overall = np.concatenate([so3, exp_lipsync], axis=1)  # (N_ok, 42)
        groups = {
            "overall": overall,
            "rot":     overall[:, 0:3],
            "exp":     overall[:, 3:42],   # brow⊕eyes⊕mouth (39d)
        }
        return _maybe_resample(groups, src_fps, target_fps)

    raise ValueError(f"unknown source: {source!r}; expected 'emoca' or 'lp'")


def cache_dir_for(source: str, base_split_dir: Path) -> Path:
    """`<split>_emoca` or `<split>_liveportrait` next to a video dir."""
    suffix = "emoca" if source == "emoca" else "liveportrait"
    return base_split_dir.parent / f"{base_split_dir.name}_{suffix}"
