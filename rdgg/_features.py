"""Feature and mask loading for R-DGG.

Each function reads one encoder cache off disk and puts it on the 25 fps metric
grid. The corpus paths come from _config; the mutated ones are read as
`_config.CORPUS` and `_config.SPLIT`.
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extractors/vad"))
from vad_mask import reactive_segments_mask, DEFAULT_MIN_SEG_SEC, DEFAULT_PAD_LISTENER_SILENCE_SEC

import _config
from _config import GRID_FPS, HUB_LAYER, LAG_NP, LSPK, MINSEG, MODELS_ALL_DIRS, P, REGION, parallel_map


def load_npz(dir_, stem, key):
    path = f"{dir_}/{stem}.npz"
    return np.load(path)[key].astype(np.float64) if os.path.exists(path) else None


def fit_pca(mats, k):
    X = np.concatenate(mats, 0)
    X = X[np.isfinite(X).all(1)]
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-8)
    Xc = (X - mu) / sd
    idx = np.random.default_rng(0).choice(len(Xc), size=min(40000, len(Xc)), replace=False)
    _, sv, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    kk = min(k, Vt.shape[0])
    return mu, sd, Vt[:kk].T


def pca_fit_sample(stems, seed):
    n = 60
    stems = list(stems)
    if len(stems) <= n:
        return stems
    idx = np.random.default_rng(seed).choice(len(stems), size=n, replace=False)
    return [stems[i] for i in sorted(idx)]


def reactive_mask(ls, sp, n_grid):
    mask = reactive_segments_mask(
        Path(f"{_config.VAD}/{ls}.npz"),
        Path(f"{_config.VAD}/{sp}.npz"),
        n_grid + 4,
        min_seg_sec=DEFAULT_MIN_SEG_SEC,
        pad_sec=DEFAULT_PAD_LISTENER_SILENCE_SEC,
    )
    return None if mask is None else mask[2 : n_grid + 2]


def runs_of(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def length_gate(n_src, n_grid):
    """Is this source close enough to the grid to be a rate difference rather than a broken generation?

    One function for all three call sites, gating on n_grid so the alignment table never describes a stem the design
    has dropped -- such as a truncated generation of 1387 frames against a 7896-frame grid.
    """
    return 0.5 * n_grid < n_src < 2.5 * n_grid


def align_rate(arr, n_dst, src_fps, dst_fps, pad_value=None):
    n_src = arr.shape[0]
    ratio = 1.0 if (src_fps is None or dst_fps is None) else float(src_fps) / float(dst_fps)
    if n_src == n_dst and abs(ratio - 1.0) < 1e-9:
        return arr
    pos = np.arange(n_dst) * ratio
    src_x = np.arange(n_src)
    beyond = {} if pad_value is None else {"right": pad_value}
    return np.stack([np.interp(pos, src_x, arr[:, col], **beyond) for col in range(arr.shape[1])], 1)


def to_grid(arr, n_dst, src_fps, pad_value=None):
    return align_rate(arr, n_dst, src_fps, GRID_FPS, pad_value)


def load_regions_full(npz_path, source):
    if source == "combo":
        ereg, efok, efps = load_regions_full(npz_path, "emoca")
        lp = np.load(npz_path.replace(f"{_config.SPLIT}_emoca", f"{_config.SPLIT}_liveportrait"))
        Tl = int(lp["n_frames"])
        lfok = lp["face_ok"][:Tl].astype(bool)
        Rr = lp["R"][:Tl].astype(np.float64)
        expo = lp["exp"][:Tl].astype(np.float64)
        so3 = np.zeros((Tl, 3))
        if lfok.any():
            so3[lfok] = Rotation.from_matrix(Rr[lfok]).as_rotvec()
        t_ = lp["t"][:Tl].astype(np.float64)
        sc = lp["scale"][:Tl].astype(np.float64).reshape(Tl, 1)
        lov = np.concatenate([so3, expo.reshape(Tl, -1), t_, sc], 1)
        eov = ereg[REGION]
        lp_fps = float(lp["fps"]) if "fps" in lp.files else None
        lov = align_rate(lov, len(eov), lp_fps, efps)
        lfok_r = align_rate(lfok[:, None].astype(float), len(eov), lp_fps, efps, pad_value=0.0)[:, 0] > 0.5
        return {REGION: np.concatenate([eov, lov], 1)}, (efok & lfok_r), efps
    npz = np.load(npz_path)
    n_frames = int(npz["n_frames"])
    face_ok = npz["face_ok"][:n_frames].astype(bool)
    fps = float(npz["fps"]) if "fps" in npz.files else None
    assert source == "emoca", f"unknown motion source {source!r} (only 'combo' and its internal 'emoca' half exist)"
    pose = npz["posecode"][:n_frames].astype(np.float64)
    exp = npz["expcode"][:n_frames].astype(np.float64)
    return {REGION: np.concatenate([pose, exp], 1)}, face_ok, fps


def speaker_raw(sp):
    """The speech features on the metric grid -- and the definition of that grid's length.

    The timeline is the speech stream decimated by two: extractors/hubert emits exactly 50 frames per second, so half of
    that is exactly the 25 fps this metric works on, and every slot reads a real frame rather than a blend of two.
    Interpolating instead would sit about half a slot off, and these features change by ~39% across half a slot.

    The grid length is derived here rather than taken from an external SyncNet embedding: the integer division is exact
    on every stem of this corpus, including the two whose duration is not a whole number of slots, where it absorbs the
    half slot cleanly.
    """
    hp = f"{_config.HUB}/{sp}.npz"
    if not os.path.exists(hp):
        return None
    with np.load(hp) as npz:
        hubert = npz[HUB_LAYER].astype(np.float64)
        rate = float(npz["fps"]) if "fps" in npz.files else None
    assert rate is not None and abs(rate - 2 * GRID_FPS) < 1e-6, (
        f"{sp}: speech features are at {rate} Hz, not the exact {2 * GRID_FPS:.0f} that decimating by two onto a "
        f"{GRID_FPS:.0f} fps grid requires. Extract with extractors/hubert, which pads by one hop."
    )
    if len(hubert) < 2 * (P + 12):
        return None
    return hubert[::2][2 : len(hubert) // 2 - 2]


def load_all_motion(dir_, source, corpus):
    sts = list(corpus.listeners)

    def load_one_stem(st):
        path = f"{dir_}/{st}.npz"
        if not os.path.exists(path):
            return None
        try:
            return load_regions_full(path, source)
        except Exception:
            return None

    return {
        st: loaded for st, loaded in zip(sts, parallel_map(load_one_stem, sts)) if loaded is not None
    }


def _src_fok(model_dir, ls, suffix, source):
    path = f"{_config.CORPUS}/{model_dir}/{_config.SPLIT}_{suffix}/{ls}.npz"
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as npz:
            n_frames = int(npz["n_frames"])
            fk = npz["face_ok"][:n_frames].astype(bool)
            fps = float(npz["fps"]) if "fps" in npz.files else None
    except Exception:
        return None
    if source == "combo":
        q2 = path.replace(f"{_config.SPLIT}_emoca", f"{_config.SPLIT}_liveportrait")
        if not os.path.exists(q2):
            return None
        try:
            with np.load(q2) as z2:
                n2 = int(z2["n_frames"])
                fk2 = z2["face_ok"][:n2].astype(bool)
                fps2 = float(z2["fps"]) if "fps" in z2.files else None
        except Exception:
            return None
        fk = fk & (align_rate(fk2[:, None].astype(float), n_frames, fps2, fps, pad_value=0.0)[:, 0] > 0.5)
    return n_frames, fk, fps


def build_eval_mask(suffix, source, corpus):
    """The frame set every model is scored on: valid where ground truth AND every model tracked the face.

    Rebuilt every run rather than cached: a rebuild measures 1.4 s against a 900 s run, so a cache buys nothing, and
    caching this mask is a rich source of staleness -- a mask built for one grid length silently truncated to another,
    an empty set from a failed run frozen in, or a stale artefact reloaded after the code that wrote it changed. It
    also keeps the metric out of the user's data directory. Adding a model redefines the row set and re-scores every
    model by itself, with no cache file for the operator to delete by hand.
    """
    out = {}
    drop = {}
    for ls in corpus.listeners:
        sp = corpus.pairs.get(ls)
        Sp = corpus.SPK_CACHE.get(sp)
        if Sp is None:
            drop[ls] = "no-speaker-audio"
            continue
        n_grid = len(Sp)
        valid = np.ones(n_grid, bool)
        why = None
        for lbl, model_dir in MODELS_ALL_DIRS:
            probe = _src_fok(model_dir, ls, suffix, source)
            if probe is None:
                why = f"{lbl}:missing"
                break
            n_frames, fk, fps = probe
            if not length_gate(n_frames, n_grid):
                why = f"{lbl}:length-gate({n_frames}/{n_grid})"
                break
            valid &= (
                to_grid(fk[:, None].astype(float), n_grid + 4, fps, pad_value=0.0)[2 : n_grid + 2, 0] > 0.5
            )
        if why:
            drop[ls] = why
        else:
            out[ls] = valid
    n_valid = sum(int(valid.sum()) for valid in out.values())
    n_total = sum(len(valid) for valid in out.values())
    print(
        f"  [evalmask] {len(out)}/{len(corpus.listeners)} stems kept, "
        f"{100 * n_valid / max(n_total, 1):.2f}% of their frames valid in ALL models; dropped {len(drop)}",
        flush=True,
    )
    for ls, why in sorted(drop.items()):
        print(f"    [evalmask] DROP {ls}  ({why})", flush=True)
    assert out, (
        f"no listener survived: every one was dropped for the reason above. All of "
        f"{[d for _l, d in MODELS_ALL_DIRS]} need a {_config.SPLIT}_{suffix} and a {_config.SPLIT}_liveportrait directory under "
        f"{_config.CORPUS}"
    )
    return out


def gtonly_eval_mask(suffix, source, corpus):
    """Frame validity from GROUND TRUTH's face_ok ALONE, dropping the all-model intersection that build_eval_mask
    imposes. This is the mask the GT reference operator B is fit on, and the mask each cell of the pairwise matrix
    uses (a design then intersects its OWN face_ok on top). Fitting B on the fullest GT decouples the reference
    from whether the *models* tracked -- adding or failing a model does not perturb the operator. Mirrors
    build_eval_mask's per-listener grid projection, restricted to GT.
    """
    out = {}
    for ls in corpus.listeners:
        sp = corpus.pairs.get(ls)
        Sp = corpus.SPK_CACHE.get(sp)
        if Sp is None:
            continue
        n_grid = len(Sp)
        probe = _src_fok("GT", ls, suffix, source)
        if probe is None:
            continue
        n_frames, fk, fps = probe
        if not length_gate(n_frames, n_grid):
            continue
        out[ls] = to_grid(fk[:, None].astype(float), n_grid + 4, fps, pad_value=0.0)[2 : n_grid + 2, 0] > 0.5
    return out


def alignment_stats(mcache, corpus, eval_mask):
    """Per-source: declared rate, frame surplus against the grid, and grid slots with no source frame behind them.

    Reports ONLY stems that are actually scored. Without the eval-mask filter it described a broken generation -- 1387
    frames against a 7896-frame grid -- that the length gate had already dropped, which is exactly the kind of
    misdescription this table exists to catch.
    """
    rates, offsets, uncovered = set(), [], []
    for ls, (reg, _face_ok, fps) in mcache.items():
        mat = reg.get(REGION)
        Sp_real = corpus.SPK_CACHE.get(corpus.pairs.get(ls, ""))
        if mat is None or Sp_real is None or ls not in eval_mask:
            continue
        n_src, n_dst = len(mat), len(Sp_real) + 4
        if not length_gate(n_src, len(Sp_real)):
            continue
        rate = GRID_FPS if fps is None else float(fps)
        rates.add(round(rate, 3))
        offsets.append(n_src - n_dst)
        uncovered.append(max(0, int(np.ceil((n_dst - 1) * (rate / GRID_FPS) - (n_src - 1)))))
    return rates, offsets, uncovered


def collect_design(mcache, ctx, eval_mask=None):
    """Build the per-listener regression design. `eval_mask` selects which grid frames are admissible (default
    ctx.eval_mask = the COMMON all-model set for the leaderboard; the B-fit and the pairwise matrix pass the
    GT-only mask instead)."""
    if eval_mask is None:
        eval_mask = ctx.eval_mask

    def design_for_listener(ls):
        sp = ctx.corpus.pairs[ls]
        Sp_real = ctx.corpus.SPK_CACHE.get(sp)
        if Sp_real is None or ls not in mcache:
            return None
        reg, face_ok, fps = mcache[ls]
        mat = reg.get(REGION)
        if mat is None or len(mat) < 2:
            return None
        n_grid = len(Sp_real)
        if not length_gate(len(mat), n_grid):
            return None
        Lp = to_grid(mat, n_grid + 4, fps)[2 : n_grid + 2]
        frame_ok = to_grid(face_ok[:, None].astype(float), n_grid + 4, fps, pad_value=0.0)[2 : n_grid + 2, 0] > 0.5
        Lp = Lp - (Lp[frame_ok].mean(0, keepdims=True) if frame_ok.any() else Lp.mean(0, keepdims=True))
        fok = frame_ok
        common_valid = eval_mask.get(ls)
        if common_valid is None:
            return None
        assert len(common_valid) == len(fok), f"eval mask {len(common_valid)} vs grid {len(fok)} for {ls}"
        fok = fok & common_valid
        mask = reactive_mask(ls, sp, n_grid)
        if mask is None:
            return None
        _k2 = min(len(mask), len(fok))
        mask = mask[:_k2] & fok[:_k2]
        if mask.sum() < P + 10:
            return None
        segments = runs_of(mask)
        for seg_a, seg_b in segments:
            if seg_b - seg_a >= 25:
                Lp[seg_a:seg_b] = Lp[seg_a:seg_b] - Lp[seg_a:seg_b].mean(0, keepdims=True)
        frame_and_segstart = [
            (frame, seg_a)
            for (seg_a, seg_b) in segments
            if (seg_b - seg_a) >= MINSEG
            for frame in range(max(seg_a + P, LSPK), seg_b)
        ]
        if not frame_and_segstart:
            return None
        tid = np.asarray([frame for frame, seg_a in frame_and_segstart])
        seg0 = np.asarray([seg_a for frame, seg_a in frame_and_segstart])
        depth = tid - seg0
        own_lags = LAG_NP
        XO = Lp[tid[:, None] - own_lags[None, :]].reshape(len(tid), -1)
        return (XO.astype(np.float32), Lp[tid].astype(np.float32), tid, depth)

    return {ls: built for ls in ctx.corpus.listeners if (built := design_for_listener(ls)) is not None}


def gridded_face_ok(mcache, corpus):
    """Each listener's face_ok, projected onto the speaker grid (the same grid projection collect_design applies to build frame_ok).
    A PAIRED 8x8 cell (A,B) scores both on GT&A&B frames: capture A with eval_mask = gtonly & B's grid face_ok
    (A's own face_ok is applied inside collect_design), and symmetrically for B -- both land on GT&A&B."""
    out = {}
    for ls, (reg, face_ok, fps) in mcache.items():
        sp = corpus.pairs.get(ls)
        Sp = corpus.SPK_CACHE.get(sp)
        if Sp is None or face_ok is None:
            continue
        n_grid = len(Sp)
        if not length_gate(len(face_ok), n_grid):
            continue
        out[ls] = to_grid(face_ok[:, None].astype(float), n_grid + 4, fps, pad_value=0.0)[2 : n_grid + 2, 0] > 0.5
    return out


def speaker_motion_raw(sp, suffix, source, n_target):
    path = f"{_config.CORPUS}/GT/{_config.SPLIT}_{suffix}/{sp}.npz"
    if not os.path.exists(path):
        return None
    try:
        reg, face_ok, fps = load_regions_full(path, source)
    except Exception:
        return None
    motion = reg.get(REGION)
    if motion is None or len(motion) < 2 or not length_gate(len(motion), n_target):
        return None
    ok = face_ok[: len(motion)]
    if not ok.any():
        return None
    motion = np.where(ok[:, None], motion, motion[ok].mean(0, keepdims=True))
    out = to_grid(motion, n_target + 4, fps)[2 : n_target + 2]
    return out - out.mean(0, keepdims=True)
