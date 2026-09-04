"""Configuration and shared constants for R-DGG.

This module holds the fixed operating point, the mutable corpus paths, and the
small shared helpers. Every other module reads its constants from here. The four
path globals (CORPUS, SPLIT, VAD, HUB) are mutated by set_paths, so reader modules
access them as `_config.CORPUS` and never import them by value.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


stale_env = sorted(k for k in os.environ if k.startswith("GRANGER_"))
if stale_env:
    raise SystemExit(
        "ABORT: this metric has no knobs — the operating point is fixed, and every alternative was\n"
        "measured and rejected. Remove from your command line:\n  "
        + "\n  ".join(stale_env)
        + "\nTo try something different, edit the constants on a branch."
    )


DEV = "cuda"
DT = torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
N_IO_THREADS = 12


def parallel_map(fn, items):
    items = list(items)
    if N_IO_THREADS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=N_IO_THREADS) as ex:
        return list(ex.map(fn, items))


CORPUS = str(Path(__file__).resolve().parents[1] / "data")   # repo-relative default; override with --corpus / $CORPUS
SPLIT = "LS_AL"
VAD = HUB = ""
HUB_LAYER = "feat_l09"


def set_paths(corpus, split):
    global CORPUS, SPLIT, VAD, HUB
    CORPUS, SPLIT = corpus, split
    VAD = f"{CORPUS}/GT/{SPLIT}_vad"
    HUB = f"{CORPUS}/GT/{SPLIT}_hubert"


set_paths(CORPUS, SPLIT)


K_A_TARGET = 64
KMV = 32
P = 10
LMAX = 100
LSPK_MIN = 1
RIDGE = 1e-4
SCHUR_RIDGE = 1e-1  # ridge on the DRIVER block only (residual_with Schur), NOT the operator fit or own-block. When
                    # the operator is starved, s_hat_audio ~= s_hat_motion (co-speech collinear) and the Schur goes
                    # barely-PD -> beta_s blows up -> both_r/mot_r swing 30-100x. Raising just this ridge tames the
                    # worst outliers (e.g. DyStream) without shrinking the operator. Set == RIDGE to use the operator fit's ridge.
N_TIMESHIFT = 100      # timeshift nulls per model; the floor is their mean AND the bootstrap uses ALL of them
BOOTSTRAP = 20000      # cluster resamples; cheap now the bootstrap is numpy over pre-summed cluster energies
MINSEG = 4 * 25
FIDFLOOR = 0.9
NFOLD_MAX = 50   # cap on leave-one-identity-out folds (one fold per identity component); set well above the SI-184 identity-component count so it never binds. More identities is the point, cost accepted
PAIRED_NULLS = 40   # timeshift nulls per 8x8 PAIRED cell (< N_TIMESHIFT): each cell re-captures A & B on GT&A&B frames, so the floor is paid ~56x; the pairwise DELTA cancels most floor noise, so a reduced set is the "fast" 8x8


def make_lag_grid(lmin, lmax):
    base = list(range(1, 13)) + [14, 16, 18, 20, 24, 28, 32, 40, 48, 60, 72, 88, 100]
    return np.array(sorted(lag for lag in base if lmin <= lag <= lmax))


LAG_NP = make_lag_grid(LSPK_MIN, LMAX)
LSPK = int(LAG_NP.max())
MINSHIFT = LSPK   # min timeshift >= the lag window: below LSPK a null's lag-window {t-L-off} overlaps the aligned
                  # {t-L}, leaking true alignment into the floor. Ties to LSPK, not a literal.


MODELS = [
    ("AVTR-1", "AVTR-1"),
    ("SoulX Lite", "SoulX_FlashHead_Lite"),
    ("SoulX Pro", "SoulX_FlashHead_Pro"),
    ("Ditto", "ditto"),
    ("DyStream", "dystream"),
    ("FLOAT", "FLOAT"),
    ("AvatarForcing", "AvatarForcing"),
]
DYADIC = {"AVTR-1", "DyStream", "AvatarForcing"}
SURROGATE = "GT\u00d7other"


MODELS_ALL_DIRS = [("GT", "GT")] + MODELS
SOURCES = [("combo", "emoca")]
REGION = "overall"


MODELS_ALL = [("GT", "GT")] + MODELS + [(SURROGATE, None)]


GRID_FPS = 25.0


@dataclass
class Corpus:
    """Everything read from the corpus once, before any encoder is scored; immutable for the rest of the run.

    Passed explicitly so that nothing here depends on where the source loop has got to, and the loading sits in one
    place."""

    pairs: dict
    listeners: list
    speakers: list
    FOLD: dict
    NFOLD: int
    LAG: object
    timeshifts: list
    SPK_CACHE: dict
    SPK_G_AUD: dict
    k_audio: int
    SURRPERM: dict


@dataclass
class SourceContext:
    """Everything that depends on WHICH encoder is being scored, built once per source and passed explicitly.

    None of these are constant across sources, so they are passed explicitly rather than held as module-level state
    that a function's behaviour could depend on.
    """

    driver: dict
    driver_all: object
    speaker_row0: dict
    channel_cols: dict
    eval_mask: dict
    corpus: Corpus
    target_scale_inv: object = None   # (n_target,) = 1/sqrt(var + median-var floor), GT-derived; balances the target
                                      # dims in the regression (EMOCA vs LP). None only when there is no GT design.


def model_kind(label):
    if label == "GT":
        return "GT"
    if label == SURROGATE:
        return "surrogate"
    return "dyadic" if label in DYADIC else "non-dyad"


def _sess_key(stem):
    return "_".join(stem.split("_")[:2])


def _ident_key(stem):
    return stem.split("_")[3]


def _intr_key(stem):
    return "_".join(stem.split("_")[:3])
