"""Shared VAD mask helpers for talking-head metrics.

Two ways to use VAD npz caches downstream:

  • per_frame_keep_mask  — bool[T] from a single VAD npz at any FPS.
    Used by metrics that filter frames independently on each side
    (rPCC, TLCC, SID, variance, PFD plain mask mode).

  • reactive_segments_mask — bool[T] from a (listener, speaker) VAD pair.
    Keeps only frames inside *maximal listener-silence intervals* that
    contain at least one speaker-speech frame and are at least
    `min_seg_sec` long. Listener-silence is inward-padded by `pad_sec`
    on each edge to discard VAD edge noise. Short speaker silences
    inside the listener-silent stretch are KEPT — only listener.speech
    breaks a segment. This is the "is this a real reactive listening
    moment?" mask used by R-DGG and the reactive variant of PFD.

VAD npz layout (produced by `extractors/vad/extract.py`):
  speech_starts (N,) float32   start of speech segment, seconds
  speech_ends   (N,) float32   end of speech segment, seconds
  audio_seconds scalar float   total audio length
  mask_25fps    (T0,) uint8    pre-rendered 25-fps mask (unused here —
                               we resample to caller-supplied T from
                               the float timestamps for arbitrary FPS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Default segment parameters (kept here so all callers share the same
# defaults unless they explicitly override).
DEFAULT_MIN_SEG_SEC = 3.0
DEFAULT_PAD_LISTENER_SILENCE_SEC = 0.2


def _speech_mask_at(vad_path: Path, T: int,
                    pad_sec: float = 0.0) -> "np.ndarray | None":
    """bool[T] speech mask at feature FPS = T / audio_seconds.

    `pad_sec` dilates each speech segment by this many seconds on each
    edge — used by reactive_segments_mask to *shrink* listener-silence
    inward. Returns None if VAD is missing or audio_seconds is 0.
    """
    if not vad_path.is_file():
        return None
    z = np.load(vad_path)
    audio_s = float(z["audio_seconds"])
    if audio_s <= 0:
        return None
    fps = T / audio_s
    pad = int(round(pad_sec * fps))
    m = np.zeros(T, dtype=bool)
    for s, e in zip(z["speech_starts"].astype(np.float64),
                    z["speech_ends"].astype(np.float64)):
        f0 = int(np.floor(s * fps)) - pad
        f1 = int(np.ceil(e * fps)) + pad
        m[max(0, f0):min(T, f1)] = True
    return m


def per_frame_keep_mask(vad_path: Path, mode: str, T: int
                        ) -> "np.ndarray | None":
    """bool[T] per-frame keep mask from a single VAD npz.

    mode ∈ {"speech", "silence", "both", "none"}:
      speech  → keep frames where this stem is speaking
      silence → keep frames where this stem is silent
      both / none → return None (no mask, caller treats as identity)

    Returns None if the npz is missing or mode is identity. Callers
    that want a strict "no VAD → drop pair" semantics should check
    explicitly for None.
    """
    if mode in ("both", "none"):
        return None
    speech = _speech_mask_at(vad_path, T, pad_sec=0.0)
    if speech is None:
        return None
    if mode == "silence":
        return ~speech
    if mode == "speech":
        return speech
    raise ValueError(f"unknown vad mode: {mode!r}")


def reactive_segments_mask(
    ls_vad_path: Path,
    sp_vad_path: Path,
    T: int,
    min_seg_sec: float = DEFAULT_MIN_SEG_SEC,
    pad_sec: float = DEFAULT_PAD_LISTENER_SILENCE_SEC,
) -> "np.ndarray | None":
    """bool[T] mask of reactive-listening frames.

    A frame t is kept iff it lies inside a maximal listener-silence
    interval that
      (a) contains at least one speaker-speech frame, and
      (b) is at least `min_seg_sec` long (≈ a "real" listening window),
    after the listener-silence is shrunk by `pad_sec` on each edge to
    discard VAD edge noise.

    Short speaker silences *inside* the listener-silent stretch are
    KEPT — only listener.speech breaks a segment. This matches the
    intuition that a listener is "listening" through the speaker's
    inhale-pauses.

    Returns None if either VAD npz is missing.
    """
    sp_speech = _speech_mask_at(sp_vad_path, T, pad_sec=0.0)
    ls_speech = _speech_mask_at(ls_vad_path, T, pad_sec=pad_sec)
    if sp_speech is None or ls_speech is None:
        return None
    sil = ~ls_speech

    audio_s = float(np.load(ls_vad_path)["audio_seconds"])
    fps = T / audio_s
    min_len = int(round(min_seg_sec * fps))

    out = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
        if sil[i]:
            j = i
            while j < T and sil[j]:
                j += 1
            if (j - i) >= min_len and sp_speech[i:j].any():
                out[i:j] = True
            i = j
        else:
            i += 1
    return out
