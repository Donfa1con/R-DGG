"""Silero-VAD speech-timestamp cache.

For each `<stem>.wav` we run silero-vad and persist:

  speech_starts  (N,)     float32  start-of-segment, seconds
  speech_ends    (N,)     float32  end-of-segment, seconds
  audio_seconds  scalar   float32  total audio length
  mask_25fps     (T,)     uint8    pre-rendered per-frame mask at 25 fps
                                   (T = floor(audio_seconds * 25)); 1 = speech
  fps            scalar   int32    25 (the framerate of the pre-rendered mask)

Why an npz: matches the storage convention of every other extractor cache in
this repo (`<split>_insightface/`, `_emoca/`, ...) so downstream metrics can
read it the same way. The pre-rendered 25 fps mask is the common form
metrics need — anything else can be reconstructed from the float
timestamps.

Usage:
  python extract.py --wavs_dir <corpus>/<src>/<split>_wav \\
                    --out_dir  <corpus>/<src>/<split>_vad \\
                    [--threshold 0.5] [--min_speech_dur_s 0.25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _load_vad():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def _speech_timestamps(wav_path: Path, model, threshold: float,
                       min_speech_dur_s: float) -> tuple[list[dict], float]:
    from silero_vad import read_audio, get_speech_timestamps
    audio = read_audio(str(wav_path), sampling_rate=16000)
    ts = get_speech_timestamps(
        audio, model,
        sampling_rate=16000,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_dur_s * 1000),
    )
    return ts, float(len(audio) / 16000)


def render_mask(speech_ts: list[dict], audio_seconds: float, fps: int = 25) -> np.ndarray:
    """Per-frame speech mask at `fps`. Length = floor(audio_seconds * fps).
    Frame i is 1 iff its center timestamp falls inside a speech segment.
    """
    n = int(audio_seconds * fps)
    mask = np.zeros(n, dtype=np.uint8)
    for s in speech_ts:
        f0 = int(np.floor(s["start"] * fps))
        f1 = int(np.ceil(s["end"]   * fps))
        mask[max(0, f0):min(n, f1)] = 1
    return mask


def process(wav_path: Path, out_path: Path, model,
            threshold: float, min_speech_dur_s: float,
            overwrite: bool) -> str:
    if out_path.exists() and not overwrite:
        return "skip"
    ts, audio_seconds = _speech_timestamps(
        wav_path, model, threshold, min_speech_dur_s)
    starts = np.array([s["start"] for s in ts], dtype=np.float32)
    ends   = np.array([s["end"]   for s in ts], dtype=np.float32)
    mask   = render_mask(ts, audio_seconds, fps=25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        speech_starts=starts,
        speech_ends=ends,
        audio_seconds=np.float32(audio_seconds),
        mask_25fps=mask,
        fps=np.int32(25),
    )
    return "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wavs_dir", required=True, type=Path,
                   help="Directory of <stem>.wav files (16 kHz mono preferred).")
    p.add_argument("--out_dir",  required=True, type=Path,
                   help="Where to write <stem>.npz with VAD timestamps + mask.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="silero-vad confidence threshold (default 0.5).")
    p.add_argument("--min_speech_dur_s", type=float, default=0.25,
                   help="Drop speech segments shorter than this (s).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.wavs_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        print(f"no *.wav under {args.wavs_dir}")
        return

    print("[load] silero-vad", file=sys.stderr)
    model = _load_vad()

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for w in tqdm(wavs, ncols=80, desc=args.wavs_dir.name):
        out = args.out_dir / (w.stem + ".npz")
        try:
            r = process(w, out, model,
                        args.threshold, args.min_speech_dur_s, args.overwrite)
        except Exception as e:
            print(f"[fail] {w.name}: {type(e).__name__}: {e}", file=sys.stderr)
            r = "fail"
        stats[r] = stats.get(r, 0) + 1
    print(f"done — {stats}")


if __name__ == "__main__":
    main()
