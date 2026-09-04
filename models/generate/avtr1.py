"""Batch-generate the AVTR-1 model: drive each listener reference with its own
audio (self / lip-sync) + its speaker's audio (other / listening), at CFG 3/3/3, one video at
a time. Output mirrors the scored-model layout: <corpus>/AVTR-1/LS_AL/<listener_stem>.mp4.

Run under the avtr-1 submodule's `renderer` pixi env so the host-built TensorRT engines deserialize
with a matching runtime (the engines are NOT portable and are NOT shipped -- build them on THIS host
first; see models/README.md). From the repo root:
    cd models/avtr-1 && pixi run -e renderer python ../generate/avtr1.py

Semantics (avatar = the listener whose reference we animate):
  audio_speech = listener's own wav   -> the avatar's own lip / talking motion
  audio_listen = speaker's wav        -> the avatar's listening / reactive motion
Pairing is data/pairs184.txt: "<listener>.wav <speaker>.wav". References are 3-channel RGB, so
AvatarLoader sets no_matting=True and putback keeps the reference's own background unchanged
(bg_id is required by the API but its tensor is unused on that path). Everything not named here
is a repo default (align-to-longer track, noise_alpha=2.0, noise_trunc_z=1.2, streaming frames).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned, in-repo git submodule at models/avtr-1 (override with AVTR1_REPO). Its
# Python package lives under src/ (setuptools packages.find where=["src"]); add it so imports resolve
# without an editable install. The reviewer builds the renderer pixi env + TensorRT engines INTO this
# submodule (models/README: pixi install -e renderer -> pixi run download -> pixi run
# build-trt-engines), which writes the engines under <REPO>/artifacts/{HF_REVISION}/.
REPO = os.environ.get("AVTR1_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "avtr-1")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
sys.path.insert(0, os.path.join(REPO, "src"))
# Anchor the artifact/engine storage root at REPO/artifacts (get_storage_root yields
# {AVTR1_LOCAL_STORAGE}/{HF_REVISION}) — the same location build-trt-engines writes to — so engine
# resolution is pinned to REPO regardless of how avtr1_renderer's __file__ resolves.
os.environ.setdefault("AVTR1_LOCAL_STORAGE", os.path.join(REPO, "artifacts"))

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

from avtr1_renderer.avatar_loader import AvatarLoader  # noqa: E402
from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager  # noqa: E402
from avtr1_renderer.pipeline import Pipeline  # noqa: E402
from avtr1_renderer.types import Chunk, RenderOptions  # noqa: E402

OUT_H = OUT_W = 1080                # = reference size; keeps the reference's own frame 1:1
SAMPLE_RATE = 16_000
FPS = 25
CFG = dict(cfg_self_audio=3.0, cfg_other_audio=3.0, cfg_kp=3.0)   # CFG 3/3/3
MAX_CONSECUTIVE_FAIL = 3            # a systemic error hits every stem; abort rather than burn the corpus

# I/O paths are argparse-populated in main() (defaults point at the repo's data/).
REFDIR = WAVDIR = PAIRS = OUTDIR = PARTIAL = ""
LIMIT: int | None = None


def load_mono_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return audio.astype(np.float32)


def fit(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) == n:
        return audio
    if len(audio) > n:
        return audio[:n]
    return np.concatenate([audio, np.zeros(n - len(audio), dtype=audio.dtype)])


def slice_chunks(audio: np.ndarray, window: int, step: int) -> list[np.ndarray]:
    n_steps = max(1, (len(audio) + step - 1) // step)
    out = []
    for i in range(n_steps):
        piece = audio[i * step : i * step + window]
        if len(piece) < window:
            piece = np.pad(piece, (0, window - len(piece)))
        out.append(piece)
    return out


def read_pairs() -> dict[str, str]:
    lis2spk = {}
    for line in Path(PAIRS).read_text().splitlines():
        a, b = line.split()
        lis2spk[a[:-4] if a.endswith(".wav") else a] = b[:-4] if b.endswith(".wav") else b
    return lis2spk


def main() -> None:
    global REFDIR, WAVDIR, PAIRS, OUTDIR, PARTIAL, LIMIT
    ap = argparse.ArgumentParser(
        description="Batch AVTR-1 (CFG 3/3/3) dyadic listener generation")
    ap.add_argument("--references", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png avatar (listener) portraits (3-channel RGB)")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav (listener own audio + speaker audio, keyed by --pairs)")
    ap.add_argument("--pairs", default=os.path.join(_DATA, "pairs184.txt"),
                    help="'<listener>.wav <speaker>.wav' per line (dyadic pairing)")
    ap.add_argument("--out_dir",
                    default=os.path.join(_DATA, "AVTR-1", "LS_AL"),
                    help="output videos dir (<listener_stem>.mp4)")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems); falls back to all --pairs keys if absent")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("AVTR1_LIMIT", "0")),
                    help="smoke: first N stems only (0 = all)")
    args = ap.parse_args()
    REFDIR, WAVDIR, PAIRS, OUTDIR = args.references, args.wav_dir, args.pairs, args.out_dir
    PARTIAL = f"{OUTDIR}/.partial"
    LIMIT = args.limit or None

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PARTIAL, exist_ok=True)
    lis2spk = read_pairs()
    if args.stems_file and os.path.exists(args.stems_file):
        # listener stem = first whitespace token of each non-blank non-#-comment line, trailing .wav
        # stripped; dedup first-seen.
        wanted, _seen = [], set()
        for s in open(args.stems_file):
            s = s.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0]
            if tok.endswith(".wav"):
                tok = tok[:-4]
            if tok not in _seen:
                _seen.add(tok)
                wanted.append(tok)
        stems = [s for s in wanted if s in lis2spk]
    else:
        stems = sorted(lis2spk)
    if LIMIT:
        stems = stems[:LIMIT]

    print(f"Loading AVTR-1 pipeline (out_size={OUT_H}x{OUT_W}, CFG={CFG}) ...", flush=True)
    pipeline, _ = Pipeline.from_artifacts(
        avatar_ids=[], portraits_dir=REFDIR, background_paths={}, out_size=(OUT_H, OUT_W)
    )
    mg = pipeline._motion_generator
    window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    step = mg.chunk_size * mg.frame_len

    mgr = get_artifact_manager()
    mask_path = mgr.get_artifact_path("pasteback_mask") if "pasteback_mask" in mgr._artifacts else None
    loader = AvatarLoader(
        engine_files={
            k: mgr.get_artifact_path(k)
            for k in ("insightface_det", "landmark106", "landmark203",
                      "appearance_extractor", "motion_extractor")
        },
        mask_template_path=mask_path,
        out_h=OUT_H, out_w=OUT_W, max_dim=max(OUT_H, OUT_W),
    )
    opts = RenderOptions(pixel_format="yuv_i420", bg_id="transparent", **CFG)

    done = skipped = 0
    failed: list[tuple[str, str]] = []
    consecutive_fail = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems):
        out_path = f"{OUTDIR}/{stem}.mp4"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue
        tmp = f"{PARTIAL}/{stem}.mp4"
        writer = None
        t0 = time.perf_counter()
        try:
            spk = lis2spk[stem]
            self_wav = f"{WAVDIR}/{stem}.wav"
            other_wav = f"{WAVDIR}/{spk}.wav"
            speech = load_mono_16k(self_wav)
            listen = load_mono_16k(other_wav)
            n = max(len(speech), len(listen))
            speech, listen = fit(speech, n), fit(listen, n)
            sp_chunks = slice_chunks(speech, window, step)
            ls_chunks = slice_chunks(listen, window, step)

            avatar = loader.load(f"{REFDIR}/{stem}.png", avatar_id=stem)
            out_h, out_w = avatar.source.shape[-2:]

            writer = imageio_ffmpeg.write_frames(
                tmp, size=(out_w, out_h), fps=FPS, codec="libx264",
                pix_fmt_in="yuv420p", pix_fmt_out="yuv420p",
                quality=8, macro_block_size=1,
                audio_path=self_wav, audio_codec="aac",
            )
            writer.send(None)

            state = None
            produced = 0
            for sp, ls in zip(sp_chunks, ls_chunks):
                state, frames = pipeline.process_chunk(
                    avatar, Chunk(audio_speech=sp, audio_listen=ls), state, opts
                )
                for frame in frames:
                    writer.send(frame.data.tobytes())
                    produced += 1
            writer.close()
            writer = None
            del avatar

            if produced == 0:
                raise RuntimeError("no frames produced")
            os.replace(tmp, out_path)     # atomic; nothing half-written ever lands as OUTDIR/<stem>.mp4
            done += 1
            consecutive_fail = 0
            dt = time.perf_counter() - t0
            eta = (time.perf_counter() - t_start) / max(done, 1) * (len(stems) - skipped - done)
            print(f"[{idx + 1}/{len(stems)}] {stem}: {produced}f "
                  f"({produced / FPS:.1f}s) in {dt:.1f}s -> {out_path}  ETA {eta / 60:.0f}m",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - collect per-stem, abort only if systemic
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if os.path.exists(tmp):
                os.remove(tmp)
            failed.append((stem, repr(exc)))
            consecutive_fail += 1
            print(f"[{idx + 1}/{len(stems)}] FAIL {stem}: {exc}", flush=True)
            traceback.print_exc()
            if consecutive_fail > MAX_CONSECUTIVE_FAIL:
                print(f"\nABORT: {consecutive_fail} consecutive failures -> systemic error, stopping.",
                      flush=True)
                break

    print(f"\nDONE: {done} generated, {skipped} pre-existing, {len(failed)} failed "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    for stem, err in failed:
        print("  FAIL", stem, err, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
