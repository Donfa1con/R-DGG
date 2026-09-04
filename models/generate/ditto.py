"""Batch-generate Ditto on SI-184: NON-DYADIC lip-sync. Drive each SI-184/LS_AL reference portrait
with THAT SAME person's own audio (no partner / listener audio anywhere -- this is a plain talking-head
lip-sync pass). Output mirrors the scored-model layout: SI-184/ditto/LS_AL/<stem>.mp4 (native 1080x1080,
25 fps, muxed with the driving wav).

Faithfulness to the model's real inference path (the pinned ditto submodule at models/ditto):
  * StreamSDK(cfg_pkl, data_root) is built EXACTLY ONCE (it deserializes the TRT engines under
    data_root -- reloading per stem is prohibitively slow). It is reused for all stems: per stem
    inference.run(SDK, audio, source, out) internally calls SDK.setup()+SDK.close(), and setup() rebuilds
    the queues/worker-threads/writer and resets worker_exception=None while close() joins+tears them down,
    so one SDK is safe to reuse (verified in stream_pipeline_offline.py:setup/close).
  * 25 fps is native: core/atomic_components/writer.py VideoWriterByImageIO defaults fps=25 and StreamSDK
    constructs it as VideoWriterByImageIO(tmp_output_path) with no override. We never re-encode.
  * inference.run() writes EXACTLY its output_path via an ffmpeg mux of the video-only intermediate
    (SDK.tmp_output_path == output_path + ".tmp.mp4") against the driving wav, and leaves that
    "<output_path>.tmp.mp4" behind for us to delete. The mux is os.system(...) so a failed mux does NOT
    raise -- we therefore pass run() a TEMP path, then verify frames+audio and os.replace onto the final.

Atomicity + verification: run() muxes to PARTIAL/<stem>.mp4; we require >= MIN_FRAMES
decodable frames (cv2) AND an audio stream (ffprobe, ffmpeg-stderr fallback) before os.replace to the
final <stem>.mp4. A lone PARTIAL/<stem>.mp4.tmp.mp4 with no muxed PARTIAL/<stem>.mp4 == the mux step
failed -> counted as a failure. Every leftover tmp for the stem is swept after each stem. Idempotent
(skip if the final mp4 already decodes >= MIN_FRAMES); aborts after > MAX_CONSECUTIVE_FAIL consecutive
failures (a systemic error -- wrong PATH, missing engine, OOM on stem 1 -- hits every stem).

Launch UNDER the model's activated env so run()'s subprocess ffmpeg is on PATH. From the repo root
(inputs/outputs default to data/; exact commands + CFG are in models/README.md):
    conda activate ditto && python models/generate/ditto.py
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
import traceback

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned git submodule at models/ditto (override with DITTO_REPO for a checkout elsewhere).
DITTO_REPO = os.environ.get("DITTO_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "ditto")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
sys.path.insert(0, DITTO_REPO)
from inference import run  # noqa: E402  (models/ditto/inference.py)
from stream_pipeline_offline import StreamSDK  # noqa: E402  (models/ditto/stream_pipeline_offline.py)

# ---- weights (override via env only) ----
DATA_ROOT = os.environ.get("DITTO_DATA_ROOT", f"{DITTO_REPO}/checkpoints/ditto_trt_Ampere_Plus")
CFG_PKL = os.environ.get("DITTO_CFG_PKL", f"{DITTO_REPO}/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")

MIN_FRAMES = 10                       # a valid clip is hundreds+ of frames; below this is degenerate/truncated
MAX_CONSECUTIVE_FAIL = 3              # abort on the 4th consecutive failure -- systemic error hits every stem

# I/O paths are argparse-populated in main() (defaults point at the repo's data/).
REFDIR = WAVDIR = OUTDIR = PARTIAL = ""
LIMIT: int | None = None


def resolve_stems(stems_file: str, refdir: str) -> list[str]:
    """Which stems to generate: from `stems_file` if it exists (data/pairs184.txt), else every <stem>.png
    in `refdir`. Each list line is "<listener>.wav <speaker>.wav"; the listener stem is the first whitespace
    token with a trailing .wav stripped. Blank and #-comment lines are skipped and the stems keep their
    first-seen order after deduplication. Either way a stem is a basename with no extension."""
    if stems_file and os.path.exists(stems_file):
        stems, seen = [], set()
        for ln in open(stems_file):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            stem = ln.split()[0]
            if stem.endswith(".wav"):
                stem = stem[:-4]
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
        return stems
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(refdir, "*.png")))


def video_nframes(path: str) -> int:
    """Frame count of a written mp4, or 0 if it can't be opened / has no decodable first frame.
    Guards against a nonzero-but-degenerate file (video-only mux fallback, 0/1-frame, truncated)
    being promoted to <stem>.mp4 and then skipped forever on resume."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return n if ok else 0


def has_audio_stream(path: str) -> bool:
    """True iff the file carries at least one audio stream. run() muxes audio via os.system(ffmpeg ...),
    which does NOT raise on failure, so a video-only (audio-less) file can silently land -- reject it."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120,
        )
        return "audio" in r.stdout
    except FileNotFoundError:
        # ffprobe not shipped with this env's ffmpeg -> parse `ffmpeg -i` stderr (nonzero rc is expected).
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=120)
        return "Audio:" in r.stderr


def sweep_tmp(stem: str) -> None:
    """Remove this stem's tmp artifacts from PARTIAL: the video-only intermediate
    <stem>.mp4.tmp.mp4 that Ditto always leaves, and any partial mux <stem>.mp4. The `.mp4*`
    anchor keeps the glob to THIS stem's files (finals live in OUTDIR, never in PARTIAL)."""
    for p in glob.glob(os.path.join(PARTIAL, glob.escape(stem) + ".mp4*")):
        try:
            os.remove(p)
        except OSError:
            pass


def main() -> None:
    global REFDIR, WAVDIR, OUTDIR, PARTIAL, LIMIT
    ap = argparse.ArgumentParser(description="Batch Ditto (non-dyadic lip-sync) -- SI-184 reproduction")
    ap.add_argument("--references", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png source portraits")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav driving audio (each person's OWN audio; non-dyadic)")
    ap.add_argument("--out_dir",
                    default=os.path.join(_DATA, "ditto", "LS_AL"),
                    help="output videos dir (<stem>.mp4)")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems); falls back to globbing --references if absent")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("DITTO_LIMIT", "0")),
                    help="smoke: first N stems only (0 = all)")
    args = ap.parse_args()
    REFDIR, WAVDIR, OUTDIR = args.references, args.wav_dir, args.out_dir
    PARTIAL = f"{OUTDIR}/.partial"
    LIMIT = args.limit or None

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PARTIAL, exist_ok=True)

    stems = resolve_stems(args.stems_file, REFDIR)
    if LIMIT:
        stems = stems[:LIMIT]

    print(f"[ditto] {len(stems)} stems  REF={REFDIR}  WAV={WAVDIR}\n"
          f"        OUT={OUTDIR}\n"
          f"        data_root={DATA_ROOT}\n        cfg_pkl={CFG_PKL}", flush=True)
    # expected count = the number of unique listener stems in the stems file, derived at runtime.
    n_expected = len(resolve_stems(args.stems_file, REFDIR)) \
        if args.stems_file and os.path.exists(args.stems_file) else None
    if n_expected is not None and len(stems) != n_expected:
        print(f"[ditto] WARNING: expected {n_expected} stems (from {os.path.basename(args.stems_file)}), "
              f"found {len(stems)}", flush=True)

    # Build the SDK exactly once (deserializes the TRT engines under data_root) and reuse it.
    print("[ditto] loading StreamSDK (TRT engines) ...", flush=True)
    t_load = time.perf_counter()
    SDK = StreamSDK(CFG_PKL, DATA_ROOT)
    print(f"[ditto] SDK ready in {time.perf_counter() - t_load:.1f}s", flush=True)

    done = skipped = 0
    failed: list[tuple[str, str]] = []
    consecutive = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems):
        out_path = f"{OUTDIR}/{stem}.mp4"
        if os.path.exists(out_path) and video_nframes(out_path) >= MIN_FRAMES:
            skipped += 1
            continue

        ref_png = f"{REFDIR}/{stem}.png"
        own_wav = f"{WAVDIR}/{stem}.wav"     # NON-DYADIC: the reference person's OWN audio
        tmp_out = f"{PARTIAL}/{stem}.mp4"    # run() muxes here; leaves tmp_out + ".tmp.mp4" behind
        sweep_tmp(stem)                      # clear any stale tmp from a previously crashed attempt
        t0 = time.perf_counter()
        try:
            for path, nm in ((ref_png, "ref_png"), (own_wav, "own_wav")):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nm} missing: {path}")

            # run(SDK, audio_path, source_path, output_path): SDK.setup()+setup_Nd()+audio2motion+close()
            # then ffmpeg-mux the driving wav -> writes EXACTLY tmp_out (native 25fps, no re-encode).
            run(SDK, own_wav, ref_png, tmp_out)

            # A missing muxed file == the os.system(ffmpeg ...) mux failed (only the .tmp.mp4 exists).
            if not os.path.exists(tmp_out):
                raise RuntimeError("mux produced no output (ffmpeg mux failed; only video-only .tmp.mp4 left)")
            nf = video_nframes(tmp_out)
            if nf < MIN_FRAMES:
                raise RuntimeError(f"degenerate output: {nf} decodable frames (< {MIN_FRAMES})")
            if not has_audio_stream(tmp_out):
                raise RuntimeError("no audio stream in muxed output (mux dropped audio)")

            os.replace(tmp_out, out_path)    # atomic; nothing half-written/audio-less lands as <stem>.mp4
            done += 1
            consecutive = 0
            dt = time.perf_counter() - t0
            eta = (time.perf_counter() - t_start) / max(done, 1) * (len(stems) - skipped - done)
            print(f"[{idx + 1}/{len(stems)}] {stem}: {nf}f ({nf / 25:.1f}s) in {dt:.1f}s "
                  f"-> {out_path}  ETA {eta / 60:.0f}m", flush=True)
        except Exception as exc:  # noqa: BLE001 - collect per-stem, abort only if systemic
            failed.append((stem, repr(exc)))
            consecutive += 1
            print(f"[{idx + 1}/{len(stems)}] FAIL {stem}: {exc}", flush=True)
            traceback.print_exc()
            if consecutive > MAX_CONSECUTIVE_FAIL:
                print(f"\nABORT: {consecutive} consecutive failures -> systemic error, stopping.", flush=True)
                sweep_tmp(stem)
                break
        finally:
            sweep_tmp(stem)                  # always drop this stem's <stem>.mp4.tmp.mp4 (+ any partial mux)

    print(f"\nDONE: {done} generated, {skipped} pre-existing, {len(failed)} failed "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    for stem, err in failed:
        print("  FAIL", stem, err, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
