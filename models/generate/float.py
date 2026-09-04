"""Batch-generate FLOAT on SI-184, NON-DYADIC: drive each listener reference with THAT SAME
person's own audio (self / lip-sync). There is no partner/other audio anywhere in this run --
FLOAT is a single-audio talking-head model (one reference portrait + one wav -> one video).
Output mirrors the scored-model layout: SI-184/FLOAT/LS_AL/<stem>.mp4 (512x512, 25fps, muxed
with the stem's own audio).

Faithfulness: uses generate.py's own InferenceAgent.run_inference in-process, with the
call:
    agent.run_inference(res_video_path=<tmp>, ref_path=<png>, audio_path=<own wav>,
                        a_cfg_scale=2, r_cfg_scale=1, e_cfg_scale=1, emo=None,
                        nfe=10, no_crop=False)
Everything else is a generate.py/base_options default: input_size=512, fps=25.0 (opt.fps drives
both the frame tensor and the ffmpeg mux -> native 25fps, no re-encode), sampling_rate=16000,
euler ODE with opt.nfe=10 function evals, wav2vec2 audio encoder + audio2emotion. emo=None takes
FLOAT's audio-predicted-emotion path (sample(): label2id.get(str(None).lower()) -> None ->
predict_emotion), matching generate.py's own __main__ default (opt.emo defaults to None).

Load-once: InferenceAgent(opt) is constructed ONCE before the loop. Its __init__ loads the
826 MB float.pth checkpoint into the FLOAT graph AND builds the DataProcessor, which constructs
face_alignment.FaceAlignment(TWO_D) -- an s3fd face detector + a 2DFAN4 landmark net whose
weights download to ~/.cache on first construction. Both the checkpoint and the face detector are
built exactly once and reused for all stems (reconstructing per stem is prohibitively slow).

Reproducibility: this driver injects no seed. FLOAT draws its noise from the global torch RNG on
its default path (sample(): `x0 = torch.randn(..., device=opt.rank)` when opt.fix_noise_seed is
False -- left untouched at its default), so outputs follow the model's own default RNG behaviour.

Idempotent (skip if <stem>.mp4 already decodes >= MIN_FRAMES). Atomic + verified: FLOAT muxes into
a driver-owned tmp under OUTDIR/.partial; we verify >= MIN_FRAMES decodable frames AND an audio
stream, THEN os.replace onto <stem>.mp4 (same filesystem -> atomic). A missing/degenerate/audioless
tmp is a failure, never promoted. Aborts after >MAX_CONSECUTIVE_FAIL consecutive failures (a
systemic error -- missing weights, broken env, OOM on stem 1 -- hits every stem).

From the repo root (inputs/outputs default to data/; exact commands + CFG are in models/README.md):
    conda activate FLOAT && python models/generate/float.py

The driver must run under the ACTIVATED FLOAT env so the subprocess `ffmpeg`/`ffprobe`
that generate.py's save_video and our audio check shell out to are on PATH (no nohup/abs-python).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned git submodule at models/float (override with FLOAT_REPO for a checkout elsewhere).
FLOAT_REPO = os.environ.get("FLOAT_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "float")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
# generate.py does `from models.float.FLOAT import FLOAT` / `from options.base_options import ...`,
# and base_options' default checkpoint paths are relative ("./checkpoints/..."), so the repo must
# be both importable and the cwd. We pin every model path to an absolute below regardless.
sys.path.insert(0, FLOAT_REPO)
os.chdir(FLOAT_REPO)

import cv2  # noqa: E402  (env cv2; used only for frame-count verification)

from generate import InferenceAgent, InferenceOptions  # noqa: E402

CKPT_PATH = os.environ.get("FLOAT_CKPT", f"{FLOAT_REPO}/checkpoints/float.pth")
WAV2VEC_DIR = f"{FLOAT_REPO}/checkpoints/wav2vec2-base-960h"
A2E_DIR = f"{FLOAT_REPO}/checkpoints/wav2vec-english-speech-emotion-recognition"

# I/O paths are argparse-populated in main() (defaults point at the repo's data/).
REFDIR = WAVDIR = OUTDIR = PARTIAL = ""
LIMIT: int | None = None

# ---- run_inference args (generate.py signature) ----
A_CFG_SCALE = 2.0
R_CFG_SCALE = 1.0
E_CFG_SCALE = 1.0
EMO = None
NFE = 10
NO_CROP = False           # MUST be False -- FLOAT expects a face-cropped reference (process_img)

MIN_FRAMES = 10           # a valid clip is hundreds+ frames; anything below this is a degenerate output
MAX_CONSECUTIVE_FAIL = 3  # abort once failures EXCEED this (on the 4th) -- a systemic error hits every stem


def build_opt():
    """Construct generate.py's opt WITHOUT touching sys.argv (parse([]) == all base_options
    defaults), then pin device + every model path to an absolute so cwd/argv can't perturb them."""
    io = InferenceOptions()
    parser = io.initialize(argparse.ArgumentParser())
    opt = parser.parse_args([])            # defaults only; no CLI coupling
    opt.rank, opt.ngpus = 0, 1             # generate.py's __main__ sets these; run on cuda:0
    opt.ckpt_path = CKPT_PATH
    opt.wav2vec_model_path = WAV2VEC_DIR
    opt.audio2emotion_path = A2E_DIR
    opt.pretrained_dir = f"{FLOAT_REPO}/checkpoints"
    # defaults we depend on and assert-by-setting for clarity (all already the base_options default):
    opt.input_size = 512                   # -> 512x512 output frames
    opt.fps = 25.0                         # -> torchvision write_video fps AND ffmpeg copy -> native 25fps
    opt.sampling_rate = 16000
    opt.nfe = NFE                          # sample() uses opt.nfe for the euler time grid
    return opt


def video_nframes(path: str) -> int:
    """Frame count of a written mp4, or 0 if it can't be opened / has no decodable first frame.
    Guards a nonzero-but-degenerate file (video-only mux fallback, 0/1-frame, truncated) from being
    promoted to <stem>.mp4 and then skipped forever on resume."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return n if ok else 0


def has_audio_stream(path: str) -> bool:
    """True iff ffprobe finds at least one audio stream. FLOAT's save_video muxes `-c:a aac`; a
    file with no audio stream means the ffmpeg mux silently produced a video-only file -- reject it
    so an audioless mp4 never lands as the final artefact (downstream checks read the muxed audio track)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
        return "audio" in r.stdout
    except Exception:
        return False


def _tmp_mp4_snapshot() -> set:
    """FLOAT.save_video writes an intermediate video-only file via tempfile.NamedTemporaryFile(
    suffix='.mp4', delete=False) in the system temp dir and only unlinks it when the ffmpeg mux
    SUCCEEDS. On a mux failure it is left behind. Snapshot the temp dir's tmp*.mp4 before each call
    so we can delete exactly what this stem leaked (a lone tmp with no final mp4 == mux failed)."""
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.mp4")))


def cleanup_tmp(before: set) -> None:
    # new NamedTemporaryFile leftovers in the system temp dir
    for f in _tmp_mp4_snapshot() - before:
        try:
            os.remove(f)
        except OSError:
            pass
    # any _tmp / .tmp.mp4 stragglers in our own dirs (defensive; FLOAT shouldn't create these here)
    for pat in ("*_tmp*.mp4", "*.mp4.tmp.mp4"):
        for f in glob.glob(os.path.join(OUTDIR, pat)) + glob.glob(os.path.join(PARTIAL, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def list_stems(stems_file: str = "") -> list:
    """From `stems_file` if it exists (data/pairs184.txt), else every <stem>.png in REFDIR. Each list line is
    "<listener>.wav <speaker>.wav"; the listener stem is the first whitespace token with a trailing .wav
    stripped. Blank and #-comment lines are skipped and the stems keep first-seen order after deduplication.
    Then apply LIMIT."""
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
    else:
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(REFDIR, "*.png")))
    if LIMIT:
        stems = stems[:LIMIT]
    return stems


def main() -> None:
    global REFDIR, WAVDIR, OUTDIR, PARTIAL, LIMIT
    ap = argparse.ArgumentParser(description="Batch FLOAT (non-dyadic lip-sync) -- SI-184 reproduction")
    ap.add_argument("--references", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png source portraits")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav driving audio (each person's OWN audio; non-dyadic)")
    ap.add_argument("--out_dir",
                    default=os.path.join(_DATA, "FLOAT", "LS_AL"),
                    help="output videos dir (<stem>.mp4)")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems); falls back to globbing --references if absent")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("FLOAT_LIMIT", "0")),
                    help="smoke: first N stems only (0 = all)")
    args = ap.parse_args()
    REFDIR, WAVDIR, OUTDIR = args.references, args.wav_dir, args.out_dir
    PARTIAL = f"{OUTDIR}/.partial"
    LIMIT = args.limit or None

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PARTIAL, exist_ok=True)
    stems = list_stems(args.stems_file)

    print(f"[float] {len(stems)} stems  REF={REFDIR}  WAV={WAVDIR}\n"
          f"        OUT={OUTDIR}  ckpt={CKPT_PATH}\n"
          f"        a_cfg={A_CFG_SCALE} r_cfg={R_CFG_SCALE} e_cfg={E_CFG_SCALE} "
          f"emo={EMO} nfe={NFE} no_crop={NO_CROP}", flush=True)

    # ---- load the model + face detector ONCE (checkpoint + face_alignment weights) ----
    opt = build_opt()
    print("[float] constructing InferenceAgent (loads float.pth + builds DataProcessor/"
          "face_alignment; may download s3fd+2DFAN4 to ~/.cache on first run) ...", flush=True)
    agent = InferenceAgent(opt)
    print("[float] model + face detector ready.", flush=True)

    done = skipped = 0
    failed = []              # list[(stem, repr(exc))]
    consecutive = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems):
        out_path = f"{OUTDIR}/{stem}.mp4"
        if os.path.exists(out_path) and video_nframes(out_path) >= MIN_FRAMES:
            skipped += 1
            continue

        ref_png = f"{REFDIR}/{stem}.png"
        own_wav = f"{WAVDIR}/{stem}.wav"      # NON-DYADIC: the stem's OWN audio, no partner wav
        tmp = f"{PARTIAL}/{stem}.mp4"
        before_tmp = _tmp_mp4_snapshot()
        t0 = time.perf_counter()
        try:
            for path, nm in ((ref_png, "ref_png"), (own_wav, "own_wav")):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nm} missing: {path}")
            if os.path.exists(tmp):
                os.remove(tmp)               # stale partial from a previous crash

            agent.run_inference(
                res_video_path=tmp,
                ref_path=ref_png,
                audio_path=own_wav,
                a_cfg_scale=A_CFG_SCALE,
                r_cfg_scale=R_CFG_SCALE,
                e_cfg_scale=E_CFG_SCALE,
                emo=EMO,
                nfe=NFE,
                no_crop=NO_CROP,
            )

            # a lone /tmp leftover with no tmp here == the ffmpeg mux step never produced a file
            if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                raise RuntimeError("run_inference produced no output (ffmpeg mux likely failed)")
            nf = video_nframes(tmp)
            if nf < MIN_FRAMES:
                raise RuntimeError(f"degenerate output: {nf} decodable frames (< {MIN_FRAMES})")
            if not has_audio_stream(tmp):
                raise RuntimeError("output has no audio stream (mux dropped audio)")

            os.replace(tmp, out_path)        # atomic (same fs); nothing half-written lands as <stem>.mp4
            done += 1
            consecutive = 0
            dt = time.perf_counter() - t0
            eta = (time.perf_counter() - t_start) / max(done, 1) * (len(stems) - skipped - done)
            print(f"[{idx + 1}/{len(stems)}] {stem}: {nf}f ({nf / opt.fps:.1f}s) "
                  f"in {dt:.1f}s -> {out_path}  ETA {eta / 60:.0f}m", flush=True)
        except Exception as exc:  # noqa: BLE001 - collect per-stem, abort only if systemic
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            failed.append((stem, repr(exc)))
            consecutive += 1
            print(f"[{idx + 1}/{len(stems)}] FAIL {stem}: {exc}", flush=True)
            traceback.print_exc()
            if consecutive > MAX_CONSECUTIVE_FAIL:
                cleanup_tmp(before_tmp)
                print(f"\nABORT: {consecutive} consecutive failures -> systemic error, stopping.",
                      flush=True)
                break
        finally:
            cleanup_tmp(before_tmp)

    print(f"\nDONE: {done} generated, {skipped} pre-existing, {len(failed)} failed "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    for stem, err in failed:
        print("  FAIL", stem, err, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
