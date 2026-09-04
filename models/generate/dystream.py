"""Batch-generate DyStream on SI-184: drive each listener reference with its own audio (self / lip-sync)
plus its speaker's audio (other / listening), at the DyStream demo defaults, one video at a time.
Output mirrors the scored-model layout: SI-184/dystream/LS_AL/<listener_stem>.mp4 (512x512, 25fps,
muxed with the listener's own audio).

Faithfulness: uses app.py's proven in-process run_inference, which is documented step-by-step as
"Matches main.py _inference_one_file exactly". Preprocessing is process_image(crop=True,
union_bbox_scale=1.6) -- identical to main.py _get_latent's img_to_mask (--crop True
--union_bbox_scale 1.6) + img_to_latent on the RESIZED (clean) image. No training/Trainer/wandb
scaffolding (main.py's) is touched.

Roles (avatar = the listener whose reference we animate):
  speaker_audio_path  = listener's own wav  -> avatar's own lip / talking motion   (app.py "self")
  listener_audio_path = speaker's wav       -> avatar's listening / reactive motion (app.py "other")
Pairing is data/pairs184.txt: "<listener>.wav <speaker>.wav" (both directions, one line each).

CFG = app.py demo defaults (cfg_audio=0.5, cfg_audio_other=0.5, cfg_anchor=0.0, cfg_all=1.0,
denoising_steps=5). Idempotent (skip if <stem>.mp4 exists); atomic (.partial + os.replace); aborts
after MAX_CONSECUTIVE_FAIL consecutive failures (a systemic error hits every stem).

    conda activate dystream_py11 && python models/generate/dystream.py   # from the repo root
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

# ---- stub gradio (this env has no gradio; run_inference only touches gr.Progress default + gr.Error) ----
class _GrDummy(Exception):
    def __init__(self, *a, **k):
        super().__init__(*a)          # keep the message so a raised gr.Error("...") is legible in the log

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, n):
        return _GrDummy()


def _gr_module_getattr(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)   # keep the module introspectable (inspect walks sys.modules)
    return _GrDummy


_gr = types.ModuleType("gradio")
_gr.__file__ = "<gradio-stub>"
_gr.__spec__ = None
_gr.__path__ = []
_gr.__getattr__ = _gr_module_getattr
sys.modules["gradio"] = _gr

_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned git submodule at models/dystream (override with DYSTREAM_REPO).
REPO = os.environ.get("DYSTREAM_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "dystream")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
sys.path.insert(0, REPO)
os.chdir(REPO)   # app.py opens repo-relative config/checkpoint paths at import (documented launch: cwd == repo)
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
import app  # noqa: E402

# DyStream demo defaults (app.py sliders) == sample.yaml model.cfg_* block
DENOISE, CFG_A, CFG_AO, CFG_ANC, CFG_ALL = 5, 0.5, 0.5, 0.0, 1.0
MIN_FRAMES = 10                     # a valid clip is thousands of frames; anything below this is a degenerate output
MAX_CONSECUTIVE_FAIL = 3            # abort once failures EXCEED this (i.e. on the 4th) -- a systemic error hits every stem

# I/O paths are argparse-populated in main() (defaults point at the repo's data/).
REFDIR = WAVDIR = PAIRS = OUTDIR = PARTIAL = ""
LIMIT: int | None = None


def _noop(*a, **k):
    return None


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


def read_pairs() -> dict[str, str]:
    lis2spk: dict[str, str] = {}
    for line in Path(PAIRS).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        key = a[:-4] if a.endswith(".wav") else a
        val = b[:-4] if b.endswith(".wav") else b
        lis2spk[key] = val
    return lis2spk


def main() -> None:
    global REFDIR, WAVDIR, PAIRS, OUTDIR, PARTIAL, LIMIT
    ap = argparse.ArgumentParser(
        description="Batch DyStream (dyadic listener generation) -- SI-184 reproduction")
    ap.add_argument("--references", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png avatar (listener) portraits")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav (listener own audio + speaker audio, keyed by --pairs)")
    ap.add_argument("--pairs", default=os.path.join(_DATA, "pairs184.txt"),
                    help="'<listener>.wav <speaker>.wav' per line (dyadic pairing)")
    ap.add_argument("--out_dir",
                    default=os.path.join(_DATA, "dystream", "LS_AL"),
                    help="output videos dir (<listener_stem>.mp4)")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems); falls back to all --pairs keys if absent")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("DS_LIMIT", "0")),
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

    print(f"[dystream] {len(stems)} stems  REF={REFDIR}  WAV={WAVDIR}\n"
          f"           OUT={OUTDIR}  CFG(a={CFG_A},ao={CFG_AO},anc={CFG_ANC},all={CFG_ALL},steps={DENOISE})",
          flush=True)

    app.load_dystream_model()
    app.load_visualization_model()

    done = skipped = 0
    failed: list[tuple[str, str]] = []
    consecutive = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems):
        out_path = f"{OUTDIR}/{stem}.mp4"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue
        spk = lis2spk[stem]
        ref_png = f"{REFDIR}/{stem}.png"
        own_wav = f"{WAVDIR}/{stem}.wav"
        other_wav = f"{WAVDIR}/{spk}.wav"
        t0 = time.perf_counter()
        try:
            for path, nm in ((ref_png, "ref_png"), (own_wav, "own_wav"), (other_wav, "other_wav")):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nm} missing: {path}")

            ref_pil = Image.open(ref_png).convert("RGB")
            out_tmp, *_ = app.run_inference(
                ref_pil, own_wav, other_wav,
                DENOISE, CFG_A, CFG_AO, CFG_ANC, CFG_ALL,
                progress=_noop, video_audio_path=own_wav,
            )
            if not (out_tmp and os.path.exists(out_tmp) and os.path.getsize(out_tmp) > 0):
                raise RuntimeError("run_inference produced no output file")
            nf = video_nframes(out_tmp)
            if nf < MIN_FRAMES:
                raise RuntimeError(f"degenerate output: {nf} decodable frames (< {MIN_FRAMES})")

            tmp = f"{PARTIAL}/{stem}.mp4"
            shutil.move(out_tmp, tmp)      # cross-fs move out of /tmp onto the data disk
            os.replace(tmp, out_path)      # atomic rename on the data fs; nothing half-written lands as <stem>.mp4
            done += 1
            consecutive = 0
            dt = time.perf_counter() - t0
            eta = (time.perf_counter() - t_start) / max(done, 1) * (len(stems) - skipped - done)
            print(f"[{idx + 1}/{len(stems)}] {stem} <- spk {spk}: {nf}f in {dt:.1f}s -> {out_path}  ETA {eta / 60:.0f}m",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - collect per-stem, abort only if systemic
            failed.append((stem, repr(exc)))
            consecutive += 1
            print(f"[{idx + 1}/{len(stems)}] FAIL {stem}: {exc}", flush=True)
            traceback.print_exc()
            if consecutive > MAX_CONSECUTIVE_FAIL:
                print(f"\nABORT: {consecutive} consecutive failures -> systemic error, stopping.", flush=True)
                break

    print(f"\nDONE: {done} generated, {skipped} pre-existing, {len(failed)} failed "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    for stem, err in failed:
        print("  FAIL", stem, err, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
