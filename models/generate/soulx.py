"""Batch-generate SoulX-FlashHead: NON-DYADIC lip-sync, drive each reference portrait
with THAT SAME PERSON's own audio (self / lip-sync). No partner audio anywhere -- this is the
non-dyadic control, unlike the dyadic drivers (AVTR-1 / DyStream) that also feed a listen track.
Output mirrors the scored-model layout: <corpus>/<MODEL_DIR>/LS_AL/<stem>.mp4 (512x512 native, 25fps,
muxed with the stem's own audio). --variant {lite|pro} selects the checkpoint AND the output dir, so
the same driver produces SoulX_FlashHead_Lite and SoulX_FlashHead_Pro (each loads its pipeline once).

Faithfulness: the model / pipeline is loaded and torch.compile'd EXACTLY ONCE via
flash_head.inference.get_pipeline(); reloading per stem would re-run torch.compile (multi-minute) and
is unacceptable. The per-stem generation loop reproduces generate_video.py:generate()'s
audio_encode_mode=='stream' branch verbatim (deque of cached_audio_duration seconds, per-chunk
streaming audio encode, video[motion_frames_num:] slice each chunk) -- the sole SoulX inference path
that keeps streaming motion-frame continuity. sample_steps is hard-set to 4 inside get_pipeline for
lite/pro. Nothing here touches training/gradio/multi-gpu scaffolding.

Writing is atomic + verified: save_video() (imported from generate_video.py) writes <name>_tmp.mp4
then ffmpeg-muxes audio into the final path and rm's the _tmp -- but with NO check=True, so a failed
mux leaves a lone _tmp and no final. This driver writes into <OUTDIR>/.partial, then INDEPENDENTLY
verifies the finished file (>=10 decodable frames via cv2 AND an audio stream via ffprobe) and treats
any leftover _tmp as a mux failure, only THEN os.replace()'ing to <OUTDIR>/<stem>.mp4. Idempotent
(skip if the final mp4 already has >=10 decodable frames); aborts after >3 consecutive failures (a
systemic error hits every stem).

    conda activate flashhead   # from the repo root; inputs/outputs default to data/
    python models/generate/soulx.py --variant lite
    python models/generate/soulx.py --variant pro

If torch.compile errors at load, set SOULX_NO_COMPILE=1 (this flips COMPILE_MODEL/COMPILE_VAE to
False in flash_head/src/pipeline/flash_head_pipeline.py without editing the vendored file).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
import traceback
from collections import deque

import numpy as np

# --- repo layout: inference.py opens "flash_head/configs/infer_params.yaml" with a RELATIVE path at
#     import time, and ckpt_dir/wav2vec_dir below are repo-relative -- so cwd MUST be the repo root
#     before importing flash_head.inference. chdir here so the driver works regardless of launch dir. ---
_HERE = os.path.dirname(os.path.abspath(__file__))
# Model source = the pinned git submodule at models/soulx-flashhead (override with SOULX_REPO).
REPO = os.environ.get("SOULX_REPO", os.path.abspath(os.path.join(_HERE, os.pardir, "soulx-flashhead")))
_DATA = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "data"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import cv2  # noqa: E402  (frame-count verification; opencv-python is a repo requirement)
import librosa  # noqa: E402  (same loader generate_video.py uses; mono 16k)

# optional compile escape hatch -- must be flipped BEFORE get_pipeline instantiates FlashHeadPipeline
if os.environ.get("SOULX_NO_COMPILE", "0") == "1":
    import flash_head.src.pipeline.flash_head_pipeline as _fhp  # noqa: E402
    _fhp.COMPILE_MODEL = False
    _fhp.COMPILE_VAE = False

from flash_head.inference import (  # noqa: E402
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)
from generate_video import save_video  # noqa: E402  (repo-root module; reuse its exact writer+mux)

CKPT_DIR = "models/SoulX-FlashHead-1_3B"                      # repo-relative (cwd == REPO)
WAV2VEC_DIR = "models/wav2vec2-base-960h"                     # repo-relative
AUDIO_ENCODE_MODE = "stream"                                  # SoulX streaming path (motion-frame continuity)
MIN_FRAMES = 10                                               # a valid clip is hundreds of frames; below this is degenerate
MAX_CONSECUTIVE_FAIL = 3                                      # abort once failures EXCEED this (on the 4th): systemic error hits every stem

# --variant -> output model dir (scored-model layout)
VARIANT_TO_MODEL_DIR = {"lite": "SoulX_FlashHead_Lite", "pro": "SoulX_FlashHead_Pro"}

# I/O paths are argparse-populated in main() (defaults point at the repo's data/).
REFDIR = WAVDIR = ""
LIMIT: int | None = None


def video_nframes(path: str) -> int:
    """Decodable frame count of an mp4, or 0 if it can't be opened / the first frame won't decode.
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
    """True iff ffprobe reports at least one audio stream. save_video muxes AAC in; a missing audio
    stream means the ffmpeg mux (run without check=True) silently failed."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and out.stdout.strip() != ""


def sweep_tmp(*dirs: str, stem: str) -> list[str]:
    """Remove and return any leftover model tmp files for this stem across the given dirs.
    SoulX leaves <name>_tmp.mp4 on a failed/interrupted mux; also sweep the *.mp4.tmp.mp4 /
    *_temp*.mp4 shapes other models leave, for defensive parity with the sibling drivers."""
    found: list[str] = []
    for d in dirs:
        for pat in (f"{stem}_tmp*.mp4", f"{stem}*.tmp.mp4", f"{stem}_temp*.mp4", f"{stem}*_tmp.mp4"):
            for f in glob.glob(os.path.join(d, pat)):
                found.append(f)
                try:
                    os.remove(f)
                except OSError:
                    pass
    return found


def generate_frames(pipeline, ip: dict, audio_path: str) -> list:
    """Reproduce generate_video.py:generate()'s audio_encode_mode=='stream' branch: encode the audio
    in fixed cached-window chunks, run the pipeline per chunk, drop the motion_frames_num carry-over
    frames each chunk, and collect CPU frame tensors. Returns generated_list (list of (T,H,W,C) uint8-
    range float tensors)."""
    sample_rate = ip["sample_rate"]
    tgt_fps = ip["tgt_fps"]
    cached_audio_duration = ip["cached_audio_duration"]
    frame_num = ip["frame_num"]
    motion_frames_num = ip["motion_frames_num"]           # set by get_pipeline from vae_stride (lite=9, pro=5)
    slice_len = frame_num - motion_frames_num

    human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    # pad audio with silence so the last chunk isn't truncated (verbatim from generate_video.py)
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate(
            [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
        )

    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

    generated_list = []
    for human_speech_array in human_speech_array_slices:
        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
        video = run_pipeline(pipeline, audio_embedding)
        video = video[motion_frames_num:]                 # drop the streaming carry-over frames
        generated_list.append(video.cpu())
    return generated_list


def main() -> None:
    global REFDIR, WAVDIR, LIMIT
    ap = argparse.ArgumentParser(
        description="Batch SoulX-FlashHead (non-dyadic lip-sync) -- SI-184 reproduction")
    ap.add_argument("--variant", required=True, choices=["lite", "pro"],
                    help="lite -> SoulX_FlashHead_Lite (motion_frames_num 9), "
                         "pro -> SoulX_FlashHead_Pro (5); shared repo+checkpoint, variant picks model_type")
    ap.add_argument("--references", default=os.path.join(_DATA, "references"),
                    help="dir of <stem>.png source portraits")
    ap.add_argument("--wav_dir", default=os.path.join(_DATA, "GT", "LS_AL_wav"),
                    help="dir of <stem>.wav driving audio (each person's OWN audio; non-dyadic)")
    ap.add_argument("--out_dir", default="",
                    help="output videos dir; default data/SoulX_FlashHead_{Lite,Pro}/LS_AL")
    ap.add_argument("--stems_file", default=os.path.join(_DATA, "pairs184.txt"),
                    help="pairing list (col1 = listener stems); falls back to globbing --references if absent")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("SOULX_LIMIT", "0")),
                    help="smoke: first N stems only (0 = all)")
    args = ap.parse_args()
    REFDIR, WAVDIR = args.references, args.wav_dir
    LIMIT = args.limit or None

    model_dir = VARIANT_TO_MODEL_DIR[args.variant]
    outdir = args.out_dir or os.path.join(_DATA, model_dir, "LS_AL")
    partial = f"{outdir}/.partial"                         # tmp writes land here; downstream globs OUTDIR/*.mp4 skip it
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(partial, exist_ok=True)

    if args.stems_file and os.path.exists(args.stems_file):
        # each line is "<listener>.wav <speaker>.wav"; the listener stem is the first whitespace token with a
        # trailing .wav stripped. Skip blank/#-comment lines, dedup first-seen.
        stems, _seen = [], set()
        for ln in open(args.stems_file):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            _s = ln.split()[0]
            if _s.endswith(".wav"):
                _s = _s[:-4]
            if _s not in _seen:
                _seen.add(_s)
                stems.append(_s)
    else:
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(f"{REFDIR}/*.png"))
    if LIMIT:
        stems = stems[:LIMIT]

    print(f"[soulx:{args.variant}] {len(stems)} stems  REF={REFDIR}  WAV={WAVDIR}\n"
          f"           OUT={outdir}  ckpt={CKPT_DIR}  mode={AUDIO_ENCODE_MODE}",
          flush=True)

    # ---- load the pipeline (and torch.compile) EXACTLY ONCE for all stems ----
    world_size = int(os.environ.get("WORLD_SIZE", 1))     # single-GPU: 1 (no torchrun / no dist init)
    print(f"[soulx:{args.variant}] loading pipeline (compiles once; first stem warms it up)...", flush=True)
    pipeline = get_pipeline(
        world_size=world_size, ckpt_dir=CKPT_DIR, model_type=args.variant, wav2vec_dir=WAV2VEC_DIR,
    )
    ip = get_infer_params()                               # AFTER get_pipeline: carries motion_frames_num + sample_steps=4
    assert ip.get("sample_steps") == 4, f"expected sample_steps==4, got {ip.get('sample_steps')}"
    tgt_fps = ip["tgt_fps"]                               # 25 (native); save_video writes at this fps, no re-encode

    done = skipped = 0
    failed: list[tuple[str, str]] = []
    consecutive = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems):
        out_path = f"{outdir}/{stem}.mp4"
        if os.path.exists(out_path) and video_nframes(out_path) >= MIN_FRAMES:
            skipped += 1
            continue

        ref_png = f"{REFDIR}/{stem}.png"
        own_wav = f"{WAVDIR}/{stem}.wav"                  # NON-DYADIC: the stem's OWN audio, nothing else
        partial_final = f"{partial}/{stem}.mp4"
        model_tmp = partial_final.replace(".mp4", "_tmp.mp4")   # exactly what save_video writes+removes
        t0 = time.perf_counter()
        try:
            for path, nm in ((ref_png, "ref_png"), (own_wav, "own_wav")):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nm} missing: {path}")

            # clear any stale partials from a prior interrupted run for this stem
            sweep_tmp(partial, outdir, stem=stem)
            if os.path.exists(partial_final):
                os.remove(partial_final)

            # per-stem: bind the new reference (does NOT recompile), then stream
            get_base_data(pipeline, cond_image_path_or_dir=ref_png, base_seed=42, use_face_crop=True)
            generated_list = generate_frames(pipeline, ip, own_wav)

            total_frames = int(sum(int(v.shape[0]) for v in generated_list))
            if total_frames < MIN_FRAMES:
                raise RuntimeError(f"degenerate generation: {total_frames} frames (< {MIN_FRAMES})")

            # save_video writes partial_final_tmp.mp4 then ffmpeg-muxes own_wav in -> partial_final
            save_video(generated_list, partial_final, own_wav, fps=tgt_fps)

            # verify the mux result INDEPENDENTLY (save_video has no check=True)
            leftovers = [f for f in (model_tmp,) if os.path.exists(f)]
            leftovers += glob.glob(os.path.join(partial, f"{stem}_tmp*.mp4"))
            if leftovers:
                raise RuntimeError(f"leftover tmp after save_video -> mux step failed: {leftovers}")
            if not os.path.exists(partial_final):
                raise RuntimeError("save_video produced no final mp4 (mux failed)")
            nf = video_nframes(partial_final)
            if nf < MIN_FRAMES:
                raise RuntimeError(f"degenerate output file: {nf} decodable frames (< {MIN_FRAMES})")
            if not has_audio_stream(partial_final):
                raise RuntimeError("final mp4 has no audio stream (mux dropped audio)")

            os.replace(partial_final, out_path)           # atomic; nothing half-written/audio-less lands as <stem>.mp4
            done += 1
            consecutive = 0
            dt = time.perf_counter() - t0
            eta = (time.perf_counter() - t_start) / max(done, 1) * (len(stems) - skipped - done)
            print(f"[{idx + 1}/{len(stems)}] {stem}: {nf}f ({nf / tgt_fps:.1f}s) "
                  f"in {dt:.1f}s -> {out_path}  ETA {eta / 60:.0f}m", flush=True)
        except Exception as exc:  # noqa: BLE001 - collect per-stem, abort only if systemic
            failed.append((stem, repr(exc)))
            consecutive += 1
            print(f"[{idx + 1}/{len(stems)}] FAIL {stem}: {exc}", flush=True)
            traceback.print_exc()
            if consecutive > MAX_CONSECUTIVE_FAIL:
                print(f"\nABORT: {consecutive} consecutive failures -> systemic error, stopping.", flush=True)
                break
        finally:
            # sweep any tmp/partial this stem left behind (never touches the finalised out_path)
            sweep_tmp(partial, outdir, stem=stem)
            if os.path.exists(partial_final):
                try:
                    os.remove(partial_final)
                except OSError:
                    pass

    print(f"\nDONE: {done} generated, {skipped} pre-existing, {len(failed)} failed "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    for stem, err in failed:
        print("  FAIL", stem, err, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
