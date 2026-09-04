#!/usr/bin/env python3
"""Generation smoke check: does a model's driver produce a VALID video?

Generation has seed noise, so there is NO pixel/number golden here (the numeric golden for the extractors is
tests/golden/P1293A.json). This check
only asserts that a produced clip is a *valid video*:

  * it decodes and has >= MIN_FRAMES decodable frames,
  * resolution + fps are plausible (sane ranges),
  * its duration is within tolerance of the reference clip (~138 s for the
    golden stem V00_S2035_I00000998_P1293A),
  * it carries an audio track (all 7 drivers mux audio) -- unless --no-audio.

It reads frame count / HxW / fps with cv2 (falls back to imageio-ffmpeg's probe),
and detects the audio track by walking the MP4 box tree for a 'soun' handler
(stdlib only -- no ffprobe/ffmpeg needed on PATH). Run it under any env that has
opencv (e.g. a model's conda env, or `cd extractors/<x> && pixi run python`).

Usage:
  # by explicit path
  python tests/check_generation.py data/ditto/LS_AL/V00_S2035_I00000998_P1293A.mp4

  # by model name -> data/<model>/LS_AL/<golden-stem>.mp4
  python tests/check_generation.py --model ditto
  python tests/check_generation.py --model SoulX_FlashHead_Lite

  # every model's golden-stem output at once (PASS/FAIL table)
  python tests/check_generation.py --all

Exit code is nonzero if any checked clip FAILs.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # tests/
_REPO = os.path.dirname(_HERE)

GOLDEN_STEM = "V00_S2035_I00000998_P1293A"
REF_DURATION_S = 138.0          # golden dyad clip length (GT: 4140 frames @ 30 fps; wav: 138.00 s)

MIN_FRAMES = 10                 # a real clip is thousands of frames; below this is degenerate/truncated
W_RANGE = (64, 8192)            # plausible frame width
H_RANGE = (64, 8192)            # plausible frame height
FPS_RANGE = (1.0, 240.0)        # plausible container fps

# model name -> (output subdir under data/, informative expected {w,h,fps,audio})
# The expected column is documentation only; PASS/FAIL uses the generic ranges +
# duration tolerance above (generation is stochastic, but geometry is deterministic).
MODELS: dict[str, dict] = {
    "ditto":                {"dir": "ditto/LS_AL",                "wh": (1080, 1080), "fps": 25, "audio": True},
    "FLOAT":                {"dir": "FLOAT/LS_AL",                "wh": (512, 512),   "fps": 25, "audio": True},
    "SoulX_FlashHead_Lite": {"dir": "SoulX_FlashHead_Lite/LS_AL", "wh": (512, 512),   "fps": 25, "audio": True},
    "SoulX_FlashHead_Pro":  {"dir": "SoulX_FlashHead_Pro/LS_AL",  "wh": (512, 512),   "fps": 25, "audio": True},
    "avatarforcing":        {"dir": "AvatarForcing/LS_AL",        "wh": (512, 512),   "fps": 25, "audio": True},
    "dystream":             {"dir": "dystream/LS_AL",             "wh": (512, 512),   "fps": 25, "audio": True},
    "avtr-1":               {"dir": "AVTR-1/LS_AL",               "wh": (1080, 1080), "fps": 25, "audio": True},
}


# ---- video geometry (cv2, else imageio-ffmpeg) ------------------------------
def probe_video(path: str) -> dict:
    """Return {'frames','width','height','fps','decodable'} for an mp4.
    `frames` is the container's frame count; `decodable` is whether the first
    frame actually decodes (guards a nonzero-but-degenerate/truncated file)."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"frames": 0, "width": 0, "height": 0, "fps": 0.0, "decodable": False}
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ok, _ = cap.read()
        cap.release()
        return {"frames": n, "width": w, "height": h, "fps": fps, "decodable": bool(ok)}
    except ImportError:
        pass
    # fallback: imageio-ffmpeg reader (bundles its own ffmpeg)
    import imageio.v2 as imageio
    rd = imageio.get_reader(path, "ffmpeg")
    meta = rd.get_meta_data()
    fps = float(meta.get("fps", 0.0))
    nframes = meta.get("nframes")
    size = meta.get("size", (0, 0))
    decodable = False
    try:
        rd.get_data(0)
        decodable = True
    except Exception:
        pass
    if not isinstance(nframes, int) or nframes <= 0:
        dur = meta.get("duration")
        nframes = int(dur * fps) if (dur and fps) else 0
    rd.close()
    return {"frames": int(nframes), "width": int(size[0]), "height": int(size[1]),
            "fps": fps, "decodable": decodable}


# ---- audio-track detection (MP4 box walk, stdlib only) ----------------------
def _walk_boxes(fh, start: int, end: int, want: bytes, found: list, depth: int = 0):
    """Recurse the ISO-BMFF box tree from [start,end); collect payload offsets of
    every `want` box. Only descends the container boxes on the path to hdlr."""
    CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}
    i = start
    while i + 8 <= end and depth < 12:
        fh.seek(i)
        hdr = fh.read(8)
        if len(hdr) < 8:
            break
        size = struct.unpack(">I", hdr[:4])[0]
        typ = hdr[4:8]
        payload = i + 8
        if size == 1:                       # 64-bit largesize
            ext = fh.read(8)
            if len(ext) < 8:
                break
            size = struct.unpack(">Q", ext)[0]
            payload = i + 16
        elif size == 0:                     # box extends to end of file
            size = end - i
        if size < 8 or i + size > end:
            break
        if typ == want:
            found.append(payload)
        elif typ in CONTAINERS:
            _walk_boxes(fh, payload, i + size, want, found, depth + 1)
        i += size


def has_audio_track(path: str) -> bool | None:
    """True/False if the mp4 has/hasn't a 'soun' handler track; None if the file
    can't be parsed as ISO-BMFF (caller decides how to treat 'unknown')."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            hdlrs: list = []
            _walk_boxes(fh, 0, size, b"hdlr", hdlrs)
            if not hdlrs:
                return None                 # not an mp4 / no moov parsed
            for off in hdlrs:
                fh.seek(off + 8)            # version(1)+flags(3)+pre_defined(4) -> handler_type
                if fh.read(4) == b"soun":
                    return True
            return False
    except Exception:
        return None


# ---- one clip --------------------------------------------------------------
def check_clip(path: str, expect_duration: float, tol_s: float,
               require_audio: bool) -> tuple[bool, str, dict]:
    """Validate a single produced clip. Returns (ok, one_line_summary, info)."""
    info: dict = {"path": path}
    if not os.path.isfile(path):
        return False, "missing file", info

    p = probe_video(path)
    info.update(p)
    reasons: list[str] = []

    if not p["decodable"]:
        reasons.append("first frame does not decode")
    if p["frames"] < MIN_FRAMES:
        reasons.append(f"only {p['frames']} frames (< {MIN_FRAMES})")
    if not (W_RANGE[0] <= p["width"] <= W_RANGE[1] and H_RANGE[0] <= p["height"] <= H_RANGE[1]):
        reasons.append(f"implausible resolution {p['width']}x{p['height']}")
    if not (FPS_RANGE[0] <= p["fps"] <= FPS_RANGE[1]):
        reasons.append(f"implausible fps {p['fps']:.3f}")

    dur = p["frames"] / p["fps"] if p["fps"] > 0 else 0.0
    info["duration_s"] = dur
    if p["fps"] > 0 and abs(dur - expect_duration) > tol_s:
        reasons.append(f"duration {dur:.1f}s not within +/-{tol_s:.0f}s of {expect_duration:.0f}s")

    aud = has_audio_track(path)
    info["audio"] = aud
    if require_audio:
        if aud is False:
            reasons.append("no audio track (driver muxes audio; mux dropped it)")
        # aud is None (unparseable container) -> can't confirm; note but don't hard-fail on that alone
        elif aud is None:
            reasons.append("audio track: could not parse container")

    summary = (f"{p['width']}x{p['height']} @ {p['fps']:.2f}fps  "
               f"{p['frames']}f  {dur:.1f}s  audio={aud}")
    return (not reasons), (summary if not reasons else summary + "  ||  " + "; ".join(reasons)), info


def resolve_path(model: str, stem: str, out_root: str) -> str:
    sub = MODELS.get(model, {}).get("dir", model)
    return os.path.join(out_root, sub, f"{stem}.mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="?", help="explicit path to a produced .mp4")
    ap.add_argument("--model", help=f"resolve data/<dir>/<stem>.mp4 for one of: {', '.join(MODELS)}")
    ap.add_argument("--all", action="store_true", help="check every model's golden-stem output")
    ap.add_argument("--stem", default=GOLDEN_STEM, help="stem to check (default: golden stem)")
    ap.add_argument("--out-root", default=os.path.join(_REPO, "data"), help="root of model output dirs")
    ap.add_argument("--expected-duration", type=float, default=REF_DURATION_S)
    ap.add_argument("--duration-tol", type=float, default=20.0, help="+/- seconds allowed vs reference")
    ap.add_argument("--no-audio", action="store_true", help="do not require an audio track")
    args = ap.parse_args()

    require_audio = not args.no_audio
    jobs: list[tuple[str, str]] = []           # (label, path)
    if args.all:
        for name in MODELS:
            jobs.append((name, resolve_path(name, args.stem, args.out_root)))
    elif args.model:
        jobs.append((args.model, resolve_path(args.model, args.stem, args.out_root)))
    elif args.video:
        jobs.append((os.path.basename(args.video), args.video))
    else:
        ap.error("give a video path, --model NAME, or --all")

    print(f"generation smoke  stem={args.stem}  expect ~{args.expected_duration:.0f}s "
          f"(+/-{args.duration_tol:.0f}s)  audio_required={require_audio}\n")
    any_fail = False
    width = max(len(lbl) for lbl, _ in jobs)
    for label, path in jobs:
        ok, summary, _ = check_clip(path, args.expected_duration, args.duration_tol, require_audio)
        tag = "PASS" if ok else "FAIL"
        any_fail |= not ok
        print(f"  [{tag}] {label:<{width}}  {summary}")
    print()
    if any_fail:
        print("RESULT: FAIL (>=1 clip invalid or missing)")
        sys.exit(1)
    print("RESULT: PASS (all checked clips are valid videos)")


if __name__ == "__main__":
    main()
