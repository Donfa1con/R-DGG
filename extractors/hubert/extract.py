"""Extract per-wav HuBERT hidden states — the audio driver for the R-DGG metric (rdgg/).

One `<stem>.npz` per input wav, holding the chosen transformer layers' hidden states at HuBERT's native 50 Hz:

    feat_l09   (T, 768)  float16   hidden state after transformer layer 9   (one key per --layers entry)
    layers     (L,)      int64     which layers were written
    fps        scalar    float64   T / (samples / sample_rate) -- see below

`fps` is DERIVED, not the nominal 50: HuBERT's convolutional front end has a 400-sample receptive field and a
320-sample hop, so T is a couple of frames short of duration x 50 and the true frame rate of what was written is
T/duration (49.99 on this corpus, not 50.00). Consumers resample by this value, so storing the nominal rate would
shift the features against the video by the length of that shortfall.

No `face_ok` / `h` / `w` / `n_frames`: those cross-cutting npz fields describe a video, and this is an audio
extractor, same as extractors/vad.

Whole files are processed in one forward pass rather than chunked. Chunking a transformer changes the values of the
frames near every boundary, and the longest clip here (346 s, 17.3k frames) fits comfortably with SDPA attention,
whose memory is linear in sequence length. If a corpus ever exceeds the device, the honest fix is a bigger device or
an explicit, documented overlap-and-discard — not a silent one.

Usage:
    python extract.py --wavs_dir <in> --out_dir <out> [--layers 9] [--overwrite] [--limit N]
    python extract.py --pairs <in1>:<out1> <in2>:<out2> ...      # several directories, one model load
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import HubertModel

# The checkpoint is a constant, not an argument. hubert-large-ll60k was evaluated at two layers and rejected: it loses
# the consuming metric's held-out anchor on every channel. It is also not a drop-in -- its layer-norm front end is not
# gain-invariant, so it REQUIRES the waveform normalisation that this one must not have (see NORMALIZE_INPUT). An
# argument here would let those two settings drift apart silently.
CKPT = "facebook/hubert-base-ls960"
HOP = 320  # the conv stride: 320 samples at 16 kHz = 20 ms per frame
TARGET_SR = 16000
# Waveform zero-mean/unit-variance. False, and measured rather than chosen -- see the note beside its use in main().
NORMALIZE_INPUT = False
# `output_hidden_states` returns 13 tensors: index 0 is the convolutional/positional output, 1..12 are the transformer
# layers, so layer N is hidden_states[N].
HIDDEN_STATE_OFFSET = 0


def load_wav_16k(path):
    """Mono float32 at 16 kHz, plus the ORIGINAL duration in seconds.

    The duration comes from the source file, not from the resampled array: it is what `fps` is derived against, and a
    resampler is free to return a sample more or fewer than the exact ratio.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    duration = wav.shape[0] / sr
    mono = wav.mean(1)
    if sr != TARGET_SR:
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, TARGET_SR).numpy()
    # One hop of silence at the end, so the frame count lands on exactly 50 per second instead of one short.
    # T = floor((n - 400)/320) + 1, so for n a multiple of the hop the unpadded count is n/320 - 1: the 400-sample
    # receptive field costs a frame the stride would otherwise have given. Adding one hop makes T = n/320 exactly.
    # This is not cosmetic. A consumer whose grid is 25 frames per second can then decimate by two and read a real
    # frame at every slot, instead of interpolating and sitting half a slot off -- and at 40 ms slots these features
    # change by ~39% across half a slot. Only the final frame sees the padding.
    mono = np.concatenate([mono, np.zeros(HOP, dtype=mono.dtype)])
    return np.ascontiguousarray(mono), duration


def encode(model, wav16, layers, device, normalize):
    x = wav16
    if normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        out = model(torch.from_numpy(x)[None].to(device), output_hidden_states=True)
    n_states = len(out.hidden_states)
    picked = {}
    for layer in layers:
        idx = layer + HIDDEN_STATE_OFFSET
        assert 0 <= idx < n_states, (
            f"layer {layer} maps to hidden_states[{idx}], but the checkpoint exposes {n_states} states "
            f"(index 0 is the convolutional output, 1..{n_states - 1} are transformer layers)"
        )
        picked[layer] = out.hidden_states[idx][0].to(torch.float16).cpu().numpy()
    return picked


def process_dir(model, wavs_dir, out_dir, layers, device, overwrite, limit, normalize, ckpt):
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(wavs_dir.glob("*.wav"))
    assert wavs, f"no *.wav under {wavs_dir}"
    if limit:
        wavs = wavs[:limit]
    n_done = n_skipped = 0
    t0 = time.time()
    for i, wav_path in enumerate(wavs, 1):
        dst = out_dir / f"{wav_path.stem}.npz"
        if dst.exists() and not overwrite:
            n_skipped += 1
            continue
        wav16, duration = load_wav_16k(wav_path)
        feats = encode(model, wav16, layers, device, normalize)
        n_frames = len(next(iter(feats.values())))
        payload = {f"feat_l{layer:02d}": mat for layer, mat in feats.items()}
        payload["layers"] = np.asarray(sorted(layers), dtype=np.int64)
        payload["ckpt"] = np.asarray(ckpt)  # so a cache can never be mistaken for one from a different checkpoint
        payload["fps"] = np.float64(n_frames / duration)
        assert abs(n_frames / duration - TARGET_SR / HOP) < 1e-6, (
            f"{wav_path.stem}: {n_frames} frames over {duration}s is {n_frames / duration:.4f} Hz, not the exact "
            f"{TARGET_SR / HOP:.0f} the one-hop pad produces -- the duration is not a whole number of hops"
        )
        # write beside the target and rename, so an interrupted run cannot leave a half-written npz that later looks
        # like a finished one. The temporary name deliberately keeps the .npz.partial tail rather than a `<stem>_tmp`
        # form -- a `_tmp` infix inside the stem is how a previous incident propagated into every downstream cache.
        partial = dst.with_suffix(".npz.partial")
        try:
            with open(partial, "wb") as fh:
                np.savez(fh, **payload)  # a file object, because np.savez APPENDS .npz to a path that lacks it
            os.replace(partial, dst)
        finally:
            if partial.exists():
                partial.unlink()
        n_done += 1
        if n_done == 1 or n_done % 20 == 0 or i == len(wavs):
            rate = n_done / max(time.time() - t0, 1e-9)
            print(
                f"  [{i}/{len(wavs)}] {wav_path.stem}: {n_frames} frames @ {n_frames / duration:.4f} Hz "
                f"({duration:.0f}s audio) | {rate:.2f} wav/s",
                flush=True,
            )
    print(f"  {out_dir}: wrote {n_done}, skipped {n_skipped} already present", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="R-DGG (rdgg/) reads layer 9. --layers takes more only if something else needs them.",
    )
    ap.add_argument("--wavs_dir", type=Path, help="directory of <stem>.wav")
    ap.add_argument("--out_dir", type=Path, help="directory to write <stem>.npz into")
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="IN:OUT",
        help="several <wavs_dir>:<out_dir> pairs, processed in one model load",
    )
    ap.add_argument("--layers", nargs="+", type=int, default=[9], help="transformer layers to write (1-based)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true", help="re-extract stems that already have an npz")
    ap.add_argument("--limit", type=int, default=0, help="stop after N wavs per directory (smoke tests)")
    args = ap.parse_args()

    if args.pairs:
        jobs = []
        for spec in args.pairs:
            assert ":" in spec, f"--pairs takes <wavs_dir>:<out_dir>, got {spec!r}"
            src, dst = spec.rsplit(":", 1)
            jobs.append((Path(src), Path(dst)))
    else:
        assert args.wavs_dir and args.out_dir, "pass either --wavs_dir with --out_dir, or --pairs"
        jobs = [(args.wavs_dir, args.out_dir)]
    for src, _dst in jobs:
        assert src.is_dir(), f"not a directory: {src}"
    if args.device == "cuda":
        assert torch.cuda.is_available(), "no CUDA device; pass --device cpu (much slower) if that is intended"

    print(f"loading {CKPT} on {args.device} ...", flush=True)
    model = HubertModel.from_pretrained(CKPT, attn_implementation="sdpa").to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    bad = [layer for layer in args.layers if not 1 <= layer <= n_layers]
    assert not bad, f"{CKPT} has {n_layers} transformer layers; --layers {bad} is out of range"
    # Waveform zero-mean/unit-variance, and NOT read from the checkpoint config even though the config asks for it.
    # Measured: base's own Wav2Vec2FeatureExtractor ships do_normalize=True, and following it changes the features by
    # 8.8% mean absolute -- which is not a detail, it moves a non-dyadic control from covering zero to EXCLUDING it, i.e.
    # it breaks the consuming metric's validation. The shipped configuration is the one that validates, not the one the
    # checkpoint prefers. A different checkpoint may genuinely require normalisation (large uses a layer-norm front end
    # and is not gain-invariant), which is one more reason the checkpoint is a deliberate choice and not a knob to turn.
    normalize = NORMALIZE_INPUT
    print(
        f"  {n_layers} layers, hidden {model.config.hidden_size}, do_normalize={normalize}; "
        f"writing layers {sorted(args.layers)}",
        flush=True,
    )

    for src, dst in jobs:
        print(f"{src} -> {dst}", flush=True)
        process_dir(model, src, dst, args.layers, args.device, args.overwrite, args.limit, normalize, CKPT)
    print("HUBERT_DONE")


if __name__ == "__main__":
    sys.exit(main())
