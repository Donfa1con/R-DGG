# vad extractor

Per-wav silero-VAD cache. Output is one `<stem>.npz` per input wav with:

| key | shape | dtype | meaning |
|---|---|---|---|
| `speech_starts`  | `(N,)`  | float32 | start of each detected speech segment (seconds) |
| `speech_ends`    | `(N,)`  | float32 | end of each detected speech segment (seconds) |
| `audio_seconds`  | scalar  | float32 | total duration (so consumers can re-render the mask at any fps) |
| `mask_25fps`     | `(T,)`  | uint8   | pre-rendered per-frame mask at 25 fps (1 = speech) |
| `fps`            | scalar  | int32   | 25 (the framerate of `mask_25fps`) |

Float timestamps are the source of truth; the 25 fps mask is a
convenience so simple consumers don't have to know the source video's
framerate. Downstream metrics that need a different framerate compute
their own mask from `speech_starts`/`speech_ends` and `audio_seconds`.

## Setup

```bash
pixi install
```

The env pins `torch==2.4.0` + `torchaudio==2.4.0` with the
`soundfile` backend — the only combination we found where torchaudio's
ABI-tagged native lib loads cleanly. CPU-only is fine: silero-VAD runs
at hundreds of times realtime on CPU.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run python extract.py \
    --wavs_dir $ROOT/GT/LS_AL_wav \
    --out_dir  $ROOT/GT/LS_AL_vad \
    [--threshold 0.5] [--min_speech_dur_s 0.25] [--overwrite]
```

`--threshold` and `--min_speech_dur_s` are passed straight through to
silero's `get_speech_timestamps`. Defaults match silero's recommended
production preset. Idempotent — existing npzs are skipped unless
`--overwrite`.

## Consumers

- `classic_metrics/{rpcc,tlcc,sid,variance}/compute.py` accept
  `--vad_dir <this-output> --vad_mode {silence,speech,none}` to
  restrict the metric to listener-silent (= actually-listening) or
  listener-speaking frames.
- `classic_metrics/pfd/compute.py` accepts `--vad_dir <this-output>`
  with `--mask_speaker` / `--mask_listener {both,speech,silence}` (and
  `--reactive_segments`) to restrict the paired distribution to one
  behavioural regime.

  When VAD is enabled, rPCC, TLCC, SID, and PFD add an `n_kept` column
  recording the surviving frame count, so you can spot stems where too
  few frames survived to be meaningful.

- `pipeline/3_extract.sh` runs this extractor on the GT wavs.
  `pipeline/5_classic.sh` then computes both the full and the listening
  (listener-silent) variants of rPCC / TLCC / PFD / SID / variance.

## Shared masking helpers (`vad_mask.py`)

Every metric in the toolkit reads VAD through one module:
`extractors/vad/vad_mask.py`. Two helpers:

- `per_frame_keep_mask(vad_path, mode, T)` — bool keep-mask at the
  caller's FPS. `mode ∈ {speech, silence, both, none}`. Used by
  rPCC, TLCC, SID, variance, and PFD's per-frame masking mode.

- `reactive_segments_mask(ls_vad, sp_vad, T, min_seg_sec=3.0, pad_sec=0.2)`
  — bool mask of frames inside maximal listener-silence intervals
  that contain ≥1 speaker-speech frame and are ≥ `min_seg_sec` long,
  with `pad_sec` inward shrink on listener silence. Used by PFD's
  `--reactive_segments` mode.

Defaults (3 s, 200 ms) are exported as `DEFAULT_MIN_SEG_SEC` /
`DEFAULT_PAD_LISTENER_SILENCE_SEC` and shared across every caller, so
changing them in one file updates the whole toolkit.

## Why mask the listener and not the speaker

The intent of the VAD-filtered TLCC / rPCC is to isolate the regime
where the listener is *actually listening*, not co-speaking. On dyadic
data the listener's face dynamics during their own speech are
self-driven and dominate any cross-correlation with the speaker; the
listener-silent frames are where a genuine listener response appears.

For TLCC the symmetric case (mask the speaker) is also reasonable and
the same flags support it via `--vad_mode speech`, but the default and
the form the pipeline computes (`pipeline/5_classic.sh`, the "listening"
variant) is `silence` applied to the listener side.
