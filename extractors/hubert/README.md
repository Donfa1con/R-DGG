# hubert extractor

Per-wav HuBERT hidden-state cache. This is the audio driver for the R-DGG metric (`rdgg/`), whose primary channel asks
whether a model's listener responds to what the speaker **said** — so these features are that channel's entire input.

```bash
pixi install
ROOT=../../data/GT      # repo corpus GT dir
pixi run python extract.py --wavs_dir $ROOT/LS_AL_wav --out_dir $ROOT/LS_AL_hubert
# several directories in one model load:
pixi run python extract.py --pairs $ROOT/LS_AL_wav:$ROOT/LS_AL_hubert <in2>:<out2>
```

Idempotent: a stem that already has an npz is skipped unless `--overwrite`. About 1.5 s per 4-minute wav on an L40S,
so a 184-clip corpus takes a few minutes.

## Output

One `<stem>.npz` per input wav:

| key | shape | dtype | meaning |
|---|---|---|---|
| `feat_l09` | `(T, 768)` | float16 | hidden state after transformer layer 9 — one such key per `--layers` entry |
| `ckpt` | scalar | str | the checkpoint that produced it, so two caches cannot be confused |
| `layers` | `(L,)` | int64 | which layers were written |
| `fps` | scalar | float64 | **`T / (samples / sample_rate)`** — derived, not the nominal 50 |

`fps` is derived because HuBERT's convolutional front end has a 400-sample receptive field and a 320-sample hop, so `T`
lands a frame or two short of `duration × 50`. Storing the nominal rate would slide the features against the video by
that shortfall. On this corpus it comes out at 49.995, not 50.000.

No `face_ok` / `h` / `w` / `n_frames`: those cross-cutting fields in the repo's `.npz` convention describe a video, and
this is an audio extractor — same as `extractors/vad`.

R-DGG (`rdgg/`) reads **layer 9**, which is why that is the default; a two-layer cache costs 6.6 GB for 184 clips
against 3.3 GB for one.

## Configuration, and how each part of it was settled

Each choice below is validated against the consuming metric's own protocol (R-DGG, `rdgg/`), over a grid of layers and
checkpoints declared before any result was seen. The extractor is deterministic: the same wavs and the same checkpoint
produce the same npz.

| question | answer | how it was settled |
|---|---|---|
| checkpoint | `facebook/hubert-base-ls960`, a constant rather than an argument | 768 dims means a base HuBERT. `TencentGameMate/chinese-hubert-base` is the obvious suspect for Chinese conversational audio, but it is not the checkpoint R-DGG is validated with. `hubert-large-ll60k` loses the consuming metric's held-out anchor on every channel; it is also not a drop-in, since its layer-norm front end requires the waveform normalisation this checkpoint must not have (see below). So the checkpoint is not a flag: the two settings cannot drift apart. |
| which layer | `hidden_states[9]`, no offset | R-DGG reads layer 9, so it is the default. `output_hidden_states` returns 13 tensors, index 0 being the convolutional output, so "layer 9" is index 9, not 10. |
| whole file or chunks | whole file, one forward pass | Chunking a transformer changes every frame near a boundary. The longest clip here (346 s, 17.3 k frames) fits with SDPA attention, whose memory is linear in sequence length. |
| waveform normalisation | **off**, and this is load-bearing | The checkpoint's own `Wav2Vec2FeatureExtractor` ships `do_normalize: True`, so following it looks more principled. Turning it on changes the features by **8.8 %** mean absolute and moves a model that architecturally cannot see its interlocutor from covering zero to **excluding** it, which breaks the consuming metric's primary channel. It is a close call — the control's lower bound moves from −0.10 to +0.01 — so it is a measurement to repeat on a new corpus, not a law. |
| resampler | `torchaudio.functional.resample` | Worth 2 % against plain stride-3 decimation. The wavs here are 48 kHz float32, an exact 3:1 ratio to 16 kHz. |
| precision | fp16 autocast | Identical to fp32 to five decimals, and about twice as fast. |
