# `models/` — the six listener/talking-head generators

Each generator is a **git submodule** pinned to the exact upstream commit our SI-184
runs were based on. We never fork or vendor the model source; where we had to change
tracked source, the change is captured as a **patch** in `models/patches/` and applied
on top of the pinned submodule by `../setup.sh`. Our batch **generation drivers** live in
`models/generate/` — they import the model from its submodule and reproduce the exact
inference call (model, CFG, seed, resolution, fps) we used.

```
models/
├── <name>/              git submodule, pinned (see table below)
├── patches/<name>.patch tracked-source edits we applied (avatarforcing, dystream only)
├── generate/<name>.py   our batch driver (imports the submodule; params below)
└── README.md            this file
```

## Reproducing a run

1. `bash setup.sh` at the repo root — inits the submodules and applies
   `models/patches/*.patch` onto the two edited models.
2. Fetch that model's weights (per-model instructions below).
3. Set up the model's own environment (each submodule ships its own
   `requirements.txt` / `environment.sh` / `pixi.toml`).
4. Run the driver. Every driver defaults its I/O to the repo's `data/`:

   | driver arg     | default                    | meaning |
   |----------------|----------------------------|---------|
   | `--references` | `data/references/`         | `<stem>.png` source portraits (committed). avatarforcing names this flag `--reference_dir`. |
   | `--wav_dir`    | `data/GT/LS_AL_wav/`       | `<stem>.wav` driving audio; `data/download_si184.py` fetches it. |
   | `--pairs`      | `data/pairs184.txt`        | dyadic only: `<listener>.wav <speaker>.wav` per line |
   | `--stems_file` | `data/pairs184.txt`        | which stems to run (col1 = listener stems); falls back to globbing `--references` if absent |
   | `--out_dir`    | `data/<model>/LS_AL/`      | output `<stem>.mp4` (25 fps, driving audio muxed) |

   So from the repo root the minimal invocation is just `python models/generate/<name>.py`
   (run under that model's activated env so its `ffmpeg`/`ffprobe` subprocesses are on `PATH`).
   The commands below show the CFG each model was run with — **do not change the model / CFG /
   seed flags**; only paths are meant to be overridden.

**Non-dyadic** models (ditto, float, soulx-flashhead) drive each portrait with *that same
person's own* audio (plain lip-sync). **Dyadic** models (avatarforcing, dystream, avtr-1)
additionally consume the *partner's* audio (and, for avatarforcing, the partner's *video*)
selected via `--pairs`.

## Pinned submodules

| name              | upstream URL                                      | pinned commit                              | clone size¹ | our change |
|-------------------|---------------------------------------------------|--------------------------------------------|-------------|------------|
| `ditto`           | https://github.com/antgroup/ditto-talkinghead     | `c3e47eee2e626500017a0556b470d6d4182f85e8` | 1.4 MB      | driver only (source clean) |
| `float`           | https://github.com/deepbrainai-research/float     | `3b5b2dfc3e65df26e7fbba17d9adb3f43747851c` | 4.1 MB      | driver only (source clean) |
| `soulx-flashhead` | https://github.com/Soul-AILab/SoulX-FlashHead     | `9bc03de06bb0de82cd6bc477804512ae06144bf2` | 6.0 MB      | driver only (source clean) |
| `avatarforcing`   | https://github.com/TaekyungKi/AvatarForcing       | `507e3046ad539c938e0412ac7a74cf6e3ba2a61c` | 7.1 MB      | **patch** + driver |
| `dystream`        | https://github.com/RobinWitch/DyStream            | `f587541ccaa77f092d03487d7d89502a67f0a084` | 15 MB       | **patch** + driver |
| `avtr-1`          | https://github.com/avaturn-live/avtr-1            | `eb1e5a89ed7bec3debe712b090efd676b3a420bd` | 1.6 MB      | driver only (source clean) |

¹ Working tree, cloned with `GIT_LFS_SKIP_SMUDGE=1`. We fetch the LFS blobs and model weights
separately (see per-model instructions). The pinned commit is the upstream base our local
checkout sat on. We did not commit the two patched models' edits upstream. So each pin is the
clean upstream commit, and `patches/<name>.patch` reconstructs our version exactly.

---

## ditto — non-dyadic lip-sync

| | |
|---|---|
| Upstream | https://github.com/antgroup/ditto-talkinghead @ `c3e47eee2e626500017a0556b470d6d4182f85e8` |
| Weights  | HuggingFace [`digital-avatar/ditto-talkinghead`](https://huggingface.co/digital-avatar/ditto-talkinghead) — `git clone https://huggingface.co/digital-avatar/ditto-talkinghead models/ditto/checkpoints`. Driver uses the **offline** config `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl` and the TensorRT engines `checkpoints/ditto_trt_Ampere_Plus/` (Ampere-plus HW-compat; an L40S/sm_89 loads these directly. On an unsupported GPU, rebuild with the repo's `scripts/cvt_onnx_to_trt.py` and point `--data_root` at the result). |
| Our change | None to source. Driver `models/generate/ditto.py`. |
| CFG / seed | no seed passed (ditto's own default; upstream seeding stays disabled); native **1080×1080**, **25 fps**; SDK built once and reused. |

```bash
conda activate ditto
python models/generate/ditto.py           # --references / --wav_dir / --out_dir override paths only
```

---

## float — non-dyadic lip-sync

| | |
|---|---|
| Upstream | https://github.com/deepbrainai-research/float @ `3b5b2dfc3e65df26e7fbba17d9adb3f43747851c` |
| Weights  | `bash models/float/download_checkpoints.sh` (gdown → `checkpoints/float.pth`). The `wav2vec2-base-960h` audio encoder and the `wav2vec-english-speech-emotion-recognition` model go under `checkpoints/` per the FLOAT repo README. |
| Our change | None to source. Driver `models/generate/float.py`. |
| CFG / seed | `run_inference(a_cfg_scale=2, r_cfg_scale=1, e_cfg_scale=1, emo=None, nfe=10, no_crop=False)`; `emo=None` → FLOAT's audio-predicted-emotion path; **512×512**, **25 fps**; no seed passed (FLOAT's own default path; `fix_noise_seed` left at its default). |

```bash
conda activate FLOAT
python models/generate/float.py
```

---

## soulx-flashhead — non-dyadic lip-sync (Lite **and** Pro)

| | |
|---|---|
| Upstream | https://github.com/Soul-AILab/SoulX-FlashHead @ `9bc03de06bb0de82cd6bc477804512ae06144bf2` |
| Weights  | `huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/soulx-flashhead/models/SoulX-FlashHead-1_3B` and `huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/soulx-flashhead/models/wav2vec2-base-960h`. **One repo + checkpoint serves both variants**; `--variant` selects the model_type (and `motion_frames_num` 9 for Lite / 5 for Pro). |
| Our change | None to source. Driver `models/generate/soulx.py`, keeps `--variant lite|pro`. |
| CFG / seed | `audio_encode_mode="stream"`, `sample_steps=4` (asserted); **512×512**, **25 fps**; pipeline (incl. `torch.compile`) loaded once. No seed of ours is injected; `get_base_data` requires a `base_seed`, so the driver passes SoulX's own default (42, from `generate_video.py`). |

```bash
conda activate flashhead
python models/generate/soulx.py --variant lite     # -> data/SoulX_FlashHead_Lite/LS_AL/
python models/generate/soulx.py --variant pro      # -> data/SoulX_FlashHead_Pro/LS_AL/
```

---

## avatarforcing — dyadic (listener conditioned on partner audio **and** video)

| | |
|---|---|
| Upstream | https://github.com/TaekyungKi/AvatarForcing @ `507e3046ad539c938e0412ac7a74cf6e3ba2a61c` |
| Weights  | `bash models/avatarforcing/download_weights.sh` (gdown → `pretrained_dir/motion_autoencoder.pth` + `pretrained_dir/flow_transformer.pth`, plus `facebook/wav2vec2-base-960h` into `pretrained_dir/`). |
| **Patch** | `patches/avatarforcing.patch` on tracked files `inference.py` + `models/avatarforcing/flow_transformer.py`. Two changes, both to run **full-length SI clips** (upstream capped at ~30 s): `max_len` / `max_len_sr` 30 s → 600 s, `opt.*` → `self.opt.*` (an upstream bug that read the module-global `opt`), and RoPE `max_seq_len` 1024 → 16384. **No model / CFG change.** Applied by `../setup.sh`. |
| Our change | Patch above + driver `models/generate/avatarforcing.py`. Streaming decode → ffmpeg (memory-bounded); no model change. |
| CFG / seed | `--nfe 10 --a_cfg_scale 2.0 --u_cfg_scale 1.0`; no seed passed — runs UNSEEDED (its `seed_everything` lived in `inference.py`'s `__main__`, which the in-process driver bypasses); `input_size` / `fps` from `configs/inference.yaml` (512 / 25). Needs partner **video** frames (`--gt_video_dir`, default `data/GT/LS_AL`) in addition to partner audio. |

```bash
conda activate avatarforcing
python models/generate/avatarforcing.py    # dyadic: uses data/pairs184.txt + data/GT/LS_AL/ + data/GT/LS_AL_wav/
```

---

## dystream — dyadic (listener conditioned on speaker audio)

| | |
|---|---|
| Upstream | https://github.com/RobinWitch/DyStream @ `f587541ccaa77f092d03487d7d89502a67f0a084` |
| Weights  | HuggingFace [`robinwitch/DyStream`](https://huggingface.co/robinwitch/DyStream): `git clone https://huggingface.co/robinwitch/DyStream` then move its `checkpoints/` (the ~7.7 GB `last.ckpt`) and `tools/` into `models/dystream/` (see the DyStream README's "Download Checkpoints"). |
| **Patch** | `patches/dystream.patch` on tracked `app.py`: a **streaming rewrite** of `latents_to_video_frames` / `save_video_with_audio` (yield one uint8 frame at a time; direct ffmpeg stream-copy mux instead of moviepy) so long clips don't exhaust host RAM. **Per-frame math is unchanged** — the patch removes only whole-video buffering. Applied by `../setup.sh`. |
| Our change | Patch above + driver `models/generate/dystream.py`. Stubs `gradio` (import-only), reuses `app.run_inference`. |
| CFG / seed | `denoising_steps=5`, `cfg_audio=0.5`, `cfg_audio_other=0.5`, `cfg_anchor=0.0`, `cfg_all=1.0`; no seed passed — runs UNSEEDED (its `seed_everything(222)` lived in `main.py`, which the in-process driver bypasses); **512×512**, **25 fps**. |

```bash
conda activate dystream_py11
python models/generate/dystream.py          # dyadic: uses data/pairs184.txt + data/GT/LS_AL_wav/
```

---

## avtr-1 — dyadic (listener conditioned on speaker audio)

| | |
|---|---|
| Upstream | https://github.com/avaturn-live/avtr-1 @ `eb1e5a89ed7bec3debe712b090efd676b3a420bd` (public) |
| License  | The AVTR-1 model weights are licensed **non-commercial research only** (`models/avtr-1/LICENSE-MODEL.md`); the renderer and streamer code are PolyForm Noncommercial. Accept the weight license on HuggingFace before you download. Do not redistribute the weights. |
| Our change | None to source. Driver `models/generate/avtr1.py` adds `models/avtr-1/src` to `sys.path`. |
| CFG / seed | CFG **3/3/3** = `cfg_self_audio=3.0, cfg_other_audio=3.0, cfg_kp=3.0`; **1080×1080** (= reference size), **25 fps**, 16 kHz; no seed passed (AVTR-1 has no seeding of its own); repo defaults otherwise (align-to-longer, `noise_alpha=2.0`, `noise_trunc_z=1.2`, streaming chunks). |

AVTR-1 renders from TensorRT engines. A TensorRT engine is specific to one GPU
compute capability, so you build the engines on your own host. This is the
normal path, not a blocker. `models/avtr-1/pixi.toml` requires CUDA 12.8 and
TensorRT ≤ 10.12. Run these four steps:

```bash
cd models/avtr-1
pixi install -e renderer                              # 1. create the env (Python 3.12 + CUDA 12.8 + TensorRT)
pixi run download                                     # 2. log in to HuggingFace, then download the weights
pixi run build-trt-engines                            # 3. build every TensorRT engine locally
pixi run -e renderer python ../generate/avtr1.py      # 4. generate SI-184
```

Step 2 (`download`) runs the `hf-login` task (`hf auth login`) first, so accept
the weight license on HuggingFace before it. Step 3 writes the engines under
`models/avtr-1/artifacts/` (or `$AVTR1_LOCAL_STORAGE`); the driver loads whatever
engines it finds there.
