# Reference-Based Directed Granger Gain for Evaluating Speech-Conditioned Listener Motion

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13)](https://openreview.net/forum?id=uZdMPMmHxa)

Official evaluation code for the paper.

## Requirements

We ran the pipeline on one machine with one NVIDIA L40S GPU (48 GB VRAM) and 30 GB of
system RAM.

| Item | What you need |
|---|---|
| GPU | One NVIDIA GPU with about 46 GB of VRAM or more. The heaviest steps — some generators and the R-DGG metric assembly — use about 46 GB. The reference GPU is an L40S. |
| System RAM | About 30 GB or more (the reference machine had 30 GB). The R-DGG assembly and some models use a lot of host RAM. |
| Disk | A few hundred GB of free space. Keep room for the SI-184 corpus (video and audio), the per-model weights (for example, the DyStream checkpoint is about 7.7 GB), and the extracted feature caches. |

You do not install CUDA, the build tools, or TensorRT. The pixi and conda environments
pull the whole CUDA stack themselves. You supply only the GPU and an NVIDIA driver new
enough for CUDA 12.8 — the highest version any environment uses.

`pixi` manages the environments. The six metrics share one env (`rdgg/pixi.toml`); each
extractor keeps its own; some generator models use their own conda environment (see
`models/README.md`).

## Getting the code

The listener generators under `models/` are **git submodules**, so clone
*recursively* — a plain `git clone` leaves them empty:

```bash
git clone --recurse-submodules https://github.com/Donfa1con/R-DGG.git
cd R-DGG
bash setup.sh          # init submodules (if needed) + apply our generation patches
```

`setup.sh` applies `models/patches/<model>.patch` onto each model submodule (the minimal
changes we made to drive SI-184 generation). The six metrics (`rdgg/` plus the five
`classic_metrics/*`) share ONE pixi env — the manifest is `rdgg/pixi.toml`; install it
once with `cd rdgg && pixi install`. Each extractor under `extractors/*` keeps its own
env, set up on demand (`cd extractors/<x> && pixi install`). The emoca extractor is the
one exception: it needs `pixi install --frozen` (a plain solve fails on
`onnxruntime-gpu==1.16.2`).

## Layout

| path | what |
|---|---|
| `rdgg/` | the R-DGG metric — `compute.py` + `pixi.toml`, the shared env for all six metrics |
| `extractors/` | video/audio → features feeding R-DGG (insightface → emoca/liveportrait, hubert, vad) |
| `models/` | the listener generators as git submodules + the patches we applied to generate SI-184 |
| `data/` | `download_si184.py` (fetch the 184 SI videos / audio / references) + stem & pairing lists |
| `classic_metrics/` | secondary metrics — rPCC / TLCC / P-FD / SID / Var (use the shared `rdgg/pixi.toml` env) |
| `pipeline/` | end-to-end glue: download → generate → extract → R-DGG table |

## Results

**Corpus = SI-184** (184 improvised `ipc_conversation` listener stems). **Speaker = GT.**
R-DGG is a corpus-level statistic: one GT reference reaction operator, fit across all
identities. The value is the directed Granger gain of the speaker AUDIO on the listener
MOTION, with the speaker's own co-speech motion regressed out (`aud|mot`, net ×1e4). Each
system is tested against its own timeshift-null. The interval is an identity-clustered
bootstrap 95% CI.

### Leaderboard (net ×1e4)

| # | model | kind | net | 95% CI | p |
|---|---|---|---|---|---|
| 1 | **GT** (anchor) | GT | **+0.72** | [+0.32, +1.12] | **0.000** |
| 2 | **AVTR-1** | dyadic | **+0.57** | [+0.19, +0.98] | **0.002** |
| 3 | **DyStream** | dyadic | **+0.48** | [+0.06, +0.93] | **0.011** |
| 4 | **AvatarForcing** | dyadic | **+0.32** | [+0.07, +0.56] | **0.008** |
| 5 | SoulX Pro | non-dyad | +0.18 | [−0.34, +0.58] | 0.239 |
| 6 | Ditto | non-dyad | +0.06 | [−0.20, +0.31] | 0.335 |
| 7 | FLOAT | non-dyad | +0.01 | [−0.33, +0.42] | 0.491 |
| — | GT×other (surrogate) | control | −0.11 | [−0.56, +0.45] | 0.659 |
| 8 | SoulX Lite | non-dyad | −0.22 | [−0.54, +0.21] | 0.857 |

### 8×8 pairwise Δ = G_row − G_col (×1e4)

GT&A&B frame-exact paired. `*` marks a 95% CI that excludes 0.

| row \ col | GT | AVTR | DySt | AvFo | SxP | FLOAT | Ditto | SxL |
|---|---|---|---|---|---|---|---|---|
| **GT** | — | +0.13 | +0.23 | +0.36 | +0.55 | +0.68 | +0.66\* | +0.89\* |
| **AVTR-1** | −0.13 | — | +0.09 | +0.24 | +0.42 | +0.55 | +0.53\* | +0.76\* |
| **DyStream** | −0.23 | −0.09 | — | +0.15 | +0.29 | +0.49 | +0.40 | +0.70\* |
| **AvatarF** | −0.36 | −0.24 | −0.15 | — | +0.18 | +0.32 | +0.29\* | +0.52\* |
| **SoulX-P** | −0.55 | −0.42 | −0.29 | −0.18 | — | +0.13 | +0.11 | +0.34 |
| **FLOAT** | −0.68 | −0.55 | −0.49 | −0.32 | −0.13 | — | −0.02 | +0.21 |
| **Ditto** | −0.66\* | −0.53\* | −0.40 | −0.29\* | −0.11 | +0.02 | — | +0.23 |
| **SoulX-L** | −0.89\* | −0.76\* | −0.70\* | −0.52\* | −0.34 | −0.21 | −0.23 | — |

## End-to-end

```bash
bash pipeline/1_download.sh     # data → corpus layout
bash pipeline/2_generate.sh     # models → listener videos
bash pipeline/3_extract.sh      # videos+audio → features
bash pipeline/4_rdgg.sh         # features → R-DGG table
bash pipeline/5_classic.sh      # features → classic-metric tables (secondary)
```

**Fast path — skip generation and extraction.** The generated videos and the extracted
features are published as the Hugging Face dataset
[`Donfa1con/R-DGG`](https://huggingface.co/datasets/Donfa1con/R-DGG). `pipeline/0_fetch_hf.sh`
downloads them into `data/`. It needs the Hugging Face CLI (`pip install -U huggingface_hub`):

```bash
bash pipeline/0_fetch_hf.sh            # download videos + features into data/
bash pipeline/4_rdgg.sh                # → R-DGG table
bash pipeline/5_classic.sh             # → classic-metric tables (secondary)
```

You still need a GPU for the R-DGG metric (`pipeline/4_rdgg.sh`); steps 1–3 above are only
for a full from-scratch run. The ground-truth raw video and audio are not in the dataset;
fetch them with `pipeline/1_download.sh` if you need them (the metrics read the shipped GT
features).

See each component's `README.md` (where present) for details.

## Verify it works

Two checks on the single golden stem confirm a fresh clone is set up right. Run them after `setup.sh`,
before you start the full 184-stem pipeline — each needs only the one golden stem, not the whole corpus.

**1. Per-video reproduction (extractors).** `tests/check_reproduction.py` re-reads the golden stem
(`V00_S2035_I00000998_P1293A`). It compares the extractor fingerprints (emoca / liveportrait / hubert / vad)
to `tests/golden/P1293A.json` within tolerance. Run it after `pipeline/3_extract.sh` produces the golden-stem
features. The check needs numpy, so run it under the shared metrics env, from the repo root:

```bash
pixi run --manifest-path rdgg/pixi.toml python tests/check_reproduction.py
```

A PASS means the extractors reproduce. The R-DGG metric itself reproduces only at the corpus level. Run the
full-corpus pipeline, then compare to `RESULTS.md`: each value lands within its 95% CI. Landmark
detection and GPU reduction are nondeterministic, so neither the per-video rPCC / TLCC nor the metric is
bit-reproducible (see `tests/golden/README.md`).

**2. Generation smoke.** First generate a clip, then check it is a valid video. Each model needs its own
environment and weights (see `models/README.md`); the generated clips are not committed, so you generate them.
Generation has seed noise, so this check confirms a valid video, not a numeric match. A valid video decodes,
has more than zero frames, and has a plausible resolution and fps. `check_generation.py` reads
`data/<model>/LS_AL/<stem>.mp4`:

```bash
# generate only the golden stem with one model driver (under that model's env)
python models/generate/ditto.py --stems_file tests/golden/generation_smoke_stems.txt
# then check the produced clip (run under any env with opencv)
python tests/check_generation.py --model ditto
```

Use `--all` to check every model at once, after you generate all of them.

## Citation

```bibtex
@inproceedings{kravtsov2026referencebased,
  title={Reference-Based Directed Granger Gain for Evaluating Speech-Conditioned Listener Motion},
  author={Artem Kravtsov and Dmitrii Ziganshin and Vsevolod Poletaev and Anastasia Tikhonova},
  booktitle={NeurIPS 2026 Workshop Real-Time Conversational Agents: Toward Natural Multimodal Interaction},
  year={2026},
  url={https://openreview.net/forum?id=uZdMPMmHxa}
}
```

## License

This repository mixes licenses by content type:

- **Code** — the scripts and modules we wrote (`rdgg/`, `classic_metrics/`, `extractors/`,
  `models/generate/`, `pipeline/`, `data/download_si184.py`, `tests/`) are licensed **MIT**
  (see `LICENSE`).
- **Data** — the reference portraits in `data/references/`, and any Seamless Interaction
  derived data, are licensed **CC BY-NC 4.0** (non-commercial) with attribution to Seamless
  Interaction (Meta / FAIR). See `data/references/README.md`.
- **Model submodules** — each generator under `models/` keeps its own upstream license. See
  `models/README.md`.
