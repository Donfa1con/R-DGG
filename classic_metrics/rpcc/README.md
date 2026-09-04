# rPCC — reactive Pearson correlation

Listener-evaluation metric. Given a triplet `(speaker, gt_listener,
gen_listener)` of feature time series we measure how well the *generated*
listener mirrors the speaker compared to the *ground-truth* listener:

```
pcc_gt   = pearson( speaker.flatten(),  gt_listener.flatten()  )
pcc_gen  = pearson( speaker.flatten(),  gen_listener.flatten() )
rPCC     = | pcc_gt - pcc_gen |
```

Lower is better (`0` ⇒ generated listener correlates with the speaker
identically to the GT listener).

The features are flattened across `(time, dim)` before correlation, so this is
a single scalar per video per feature — not a per-dimension average.

One run of `compute.py` evaluates **one direction** (one generated dir). For a
symmetric corpus, where a model also generates the speaker, run the script a
second time with the roles swapped.

## Setup

The six metrics share one pixi env — the manifest is `rdgg/pixi.toml`. Install it
once:

```bash
cd ../../rdgg && pixi install
```

Run this metric from its own directory through the shared manifest. Every `pixi
run` example below uses this form:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py --help
```

The shared env gives `python 3.10 + numpy + scipy + scikit-learn + tqdm`.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --speaker      $ROOT/GT/LS_AL_emoca \
    --gt_listener  $ROOT/GT/LS_AL_emoca \
    --gen_listener $ROOT/<MODEL>/LS_AL_emoca \
    --pairing      from_file:$ROOT/pairs184.txt \
    --feature_source emoca \
    [--out         $ROOT/<MODEL>/_metrics/rpcc_emoca__LS_AL.csv] \
    [--overwrite]
```

On SI-184 both dyad roles share one dir (`GT/LS_AL_emoca`), so `--speaker` and
`--gt_listener` point at the same dir. `from_file:$ROOT/pairs184.txt` maps each
listener stem to its speaker stem.

If `--out` is omitted, the CSV path defaults to:

    <gen_listener>/../_metrics/rpcc_<source>__<split>.csv

where `<split>` is the gen_listener directory's basename with a trailing
`_emoca` / `_liveportrait` stripped.

The CSV is appended to incrementally — a re-run of the same command processes
only stems missing from the CSV. Pass `--overwrite` to start from scratch.

### Speaker ↔ listener pairing modes

`--pairing` controls how we look up the speaker `.npz` for a given listener
stem:

| mode                   | rule                                                             | when to use            |
|---|---|---|
| `exact` (default)      | speaker stem == listener stem                                    | listener and speaker have identical filenames (e.g. monologue, or you've pre-aligned them) |
| `flip01`               | flip the trailing `.0` ↔ `.1` of the listener stem               | paired-split corpora (for example ViCo, not shipped here) |
| `from_file:<pairs.txt>`| read PATH (whitespace `<listener>.wav <speaker>.wav` per line)   | SI-184 (both roles in one dir; mapping is `data/pairs184.txt`) |

Stems whose speaker can't be resolved (no flip target / not in pairs.txt /
no `<sp_stem>.npz` on disk) are silently skipped; the run summary prints a
count.

### CSV columns

```
stem, n_frames, n_kept,
rPCC_<g1>, pcc_gt_<g1>, pcc_gen_<g1>,
rPCC_<g2>, pcc_gt_<g2>, pcc_gen_<g2>, ...
```

`n_frames` is the original triplet length (after truncation to common
extent). `n_kept` is how many frames survived the VAD filter (see below);
equals `n_frames` when `--vad_mode none`.

`pcc_gt_<g>` and `pcc_gen_<g>` are the **signed** raw Pearson
correlations: `pcc(speaker, gt_listener)` and `pcc(speaker, gen_listener)`.
These trace columns let downstream consumers derive the signed
direction `pcc_gen − pcc_gt` (proactivity: positive ⇒ model couples
to speaker MORE strongly than the GT listener) without recomputing.
The headline `rPCC_<g>` is `|pcc_gt − pcc_gen|`.

Empty cell where any of the three series has zero variance for that feature.

### Silence-only / speech-only via VAD

For listener-aware analysis it's often more meaningful to restrict the
correlation to frames where the listener is **silent** (= they're actually
listening, not self-speaking). Two flags do that:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    ... \
    --vad_dir <corpus>/GT/<split>_vad \
    --vad_mode silence \
    --out <corpus>/<MODEL>/_metrics/rpcc_<src>_silence__<split>.csv
```

* `--vad_dir` is a directory of `<stem>.npz` files written by
  `extractors/vad/extract.py` (silero-VAD output: per-segment timestamps
  + a pre-rendered 25 fps mask).
* `--vad_mode silence` (default `none`) flips the speech mask so we keep
  listener-silent frames; `--vad_mode speech` keeps the inverse.
* The mask is resampled per-stem to match the feature-frame length
  (`audio_seconds * feat_fps`), so different source-video fps work
  without configuration.
* `n_kept` in the CSV records the surviving frame count — useful for
  weighting or for noticing stems where VAD threw away too much.

### Worked example

SI-184 (both dyad roles share `GT/LS_AL_emoca`; `pairs184.txt` maps
listener → speaker):

```
ROOT=../../data
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --speaker      $ROOT/GT/LS_AL_emoca \
    --gt_listener  $ROOT/GT/LS_AL_emoca \
    --gen_listener $ROOT/<MODEL>/LS_AL_emoca \
    --pairing      from_file:$ROOT/pairs184.txt
# → $ROOT/<MODEL>/_metrics/rpcc_emoca__LS_AL.csv
```

#### Other corpora (not in this repo)

The `flip01` pairing targets paired-split corpora (for example ViCo, with
separate `LS`/`AL` dirs and `.0`/`.1` stems). Those corpora are not shipped
here; run the script once per direction with `--pairing flip01`.
