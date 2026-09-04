# tlcc — Time-Lagged Cross-Correlation

Behavioural sync between speaker and listener motion features.
Adapted from `baseline_react2024/metric/TLCC.py`. For each (speaker,
listener) pair and each lag `l ∈ [-2s, +2s]`, compute the mean
per-dim Pearson correlation between aligned `speaker[t]` and
`listener[t+l]`. Report:

| | meaning |
|---|---|
| `tlcc_peak_<group>`   | max correlation across lags (higher = stronger sync) |
| `tlcc_center_<group>` | correlation at `lag=0` |
| `tlcc_offset_<group>` | `argmax_lag` in source-fps frames; `+ve` ⇒ listener lags speaker |

Computed **per group** for both feature backends (`--feature_source
{emoca, lp}`) and emitted to two CSVs side-by-side, per the dual-source
convention.

## Pairing modes

| mode                       | rule                                                   |
|---|---|
| `exact` (default)          | speaker stem == listener stem                          |
| `flip01`                   | flip trailing `.0` ↔ `.1` (paired-split corpora, not shipped here) |
| `from_file:<pairs.txt>`    | whitespace map `<listener>.wav <speaker>.wav` (SI-184: `data/pairs184.txt`) |

On SI-184 both dyad roles share one dir; `from_file:data/pairs184.txt`
maps each listener stem to its speaker stem.

## Environment

The six metrics share one pixi env — the manifest is `rdgg/pixi.toml`. Install
it once with `cd ../../rdgg && pixi install`. Every `pixi run` example below
runs through the shared manifest.

## Usage

```bash
ROOT=../../data      # repo corpus root (holds GT/ and one dir per model)
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    --speaker        $ROOT/GT/LS_AL_emoca \
    --listener       $ROOT/<MODEL>/LS_AL_emoca \
    --feature_source emoca \
    --pairing        from_file:$ROOT/pairs184.txt \
    [--max_lag_sec 2] [--target_fps 25] [--adaptive_max_lag] \
    [--out $ROOT/<MODEL>/_metrics/tlcc_emoca__LS_AL.csv] \
    [--overwrite]
```

`--fps` (source fps) is read from the first npz if not passed;
`--target_fps` resamples before correlation (the pipeline uses 25). Per-dim
Pearson is computed in numpy; about 1.5 sec per video (300 frames @ 30 fps,
±2 s ⇒ 121 lags). Idempotent: existing stems skipped on re-run.

### `--adaptive_max_lag`

By default the strict React2024 check `T ≥ 2*max_lag + 4` drops clips
shorter than ~4 sec at the default `max_lag_sec=2`. With this flag the
window shrinks per-video to `min(max_lag, (T-4)//2)` so short clips
contribute a narrower-window measurement instead of being dropped. The
CSV column `eff_max_lag` records what was actually used so consumers
can weight short-clip results appropriately.

### `--vad_dir` / `--vad_mode` — silence/speech-aware TLCC

On dyadic data, the labelled "listener" is often self-speaking. To
restrict TLCC to listener-silent frames (the regime where a genuine
listener response appears), point the script at a VAD cache from
`extractors/vad/extract.py`:

```bash
pixi run --manifest-path ../../rdgg/pixi.toml python compute.py \
    ... \
    --vad_dir $ROOT/GT/LS_AL_vad \
    --vad_mode silence \
    --out $ROOT/<MODEL>/_metrics/tlcc_<src>_silence__LS_AL.csv
```

`--vad_mode silence` keeps frames where the listener is silent;
`speech` keeps the inverse; `none` (default) keeps the full video
(the react2024 metric). The mask is resampled per-stem to feature-frame
length; new CSV column `n_kept` records how many frames survived.
Pair every `*_<split>.csv` with its listening variant to read the two
regimes side-by-side. `pipeline/5_classic.sh` computes both by default.

## Caveat: long bidirectional dialogues

TLCC assumes role stability across the segment it correlates. On long
SI clips with turn-taking the labelled speaker is sometimes the
listener and vice versa, which dilutes the lagged correlation. The
`--vad_mode silence` variant addresses this for the listener side
(by restricting to frames where they're not self-speaking) — use it
as the primary read on dyadic data; the full-video score remains a
useful sanity check.
