# `tests/golden/` — reproduction fingerprints

## What `check_reproduction.py` verifies

`tests/check_reproduction.py` recomputes a small **fingerprint** for the single
shortest SI-184 clip and diffs it against the committed golden JSON here, within
tolerances (`--rtol`, default `1e-2`; `--atol`, default `1e-4`):

- **golden stem (listener):** `V00_S2035_I00000998_P1293A`
- **its speaker (dyadic partner):** `V00_S2035_I00000998_P1292A`

The fingerprint covers one thing: the **extractors**. It is the shape + mean + std
of every array in the GT `emoca` / `liveportrait` / `hubert` / `vad` npz for the
golden stem. A mismatch shows the feature extractors drifted. So this per-video
check verifies the EXTRACTORS reproduce on the shortest clip — cheap enough to run
on one video.

The check does **not** compare per-video `rPCC` / `TLCC`. Upstream landmark
detection is GPU-nondeterministic. The rPCC is a difference of correlations and the
TLCC is an argmax lag, so both turn sub-pixel landmark jitter into a large,
sometimes sign-flipped change; no tolerance passes per video. The R-DGG metric's
reproducibility is therefore **corpus-level** (a full-corpus run compared to
`RESULTS.md`), not per video — see the next section.

## Why there is no per-video golden for R-DGG

R-DGG is a **corpus-level** statistic: it fits a single GT reference reaction
operator across all identities and reports leaderboard / pairwise numbers over
the whole SI-184 set. It has no meaningful per-video value, so it is deliberately
**not** part of the per-video `P1293A.json` check. R-DGG's reproducibility golden
is the **committed SI-184 per-model numbers**, verified by a full-corpus
`pipeline/4_rdgg.sh` run against the reference table — not a single-clip
fingerprint.

## Using the golden

`P1293A.json` ships **committed** in this directory. `tests/check_reproduction.py`
compares against it by default, so no setup is needed:

```bash
# after pipeline step 3 has produced the golden-stem features:
python tests/check_reproduction.py             # compare this run to the committed golden
```

Run it under the shared metrics env (it needs numpy), from the repo root:
`pixi run --manifest-path rdgg/pixi.toml python tests/check_reproduction.py`.

`--write` regenerates the golden from the current run. Only the repo owner runs
it, to refresh the committed fingerprint after an intended extractor change:

```bash
python tests/check_reproduction.py --write     # overwrite tests/golden/P1293A.json (owner only)
```

If the golden is ever absent (for example, a stripped checkout), the check
prints `SKIP: golden not yet populated` and exits 0. That is a fallback, not the
normal state.
