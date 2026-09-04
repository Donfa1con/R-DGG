# CLAUDE.md — the IMMUNE design charter

This repo (`r-dgg-submission`) is the anonymous release of **R-DGG** and the
classic dyadic-listener metrics on SI-184. It is organised around one idea,
borrowed from immunology: **the cells are disposable, the memory is not.** Any
one script here can be thrown away and regenerated; what must survive is the
recorded *knowledge* — the contracts, the conventions, the reasons — written
down where a fresh clone (or a fresh agent) can read it. The charter below is
that memory.

## IMMUNE

- **I — Isolation.** Components are self-contained, like compartmentalised cells.
  Every `extractors/*` owns its `pixi.toml` + `pixi.lock`, and no extractor
  imports another's code. The six metrics — `rdgg/` and the five
  `classic_metrics/*` — share ONE pixi env: the single manifest `rdgg/pixi.toml`.
  The five classic metrics also share one `classic_metrics/_motion.py` feature
  loader. A broken extractor env cannot infect another extractor. The six metrics
  use one env on purpose, because their dependencies are compatible.
- **M — Memory (knowledge outlives code).** The durable asset is documentation:
  per-component `README.md`, `models/README.md`, this charter. Code is
  regenerable; the *knowledge of why it is the way it is* is not. Record decisions
  where they will be re-read, not in a commit that scrolls away.
- **M — Minimalism (no crutches).** Fix the root cause; do not stack a patch on a
  fragile mechanism to prop it up. Reproduce a model's real inference (model, CFG,
  seed, resolution, fps) and nothing more.
- **U — Uncertainty is loud (fail loud, never guess).** When an input is missing
  or a mapping is unverified, stop and surface it — never silently skip, substitute,
  or fabricate. `data/download_si184.py` ships an explicit `derive_file_id`, a
  `--dry-run` to inspect the (unverified) stem→file_id mapping, and re-raises with
  guidance on any download error. Pipeline scripts are `set -euo pipefail` and
  fail loud on missing inputs.
- **N — Nonredundant source of truth (SSOT).** One canonical list drives
  everything: `data/pairs184.txt` (184 directed `<listener>.wav <speaker>.wav`
  pairs). Its first column is the 184-stem listener universe; every consumer
  derives the stem list from that column at runtime. Counts are *derived* from the
  list, never hardcoded (the generation drivers read their expected N from it).
- **E — Evident & portable.** Paths come from args/env with **repo-relative
  defaults** — no absolute host paths baked in, so the code runs from a fresh
  clone. Outward-facing docs use one name, **R-DGG**, coherently.

## How this repo applies IMMUNE

- **Isolation:** `cd rdgg && pixi install` builds the shared metrics env; each
  extractor is installed on demand in its own env (`cd extractors/<x> && pixi
  install`). `pipeline/4_rdgg.sh` and `pipeline/5_classic.sh` run the six metrics
  through the shared manifest (`pixi run --manifest-path rdgg/pixi.toml`); the
  other `pipeline/*.sh` steps run each extractor or model in its own `pixi run`.
- **Memory:** each component's `README.md` (where present) states its CLI + `.npz`/CSV contract;
  `models/README.md` pins every submodule commit and records the exact CFG/seed
  each generator was run with.
- **Minimalism:** the R-DGG metric lives in `rdgg/`; the generation drivers
  wire corpus paths only and change no model/CFG/seed; the five classic metrics
  load features through one shared `classic_metrics/_motion.py`.
- **Uncertainty loud:** `data/download_si184.py` (dry-run + fail-loud on the
  unverified mapping); `pipeline/1..5_*.sh` pre-check inputs and abort with a
  clear message; `5_classic.sh` logs and counts per-metric failures and exits
  non-zero if any failed.
- **SSOT:** the drivers and both metric layers all read `data/pairs184.txt` (its
  first column is the listener universe); the non-dyadic drivers derive their
  expected stem count from that column, not a hardcoded constant.
- **Evident & portable:** every script defaults its I/O under the repo's `data/`
  and takes `--corpus` / `$CORPUS` to relocate it.

## Deliberate deviations (honest)

These break a common software norm on purpose; each buys something the charter
values more.

- **Internal symbols and output filenames keep the `reactivity_*`
  names** (e.g. `reactivity_matrix__LS_AL.{json,npz}`). Only the outward-facing
  docs say **R-DGG**; renaming the internals would churn caches and cross-refs
  for no scientific gain.
- **Vendored metric code keeps its own `try/except`.** Fail-loud (U) is a rule
  for *our glue*, not a license to rewrite error handling inside code we vendored;
  its authors' contract stands.

## Writing (prose, docs, comments)

All prose — the `README.md` files, this charter, docstrings, comments, and commit
messages — follows **ASD-STE100 (Simplified Technical English)**:

- Write short sentences: 20 words maximum for an instruction, 25 for a
  description. Put one idea in one sentence and one instruction in one sentence.
- Use the active voice. Use the present tense, or the imperative for a step.
- Give one word one meaning and one meaning one word. Use the same term for the
  same thing; do not use synonyms. Keep a noun cluster to 3 words or fewer.
- Use plain, approved words. Do not use slang or unexplained jargon. Do not drop
  articles. Do not use an ambiguous pronoun — name the thing.

The goal is the charter's goal: text that a new reader, or a new agent, reads
once and does not have to guess at.
