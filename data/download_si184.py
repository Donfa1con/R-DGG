#!/usr/bin/env python3
"""Fetch the SI-184 corpus (video + audio) from the official Seamless Interaction release.

This populates the corpus layout the rest of the repo expects:

    <out_root>/GT/LS_AL/<stem>.mp4          GT / speaker video (both dyad roles)
    <out_root>/GT/LS_AL_wav/<stem>.wav      driving / partner audio (both roles)

`data/references/<stem>.png` is committed already; this script only fetches the
mp4 + wav that are *not* committed (see .gitignore).

--------------------------------------------------------------------------------
SI access parameters (owner-confirmed)
--------------------------------------------------------------------------------
The Seamless Interaction selection for SI-184 is, per the dataset owner:

    vendor            = V00
    interaction_type  = ipc_conversation
    label             = improvised     -> DatasetConfig(label=...)
    split             = test           -> DatasetConfig(split=...)

`label` and `split` are the only two of these that
`seamless_interaction.fs.DatasetConfig` accepts as filters, and this script passes
them (`--label` / `--split` default to these confirmed values). `DatasetConfig`
has **no** `vendor` or `interaction_type` field, so those two are NOT passed to the
API — the first column of `data/pairs184.txt` is the listener stems: every file_id
begins `V00_` (vendor V00), and the list is the 184 `ipc_conversation` stems (the
186 improvised stems minus the 2 `grounded_gesture` ones). Fetches are by explicit
file_id, so the vendor is pinned by the id itself.

--------------------------------------------------------------------------------
Prerequisites / caveats (READ BEFORE RUNNING)
--------------------------------------------------------------------------------
* Uses the official `seamless_interaction` package
  (https://github.com/facebookresearch/seamless_interaction). It is NOT on PyPI
  as a drop-in; install it from that repo, e.g.
      git clone https://github.com/facebookresearch/seamless_interaction
      pip install -e seamless_interaction
* The underlying Hugging Face dataset (`facebook/seamless-interaction`) may be
  **gated** — you must request/accept access on HF and be logged in
  (`huggingface-cli login`) before `gather_file_id_data_from_s3` will succeed.
* **The stem -> SI file_id mapping is the one thing still to confirm on the first
  real download.** Our stems carry a trailing role letter (e.g. `..._P1293A`) that
  the SI `file_id` does NOT have. This script derives the file_id by dropping that
  trailing letter (see `derive_file_id`). The label/split/vendor/interaction_type
  above are owner-confirmed; this letter-strip derivation is NOT yet confirmed
  against the release. Run `--dry-run` first and inspect every stem -> file_id line
  before downloading anything.

IMMUNE-U (fail loud, never guess): if a real download errors, this script prints
the attempted file_id + guidance and RE-RAISES. It never skips a stem silently
and never fabricates a substitute.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../data

# Our stem shape: V<vendor>_S<session>_I<interaction>_P<participant><role-letter>
# The SI file_id drops the trailing role letter.  Both fields are validated so a
# malformed stem fails loud instead of producing a bogus file_id.
_STEM_RE = re.compile(r"^(V\d+_S\d+_I\d+_P\d+)([A-Za-z])$")


def derive_file_id(stem: str) -> str:
    """stem -> SI file_id (UNVERIFIED mapping; see module docstring).

    `V00_S2035_I00000998_P1293A` -> `V00_S2035_I00000998_P1293`
    (strip the single trailing role letter from the participant field).

    Raises ValueError (fail loud) on any stem that does not match the expected
    shape, rather than silently emitting a wrong file_id.
    """
    m = _STEM_RE.match(stem.strip())
    if not m:
        raise ValueError(
            f"stem {stem!r} does not match the expected "
            f"'V<d>_S<d>_I<d>_P<d><role-letter>' shape; refusing to guess a file_id"
        )
    return m.group(1)


def read_stems(stems_file: str) -> list[str]:
    """The listener stems from data/pairs184.txt. Each line is "<listener>.wav <speaker>.wav"; the listener
    stem is the first whitespace token with a trailing .wav stripped. Skip comment (`#`) and blank lines,
    dedup keeping first-seen order."""
    if not os.path.isfile(stems_file):
        sys.exit(f"stems file not found: {stems_file}")
    stems, seen = [], set()
    for ln in open(stems_file):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        stem = ln.split()[0]
        if stem.endswith(".wav"):
            stem = stem[:-4]
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    if not stems:
        sys.exit(f"no stems in {stems_file} (only comments/blanks?)")
    return stems


def _find_one(root: str, file_id: str, exts: tuple[str, ...]) -> str | None:
    """Locate a downloaded file for `file_id` with one of `exts` under `root`.

    The SI package's on-disk layout is not part of a stable contract we can rely
    on, so we search rather than assume a path. Returns the first match (shortest
    path wins for determinism), or None."""
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if file_id.lower() in low and low.endswith(exts):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits, key=lambda p: (len(p), p))[0] if hits else None


def place(stem: str, file_id: str, stage_dir: str, out_root: str, overwrite: bool) -> None:
    """Copy the fetched mp4/wav for `file_id` into the corpus layout under our stem name."""
    dst_mp4 = os.path.join(out_root, "GT", "LS_AL", f"{stem}.mp4")
    dst_wav = os.path.join(out_root, "GT", "LS_AL_wav", f"{stem}.wav")
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    os.makedirs(os.path.dirname(dst_wav), exist_ok=True)

    src_mp4 = _find_one(stage_dir, file_id, (".mp4",))
    src_wav = _find_one(stage_dir, file_id, (".wav",))
    missing = [ext for ext, src in ((".mp4", src_mp4), (".wav", src_wav)) if src is None]
    if missing:
        raise FileNotFoundError(
            f"downloaded {file_id} but could not locate {missing} under {stage_dir!r}. "
            f"Inspect that directory: the SI on-disk naming may differ from the file_id, "
            f"or a modality may not have been fetched."
        )
    for src, dst in ((src_mp4, dst_mp4), (src_wav, dst_wav)):
        if os.path.exists(dst) and not overwrite:
            continue
        shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stems_file", default=os.path.join(_HERE, "pairs184.txt"),
                    help="pairing list; col1 = listener stems (default: data/pairs184.txt)")
    ap.add_argument("--out_root", default=_HERE,
                    help="corpus root to populate (default: data/)")
    ap.add_argument("--stage_dir", default=None,
                    help="where seamless_interaction downloads land before we copy them into the "
                         "corpus layout (default: <out_root>/.si_download_cache)")
    ap.add_argument("--label", default="improvised",
                    help="SI DatasetConfig label (owner-confirmed: improvised)")
    ap.add_argument("--split", default="test",
                    help="SI DatasetConfig split (owner-confirmed: test)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N stems only (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-copy even if <stem>.mp4/.wav already exist in the corpus")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print stem -> derived file_id for every stem and exit; downloads NOTHING")
    args = ap.parse_args()

    stems = read_stems(args.stems_file)
    if args.limit:
        stems = stems[: args.limit]

    # ---- dry-run: pure stdlib, no package import, no network. Inspect this first. ----
    if args.dry_run:
        print(f"# {len(stems)} stems  (stem -> derived SI file_id; UNVERIFIED mapping)")
        for stem in stems:
            print(f"{stem}\t{derive_file_id(stem)}")
        print("\n# dry run: nothing downloaded. Confirm the mapping before a real run.")
        return

    # ---- real run: needs the (possibly gated) package + HF access. ----
    try:
        from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS
    except ImportError as exc:
        sys.exit(
            f"could not import seamless_interaction ({exc}).\n"
            "Install it from https://github.com/facebookresearch/seamless_interaction "
            "(pip install -e <clone>), and make sure you have accepted access to the gated\n"
            "Hugging Face dataset and run `huggingface-cli login`."
        )

    stage_dir = args.stage_dir or os.path.join(args.out_root, ".si_download_cache")
    os.makedirs(stage_dir, exist_ok=True)

    try:
        # Owner-confirmed SI-184 selection: label=improvised, split=test (V00 /
        # ipc_conversation are enforced by the stem list, not DatasetConfig fields).
        config = DatasetConfig(label=args.label, split=args.split)
        fs = SeamlessInteractionFS(config=config, local_dir=stage_dir,
                                   num_workers=args.num_workers)
    except Exception as exc:  # noqa: BLE001 -- construction contract may differ across versions
        sys.exit(
            f"failed to construct SeamlessInteractionFS ({type(exc).__name__}: {exc}).\n"
            "The package's constructor/DatasetConfig signature may differ from what this script "
            "assumes (label/split/local_dir/num_workers). Check the installed version's fs.py and "
            "adjust the two lines above; do not guess."
        )

    print(f"[si184] downloading {len(stems)} stems -> {args.out_root}/GT/  (stage: {stage_dir})",
          flush=True)
    for i, stem in enumerate(stems, 1):
        file_id = derive_file_id(stem)
        print(f"[{i}/{len(stems)}] {stem}  (file_id={file_id})", flush=True)
        try:
            fs.gather_file_id_data_from_s3(file_id)
            place(stem, file_id, stage_dir, args.out_root, args.overwrite)
        except Exception as exc:  # noqa: BLE001 -- fail loud, never skip silently (IMMUNE-U)
            print(
                f"\n!! FAILED on stem {stem} (attempted file_id={file_id}): "
                f"{type(exc).__name__}: {exc}\n"
                "   Guidance: (1) confirm the stem->file_id mapping (--dry-run: does this id\n"
                "   exist in the release?); (2) confirm HF access to the gated dataset + login;\n"
                "   (3) inspect the stage dir for the on-disk naming. Aborting (not skipping).",
                file=sys.stderr, flush=True,
            )
            raise

    print(f"[si184] done: {len(stems)} stems placed under {args.out_root}/GT/", flush=True)


if __name__ == "__main__":
    main()
