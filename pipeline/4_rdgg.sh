#!/usr/bin/env bash
# [4/5] Compute the R-DGG leaderboard + 8x8 pairwise matrix over the corpus.
# Writes <corpus>/GT/_metrics/reactivity_matrix__<split>.{json,npz} (the output
# filenames use the internal `reactivity_*` name).
#
# Runs in the shared metrics env (rdgg/pixi.toml). Repo-relative; override with
# $CORPUS / $SPLIT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="${CORPUS:-$REPO/data}"
SPLIT="${SPLIT:-LS_AL}"
MANIFEST="$REPO/rdgg/pixi.toml"   # the six metrics share this one pixi env

# pairs184.txt (committed at the corpus root) decides who is scored.
for f in "$CORPUS/pairs184.txt"; do
  [ -f "$f" ] || { echo "!! [4/5] missing pairs184.txt: $f" >&2; exit 1; }
done

# R-DGG fits the GT reference operator on the GT features + reads GT hubert/vad,
# so those must be present.
for d in "$CORPUS/GT/${SPLIT}_emoca" "$CORPUS/GT/${SPLIT}_liveportrait" \
         "$CORPUS/GT/${SPLIT}_hubert" "$CORPUS/GT/${SPLIT}_vad"; do
  [ -d "$d" ] || { echo "!! [4/5] missing GT feature dir: $d (run pipeline/3_extract.sh)" >&2; exit 1; }
done

echo ">> [4/5] R-DGG on $CORPUS (split $SPLIT) -> $CORPUS/GT/_metrics/reactivity_matrix__${SPLIT}.{json,npz}"
pixi run --manifest-path "$MANIFEST" python "$REPO/rdgg/compute.py" --corpus "$CORPUS" --split "$SPLIT"

echo ">> [4/5] done"
