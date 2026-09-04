#!/usr/bin/env bash
# Fetch the pre-computed SI-184 data (generated videos + extracted features) from
# the Hugging Face dataset into data/, so you can run pipeline/4_rdgg.sh +
# 5_classic.sh WITHOUT generation or extraction.
#
# The data lives in the Hugging Face dataset Donfa1con/R-DGG. The dataset holds one
# directory per source (GT, AVTR-1, ...). This script downloads the dataset and
# copies each source directory under data/.
#
#   bash pipeline/0_fetch_hf.sh
#
# Set HF_REPO to use a different dataset id. The script needs the Hugging Face CLI:
#   pip install -U huggingface_hub
#
# The ground-truth raw video and audio are NOT in this dataset. Get them with
# data/download_si184.py (see pipeline/1_download.sh). This dataset carries the GT
# features, the seven systems' videos, and every source's features.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DATA="${CORPUS:-$REPO/data}"
HF_REPO="${HF_REPO:-Donfa1con/R-DGG}"
SOURCES=(GT AVTR-1 AvatarForcing FLOAT SoulX_FlashHead_Lite SoulX_FlashHead_Pro ditto dystream)

die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

# Pick a Hugging Face downloader: the new `hf` CLI, or the older huggingface-cli.
if command -v hf >/dev/null 2>&1; then
  HF=(hf download "$HF_REPO" --repo-type dataset)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download "$HF_REPO" --repo-type dataset)
else
  die "the Hugging Face CLI is not installed. Install it: pip install -U huggingface_hub"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo ">> [0/5] downloading dataset $HF_REPO from Hugging Face"
"${HF[@]}" --local-dir "$TMP" || die "download failed: $HF_REPO"

# Anchor on pairs184.txt inside the download, in case the tree nests one level.
ROOT="$(dirname "$(find "$TMP" -type f -name pairs184.txt | head -1)")"
[ -n "$ROOT" ] && [ -d "$ROOT" ] || die "no pairs184.txt in the dataset -- unexpected layout"

echo ">> [0/5] placing sources under $DATA"
placed=0
for src in "${SOURCES[@]}"; do
  [ -d "$ROOT/$src" ] || { echo "   skip $src (not in dataset)"; continue; }
  mkdir -p "$DATA/$src"
  cp -r "$ROOT/$src/." "$DATA/$src/"
  placed=$((placed + 1))
done
[ "$placed" -gt 0 ] || die "no known source dirs in the dataset"

# pairs184.txt and data/references/ ship with the code repo (the SSOT). This script
# copies only the source dirs, so it does not overwrite them.
echo ">> [0/5] done: $placed source dirs placed under data/. GT video/audio still come from data/download_si184.py."
