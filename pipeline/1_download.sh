#!/usr/bin/env bash
# [1/5] Fetch the SI-184 corpus (mp4 + wav) into the corpus layout, via
# data/download_si184.py. Reference PNGs are committed; this fetches only the
# (gitignored) video + audio.
#
# Corpus/out are repo-relative; override with $CORPUS. Extra args pass through to
# the python script, so a mapping dry-run is:
#     bash pipeline/1_download.sh --dry-run
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="${CORPUS:-$REPO/data}"
PY="${PY:-python}"

STEMS="$CORPUS/pairs184.txt"
[ -f "$STEMS" ] || { echo "!! [1/5] missing pairs184.txt: $STEMS" >&2; exit 1; }

echo ">> [1/5] download SI-184 -> $CORPUS/GT/{LS_AL,LS_AL_wav}  (stems: $STEMS)"
echo "   note: the stem->SI file_id mapping is UNVERIFIED — dry-run and inspect it first"
echo "   (bash pipeline/1_download.sh --dry-run); needs the seamless_interaction package + HF access."

"$PY" "$REPO/data/download_si184.py" --stems_file "$STEMS" --out_root "$CORPUS" "$@"

echo ">> [1/5] done"
