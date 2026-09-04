#!/usr/bin/env bash
# [3/5] Extract per-frame features feeding R-DGG + the classic metrics.
#
# For GT and each model dir (default: GT + the 7 generated model dirs):
#     insightface  ->  <src>/LS_AL_insightface      (cached lm106 / crop / ArcFace / face_ok)
#     emoca        ->  <src>/LS_AL_emoca            (reuses the insightface crop)
#     liveportrait ->  <src>/LS_AL_liveportrait     (reuses the insightface lm106)
# Then, once, on the GT wavs (all models are conditioned on the GT audio, so R-DGG
# reads audio features from GT only):
#     hubert       ->  GT/LS_AL_hubert
#     vad          ->  GT/LS_AL_vad
#
# Each extractor runs in its OWN pixi env (extractors/<name>). Repo-relative;
# override corpus with $CORPUS. Pass src names to extract a subset, e.g.
#     bash pipeline/3_extract.sh GT ditto
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="${CORPUS:-$REPO/data}"
SPLIT="${SPLIT:-LS_AL}"

EMOCA_BS="${EMOCA_BS:-32}"; EMOCA_NW="${EMOCA_NW:-4}"
LP_BS="${LP_BS:-64}";       LP_NW="${LP_NW:-4}"
IF_NW="${IF_NW:-4}"

# each extractor lives in its own env; run compute in that env
ext() { local name="$1"; shift; ( cd "$REPO/extractors/$name" && pixi run python extract.py "$@" ); }

SRCS=("$@")
[ ${#SRCS[@]} -eq 0 ] && SRCS=(GT AVTR-1 AvatarForcing FLOAT SoulX_FlashHead_Lite SoulX_FlashHead_Pro ditto dystream)

for s in "${SRCS[@]}"; do
  vids="$CORPUS/$s/$SPLIT"
  [ -d "$vids" ] || { echo "!! [3/5] missing videos dir for '$s': $vids (generate it first)" >&2; exit 1; }
  insf="$CORPUS/$s/${SPLIT}_insightface"
  emoc="$CORPUS/$s/${SPLIT}_emoca"
  lp="$CORPUS/$s/${SPLIT}_liveportrait"

  echo ">> [3/5] $s: insightface -> $insf"
  ext insightface --videos_dir "$vids" --out_dir "$insf" --num_workers "$IF_NW"

  echo ">> [3/5] $s: emoca -> $emoc"
  # emoca only: the lock is authoritative. A plain 'pixi run' re-resolves and fails
  # on onnxruntime-gpu==1.16.2, so pin the solve with --frozen for this extractor.
  ( cd "$REPO/extractors/emoca" && pixi run --frozen python extract.py \
      --videos_dir "$vids" --insightface_dir "$insf" --out_dir "$emoc" \
      --batch_size "$EMOCA_BS" --num_workers "$EMOCA_NW" )

  echo ">> [3/5] $s: liveportrait -> $lp"
  ext liveportrait --videos_dir "$vids" --insightface_dir "$insf" --out_dir "$lp" \
      --batch_size "$LP_BS" --num_workers "$LP_NW"
done

# ---- audio features: GT wavs only (all models use GT audio; R-DGG reads GT) ----
WAVS="$CORPUS/GT/${SPLIT}_wav"
[ -d "$WAVS" ] || { echo "!! [3/5] missing GT wavs: $WAVS (run pipeline/1_download.sh first)" >&2; exit 1; }

echo ">> [3/5] GT: hubert -> $CORPUS/GT/${SPLIT}_hubert"
ext hubert --wavs_dir "$WAVS" --out_dir "$CORPUS/GT/${SPLIT}_hubert"

echo ">> [3/5] GT: vad -> $CORPUS/GT/${SPLIT}_vad"
ext vad --wavs_dir "$WAVS" --out_dir "$CORPUS/GT/${SPLIT}_vad"

echo ">> [3/5] done: ${SRCS[*]} (+ GT hubert/vad)"
