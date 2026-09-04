#!/usr/bin/env bash
# [2/5] Generate listener/talking-head videos for the 6 models into the corpus
# layout: <corpus>/<model>/LS_AL/<stem>.mp4.
#
# Each model ships its OWN environment + weights (see models/README.md); a single
# shell cannot activate all of them at once. So run this ONE MODEL AT A TIME with
# that model's env active, e.g.:
#     conda activate ditto && bash pipeline/2_generate.sh ditto
#     conda activate FLOAT && bash pipeline/2_generate.sh float
#     (cd models/avtr-1 && pixi shell -e renderer)  # then: bash ../../pipeline/2_generate.sh avtr1
# With no model args it attempts ALL of them under $PY (default: python) — only
# useful if a single env somehow satisfies every model.
#
# The python launcher is $PY (default "python"); override per model as needed,
# e.g.  PY="conda run -n ditto python" bash pipeline/2_generate.sh ditto
# AVTR-1 is special-cased to its documented `pixi run -e renderer` invocation.
#
# Repo-relative; override corpus with $CORPUS. Does NOT change any model/CFG/seed
# (those live inside each driver) — it only wires the corpus paths.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="${CORPUS:-$REPO/data}"
GEN="$REPO/models/generate"
PY="${PY:-python}"

REFS="$CORPUS/references"
WAVS="$CORPUS/GT/LS_AL_wav"
GTVID="$CORPUS/GT/LS_AL"               # partner video frames (AvatarForcing only)
PAIRS="$CORPUS/pairs184.txt"
STEMS="$CORPUS/pairs184.txt"          # col1 = listener stems (same file as PAIRS)

need() { [ -e "$1" ] || { echo "!! [2/5] missing required input: $1 ($2)" >&2; exit 1; }; }
need "$REFS"  "reference portraits (data/references/)"
need "$WAVS"  "driving audio (run pipeline/1_download.sh first)"
need "$STEMS" "listener stems (pairs184.txt)"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(ditto float soulx avatarforcing dystream avtr1)

for m in "${MODELS[@]}"; do
  case "$m" in
    ditto)
      out="$CORPUS/ditto/LS_AL"
      echo ">> [2/5] ditto (non-dyadic) env: conda activate ditto  ->  $out"
      "$PY" "$GEN/ditto.py" --references "$REFS" --wav_dir "$WAVS" \
            --stems_file "$STEMS" --out_dir "$out" ;;
    float)
      out="$CORPUS/FLOAT/LS_AL"
      echo ">> [2/5] float (non-dyadic) env: conda activate FLOAT  ->  $out"
      "$PY" "$GEN/float.py" --references "$REFS" --wav_dir "$WAVS" \
            --stems_file "$STEMS" --out_dir "$out" ;;
    soulx)
      echo ">> [2/5] soulx-flashhead lite+pro (non-dyadic) env: conda activate flashhead"
      "$PY" "$GEN/soulx.py" --variant lite --references "$REFS" --wav_dir "$WAVS" \
            --stems_file "$STEMS" --out_dir "$CORPUS/SoulX_FlashHead_Lite/LS_AL"
      "$PY" "$GEN/soulx.py" --variant pro  --references "$REFS" --wav_dir "$WAVS" \
            --stems_file "$STEMS" --out_dir "$CORPUS/SoulX_FlashHead_Pro/LS_AL" ;;
    dystream)
      out="$CORPUS/dystream/LS_AL"
      need "$PAIRS" "dyadic pairing (dystream)"
      echo ">> [2/5] dystream (dyadic) env: conda activate dystream_py11  ->  $out"
      "$PY" "$GEN/dystream.py" --references "$REFS" --wav_dir "$WAVS" --pairs "$PAIRS" \
            --stems_file "$STEMS" --out_dir "$out" ;;
    avatarforcing)
      out="$CORPUS/AvatarForcing/LS_AL"
      need "$PAIRS" "dyadic pairing (avatarforcing)"
      need "$GTVID" "partner video frames (avatarforcing --gt_video_dir)"
      echo ">> [2/5] avatarforcing (dyadic + partner video) env: conda activate avatarforcing  ->  $out"
      "$PY" "$GEN/avatarforcing.py" --reference_dir "$REFS" --wav_dir "$WAVS" \
            --gt_video_dir "$GTVID" --pairs "$PAIRS" --stems_file "$STEMS" --out_dir "$out" ;;
    avtr1)
      out="$CORPUS/AVTR-1/LS_AL"
      need "$PAIRS" "dyadic pairing (avtr1)"
      # Run under the in-repo submodule's renderer pixi env (the reviewer installs it + builds the
      # TensorRT engines there: pixi install -e renderer -> pixi run download -> build-trt-engines).
      echo ">> [2/5] avtr-1 (dyadic) env: models/avtr-1 pixi renderer  ->  $out"
      ( cd "$REPO/models/avtr-1" && pixi run -e renderer python "$GEN/avtr1.py" \
            --references "$REFS" --wav_dir "$WAVS" --pairs "$PAIRS" \
            --stems_file "$STEMS" --out_dir "$out" ) ;;
    *)
      echo "!! [2/5] unknown model '$m' (want: ditto float soulx avatarforcing dystream avtr1)" >&2
      exit 2 ;;
  esac
done

echo ">> [2/5] done: ${MODELS[*]}"
