#!/usr/bin/env bash
# [5/5] Classic dyadic-listener metrics (secondary comparison): rpcc / tlcc / pfd
# / sid / variance, on EMOCA + LP features, in FULL and LISTENING (listener-silent
# VAD) variants, then aggregate into the SI-184 tables.
#
# Writes each model's CSVs to <corpus>/<model>/_metrics/<metric>_<src>[_listening]__LS_AL.csv
# (GT is a pseudo-model: rPCC/PFD ~ 0 sanity anchor; TLCC/SID/Var give the GT baseline).
#   Speaker = GT everywhere; target_fps = 25 (matches the models; downsamples the 30fps GT).
#   Paired metrics (rpcc/tlcc/pfd): --pairing from_file:pairs184.txt.
#   Single-source metrics (sid/variance): --stems pairs184.txt (col1 = listener stems).
#   "listening" = LISTENER-silent VAD mask (rpcc/tlcc/sid/variance: --vad_mode silence;
#                 pfd: --mask_listener silence).
#
# The five classic metrics and rdgg share ONE pixi env (rdgg/pixi.toml). set -euo
# pipefail, but an individual metric failure is logged and the sweep continues
# (non-zero exit at the end if anything failed). Repo-relative; override corpus
# with $CORPUS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="${CORPUS:-$REPO/data}"
SPLIT="${SPLIT:-LS_AL}"
CM="$REPO/classic_metrics"
MANIFEST="$REPO/rdgg/pixi.toml"   # the six metrics share this one pixi env
GT="$CORPUS/GT"
VAD="$GT/${SPLIT}_vad"
PAIRS="$CORPUS/pairs184.txt"
STEMS="$CORPUS/pairs184.txt"          # col1 = listener stems (same file as PAIRS)
PAIRING="from_file:$PAIRS"
TF=25

for f in "$PAIRS" "$STEMS"; do
  [ -f "$f" ] || { echo "!! [5/5] missing pairs184.txt: $f" >&2; exit 1; }
done
[ -d "$VAD" ] || { echo "!! [5/5] missing GT VAD (needed for the listening variant): $VAD" >&2; exit 1; }

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(GT AVTR-1 AvatarForcing FLOAT SoulX_FlashHead_Lite SoulX_FlashHead_Pro ditto dystream)

FAIL=0
# run <label> <metric> <args...> : quiet on success; loud + counted on failure; never aborts the sweep.
run() {
  local label="$1"; local metric="$2"; shift 2
  if ( cd "$CM/$metric" && pixi run --manifest-path "$MANIFEST" python compute.py "$@" ) >/tmp/_cm_$$.log 2>&1; then
    echo "   ok    $label"
  else
    FAIL=$((FAIL + 1))
    echo "   !! FAILED $label" >&2
    echo "      CMD: (cd $CM/$metric && pixi run --manifest-path $MANIFEST python compute.py $*)" >&2
    tail -n 12 /tmp/_cm_$$.log | sed 's/^/      | /' >&2
  fi
  rm -f /tmp/_cm_$$.log
}

model_dir() {  # <model> <suffix>  ->  feature dir
  local m="$1" sfx="$2"
  if [ "$m" = GT ]; then echo "$GT/${SPLIT}_$sfx"; else echo "$CORPUS/$m/${SPLIT}_$sfx"; fi
}

for m in "${MODELS[@]}"; do
  for src in emoca lp; do
    if [ "$src" = lp ]; then sfx=liveportrait; else sfx=emoca; fi
    gd="$GT/${SPLIT}_$sfx"                        # GT speaker / listener / features
    md="$(model_dir "$m" "$sfx")"                # model listener / pred / features
    outd="$(dirname "$md")/_metrics"             # <corpus>/<model>/_metrics
    if [ ! -d "$md" ]; then
      echo "   skip $m/$src: missing $md"; continue
    fi
    echo ">> [5/5] $m / $src"

    # -------- FULL variant (no mask) --------
    run "$m $src rpcc full" rpcc \
        --speaker "$gd" --gt_listener "$gd" --gen_listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF \
        --out "$outd/rpcc_${src}__${SPLIT}.csv" --overwrite
    run "$m $src tlcc full" tlcc \
        --speaker "$gd" --listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF \
        --max_lag_sec 2 --adaptive_max_lag \
        --out "$outd/tlcc_${src}__${SPLIT}.csv" --overwrite
    run "$m $src pfd full" pfd \
        --gt_speaker "$gd" --gt_listener "$gd" --pred_speaker "$gd" --pred_listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF --label "$m" \
        --out "$outd/pfd_${src}__${SPLIT}.csv" --overwrite
    run "$m $src sid full" sid \
        --gt_features "$gd" --pred_features "$md" \
        --feature_source "$src" --stems "$STEMS" --target_fps $TF --label "$m" \
        --out "$outd/sid_${src}__${SPLIT}.csv" --overwrite
    run "$m $src variance full" variance \
        --features_dir "$md" \
        --feature_source "$src" --stems "$STEMS" --target_fps $TF --label "$m" \
        --out "$outd/variance_${src}__${SPLIT}.csv" --overwrite

    # -------- LISTENING variant (listener-silent VAD mask) --------
    run "$m $src rpcc listening" rpcc \
        --speaker "$gd" --gt_listener "$gd" --gen_listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF \
        --vad_dir "$VAD" --vad_mode silence \
        --out "$outd/rpcc_${src}_listening__${SPLIT}.csv" --overwrite
    run "$m $src tlcc listening" tlcc \
        --speaker "$gd" --listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF \
        --max_lag_sec 2 --adaptive_max_lag \
        --vad_dir "$VAD" --vad_mode silence \
        --out "$outd/tlcc_${src}_listening__${SPLIT}.csv" --overwrite
    run "$m $src pfd listening" pfd \
        --gt_speaker "$gd" --gt_listener "$gd" --pred_speaker "$gd" --pred_listener "$md" \
        --feature_source "$src" --pairing "$PAIRING" --target_fps $TF --label "$m" \
        --vad_dir "$VAD" --mask_listener silence \
        --out "$outd/pfd_${src}_listening__${SPLIT}.csv" --overwrite
    run "$m $src sid listening" sid \
        --gt_features "$gd" --pred_features "$md" \
        --feature_source "$src" --stems "$STEMS" --target_fps $TF --label "$m" \
        --vad_dir "$VAD" --vad_mode silence \
        --out "$outd/sid_${src}_listening__${SPLIT}.csv" --overwrite
    run "$m $src variance listening" variance \
        --features_dir "$md" \
        --feature_source "$src" --stems "$STEMS" --target_fps $TF --label "$m" \
        --vad_dir "$VAD" --vad_mode silence \
        --out "$outd/variance_${src}_listening__${SPLIT}.csv" --overwrite
  done
done

echo ">> [5/5] aggregate SI-184 tables"
pixi run --manifest-path "$MANIFEST" python "$CM/aggregate_si184.py" --corpus "$CORPUS"

if [ "$FAIL" -gt 0 ]; then
  echo "!! [5/5] $FAIL metric run(s) FAILED — see the log lines above" >&2
  exit 1
fi
echo ">> [5/5] done"
