#!/usr/bin/env bash
# Post-clone setup: bring up EVERY environment for the R-DGG evaluation.
#
# Run this once from a fresh clone:
#   git clone --recurse-submodules https://github.com/Donfa1con/R-DGG.git   # (or run section 1 to fetch submodules)
#   bash setup.sh
#
# The script is idempotent. It skips an environment that is already installed, so
# you can run it again after you add a gated asset. It is fail-loud: it stops with
# a clear message when a required, license-gated asset is absent. It never skips a
# gated step in silence.
#
# Sections:
#   1. submodules + generation patches   (owner-verified)
#   2. extractor envs                    (owner-verified)
#   3. metrics env                       (owner-verified)
#   4. model generation envs             (avtr-1 owner-verified; the five conda
#                                         models are documented-only)
#
# The final summary prints anything that still needs a gated download. The exit
# code is non-zero while a required gated asset is absent.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---- helpers ----------------------------------------------------------------
GATED=()   # license-gated assets that are still absent; the run exits non-zero
TODO=()    # documented-only steps that did not complete on this host

have()  { command -v "$1" >/dev/null 2>&1; }
die()   { printf '\n!! FATAL: %s\n' "$*" >&2; exit 1; }
gate()  { printf '\n!! GATED: %s\n' "$*" >&2; GATED+=("$*"); }
todo()  { printf '\n.. TODO:  %s\n' "$*" >&2; TODO+=("$*"); }

# pixi runs the extractor envs, the metrics env, and the avtr-1 env.
have pixi || die "pixi is not on PATH. Install pixi (https://pixi.sh), then re-run: bash setup.sh"

# ---- section 1: submodules + patches (owner-verified) -----------------------
echo ">> [1/4] initialising submodules"
git -C "$ROOT" submodule update --init --recursive || die "submodule update failed"

echo ">> [1/4] applying generation patches"
shopt -s nullglob
for patch in "$ROOT"/models/patches/*.patch; do
  m="$(basename "$patch" .patch)"
  dir="$ROOT/models/$m"
  if [ ! -d "$dir" ]; then
    echo "   !! submodule dir missing: models/$m — is it a declared submodule?"
    continue
  fi
  if git -C "$dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "   models/$m: already patched, skipping"
  elif git -C "$dir" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$dir" apply "$patch" && echo "   models/$m: patched"
  else
    echo "   !! models/$m: patch does not apply cleanly — inspect $patch"
  fi
done

# ---- section 2: extractor envs (owner-verified) -----------------------------
# Each extractor owns its own pixi env under extractors/<name>/.pixi. These blocks
# are VERIFIED on the reference host. A re-run skips a present env.
echo
echo ">> [2/4] extractor envs"

# insightface — keep the onnxruntime CUDA provider off the CPU fallback. The
# activation hook puts the env's CUDA libs on the loader path.
(
  cd "$ROOT/extractors/insightface" || exit 1
  if [ -d .pixi/envs/default ]; then
    echo "   insightface: env present, skipping install"
  else
    pixi install || exit 1
  fi
  mkdir -p .pixi/envs/default/etc/conda/activate.d
  echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"' \
    > .pixi/envs/default/etc/conda/activate.d/zz_cuda_ld.sh
) || die "insightface env setup failed"

# liveportrait — vendored source + motion-extractor weights.
(
  cd "$ROOT/extractors/liveportrait" || exit 1
  if [ -d .pixi/envs/default ]; then
    echo "   liveportrait: env present, skipping install"
  else
    pixi install || exit 1
  fi
  [ -d _vendor/src ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait _vendor || exit 1
  pixi run download-weights || exit 1
) || die "liveportrait env setup failed"

# emoca — inferno editable dependency + pytorch3d build. MUST use --frozen: the
# lock is authoritative, and a plain install re-resolves and fails on
# onnxruntime-gpu==1.16.2. The FLAME + EMOCA_v2 + DECA + FaceRecognition assets
# (~3 GB) must sit under extractors/emoca/_vendor/inferno/assets/. FLAME is
# registration-gated at flame.is.tue.mpg.de.
(
  cd "$ROOT/extractors/emoca" || exit 1
  mkdir -p _vendor
  [ -e _vendor/inferno ] || git clone https://github.com/radekd91/inferno _vendor/inferno || exit 1
  if [ -d .pixi/envs/default ]; then
    echo "   emoca: env present, skipping install"
  else
    pixi install --frozen || exit 1
  fi
  pixi run --frozen setup || exit 1   # build-pytorch3d + patch-inferno (both idempotent)
) || die "emoca env setup failed"

# Fail loud when the license-gated FLAME assets are absent.
if [ ! -d "$ROOT/extractors/emoca/_vendor/inferno/assets/FLAME" ]; then
  gate "emoca needs the FLAME assets. Register at https://flame.is.tue.mpg.de , then download FLAME + EMOCA_v2 + DECA + FaceRecognition (via inferno's inferno_apps/EMOCA/demos/download_assets.sh) into extractors/emoca/_vendor/inferno/assets/{FLAME,EMOCA,DECA,FaceRecognition}, then re-run: bash setup.sh"
fi

# hubert, vad — plain pixi envs.
for e in hubert vad; do
  (
    cd "$ROOT/extractors/$e" || exit 1
    if [ -d .pixi/envs/default ]; then
      echo "   $e: env present, skipping install"
    else
      pixi install || exit 1
    fi
  ) || die "$e env setup failed"
done

# ---- section 3: metrics env (owner-verified) --------------------------------
# rdgg/ is the single shared env for R-DGG and the five classic metrics.
echo
echo ">> [3/4] metrics env (rdgg)"
(
  cd "$ROOT/rdgg" || exit 1
  if [ -d .pixi/envs/default ]; then
    echo "   rdgg: env present, skipping install"
  else
    pixi install || exit 1
  fi
) || die "rdgg metrics env setup failed"

# ---- section 4: model generation envs ---------------------------------------
echo
echo ">> [4/4] model generation envs"

# avtr-1 — pixi env + gated weights + local TensorRT engine build. OWNER-VERIFIED.
# The weights are non-commercial-research-license-gated on HuggingFace.
(
  cd "$ROOT/models/avtr-1" || exit 1
  if [ -d .pixi/envs/renderer ]; then
    echo "   avtr-1: renderer env present, skipping install"
  else
    pixi install -e renderer || exit 1
  fi
) || die "avtr-1 pixi install failed"

if ( cd "$ROOT/models/avtr-1" && pixi run download ); then
  ( cd "$ROOT/models/avtr-1" && pixi run build-trt-engines ) \
    || die "avtr-1 TensorRT engine build failed"
  echo "   avtr-1: weights present + engines built"
else
  gate "avtr-1 weights are HuggingFace non-commercial-research-license-gated (see models/avtr-1/LICENSE-MODEL.md). Accept the license on HuggingFace, run 'hf auth login' or set HF_TOKEN, then re-run: (cd models/avtr-1 && pixi run download && pixi run build-trt-engines)"
fi

# The five conda models below are DOCUMENTED-ONLY. Their env + weight commands come
# from each submodule README and models/README.md. They are not verified on this
# host. Each step is idempotent and best-effort: a failure is recorded in the final
# summary, and it does not stop the verified sections above. The env-creation
# recipes (python version, deps) come from each submodule's own README.
if ! have conda; then
  todo "conda is not on PATH. The five model generation envs (ditto, FLOAT, flashhead, avatarforcing, dystream) need conda. Install Miniconda, then re-run: bash setup.sh"
else
  # Capture the env names first, then match, so 'set -o pipefail' does not trip on
  # a SIGPIPE from 'conda env list' when grep exits early.
  conda_env_exists() {
    local names
    names="$(conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}')"
    grep -qxF "$1" <<<"$names"
  }
  # ensure_conda_env <env-name> <create-command...>
  ensure_conda_env() {
    local env="$1"; shift
    if conda_env_exists "$env"; then
      echo "   conda env '$env': present, skipping create"
      return 0
    fi
    echo "   conda env '$env': creating (documented-only)"
    "$@"
  }
  dir_nonempty() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

  # ditto (documented-only) — env from environment.yaml; weights git-clone.
  if ensure_conda_env ditto conda env create -f "$ROOT/models/ditto/environment.yaml"; then
    if dir_nonempty "$ROOT/models/ditto/checkpoints"; then
      echo "   ditto: checkpoints present, skipping"
    else
      git clone https://huggingface.co/digital-avatar/ditto-talkinghead "$ROOT/models/ditto/checkpoints" \
        || todo "ditto weights: git clone https://huggingface.co/digital-avatar/ditto-talkinghead models/ditto/checkpoints"
    fi
  else
    todo "ditto env: conda env create -f models/ditto/environment.yaml"
  fi

  # float (documented-only) — env FLOAT (python 3.8.5) + environments.sh; weights via download_checkpoints.sh.
  if ensure_conda_env FLOAT conda create -y -n FLOAT python=3.8.5; then
    ( cd "$ROOT/models/float" && conda run -n FLOAT bash environments.sh ) \
      || todo "float deps: (cd models/float && conda run -n FLOAT bash environments.sh)"
    if [ -f "$ROOT/models/float/checkpoints/float.pth" ]; then
      echo "   float: checkpoint present, skipping"
    else
      ( cd "$ROOT/models/float" && conda run -n FLOAT bash download_checkpoints.sh ) \
        || todo "float weights: (cd models/float && conda run -n FLOAT bash download_checkpoints.sh)"
    fi
  else
    todo "float env: conda create -n FLOAT python=3.8.5 (+ bash environments.sh)"
  fi

  # soulx-flashhead (documented-only) — env flashhead (python 3.10) + requirements; weights via huggingface-cli.
  if ensure_conda_env flashhead conda create -y -n flashhead python=3.10; then
    ( cd "$ROOT/models/soulx-flashhead" && conda run -n flashhead pip install -r requirements.txt ) \
      || todo "soulx deps: (cd models/soulx-flashhead && conda run -n flashhead pip install -r requirements.txt)"
    if dir_nonempty "$ROOT/models/soulx-flashhead/models/SoulX-FlashHead-1_3B"; then
      echo "   soulx-flashhead: checkpoint present, skipping"
    else
      conda run -n flashhead huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir "$ROOT/models/soulx-flashhead/models/SoulX-FlashHead-1_3B" \
        || todo "soulx weights: huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/soulx-flashhead/models/SoulX-FlashHead-1_3B"
      conda run -n flashhead huggingface-cli download facebook/wav2vec2-base-960h \
        --local-dir "$ROOT/models/soulx-flashhead/models/wav2vec2-base-960h" \
        || todo "soulx wav2vec2: huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/soulx-flashhead/models/wav2vec2-base-960h"
    fi
  else
    todo "soulx env: conda create -n flashhead python=3.10 (+ pip install -r requirements.txt)"
  fi

  # avatarforcing (documented-only) — env avatarforcing (python 3.10) + environment.sh; weights via download_weights.sh.
  if ensure_conda_env avatarforcing conda create -y -n avatarforcing python=3.10; then
    ( cd "$ROOT/models/avatarforcing" && conda run -n avatarforcing bash environment.sh ) \
      || todo "avatarforcing deps: (cd models/avatarforcing && conda run -n avatarforcing bash environment.sh)"
    if [ -f "$ROOT/models/avatarforcing/pretrained_dir/flow_transformer.pth" ]; then
      echo "   avatarforcing: weights present, skipping"
    else
      ( cd "$ROOT/models/avatarforcing" && conda run -n avatarforcing bash download_weights.sh ) \
        || todo "avatarforcing weights: (cd models/avatarforcing && conda run -n avatarforcing bash download_weights.sh)"
    fi
  else
    todo "avatarforcing env: conda create -n avatarforcing python=3.10 (+ bash environment.sh)"
  fi

  # dystream (documented-only) — env dystream_py11 (python 3.11) + requirements; weights git-clone.
  # The last.ckpt (~7.7 GB) is HuggingFace-gated → fail loud when the checkpoints are absent.
  if ensure_conda_env dystream_py11 conda create -y -n dystream_py11 python=3.11; then
    ( cd "$ROOT/models/dystream" && conda run -n dystream_py11 pip install -r requirements.txt ) \
      || todo "dystream deps: (cd models/dystream && conda run -n dystream_py11 pip install -r requirements.txt)"
    if dir_nonempty "$ROOT/models/dystream/checkpoints"; then
      echo "   dystream: checkpoints present, skipping"
    else
      tmp="$ROOT/models/dystream/_hf_DyStream"
      if [ ! -e "$tmp" ] && git clone https://huggingface.co/robinwitch/DyStream "$tmp"; then
        mv "$tmp/checkpoints" "$ROOT/models/dystream/" 2>/dev/null
        mv "$tmp/tools"       "$ROOT/models/dystream/" 2>/dev/null
        rm -rf "$tmp"
      fi
      dir_nonempty "$ROOT/models/dystream/checkpoints" \
        || gate "dystream last.ckpt (~7.7 GB) is HuggingFace-gated. Accept the license and download https://huggingface.co/robinwitch/DyStream , then move its checkpoints/ and tools/ into models/dystream/ (see models/README.md), then re-run: bash setup.sh"
    fi
  else
    todo "dystream env: conda create -n dystream_py11 python=3.11 (+ pip install -r requirements.txt)"
  fi
fi

# ---- final summary ----------------------------------------------------------
echo
echo "=================================================================="
echo " SETUP COMPLETE — the owner-verified envs are installed:"
echo "   extractors (insightface, liveportrait, emoca, hubert, vad),"
echo "   the rdgg metrics env, and the avtr-1 pixi env."
echo "=================================================================="

if [ ${#TODO[@]} -gt 0 ]; then
  echo
  echo "Documented-only steps that did not complete on this host:"
  for t in "${TODO[@]}"; do echo "  - $t"; done
fi

if [ ${#GATED[@]} -gt 0 ]; then
  echo
  echo "GATED downloads still needed (accept the license, set the token, re-run):"
  for g in "${GATED[@]}"; do echo "  - $g"; done
  echo
  echo "Re-run 'bash setup.sh' after you add the gated assets (it is idempotent)."
  exit 1
fi

echo
echo "Next: run the pipeline in order —"
echo "  pipeline/1_download.sh -> 2_generate.sh -> 3_extract.sh -> 4_rdgg.sh -> 5_classic.sh"
exit 0
