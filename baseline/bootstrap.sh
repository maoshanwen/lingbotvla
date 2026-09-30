#!/usr/bin/env bash
# Rebuild the two venvs and download all non-Git training/evaluation artifacts.
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
TRAIN_ENV="$ROOT/.venvs/lingbot"
SIM_ENV="$ROOT/.venvs/robotwin"
export PYTHONNOUSERSITE=1
export UV_LINK_MODE="${UV_LINK_MODE:-auto}"

usage() {
  cat <<'EOF'
Usage: bash baseline/bootstrap.sh [--only-deps | --only-download]

Default: clone pinned code, create both venvs, install dependencies, download
the LingBot base model, Qwen/MoGe teachers, 50 clean ZIPs, and RoboTwin assets.
Re-running resumes downloads and keeps existing virtual environments.
EOF
}

phase=all
case "${1:-}" in
  "") ;;
  --only-deps) phase=deps ;;
  --only-download) phase=download ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi

for executable in git curl; do
  command -v "$executable" >/dev/null || { echo "Missing command: $executable" >&2; exit 2; }
done
if [[ "$phase" != download ]]; then
  if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
  command -v nvcc >/dev/null || { echo "CUDA toolkit nvcc required to build cuRobo" >&2; exit 2; }
fi
if ! command -v uv >/dev/null; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
  sh "$installer"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || { echo "uv installation failed" >&2; exit 1; }
  rm -f "$installer"
  trap - EXIT
fi

ensure_repo() {
  local directory="$1" url="$2" revision="$3"
  if [[ ! -e "$directory" ]]; then
    git clone "$url" "$directory"
  fi
  if [[ ! -e "$directory/.git" ]]; then
    echo "Expected a Git checkout at $directory" >&2
    exit 1
  fi
  if ! git -C "$directory" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    git -C "$directory" fetch origin "$revision"
  fi
  if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
    if [[ -n "$(git -C "$directory" status --porcelain)" ]]; then
      echo "Refusing to change dirty checkout: $directory" >&2
      exit 1
    fi
    git -C "$directory" checkout --detach "$revision"
  fi
}

ensure_repo "$ROOT/lingbot-vla-v2" \
  https://github.com/Robbyant/lingbot-vla-v2.git be969b8fd117fb70550c5d4bf4bc328211b5b1b6
ensure_repo "$ROOT/RoboTwin" \
  https://github.com/RoboTwin-Platform/RoboTwin.git 13c3c47ff4312dd62484bcd51be034af55c062d1

if [[ "$phase" == download && ! -x "$TRAIN_ENV/bin/python" ]]; then
  echo "Run bootstrap.sh --only-deps before --only-download" >&2
  exit 2
fi

if [[ "$phase" != download ]]; then
  if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then uv venv --python 3.12 "$TRAIN_ENV"; fi
  if [[ ! -x "$SIM_ENV/bin/python" ]]; then uv venv --python 3.11 "$SIM_ENV"; fi

  echo "Installing LingBot training/inference dependencies"
  uv pip install --python "$TRAIN_ENV/bin/python" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$TRAIN_ENV/bin/python" torchdata==0.11.0 torchcodec==0.6.0
  uv pip install --python "$TRAIN_ENV/bin/python" -r "$BASELINE_DIR/requirements.txt"
  uv pip install --python "$TRAIN_ENV/bin/python" -r "$ROOT/lingbot-vla-v2/requirements-depth.txt"
  uv pip install --python "$TRAIN_ENV/bin/python" -r "$BASELINE_DIR/requirements.txt"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps numpydantic==1.9.0
  flash_wheel="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps "$flash_wheel"

  mkdir -p "$ROOT/.deps"
  ensure_repo "$ROOT/.deps/lerobot" https://github.com/huggingface/lerobot.git \
    a445d9c9da6bea99a8972daa4fe1fdd053d711d2
  ensure_repo "$ROOT/.deps/utils3d" https://github.com/EasternJournalist/utils3d.git \
    3fab839f0be9931dac7c8488eb0e1600c236e183
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps "$ROOT/.deps/lerobot"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps "$ROOT/.deps/utils3d"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps -e "$ROOT/lingbot-vla-v2"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps -e \
    "$ROOT/lingbot-vla-v2/lingbotvla/models/vla/vision_models/lingbot-depth"
  uv pip install --python "$TRAIN_ENV/bin/python" --no-deps -e \
    "$ROOT/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe"

  echo "Installing RoboTwin simulation/data-processing dependencies"
  uv pip install --python "$SIM_ENV/bin/python" torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$SIM_ENV/bin/python" -r "$BASELINE_DIR/requirements-sim.txt"
  uv pip install --python "$SIM_ENV/bin/python" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0
  if [[ ! -f "$ROOT/RoboTwin/envs/curobo/setup.py" ]]; then
    git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git \
      "$ROOT/RoboTwin/envs/curobo"
  fi
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  # Existing RoboTwin bundles may contain cuRobo source without its .git metadata.
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-0.7.8}"
  if ! "$SIM_ENV/bin/python" - <<'PY'
import torch
from importlib.metadata import version
assert version('nvidia-curobo') == '0.7.8'
from curobo.curobolib import lbfgs_step_cu, kinematics_fused_cu, line_search_cu, tensor_step_cu, geom_cu
PY
  then
    uv pip install --python "$SIM_ENV/bin/python" --no-build-isolation --no-deps \
      -e "$ROOT/RoboTwin/envs/curobo"
  fi
  ffmpeg_exe="$("$SIM_ENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  ln -sfn "$ffmpeg_exe" "$SIM_ENV/bin/ffmpeg"

  "$TRAIN_ENV/bin/python" - <<'PY'
import torch, flash_attn, lerobot, moge, utils3d
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.version.cuda == '12.8', torch.version.cuda
print('training environment:', torch.__version__, 'CUDA:', torch.version.cuda,
      'GPU visible:', torch.cuda.is_available())
PY
  "$SIM_ENV/bin/python" - <<'PY'
import torch, sapien, mplib, curobo, warp
assert torch.__version__.split('+')[0] == '2.4.1'
assert torch.version.cuda == '12.4', torch.version.cuda
print('simulation environment:', torch.__version__, 'CUDA:', torch.version.cuda,
      'GPU visible:', torch.cuda.is_available())
PY
fi

if [[ "$phase" != deps ]]; then
  "$TRAIN_ENV/bin/python" "$BASELINE_DIR/download_artifacts.py"
  (cd "$ROOT/RoboTwin" && "$SIM_ENV/bin/python" script/update_embodiment_config_path.py)
  "$TRAIN_ENV/bin/python" "$BASELINE_DIR/prepare_data.py" verify
fi

echo "Bootstrap complete. Start the full workflow with:"
echo "  TEAM_ID=<actual-id> EVAL_COMPAT=1 bash baseline/run_all.sh"
