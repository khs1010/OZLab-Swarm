#!/usr/bin/env bash
# Fresh Linux x86_64 CUDA environment; never sync/remove an existing GPU stack.
# Downloads public packages/Python only. No model weights or model API requests.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-install}"
case "$MODE" in
  --help|-h)
    cat <<'EOF'
Usage: bash scripts/setup_qwen_gpu.sh [--plan|--bootstrap-only|--resolve-only]
Default: create/check an isolated Python environment and install locked GPU+bank
dependencies, then check CUDA/NF4 using tiny random tensors (no model download).
--plan: print selected paths/versions only; no network or installation.
--bootstrap-only: check/install pinned uv only; no GPU dependencies or weights.
--resolve-only: validate both locked dependency sets using metadata; no GPU install.
Variables: QWEN_ENV_PATH=/content/ozlab-qwen-env, QWEN_PYTHON_VERSION=3.12.14,
QWEN_BOOTSTRAP_PYTHON=python3. Existing incompatible environments are not replaced.
The user's observed success manifest used Python 3.12.16; select that patch version
explicitly to match it. Dependencies are locked for Linux x86_64 / Python 3.12.
EOF
    exit 0 ;;
  --plan|--bootstrap-only|--resolve-only|install) ;;
  *) echo 'Unknown option; use --help.' >&2; exit 2 ;;
esac
QWEN_ENV_PATH="${QWEN_ENV_PATH:-/content/ozlab-qwen-env}"
QWEN_PYTHON_VERSION="${QWEN_PYTHON_VERSION:-3.12.14}"
QWEN_BOOTSTRAP_PYTHON="${QWEN_BOOTSTRAP_PYTHON:-python3}"
[[ "$QWEN_ENV_PATH" = /* ]] || { echo 'QWEN_ENV_PATH must be absolute.' >&2; exit 2; }
[[ "$QWEN_PYTHON_VERSION" =~ ^3\.12\.[0-9]+$ ]] || { echo 'Select a Python 3.12 patch version.' >&2; exit 2; }
printf 'Environment: %s\nPython: %s\nuv: 0.12.23\nTorch: 2.9.1+cu128\n' "$QWEN_ENV_PATH" "$QWEN_PYTHON_VERSION"
if [[ "$MODE" == --plan ]]; then
  echo 'Plan only: no environment, downloads, CUDA checks, or model runs performed.'
  exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This lock targets Linux x86_64.' >&2; exit 1; }
if [[ "$MODE" == install ]]; then
  command -v nvidia-smi >/dev/null || { echo 'Attach an NVIDIA GPU runtime first; no GPU packages installed.' >&2; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo 'The host NVIDIA driver must support the CUDA 12.8 wheel. This script does not install drivers.'
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.cache/qwen-python}"
UV_TOOLS="$ROOT/.cache/qwen-uv-0.12.23"
uv_version_matches() {
  local name version details
  # Official builds can append a hash/date/platform. Pin the version token,
  # rather than rejecting correct releases because their build text differs.
  read -r name version details <<< "$1"
  [[ "$name" == uv && "$version" == 0.12.23 ]]
}
UV_BIN="$(command -v uv || true)"
if [[ -n "$UV_BIN" ]]; then
  if ! UV_REPORTED="$("$UV_BIN" --version 2>&1)" || ! uv_version_matches "$UV_REPORTED"; then
    printf 'PATH uv is not the pinned release; using isolated tools. Observed: %s\n' "$UV_REPORTED"
    UV_BIN=""
  fi
fi
if [[ -z "$UV_BIN" ]]; then
  mkdir -p "$UV_TOOLS"
  cat >"$UV_TOOLS/bootstrap-requirements.txt" <<'EOF'
uv==0.12.23 \
    --hash=sha256:02f2a0fd1cb90b5fc06b52e8cbc3169bdc4e53804e9d66ba0ad516be767424f4 \
    --hash=sha256:08b2fc8e250efdfc1e47da0483f3e19f433bb0934282361817f61c0b19cb3674 \
    --hash=sha256:11d18c1589d7907376b4f3fa31028e46cc609ac4b7d83eabc2b04cd8ca1433cc \
    --hash=sha256:4951286e68f116fa41897ecf8e248cc2836cb013be6d4f37afae4049f9ef4df7 \
    --hash=sha256:565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff \
    --hash=sha256:5b07cbea635177da0d6b902bcde9ce0304c1fa5b49a37b27ef7aa021c5c31697 \
    --hash=sha256:7a919e3420adf4cc03895b404af655c7229f646bffabbd9a16979683772a7d34 \
    --hash=sha256:855232fd95d456bd5ee17e262f73112967fbbb2803e745f1911c08ceb042e641 \
    --hash=sha256:895137194d242cc8075c3006288b485af54feeae68512f4ffc96f75b27543cc0 \
    --hash=sha256:8f021dfbe5b6081021dd47fd8372cfba386fc5d39161e73d4f33cff90ce1760c \
    --hash=sha256:9346c741175c7fc0d380f30fcfcb8d5e13f5f900b30536230c8d34208524e47a \
    --hash=sha256:99197c022afbd1edd0a295e9fdac5078c61eddf2e77657a77294cfe8ea0d94b9 \
    --hash=sha256:9a65258535db432df51ca7d26338c01431820b94b634762f5718fc55895f974e \
    --hash=sha256:b03b1c35b5f301e888671eb5d0e74b8789558905db1d411cb8e07ea4ec320268 \
    --hash=sha256:b70220be9056bd05908770179101a53687eb5c89f72a3edc1e20df0cd3f157c2 \
    --hash=sha256:c4d236bb220856027ba8e28ef693cac472ee2a740a61faef1ac2001165c7b3d5 \
    --hash=sha256:e952162a5be6847deaccf3fd5b41a628506fd9ff1ea4363b8c975bd6f927d832 \
    --hash=sha256:f05c5ad8d6cb4b2659cb6368ed02c067539415b24a2b69effd8ada576fdbf303 \
    --hash=sha256:fb8a4117a5224d73abe2204a1e744ae34b544f1b9a0b761576851deaa7215c12
EOF
  # Execute the pinned binary directly. `python -m uv` can choose another uv
  # from Python's global scripts directory even with a pinned PYTHONPATH.
  UV_BIN="$UV_TOOLS/bin/uv"
  if [[ ! -x "$UV_BIN" ]] || ! UV_REPORTED="$("$UV_BIN" --version 2>&1)" || ! uv_version_matches "$UV_REPORTED"; then
    "$QWEN_BOOTSTRAP_PYTHON" -m pip install --disable-pip-version-check --no-deps \
      --only-binary=:all: --require-hashes --index-url https://pypi.org/simple \
      --upgrade --target "$UV_TOOLS" -r "$UV_TOOLS/bootstrap-requirements.txt"
  fi
fi
uv_run() {
  "$UV_BIN" "$@"
}
if ! UV_REPORTED="$(uv_run --version 2>&1)" || ! uv_version_matches "$UV_REPORTED"; then
  printf 'Pinned uv bootstrap failed. Expected uv 0.12.23; binary: %s; observed: %s\n' "$UV_BIN" "$UV_REPORTED" >&2
  exit 1
fi
printf 'Verified uv: %s (%s)\n' "$UV_REPORTED" "$UV_BIN"
if [[ "$MODE" == --bootstrap-only ]]; then
  echo 'uv bootstrap passed. No GPU dependencies, CUDA checks or model weights were loaded.'
  exit 0
fi
if [[ "$MODE" == --resolve-only ]]; then
  mkdir -p "$ROOT/.cache"
  uv_run pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --torch-backend cu128 --no-build --generate-hashes \
    --constraints requirements-qwen-gpu.lock --constraints requirements-bank.lock \
    requirements-qwen-gpu.in requirements-bank.in \
    --output-file "$ROOT/.cache/qwen-gpu-combined-resolution.lock" >/dev/null
  echo 'GPU+bank dependency metadata resolved; no GPU packages or model weights installed.'
  exit 0
fi
if [[ -e "$QWEN_ENV_PATH" ]]; then
  [[ -f "$QWEN_ENV_PATH/pyvenv.cfg" && -x "$QWEN_ENV_PATH/bin/python" ]] || { echo 'Existing path is not an isolated environment; choose another QWEN_ENV_PATH.' >&2; exit 1; }
  "$QWEN_ENV_PATH/bin/python" - "$QWEN_PYTHON_VERSION" <<'PY'
import platform, sys
if platform.python_version() != sys.argv[1]:
    raise SystemExit('Existing Python differs; preserve it and choose a new QWEN_ENV_PATH or explicit matching QWEN_PYTHON_VERSION.')
PY
else
  uv_run python install "$QWEN_PYTHON_VERSION" --no-bin
  uv_run venv --python "$QWEN_PYTHON_VERSION" "$QWEN_ENV_PATH"
fi
# Preserve an existing working stack rather than silently change shared versions.
"$QWEN_ENV_PATH/bin/python" - "$ROOT" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
for lock in ('requirements-bank.lock', 'requirements-qwen-gpu.lock'):
    for name, expected in re.findall(r'^([\w.-]+)==([^\s\\]+)', (root / lock).read_text(), re.M):
        try:
            installed = version(name)
        except PackageNotFoundError:
            continue
        if installed != expected:
            raise SystemExit(f'Preserving existing environment: {name} is {installed}, lock needs {expected}. Choose a new QWEN_ENV_PATH.')
PY
uv_run pip install --python "$QWEN_ENV_PATH/bin/python" --torch-backend cu128 \
  --only-binary=:all: --require-hashes -r requirements-bank.lock -r requirements-qwen-gpu.lock
uv_run pip check --python "$QWEN_ENV_PATH/bin/python"
"$QWEN_ENV_PATH/bin/python" - "$QWEN_ENV_PATH" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json, platform, sys
import torch
import bitsandbytes as bnb
if not torch.cuda.is_available() or torch.version.cuda != '12.8':
    raise SystemExit('CUDA 12.8 is not usable; inspect the host GPU driver. Installed files are retained.')
with torch.inference_mode():
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                             quant_type='nf4', compress_statistics=True).cuda()
    output = layer(torch.zeros((1, 64), device='cuda', dtype=torch.float16))
    if not torch.isfinite(output).all().item():
        raise SystemExit('Tiny NF4 GPU readiness check returned non-finite values.')
    torch.cuda.synchronize()
manifest = {'python': platform.python_version(), 'torch_cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'cuda_available': True,
            'tiny_nf4_check_passed': True, 'model_weights_loaded': False,
            'model_api_calls': 0,
            'packages': {n: version(n) for n in ('torch', 'transformers', 'accelerate',
                'bitsandbytes', 'huggingface-hub', 'tokenizers', 'safetensors', 'pydantic', 'pandas')}}
target = Path(sys.argv[1]) / 'qwen-gpu-setup.json'
target.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
echo "Dependencies and tiny CUDA/NF4 check passed. Model/long-context generation is not validated by setup."
