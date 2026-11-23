#!/usr/bin/env bash
set -euo pipefail
: "${CONDA_PREFIX:?Activate layerpano3d-blackwell first}"
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=${MAX_JOBS:-4}
export TMPDIR=${TMPDIR:-/mnt/NAS/pierce/.workstation/S3063/tmp/layerpano3d}
mkdir -p "$TMPDIR"
export LAYERPANO_BUILD_ROOT="$TMPDIR/native-build"
mkdir -p "$LAYERPANO_BUILD_ROOT/wheels"
# Never reuse or overwrite the prebuilt cp39/Windows binaries shipped in git.
python - "$REPO" <<'PY'
from pathlib import Path
import os,shutil,sys
repo=Path(sys.argv[1]); root=Path(os.environ['LAYERPANO_BUILD_ROOT'])
for name in ['diff-gaussian-rasterization','simple-knn']:
    dst=root/name
    if dst.exists():
        raise SystemExit(f'{dst} already exists; set TMPDIR to a fresh NAS build directory')
    shutil.copytree(repo/'submodules'/name,dst,ignore=shutil.ignore_patterns('build','dist','*.egg-info','*.so','*.pyd','__pycache__'))
PY
python -m pip wheel --no-deps --no-build-isolation "$LAYERPANO_BUILD_ROOT/diff-gaussian-rasterization" "$LAYERPANO_BUILD_ROOT/simple-knn" -w "$LAYERPANO_BUILD_ROOT/wheels"
python -m pip install --no-deps "$LAYERPANO_BUILD_ROOT"/wheels/*.whl
cmake -S "$REPO/scripts/instaOmniDepth" -B "$LAYERPANO_BUILD_ROOT/instaOmniDepth" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DPYBIND11_FINDPYTHON=ON
cmake --build "$LAYERPANO_BUILD_ROOT/instaOmniDepth" --parallel "$MAX_JOBS"
cmake --install "$LAYERPANO_BUILD_ROOT/instaOmniDepth"
