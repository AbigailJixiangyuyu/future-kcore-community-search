#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CUDA_ROOT="${SWIFT_CUDA_ROOT:-/usr/local/cuda-11.1}"
if [[ ! -f swift-dgl/third_party/METIS/GKlib/GKlibSystem.cmake ]]; then
    echo "Missing GKlib source; see ../README.md. No system install was attempted." >&2
    exit 1
fi
python build_sampler_local.py build_ext \
    --build-lib .local-build/python --build-temp .local-build/sampler-temp
cmake -S swift-dgl -B .local-build/dgl \
    -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
    -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=86 -DCUDA_ARCH_PTX=86 \
    -DBUILD_TORCH=OFF -DUSE_LIBXSMM=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build .local-build/dgl --parallel "${SWIFT_BUILD_JOBS:-4}"
