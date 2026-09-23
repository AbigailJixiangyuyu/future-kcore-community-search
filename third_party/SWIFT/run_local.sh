#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DGLBACKEND=pytorch
export SWIFT_ALLOW_OLD_TORCH=1
export DGL_FFI=ctypes
export DGL_LIBRARY_PATH="$ROOT/.local-build/dgl"
export PYTHONPATH="$ROOT/.local-build/python:$ROOT/.local-deps:$ROOT/swift-dgl/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${SWIFT_CUDA_ROOT:-/usr/local/cuda-11.1}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec python "$@"
