#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-4 (aligned GEMM) build. Compiles gemm_aligned.cu (a derivative of
# cutlass example 48 — a Hopper warp-specialized cooperative GEMM) so the
# AC-4 measurements at three exact-wgmma-tile-multiple shapes (512^3,
# 2048^3, 8192^3) reproduce on H200.
#
# Bootstrap from a fresh sandbox:
#   rsync -az --exclude='.git' --exclude='build' \
#         ${CUTLASS_DIR}/ \
#         h200_ncu:${CUTLASS_DIR}/
# Run on H200; assumes /usr/local/cuda-12.9/bin/nvcc is available.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"

nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo \
     -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I"${CUTLASS_ROOT}/include" \
     -I"${CUTLASS_ROOT}/tools/util/include" \
     -I"$(pwd)" \
     gemm_aligned.cu \
     -o gemm_aligned
