#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# Build cutlass 88_hopper_fmha on H200 (sm_90a).
# Requires cutlass checkout at $CUTLASS_DIR.
# Default: ${CUTLASS_DIR} (local dev machine)
# H200:    /inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-src/cutlass
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PATH=/usr/local/cuda-12.9/bin:$PATH
CUTLASS_DIR=${CUTLASS_DIR:-${CUTLASS_DIR}}

if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "ERROR: CUTLASS_DIR=$CUTLASS_DIR does not contain include/cutlass/"
    echo "Set CUTLASS_DIR to a valid cutlass checkout."
    exit 1
fi

nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a \
  --expt-relaxed-constexpr -w \
  -I "$CUTLASS_DIR/include" -I "$CUTLASS_DIR/tools/util/include" \
  -I "$CUTLASS_DIR/examples/88_hopper_fmha" \
  "$SCRIPT_DIR/cutlass_88_hopper_fmha.cu" \
  -o "$SCRIPT_DIR/88_hopper_fmha"
echo "Build OK: 88_hopper_fmha (output: $SCRIPT_DIR/88_hopper_fmha)"
