#!/usr/bin/env bash
# Build flash-attention kernels for H200 (sm_90a)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Primary: TMA + wgmma implementation
nvcc -std=c++17 -O3 \
     -gencode=arch=compute_90a,code=sm_90a \
     -lineinfo \
     flash_attn_tma_wgmma.cu -lcuda \
     -o flash_attn_tma_wgmma
echo "Build OK: flash_attn_tma_wgmma (TMA + wgmma)"

# Secondary: thread-level reference (correctness baseline)
nvcc -std=c++17 -O3 \
     -gencode=arch=compute_90a,code=sm_90a \
     -lineinfo \
     flash_attn_minimal.cu \
     -o flash_attn_minimal
echo "Build OK: flash_attn_minimal (thread-level reference)"

# Descriptor validation test
nvcc -std=c++17 -O3 \
     -gencode=arch=compute_90a,code=sm_90a \
     wgmma_desc_test.cu \
     -o wgmma_desc_test
echo "Build OK: wgmma_desc_test (descriptor + fragment validation)"
