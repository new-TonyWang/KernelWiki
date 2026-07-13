#!/bin/bash
# Build script for the cutlass-free warp-specialized GEMM (Hopper sm_90a).
#
# The -gencode=arch=compute_90a,code=sm_90a form is REQUIRED -- a plain
# -arch=sm_90a generates compute_90 PTX that ptxas rejects for wgmma.
# See 40-hardware-feature/wgmma-ptx/pitfalls.md #1.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

nvcc -std=c++17 -O3 \
     -gencode=arch=compute_90a,code=sm_90a \
     -lineinfo \
     "$HERE/gemm_ws_ptx.cu" \
     -lcuda -lcublas \
     -o "$HERE/gemm_ws_ptx"

echo "built: $HERE/gemm_ws_ptx"

# Cutlass-free preprocessor gate.
echo
echo "--- cutlass-free preprocessor gate (expect 0 matches) ---"
nvcc -std=c++17 -gencode=arch=compute_90a,code=sm_90a -E "$HERE/gemm_ws_ptx.cu" 2>/dev/null \
    | grep -E 'cutlass::|cute::' | wc -l

# Cutlass-free linked-binary gate.
echo "--- cutlass-free linked-binary gate (expect 0 matches) ---"
cuobjdump --dump-elf-symbols "$HERE/gemm_ws_ptx" 2>/dev/null \
    | grep -E 'cutlass::|cute::' | wc -l
