#!/usr/bin/env bash
# Build the cutlass-free TMA throughput probe + verify cutlass-free gates.
set -euo pipefail
SRC=tma_throughput_probe.cu
BIN=tma_throughput_probe

NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}

"$NVCC" -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
        "$SRC" -lcuda -o "$BIN"

echo "=== nvcc -E grep for cutlass::/cute:: in preprocessed source ==="
PRE=$("$NVCC" -E -gencode=arch=compute_90a,code=sm_90a -DCUDA_ARCH_VAL=900 \
              -I/usr/local/cuda-12.9/include "$SRC" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
if [ "$PRE" = "0" ]; then
  echo "PASS: preprocessor produces 0 cutlass:: / cute:: matches."
else
  echo "FAIL: preprocessor matched $PRE cutlass:: / cute:: occurrences."
  exit 1
fi

echo "=== cuobjdump --dump-elf-symbols grep for cutlass::/cute:: in linked binary ==="
SYM=$(/usr/local/cuda-12.9/bin/cuobjdump --dump-elf-symbols "$BIN" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
if [ "$SYM" = "0" ]; then
  echo "PASS: linked binary has 0 cutlass:: / cute:: symbols."
else
  echo "FAIL: binary contains $SYM cutlass:: / cute:: symbols."
  exit 1
fi
