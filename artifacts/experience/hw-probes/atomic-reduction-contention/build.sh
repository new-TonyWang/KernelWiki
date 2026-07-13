#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
     -o atomic_reduction_probe atomic_reduction_probe.cu
