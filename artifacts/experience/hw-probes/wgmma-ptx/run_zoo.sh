#!/usr/bin/env bash
# Build + run the wgmma multi-config harness on H200; persist CSV.
set -euo pipefail
DATE=$(date +%Y-%m-%d)
PROFILES_DIR=profiles
mkdir -p "$PROFILES_DIR"
OUT="$PROFILES_DIR/${DATE}-wgmma-zoo.csv"

bash build_zoo.sh
./wgmma_zoo | tee "$OUT"
echo "wrote $OUT"
