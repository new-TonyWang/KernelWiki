#!/usr/bin/env bash
# Run cutlass 88_hopper_fmha example and collective configs on H200.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p profiles

echo "=== example config: B=2 H=16 Q=1024 K=1024 D=128 forward full ==="
./88_hopper_fmha --b=2 --h=16 --q=1024 --k=1024 --d=128 --verify 2>&1 | tee profiles/cutlass-fmha-example.log
echo ""
echo "=== collective config: B=2 H=16 Q=2048 K=2048 D=128 forward causal ==="
./88_hopper_fmha --b=2 --h=16 --q=2048 --k=2048 --d=128 --mask=causal --verify 2>&1 | tee profiles/cutlass-fmha-collective.log
echo ""
echo "=== done ==="
