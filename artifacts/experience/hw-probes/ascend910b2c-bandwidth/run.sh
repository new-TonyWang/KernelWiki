#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
TOTAL_BYTES="${1:-1GiB}"
REPEATS="${2:-7}"
DEVICE="${3:-0}"

source /usr/local/Ascend/cann-8.5.1/set_env.sh >/dev/null 2>&1

rm -rf "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION=Ascend910B2 \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j1

"${BUILD_DIR}/910b_bandwidth" "${TOTAL_BYTES}" "${REPEATS}" "${DEVICE}"
