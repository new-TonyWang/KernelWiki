#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
out=$(npu-smi info)
printf "%s\n" "$out" | tee /root/mte_microbench_work/preflight_aic_modes.log
if printf "%s\n" "$out" | grep -Eq '^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+'; then
  echo "NPU busy: abort" >&2; exit 1
fi
cd /root/mte_microbench_work
rm -rf aic_results; mkdir -p aic_results
: > aic_results/summary.tsv
echo -e "mode\tloops\tstatus\tduration_us\tbytes\tgbps\toutdir" >> aic_results/summary.tsv
run_one() {
  local mode=$1 loops=$2 nelems=${3:-8388608} epl=${4:-16384}
  /opt/mamba/envs/ascend-torch/bin/python make_mte_config_b.py "$mode" "$loops" "$nelems" "$epl" > "aic_results/config_mode${mode}.log"
  local outdir="/root/mte_microbench_work/aic_results/prof_mode${mode}"
  rm -rf "$outdir"
  set +e
  msprof op --config=/root/mte_microbench_work/run_b/config.json --output="$outdir" --aic-metrics=PipeUtilization --warm-up=0 > "aic_results/msprof_mode${mode}.log" 2>&1
  rc=$?
  set -e
  local dur="0" status="FAIL"
  local csv=$(find "$outdir" -name OpBasicInfo.csv -type f 2>/dev/null | head -1 || true)
  if [[ -n "$csv" ]]; then
    dur=$(awk -F, 'NR==2 {print $3}' "$csv" 2>/dev/null || echo 0)
  fi
  if grep -q "Profiling kernels result is: 1 success" "aic_results/msprof_mode${mode}.log" 2>/dev/null || awk "BEGIN{exit !($dur>0)}"; then status="OK"; fi
  local bytes=0
  case "$mode" in
    0|1|2|4|5|7) bytes=$((loops*epl*2));;
    3|6) bytes=$((loops*epl*2*2));;
    8) bytes=$((loops*epl*2*2));;
    99) bytes=0;;
  esac
  local gbps=$(awk -v b="$bytes" -v us="$dur" 'BEGIN{if(us>0) printf "%.3f", b/us/1000; else print "0"}')
  echo -e "${mode}\t${loops}\t${status}\t${dur}\t${bytes}\t${gbps}\t${outdir}" >> aic_results/summary.tsv
  return 0
}
# smoke no-op and short loops first
run_one 99 1 1024 32
for m in 0 1 2 3 4 5 6 7 8; do run_one "$m" 128 8388608 16384; done
cat aic_results/summary.tsv
