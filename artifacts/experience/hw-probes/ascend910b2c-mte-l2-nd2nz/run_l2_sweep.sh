#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
out=$(npu-smi info)
printf "%s\n" "$out" | tee /root/mte_microbench_work/preflight_l2_sweep.log
if printf "%s\n" "$out" | grep -Eq '^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+'; then
  echo "NPU busy: abort" >&2; exit 1
fi
cd /root/mte_microbench_work
rm -rf l2_results; mkdir -p l2_results
: > l2_results/summary.tsv
echo -e "mode\tworking_set_bytes\tloops\tduration_us\tlogical_bytes\tgbps\toutdir" >> l2_results/summary.tsv
# mode0 GM/L2->L0A: vary working set; repeated small set estimates hit, large set estimates miss.
for kib in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
  nelems=$((kib*1024/2))
  loops=4096
  /opt/mamba/envs/ascend-torch/bin/python make_mte_config_b.py 0 "$loops" "$nelems" 16384 > "l2_results/config_${kib}k.log"
  outdir="/root/mte_microbench_work/l2_results/prof_${kib}k"
  rm -rf "$outdir"
  msprof op --config=/root/mte_microbench_work/run_b/config.json --output="$outdir" --aic-metrics=L2Cache --warm-up=0 > "l2_results/msprof_${kib}k.log" 2>&1 || true
  csv=$(find "$outdir" -name OpBasicInfo.csv -type f 2>/dev/null | head -1 || true)
  dur=0; [[ -n "$csv" ]] && dur=$(awk -F, 'NR==2 {print $3}' "$csv" 2>/dev/null || echo 0)
  bytes=$((loops*16384*2))
  gbps=$(awk -v b="$bytes" -v us="$dur" 'BEGIN{if(us>0) printf "%.3f", b/us/1000; else print "0"}')
  echo -e "0\t$((kib*1024))\t${loops}\t${dur}\t${bytes}\t${gbps}\t${outdir}" >> l2_results/summary.tsv
done
cat l2_results/summary.tsv
