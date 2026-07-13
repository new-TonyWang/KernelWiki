
## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 256.753us
**Optimized**: 37.769us
**Improvement**: 85.3%
**Key metric**: dram__throughput.avg.pct_of_peak_sustained_elapsed (10.71% → 64.95%)
**Insight**: Shared-memory tiling converted uncoalesced or redundant global accesses into coalesced bulk transfers, cutting DRAM traffic ~18% and GPU active cycles 7.6x (381k→50k), bringing the kernel from 2x slower than PyTorch to on-par with torch.compile.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 3.917us
**Optimized**: 3.945us
**Improvement**: -0.7%
**Key metric**: sm__inst_executed.sum (688128 → 557056 (-19.0%))
**Insight**: prmt.b32 reduced instruction count by 19% and active cycles by ~5%, but the kernel is memory-latency-bound (long scoreboard stalls rose from 50.8% to 54.8%), so fewer instructions simply expose more time waiting on memory.
**Conditions**: only if the kernel is compute-bound or instruction-issue-limited; for memory-bound kernels the instruction savings do not translate to wall-clock improvement

## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 252.553us
**Optimized**: 81.539us
**Improvement**: 67.7%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (12.27% → 52.61%)
**Insight**: Shared-memory tiled transpose with +1 padding eliminated uncoalesced global writes, cutting cycles by 3x (381k→115k) and boosting SM throughput from 12% to 53%, though it introduced significant barrier stalls (0%→26%) from the required __syncthreads.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 12.685us
**Optimized**: 5.879us
**Improvement**: 53.7%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (25.0% → 100.0%)
**Insight**: Achieving 100% load sector utilization (up from 25%) eliminated 75% wasted global memory bandwidth from strided AoS access, also reducing DRAM write traffic by 70% (2.7MB to 0.8MB) and instruction count by 7.5x, bringing performance on par with torch.compile.

## Experiment: Skill 4 verification (2026-04-07)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 7316.11us
**Optimized**: 6949.994us
**Improvement**: 5.0%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (80.31% → 97.35%)
**Insight**: cudaMallocPitch dramatically improved load coalescing efficiency (80.3% → 97.4%), but the row padding increased total DRAM traffic by ~6% and degraded L2 cache hit rate from 62.7% to 50.1%, largely negating the coalescing benefit and yielding only a borderline 5% wall-clock speedup.
**Conditions**: only if rows are poorly aligned and coalescing efficiency is the bottleneck; padding overhead and reduced L2 cache hit rate can offset gains when data already fits well in cache
