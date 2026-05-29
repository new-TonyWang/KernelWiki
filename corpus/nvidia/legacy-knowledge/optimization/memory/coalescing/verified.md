
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 256.421us
**Optimized**: 60.707us
**Improvement**: 76.3%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: Baseline had 12.5% global load efficiency (1/8 sector utilization, completely uncoalesced), fixing thread-to-address mapping achieved 100% coalesced loads, reducing active cycles by 4.6x and yielding a 76% speedup.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 215.724us
**Optimized**: 286.094us
**Improvement**: -32.6%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (86.21% → 96.15%)
**Insight**: cudaMallocPitch padding aligns each row but inflates total memory footprint by ~19%, destroys L2 locality (hit rate drops 60.9%→49.1%), and adds 30% more instructions, overwhelming the coalescing gain.
**Conditions**: cudaMallocPitch improves coalescing efficiency (86%→96%) but the pitch padding increases total memory traffic and hurts L2 cache hit rate, resulting in a net slowdown

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 70.105us
**Optimized**: 14.982us
**Improvement**: 78.6%
**Key metric**: sm__cycles_active.avg (103016.73 cycles (memory-bound at 73.9% mem SOL, 10.9% compute SOL) → 26278.95 cycles (53.6% compute SOL, 21.5% mem SOL — bottleneck eliminated))
**Insight**: Shared memory staging converted strided global writes into coalesced ones, cutting DRAM write volume by 48% (1.43MB→0.74MB) and reducing active cycles by 3.9x, despite increased instruction count and barrier synchronization overhead.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 103.301us
**Optimized**: 56.851us
**Improvement**: 44.97%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (86.21% → 100.0%)
**Insight**: Aligning blockDim.x to warp size (32) ensures all threads in a warp access consecutive addresses, achieving perfect coalescing (100% sector utilization vs 86.21%), which nearly doubled DRAM throughput (22.4% → 43.6%) and cut active cycles in half.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 19.485us
**Optimized**: 20.083us
**Improvement**: -3.07%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (33.33% → 100.0%)
**Insight**: The SoA pattern achieves perfect coalescing (33%→100% sector utilization), but the shared-memory transpose workaround nearly doubled instruction count (2.75M→5.24M), introduced 11% barrier stalls, and destroyed L1 hit rate (60%→0%), causing a net 3% wall-clock regression despite eliminating strided access.
**Conditions**: only if data is laid out as SoA on the host before transfer; using shared memory to transpose AoS→SoA at runtime adds enough instruction and synchronization overhead to negate the coalescing benefit at this problem size

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 23.341us
**Optimized**: 3.907us
**Improvement**: 83.3%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: Baseline had 12.5% sector utilization (1 useful float per 32-byte sector, classic uncoalesced strided access), optimization achieved perfect 100% coalescing which reduced active cycles by 5.6x and instructions by 2.6x.
