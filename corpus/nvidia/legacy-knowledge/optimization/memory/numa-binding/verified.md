
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 5881.333us
**Optimized**: 1012.73us
**Improvement**: 82.78%
**Key metric**: dram_throughput (25.03% → 24.74%)
**Insight**: NCU kernel metrics are identical across both rounds (same cycles, instructions, throughput, stall reasons), proving zero GPU-level improvement; the 5.8x wall-clock speedup is a cold-start/warmup artifact (JIT compilation, CUDA context init), not a NUMA binding effect — real NUMA effects are typically <20%, not 480%.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2608.902us
**Optimized**: 2609.715us
**Improvement**: -0.03%
**Key metric**: dram_throughput (41.4% → 41.24%)
**Insight**: NUMA binding has no measurable effect in this benchmark because the test likely runs on a single-socket system (or single-GPU setup) where all memory is already local, and the kernel is GPU-compute/memory bound rather than host-transfer bound.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 623.516us
**Optimized**: 623.848us
**Improvement**: -0.05%
**Key metric**: dram_throughput (24.22% → 24.25%)
**Insight**: NUMA-aware memory pool placement shows no measurable benefit here because the workload fits entirely on a single GPU with no multi-GPU or cross-socket traffic, so NUMA locality is irrelevant.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 321.168us
**Optimized**: 322.004us
**Improvement**: -0.26%
**Key metric**: dram_throughput (47.08% → 46.80%)
**Insight**: NUMA binding affects host-side memory allocation and CPU-GPU transfer latency, not GPU kernel execution itself — this benchmark measures only kernel runtime, so no improvement is expected.
