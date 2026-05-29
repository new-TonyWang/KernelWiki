---
id: hw-foundation-warp-execution-model
title: Warp Execution Model
type: hardware
vendor: nvidia
architectures:
- sm90
- sm90a
tags:
- cuda-cpp
confidence: source-reported
related: []
sources: []
aliases: []
blackwell_relevance: Foundation hardware concepts apply to both Hopper and Blackwell
---
# CUDA Warp Execution Model Reference Card

Quick-reference for kernel optimization decisions. Focus: NVIDIA H200 SXM (Hopper, compute capability sm_90a). The H200 shares the same GH100 compute die as the H100, with upgraded HBM3e memory (141 GB, 4.8 TB/s).

---

## 1. Warp Fundamentals

| Parameter | Value |
|-----------|-------|
| Warp size | 32 threads |
| Max warps per SM | 64 |
| Max threads per SM | 2048 |
| Max thread blocks per SM | 32 |
| Max threads per block | 1024 |

- A warp is the fundamental scheduling and execution unit. All 32 threads execute the same instruction simultaneously (SIMT model).
- Thread blocks are partitioned into warps by consecutive thread IDs: warp 0 = threads [0..31], warp 1 = threads [32..63], etc.
- Total warps per block = ceil(threadsPerBlock / 32). Partial warps waste compute resources.
- **Always choose block sizes that are multiples of 32** to avoid under-populated warps.

---

## 2. Warp Scheduler Behavior

### H100 (Compute Capability 9.0)

- **4 warp schedulers per SM** (same as Volta, Ampere).
- Warps are **statically distributed** among the 4 schedulers at block launch time.
- Each scheduler can issue **1 instruction per cycle** from one of its assigned warps.
- At every issue cycle, each scheduler picks a warp that has a ready instruction (no dependencies, not stalled).

### Latency Hiding Requirement

| Latency Source | Typical Latency | Warps Needed to Hide (per SM) |
|---------------|----------------|-------------------------------|
| Arithmetic (register-to-register) | ~4 cycles | 16 warps (4 schedulers x 4 cycles) |
| Shared memory (no bank conflict) | ~20-30 cycles | 20-30 warps (or ILP within warps) |
| Global memory (L2 hit) | ~200 cycles | Many warps + ILP |
| Global memory (HBM) | ~400-600 cycles | Requires high occupancy + ILP |

**Key formula:** To fully hide latency L with throughput T (instructions/cycle), need L x T active warps.
- For 4 schedulers issuing 1 instr/cycle each: need 4L warps for latency L.
- With instruction-level parallelism (ILP), fewer warps suffice because multiple independent instructions from one warp can issue back-to-back.

### Independent Thread Scheduling (Volta+)

- Since Volta (CC 7.0), each thread maintains its own program counter and call stack.
- Threads within a warp can diverge and reconverge at sub-warp granularity.
- A schedule optimizer groups active threads into SIMT units.
- **Implication:** Do not assume warp-synchronous execution. Use explicit synchronization (`__syncwarp()`, `__syncthreads()`).

---

## 3. Warp Divergence

- When threads in a warp take different branch paths, the warp executes each path serially, disabling threads not on the active path.
- **Branch divergence occurs only within a warp.** Different warps are completely independent.
- If N threads take path A and (32-N) take path B, execution time = time(A) + time(B), not max(time(A), time(B)).

### Optimization Tips

- Structure conditions to align with warp boundaries: `if (threadIdx.x / 32 < threshold)` causes no divergence; `if (threadIdx.x % 2 == 0)` causes 100% divergence.
- Minimize divergent code paths; keep both branches short when divergence is unavoidable.
- Consider warp-level voting (`__ballot_sync`, `__any_sync`, `__all_sync`) to reorganize work.

---

## 4. Warp Group Concept (Hopper-Specific)

H100 introduces **warp groups** for cooperative matrix operations:

| Parameter | Value |
|-----------|-------|
| Warp group size | 4 warps (128 threads) |
| Purpose | Collaborative MMA (matrix multiply-accumulate) execution |

- **Warp groups** enable MMA instructions that act on large matrices spanning 4 warps simultaneously.
- Dynamic register capacity can be reassigned among warp groups to support larger matrix tiles.
- Operand matrices can be accessed directly from shared memory (avoiding register staging).
- This feature is exposed via inline PTX; for application code, use CUTLASS or cuBLAS.

### Warp Specialization

- Hopper enables **warp specialization** within a thread block: some warps specialize in data movement (using TMA) while others focus on computation.
- This decouples the data-feeding pipeline from the compute pipeline, improving overlapping.

---

## 5. Register File

### Specifications (H100)

| Parameter | Value |
|-----------|-------|
| Register file size per SM | 65,536 x 32-bit registers (256 KB) |
| Max registers per thread | 255 |
| Max registers per block | 65,536 |
| Register width | 32 bits (a `double` or `long long` uses 2 registers) |
| Allocation granularity | 256 registers per warp (rounded up) |

### Register Allocation Impact on Occupancy

Registers are allocated per warp (not per thread). The total registers for a block = warps_per_block x registers_per_warp.

**Example calculation (H100):**
- Kernel uses 64 registers/thread, block size = 512 threads (16 warps).
- Registers per warp = 64 x 32 = 2048, but allocated in multiples of 256, so 2048 (already aligned).
- Registers per block = 16 x 2048 = 32,768.
- Blocks per SM = floor(65,536 / 32,768) = 2 blocks = 32 warps = 50% occupancy.
- Using 65 registers/thread: per warp = 65 x 32 = 2080, rounded up to 2304 (9 x 256).
- Registers per block = 16 x 2304 = 36,864.
- Blocks per SM = floor(65,536 / 36,864) = 1 block = 16 warps = 25% occupancy.

**One extra register per thread can halve occupancy.** Use `--ptxas-options=-v` to check register usage and `-maxrregcount=N` or `__launch_bounds__()` to control it.

### Controlling Register Usage

```cpp
// Per-kernel hint: max threads per block, min blocks per SM
__global__ void __launch_bounds__(256, 4) kernel(...) { ... }

// Compilation flag: cap registers at N per thread for the whole file
// nvcc -maxrregcount=32

// Per-kernel max register qualifier (PTX-level)
__global__ void __maxnreg__(32) kernel(...) { ... }
```

---

## 6. Occupancy

**Occupancy = active warps per SM / max warps per SM (64).**

It is NOT always true that higher occupancy yields better performance, but low occupancy (<25%) typically cannot hide memory latency.

### Limiting Factors

| Resource | H100 Limit | How It Limits Occupancy |
|----------|-----------|------------------------|
| Registers per SM | 65,536 | More regs/thread -> fewer warps fit |
| Shared memory per SM | Up to 228 KB | More smem/block -> fewer blocks fit |
| Max blocks per SM | 32 | Even tiny blocks cap at 32 |
| Max warps per SM | 64 (2048 threads) | Hard ceiling |

### Occupancy Calculation Steps

1. **Register limit:** max_blocks_by_regs = floor(65536 / (regs_per_thread x threads_per_block, rounded to allocation granularity))
2. **Shared memory limit:** max_blocks_by_smem = floor(available_smem / smem_per_block)
3. **Block count limit:** max_blocks_by_hw = 32
4. **Effective blocks per SM:** min(max_blocks_by_regs, max_blocks_by_smem, max_blocks_by_hw)
5. **Occupancy:** (effective_blocks x warps_per_block) / 64

### Occupancy Calculator API

```cpp
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, dynamicSmemSize);

int blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// For cluster-based kernels (Hopper)
int maxActiveClusters;
cudaOccupancyMaxActiveClusters(&maxActiveClusters, kernel, &launchConfig);
```

### Practical Guidelines

- **128-256 threads per block** is a good starting range for experimentation.
- Use several smaller blocks rather than one large block per SM if `__syncthreads()` is frequent.
- Higher occupancy does not always help: with enough ILP, 50% occupancy can match 100%.
- Low occupancy kernels have more registers/thread available, reducing spills to local memory.

---

## 7. Warp Stall Reasons

When profiling with Nsight Compute, warps report stall reasons. Categories relevant to optimization:

| Stall Reason | Description | Optimization Strategy |
|-------------|-------------|----------------------|
| **Memory Dependency (Long Scoreboard)** | Waiting for global/local memory load to complete | Increase occupancy; prefetch data; use shared memory; improve coalescing |
| **Memory Dependency (Short Scoreboard)** | Waiting for shared memory or L1 cache operation | Reduce bank conflicts; reduce shared memory pressure |
| **Execution Dependency** | Waiting for result of prior arithmetic instruction (~4 cycles) | Increase ILP; reorder independent instructions |
| **Synchronization (Barrier)** | Waiting at `__syncthreads()` or `__syncwarp()` | Reduce work imbalance between warps in block; use multiple blocks per SM |
| **Memory Throttle** | Memory subsystem is saturated | Reduce memory requests; improve data reuse |
| **Not Selected** | Warp is ready but scheduler picked another warp | Generally okay -- indicates good latency hiding |
| **Instruction Fetch** | Waiting for instruction cache | Reduce code size; improve locality of code execution |
| **Texture** | Waiting for texture memory fetch | Optimize texture access patterns |
| **Sleeping** | Warp explicitly put to sleep (Hopper barrier wait) | Expected behavior with async barriers |
| **Dispatch Stall** | Instruction dispatch pipeline is full | Reduce instruction mix pressure |
| **Drain** | Warp completing final instructions before exit | Normal at end of kernel |
| **IMC Miss** | Instruction cache miss | Keep hot loops compact |

### Priority of Optimization

1. **Long scoreboard stalls (global memory):** Usually the dominant bottleneck. Fix via coalescing, caching, shared memory tiling.
2. **Barrier stalls:** Indicate load imbalance or over-synchronization. Consider async patterns, reduce sync points.
3. **Short scoreboard stalls (shared memory):** Fix bank conflicts, reduce smem pressure.
4. **Execution dependency:** Increase ILP or accept as inherent to the algorithm.

---

## 8. H200/H100 SM Architecture Summary

| Component | Count per SM |
|-----------|-------------|
| FP32 CUDA Cores | 128 (2x A100) |
| FP64 CUDA Cores | 64 |
| INT32 Cores | 64 |
| Tensor Cores (4th gen) | 4 |
| Warp Schedulers | 4 |
| Special Function Units (SFU) | 16 |
| Register File | 65,536 x 32-bit |
| Unified L1/Shared Memory | 256 KB |
| Constant Cache | Read-only, shared by all functional units |

### H200 SXM GPU-Level Numbers

| Parameter | Value |
|-----------|-------|
| Total SMs | 132 |
| Total FP32 Cores | 16,896 |
| Total Tensor Cores | 528 |
| L2 Cache | 50 MB |
| HBM3e | 141 GB, 4.8 TB/s |
| NVLink (4th gen) | 900 GB/s bidirectional (18 links) |
| Thread Block Cluster size | Up to 8 (portable), up to 16 (non-portable) |

---

*Sources: CUDA C++ Programming Guide 13.2 (SIMT Architecture, Compute Capabilities), CUDA C++ Best Practices Guide 13.2, Hopper Tuning Guide 13.2, NVIDIA H100 Tensor Core Hopper Architecture Whitepaper.*
