---
title: Asynchronous Data Copies (LDGSTS)
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- stencil
- scan
requires_sm: '>=8.0'
requires_features:
- cp.async
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: spec
  anchor: part-1-maximizing-memory-bandwidth
  excerpt: 'Asynchronous copies let the data go directly from global memory to shared memory, skipping the register file. Benefits: free up registers for compute, reduce L1 traffic, reduce MIO pressure.'
artifacts:
  code: artifacts/experience/hw-probes/async-copy/async_copy_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o async_copy_probe async_copy_probe.cu
  introspection: ''
  profile: ''
related_apis:
- __pipeline_memcpy_async
- __pipeline_commit
- __pipeline_wait_prior
- cuda::memcpy_async
- cuda::pipeline
- cuda::barrier
- cuda::aligned_size_t
- cooperative_groups::memcpy_async
related_skills:
- ilp
- vectorized-access
- register-pressure
- coalescing
id: skill-async-copy
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3937-L3944
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11259-L11269
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L940-L999
---
## What

Asynchronous data copies (LDGSTS) are a hardware-accelerated mechanism (Ampere / cc 8.0+) that transfers data **directly from global memory to shared memory**, bypassing the register file. Traditional synchronous copies require a two-step `LDG` (global to register) + `STS` (register to shared) sequence; LDGSTS fuses this into a single asynchronous instruction that the hardware executes in the background while the thread continues computing.

LDGSTS supports copying 4, 8, or 16 bytes per transfer. It has two modes (programming guide L11266):
- **L1 ACCESS** (4-byte or 8-byte copies): data is also cached in L1.
- **L1 BYPASS** (16-byte copies, 16-byte aligned): L1 is not polluted. Best performance when source and destination are 128-byte aligned.

The completion of async copies is signaled through either a **pipeline** (`cuda::pipeline` or `__pipeline_*` primitives) or a **shared memory barrier** (`cuda::barrier`). The choice determines the synchronization granularity: pipelines are thread-scoped (cheaper, no inter-thread sync needed for the copy itself), while barriers are block-scoped (required when a subset of threads does the copying for the whole block).

## Why: Little's Law and Bytes-in-Flight

The fundamental motivation comes from **Little's Law** applied to GPU memory:

```
bytes-in-flight = bandwidth x mean_latency
```

Bandwidth and latency are fixed by hardware; **bytes-in-flight is the only knob** the programmer controls. To achieve high DRAM bandwidth utilization, the software must keep enough bytes in-flight per SM.

!BW trends across generations

DRAM bandwidth per SM is increasing roughly 2x per generation, while SM count grows slowly. A simple `c[i] = a[i] + b[i]` kernel with 2 loads/thread, 4 bytes/load, 256 threads/block, and 8 blocks/SM produces only 16 KiB/SM of bytes-in-flight. The experimental targets for >90% BW utilization are (GTC25-S72683):

!Little's Law bytes-in-flight scaling

- **H100**: ~32 KiB/SM
- **H200**: ~64 KiB/SM
- **Blackwell (GB200-NVL)**: ~40 KiB/SM

There are three techniques to grow bytes-in-flight:
1. **ILP via unrolling** -- more independent loads per thread (see `ilp` skill).
2. **DLP via vectorized loads** -- wider loads per instruction (see `vectorized-access` skill).
3. **Asynchronous copies** -- LDGSTS and TMA (**this skill**).

The first two increase bytes-in-flight at the cost of **register pressure**:

!Register pressure vs BW utilization

Asynchronous copies avoid this tradeoff by routing data through shared memory instead of registers.

## How It Works: APIs and Programming Patterns

### Three API Layers

CUDA exposes LDGSTS through three API layers (GTC25-S72683, programming guide L3879-L3934):

1. **Primitives API** (`<cuda_pipeline.h>`):
   - `__pipeline_memcpy_async(dst, src, size)` -- initiate async copy
   - `__pipeline_commit()` -- associate a barrier with pending copies
   - `__pipeline_wait_prior(N)` -- wait until at most N stages are pending

2. **libcudacxx** (`<cuda/pipeline>`, `<cuda/barrier>`):
   - `cuda::memcpy_async(dst, src, size, pipe)` -- with pipeline completion
   - `cuda::memcpy_async(dst, src, aligned_size_t<16>(size), barrier)` -- with barrier completion
   - `cuda::pipeline<cuda::thread_scope_thread>` for thread-local staging
   - `cuda::barrier<cuda::thread_scope_block>` for block-wide sync

3. **Cooperative Groups** (`<cooperative_groups/memcpy_async.h>`):
   - `cooperative_groups::memcpy_async(group, dst, src, size)` -- group-wide
   - `cooperative_groups::wait(group)` or `wait_prior<N>(group)`

**Recommendation** (GTC25-S72683): use the Primitives API or libcudacxx for fine-grained control. Cooperative groups is convenient for large block-wide copies but provides less synchronization granularity.

### LDGSTS vs TMA Cheat-Sheet

!LDGSTS / TMA matrix

| Method | Alignment | When to use |
| --- | --- | --- |
| LDGSTS | 4, 8, or 16 bytes | Element-wise copies, tiles < 1 KiB |
| TMA 1D | 16 bytes, size multiple of 16 | Bulk copies >= 2 KiB (Hopper+) |
| TMA ND | SMEM 128B, GMEM 16B | Multi-dimensional tiles (Hopper+) |

**Rule of thumb**: LDGSTS for small or irregular transfers; TMA for large contiguous blocks. TMA instructions have higher latency so they need more data to amortize their cost.

### Basic LDGSTS Pattern (Primitives API)

The simplest usage replaces `smem[tid] = gmem[idx]` with async copies:

```cuda
__shared__ float a_buf[BLOCK_DIM];
__shared__ float b_buf[BLOCK_DIM];

// Fire async copies
__pipeline_memcpy_async(&a_buf[tid], &a[i], sizeof(float));
__pipeline_memcpy_async(&b_buf[tid], &b[i], sizeof(float));

// Commit and wait
__pipeline_commit();
__pipeline_wait_prior(0);  // wait for ALL pending copies

// Compute from shared memory
c[i] = a_buf[tid] * b_buf[tid];
```

### Multi-Stage Prefetching with `cuda::pipeline`

The real power of LDGSTS comes from **multi-stage prefetching**: while the current iteration's data is being consumed from shared memory, the next iteration's data is being loaded in the background.

```cuda
__shared__ float a_buf[NUM_STAGES][BLOCK_DIM];
__shared__ float b_buf[NUM_STAGES][BLOCK_DIM];

cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// Prologue: fill NUM_STAGES stages
for (int s = 0; s < NUM_STAGES; ++s) {
    pipe.producer_acquire();
    cuda::memcpy_async(&a_buf[s][tid], &a[offset + s*stride], sizeof(float), pipe);
    cuda::memcpy_async(&b_buf[s][tid], &b[offset + s*stride], sizeof(float), pipe);
    pipe.producer_commit();
}

// Main loop
int stage = 0;
for (int i = 0; i < iterations; ++i) {
    // Prefetch NUM_STAGES ahead
    pipe.producer_acquire();
    cuda::memcpy_async(&a_buf[stage][tid], &a[next_offset], sizeof(float), pipe);
    pipe.producer_commit();

    stage ^= 1;  // flip for 2-stage

    // Wait for the current stage
    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

    // Compute
    c[ci] = a_buf[stage][tid] * b_buf[stage][tid];

    // Release
    pipe.consumer_release();
}
```

**Critical**: `NUM_STAGES` must be a **compile-time constant** so the pipeline's internal bookkeeping instructions can be optimized away (GTC25-S72683, slide interval_0130).

### Stage-Count Formula

From the bytes-in-flight equation:

```
bytes-in-flight/SM = blocks/SM x threads/block x loads/thread x bytes/load
```

With async copies, `loads/thread = 2 x num_stages`. Solving for the target:
- **H200**: target ~64 KiB/SM. With 8 blocks/SM, 256 threads/block, 4 bytes/load: 2 stages gives 32 KiB, 4 stages gives 64 KiB. **Minimum 2 stages recommended; 4 stages to fully saturate.**
- **H100**: 2 stages suffices (32 KiB target).
- **Blackwell**: 3 stages recommended (40 KiB target).

### L1 BYPASS Mode

To enable L1 BYPASS, each thread must copy exactly 16 bytes with 16-byte alignment. The common pattern reduces the number of copying threads:

```cuda
const int memcpy_threads = BLOCK_DIM / 4;
if (tid < memcpy_threads) {
    cuda::memcpy_async(&a_buf[stage][tid * 4],
                       &a[offset + tid * 4],
                       cuda::aligned_size_t<16>(4 * sizeof(float)),
                       pipe);
}
// __syncthreads() is REQUIRED after pipeline wait
// because only a subset of threads did the copies
```

This creates a **producer-consumer pattern** where `memcpy_threads` act as producers and all threads consume. The `__syncthreads()` after the pipeline wait is mandatory.

## When to Use

Async copies are most beneficial in these scenarios:

- **Iterative kernels with prefetch opportunity**: the classic use case. If a kernel loops over tiles of global data, each tile can be prefetched while the previous tile is being computed.

- **Compute-heavy kernels at low occupancy**: when the compute per element is expensive (sqrt chains, trig functions, etc.), async copies hide the memory latency that would otherwise dominate. GTC25-S72683 showed a **1.305x speedup** on H100 for `sqrt(sqrt(a)/sqrt(b))` vs vanilla.

!2-stage benchmark a*b

!2-stage benchmark sqrt

- **Register-pressure-sensitive kernels**: when ILP/DLP techniques would push register usage too high and cause spilling or low occupancy, async copies provide bytes-in-flight without consuming registers.

- **Stencil kernels with halo loads**: conditional code paths (loading left halo, center, right halo) prevent the compiler from batching loads optimally. LDGSTS forces all loads to be in-flight regardless of control flow (programming guide L11324-L11350).

## When NOT to Use

- **Simple elementwise kernels with low compute intensity**: as our H200 probe confirms, for trivial `c[i] = a[i] * b[i]`, the overhead of shared memory staging exceeds the latency-hiding benefit. The vanilla kernel already achieves 75% BW utilization at 16 KiB bytes-in-flight.

- **Kernels already achieving >90% BW utilization**: if ILP/DLP techniques already saturate bandwidth, adding LDGSTS staging will only add overhead.

- **Kernels with very small working sets**: if the working set fits in L1/L2 cache, the memory latency is already low and async copies add unnecessary SMEM traffic.

- **Single-pass kernels without iteration**: the prefetch benefit requires at least 2 iterations to amortize the prologue cost.

## Optimization Decision Flow

!Optimization flow

The GTC25-S72683 flowchart for increasing bytes-in-flight:

```
Enough bytes-in-flight? --YES--> Do nothing
  | NO
  v
Prefetch in...?
  |-- REG  --> Unroll / vectorize (ilp, vectorized-access skills)
  |-- SMEM --> Aligned to...?
              |-- 4, 8 bytes --> LDGSTS (this skill)
              |-- 16 bytes   --> Size of tile
                                 |-- < 1 KiB        --> LDGSTS
                                 |-- 1 KiB - 2 KiB  --> LDGSTS or TMA (experiment)
                                 |-- > 2 KiB        --> TMA
```

## TMA 1D: Brief Overview (Hopper+)

TMA (Tensor Memory Accelerator) is Hopper's bulk async copy mechanism. **Full TMA coverage is deferred** to the `40-hardware-feature` layer (out of MVP scope); this section provides only the essential context.

Key differences from LDGSTS:
- TMA 1D requires 16-byte alignment and copy size as a multiple of 16 bytes.
- Programming model is **warp-uniform**: issue from one thread per warp.
- `cuda::memcpy_async` auto-uses TMA when alignment/size requirements are met (GTC25-S72683 slide interval_0177).
- `thrust::transform` with `cuda::proclaim_copyable_arguments` enables TMA "for free" -- Thrust auto-tunes to maximize bytes-in-flight.
- Use `cooperative_groups::invoke_one` to eliminate the compiler's peeling loop when issuing TMA from within an `if (threadIdx.x == 0)` block.

## Measured Characteristics

- async-copy-2stage bandwidth probe: On H200 (sm_90a, CUDA 12.9), a **vanilla elementwise kernel** (`c[i] = a[i] * b[i]`, N=256M floats) achieved **3654 GB/s** (median 0.8815 ms). A **2-stage LDGSTS prefetch version** using `__pipeline_memcpy_async` achieved **3295 GB/s** (median 0.9775 ms), approximately **10% slower**. This confirms the GTC guidance: for trivially simple compute, the shared memory staging overhead outweighs the latency-hiding benefit. The async approach delivers significant uplift only with heavier compute intensity or iterative kernels with genuine prefetch opportunities.
