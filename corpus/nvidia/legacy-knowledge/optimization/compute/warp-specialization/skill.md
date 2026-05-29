---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization)"
cross_ref:
  - "Programming Guide, Section 3.2.4.2 (Asynchronous Barriers)"
  - "Programming Guide, Section 4.10 (Pipelines)"
  - "Programming Guide, Section 4.10.7 (Producer-Consumer Pattern using Pipelines)"
  - "PTX ISA, setmaxnreg / bar.sync / bar.arrive / elect.sync instructions"
related_apis:
  - setmaxnreg.inc.sync.aligned.u32
  - setmaxnreg.dec.sync.aligned.u32
  - bar.sync
  - bar.arrive
  - elect.sync
  - "cuda::pipeline"
  - "cuda::make_pipeline"
  - "cuda::memcpy_async"
  - __mbarrier_arrive
  - __mbarrier_try_wait
  - __pipeline_memcpy_async
related_experience: []
unlocks:
  - "data-prefetch: producer warps drive async copy pipeline"
  - "tensor-core: consumer warps can dedicate all registers to MMA"
  - "barrier-optimization: split arrive/wait enables producer-consumer overlap"
conflicts_with:
  - "occupancy-tuning: warp specialization may reduce effective occupancy per role"
---

# Warp Specialization

## Skill 1: Producer-Consumer Warp Partitioning with cuda::pipeline
### When to Use
When you want to overlap data movement (global to shared) with computation by dedicating specific warps to each role. One warp fetches data while others compute.

### How to Apply
1. Assign the first warp (threads 0-31) as the producer; remaining warps as consumers.
2. Use `cuda::make_pipeline(block, &shared_state, producer_count)` to create a partitioned pipeline.
3. Producers call `producer_acquire()`, issue `cuda::memcpy_async`, then `producer_commit()`.
4. Consumers call `consumer_wait()`, process the data, then `consumer_release()`.
5. Use double-buffering (2 stages) or multi-buffering (N stages) in shared memory.

### Code Template
```cuda
#include <cooperative_groups.h>
#include <cuda/pipeline>

using pipeline = cuda::pipeline<cuda::thread_scope_block>;

__device__ void produce(pipeline &pipe, int stage, float *buffer,
                         int buffer_len, float *in, int batch) {
    pipe.producer_acquire();
    cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x,
                       in + batch * buffer_len + threadIdx.x,
                       cuda::aligned_size_t<4>(sizeof(float)), pipe);
    pipe.producer_commit();
}

__device__ void consume(pipeline &pipe, int stage, float *buffer,
                         int buffer_len, float *out, int batch) {
    pipe.consumer_wait();
    // Process buffer[stage * buffer_len ... ]
    out[batch * buffer_len + threadIdx.x] =
        buffer[stage * buffer_len + threadIdx.x] * 2.0f;
    pipe.consumer_release();
}

__global__ void warp_specialized_kernel(float *in, float *out,
                                         int N, int buffer_len) {
    auto block = cooperative_groups::this_thread_block();
    constexpr int num_stages = 2;
    __shared__ extern float buffer[];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, num_stages> shared_state;

    cuda::std::size_t producer_count = 32;  // First warp
    pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

    int num_batches = N / buffer_len;

    // Fill pipeline
    if (block.thread_rank() < producer_count) {
        for (int s = 0; s < num_stages; ++s)
            produce(pipe, s, buffer, buffer_len, in, s);
    }

    // Main loop
    int stage = 0;
    for (int b = 0; b < num_batches; ++b) {
        if (block.thread_rank() < producer_count) {
            produce(pipe, stage, buffer, buffer_len, in, b + num_stages);
        } else {
            consume(pipe, stage, buffer, buffer_len, out, b);
        }
        stage = (stage + 1) % num_stages;
    }
}
```

### Source
Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization)

## Skill 2: Producer-Consumer with mbarrier Primitives
### When to Use
When you need lower-level control than `cuda::pipeline` provides, or when targeting custom synchronization patterns between producer and consumer warps.

### How to Apply
1. Allocate `__mbarrier_t` arrays in shared memory (one per buffer stage) for "ready" and "filled" signals.
2. Producer warp: arrive on "ready" barrier, wait for it, copy data, arrive on "filled" barrier.
3. Consumer warps: arrive on "filled" barrier, wait for it, process data, arrive on "ready" barrier.
4. This allows fully asynchronous overlap between producer and consumer.

### Code Template
```cuda
#include <cuda_awbarrier_primitives.h>

__device__ void produce_mbar(__mbarrier_t ready[], __mbarrier_t filled[],
                              float *buffer, int buffer_len, float *in, int N) {
    for (int i = 0; i < N / buffer_len; ++i) {
        __mbarrier_token_t token = __mbarrier_arrive(&ready[i % 2]);
        while (!__mbarrier_try_wait(&ready[i % 2], token, 1000)) {}
        // Copy data: global -> shared
        __pipeline_memcpy_async(buffer + (i % 2) * buffer_len + threadIdx.x,
                                in + i * buffer_len + threadIdx.x,
                                cuda::aligned_size_t<4>(sizeof(float)));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __mbarrier_arrive(&filled[i % 2]);
    }
}

__device__ void consume_mbar(__mbarrier_t ready[], __mbarrier_t filled[],
                              float *buffer, int buffer_len, float *out, int N) {
    for (int i = 0; i < N / buffer_len; ++i) {
        __mbarrier_token_t token = __mbarrier_arrive(&filled[i % 2]);
        while (!__mbarrier_try_wait(&filled[i % 2], token, 1000)) {}
        // Process buffer[(i%2) * buffer_len ...]
        out[i * buffer_len + threadIdx.x] =
            buffer[(i % 2) * buffer_len + threadIdx.x] * 2.0f;
        __mbarrier_arrive(&ready[i % 2]);
    }
}
```

### Source
Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization) -- CUDA C primitives variant

## Skill 3: Dynamic Register Reallocation with setmaxnreg
### When to Use
When producer and consumer warps have very different register requirements. On sm_90+, `setmaxnreg` can dynamically adjust per-warp register allocation, giving consumers more registers (for MMA) and producers fewer (for address computation only).

### How to Apply
1. Use PTX `setmaxnreg.inc.sync.aligned.u32` to increase the register limit.
2. Use `setmaxnreg.dec.sync.aligned.u32` to decrease and free registers for other warps.
3. Must be called uniformly by all threads in the warp.
4. The total register file size per SM is conserved; one warp's reduction enables another's increase.

### Code Template
```cuda
__device__ void consumer_warp() {
    // Increase register budget for heavy compute (MMA)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 64;\n");

    // ... perform MMA-heavy computation with more registers ...

    // Release registers before exit
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n");
}

__device__ void producer_warp() {
    // Producer needs fewer registers (just address computation + async copy)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n");

    // ... perform async copies ...

    asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n");
}
```

### Source
PTX ISA, setmaxnreg.{inc/dec}.sync.aligned.u32 instruction

## Skill 4: Warp Election for Leader Selection
### When to Use
When you need to elect a single leader thread from the active threads in a warp, e.g., for one-time initialization, barrier setup, or single-threaded I/O within the producer warp.

### How to Apply
1. Use PTX `elect.sync` to choose one active thread deterministically.
2. The elected thread gets a predicate set to true; all others get false.
3. This is more efficient than `threadIdx.x % 32 == 0` because it works correctly under divergence.

### Code Template
```cuda
__device__ bool elect_leader() {
    int is_leader;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  elect.sync _|p, 0xFFFFFFFF;\n"
        "  selp.s32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(is_leader)
    );
    return is_leader != 0;
}

__device__ void setup_barrier(__mbarrier_t *bar) {
    if (elect_leader()) {
        // Only one thread initializes the barrier
        __mbarrier_init(bar, blockDim.x);
    }
    __syncwarp();
}
```

### Source
PTX ISA, elect.sync instruction

## Skill 5: Multi-Stage Buffering for Deep Pipelining
### When to Use
When the memory latency is much larger than the computation time per tile, requiring more than 2 stages to keep the pipeline full.

### How to Apply
1. Increase `num_stages` beyond 2 (e.g., 3, 4, or more).
2. Allocate `num_stages` buffers in shared memory.
3. The producer fills ahead by `num_stages` batches while consumers process the oldest.
4. More stages hide more latency but consume more shared memory.

### Code Template
```cuda
constexpr int NUM_STAGES = 4;
__shared__ float buffer[NUM_STAGES * BUFFER_LEN];
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> shared_state;

pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

// Producer pre-fills all stages
if (is_producer) {
    for (int s = 0; s < NUM_STAGES; ++s)
        produce(pipe, s, buffer, BUFFER_LEN, in, s);
}

// Main loop: producer fills stage (b + NUM_STAGES), consumer processes stage b
int stage = 0;
for (int b = 0; b < num_batches; ++b) {
    if (is_producer)
        produce(pipe, stage, buffer, BUFFER_LEN, in, b + NUM_STAGES);
    else
        consume(pipe, stage, buffer, BUFFER_LEN, out, b);
    stage = (stage + 1) % NUM_STAGES;
}
```

### Source
Programming Guide, Section 4.10.7 (Producer-Consumer Pattern using Pipelines); Section 3.2.4.3 (Pipelines)

## Cascading Opportunities (unlocks)
- **data-prefetch**: the producer warp's primary job is async data prefetch (cp.async, TMA).
- **tensor-core**: consumer warps can maximize TC utilization with all available registers dedicated to fragments.
- **barrier-optimization**: the split arrive/wait pattern of mbarrier enables fine-grained producer-consumer synchronization.

## Conflicts
- **occupancy-tuning**: warp specialization means some warps are idle during their non-role phase. Effective compute occupancy for the consumer role is reduced by the number of producer warps.

## Principles
- Warp specialization is most beneficial when memory latency significantly exceeds compute time per tile.
- The producer warp uses very few registers (just address computation), freeing registers for consumer warps.
- Double-buffering (2 stages) is the minimum; deeper pipelines may be needed for high-latency memory access patterns.
- On Hopper (sm_90+), wgmma natively supports warp-group-level specialization with TMA-based producers.
- All threads in the block must participate in pipeline creation, even if their role is different.

## Open Questions
- What is the optimal producer-to-consumer warp ratio for GEMM on Hopper (1:3 warpgroups)?
- How does `setmaxnreg` interact with the compiler's register allocation for consumer warps?
- Can work-stealing (cluster launch control) complement warp specialization for load balancing?
