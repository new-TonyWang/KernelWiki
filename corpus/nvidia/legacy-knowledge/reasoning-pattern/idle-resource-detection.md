# Reasoning Pattern: Idle Resource Detection

## Core Question
When a hardware unit or a group of threads is idle during some phase, ask yourself: "What useful work could this idle resource be doing instead?"

## Thinking Framework

### Step 1: Identify what is idle and when
- Are some warps waiting at a barrier while others compute?
- Is the memory subsystem idle while compute units are busy (or vice versa)?
- Are specific hardware units (SFU, tensor cores, load/store units) underutilized?
- Is there dead time between kernel launches or between pipeline stages?

### Step 2: Match idle resources to available work
- If threads are idle: can they prefetch data, compute statistics, or handle bookkeeping?
- If memory units are idle: can they start loading data for future iterations?
- If compute units are idle: can they process non-critical-path work (softmax, normalization)?
- If tensor cores are idle: can they do a preparatory MMA for the next tile?

### Step 3: Design the overlap
- Use warp specialization to assign different work to different thread groups
- Use async operations (TMA, WGMMA) to decouple issuance from completion
- Use producer-consumer patterns with barriers to synchronize access
- Account for resource constraints: shared memory, registers, barrier objects

## Examples from Literature

### Example 1: Warp Specialization -- Producer Warps Are Idle After TMA Issue
**From:** CUTLASS Pipelining Tutorial, Developing CUDA Kernels for GEMM on Hopper (Colfax)

**Idle resource identified:** In a GEMM mainloop, TMA load is issued by a single thread. After issuing the TMA instruction, the entire producer warpgroup has nothing to do until the next pipeline stage needs to be loaded. On Hopper, TMA is so lightweight that the producer warp spends most of its time idle.

**Solution:** Warp specialization formalizes this: dedicate one warpgroup to TMA (producer) and give the remaining warpgroups to WGMMA (consumers). Since the producer is idle most of the time, it needs very few registers. Use `setmaxnreg` to deallocate the producer's excess registers (down to 24-40 per thread) and reallocate them to consumers (up to 240-256 per thread). This converts idle thread registers into active compute resources.

**Key insight:** The producer warpgroup is "paying" 128 threads worth of occupancy for minimal work. The optimization is not about making the producer faster -- it's about reclaiming the resources those idle threads hold.

### Example 2: FlashAttention-3 Pingpong -- One Warpgroup Idle During Other's GEMM
**From:** FlashAttention-3 (Colfax)

**Idle resource identified:** With 2 consumer warpgroups doing attention, when warpgroup 1 is executing WGMMA (which occupies the tensor cores), warpgroup 2's threads are idle. The CUDA cores and SFU units are not being used during the WGMMA execution.

**Solution:** Schedule warpgroup 2 to do softmax (which uses CUDA cores for FMA and SFU for exp) while warpgroup 1 does GEMM (which uses tensor cores). Then swap. This "pingpong" schedule overlaps tensor core work with CUDA core + SFU work.

**Result:** 570 -> 620 TFLOPS improvement on H100 for FP16 attention forward pass. The idle CUDA cores/SFU during WGMMA are now doing softmax.

### Example 3: Intra-Warpgroup Overlap -- Threads Idle Between WGMMA Issue and Completion
**From:** FlashAttention-3, CUTLASS Pipelining Tutorial (Colfax)

**Idle resource identified:** WGMMA is asynchronous: after `warpgroup_commit_batch()`, the 128 threads in the warpgroup are free to do other work until `warpgroup_wait<0>()`. During this window, the threads' integer/FP ALUs are idle.

**Solution:** Insert non-MMA work (softmax partial computation, index calculation, pipeline state management) between commit and wait. This uses the threads' ALUs while the tensor cores execute WGMMA independently.

**Result:** Additional ~20-40 TFLOPS improvement in FlashAttention-3 (620 -> 660 TFLOPS). Higher register pressure is the cost.

### Example 4: Blackwell UMMA -- Single-Thread Launch Frees 127 Threads
**From:** FlashAttention-4, CUTLASS Blackwell Tutorials (Colfax)

**Idle resource identified:** On Blackwell, `tcgen05.mma` is launched by a single thread. The other 127 threads in the warpgroup are completely idle during the MMA launch sequence.

**Solution:** Those idle threads can handle softmax computation, data staging in TMEM, synchronization bookkeeping, and DSMEM exchanges for 2-CTA mode. The single-thread MMA launch makes warp specialization even more natural: one thread issues MMA, the rest do everything else.

## When This Pattern Applies
- Profiler shows low utilization of specific hardware units (tensor cores, SFU, LSU)
- Warp stall analysis shows significant "barrier wait" or "not selected" stalls
- Pipeline visualization shows gaps between stages
- One warpgroup consistently finishes before another in a multi-warpgroup design
- Async operations (TMA, WGMMA, tcgen05.mma) create natural idle windows
