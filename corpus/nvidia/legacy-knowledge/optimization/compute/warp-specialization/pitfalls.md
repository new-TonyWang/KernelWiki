# Warp Specialization -- Pitfalls

## P1: Producer Warp Stalling on Barrier Before Consumer Releases
**Symptom**: Deadlock or severe performance degradation. The producer warp waits on "ready" but the consumer has not yet signaled "release" on the previous buffer.
**Detection**: Kernel hangs or takes orders of magnitude longer than expected. Ncu shows long stalls on barrier instructions.
**Fix**: Ensure the consumer always calls `consumer_release()` (or arrives on the "ready" barrier) after processing each buffer. Use sufficient pipeline stages so the producer can advance to the next stage while the consumer processes the current one. Verify that the number of stages is at least 2.
**Source**: Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization)

## P2: All Threads Must Participate in Pipeline Creation
**Symptom**: Incorrect pipeline behavior or hang when only producer threads call `cuda::make_pipeline`.
**Detection**: Deadlock at pipeline creation.
**Fix**: `cuda::make_pipeline` is a block-wide operation. ALL threads in the block must reach the call, regardless of their producer/consumer role. The `producer_count` parameter tells the pipeline which threads are producers; this is not the same as only having producers call the function.
**Source**: Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization)

## P3: Shared Memory Exhaustion from Multi-Stage Buffering
**Symptom**: Kernel fails to launch with "too much shared memory requested" error, or occupancy drops severely.
**Detection**: Launch failure, or Ncu shows shared memory is the occupancy limiter.
**Fix**: Reduce the number of pipeline stages. Each stage requires a full buffer in shared memory (e.g., 4 stages of 16KB = 64KB). Balance the number of stages against the available shared memory per SM. Consider using `cudaFuncSetAttribute` to increase the maximum dynamic shared memory.
**Source**: Programming Guide, Section 4.10 (Pipelines); General shared memory constraints

## P4: setmaxnreg Must Be Called Uniformly Across the Warp
**Symptom**: Undefined behavior or hang when `setmaxnreg` is called from a divergent code path where not all warp threads execute it.
**Detection**: Kernel hang or incorrect register state.
**Fix**: Ensure all 32 threads in the warp execute `setmaxnreg` at the same program point. Do not place it inside a branch that only some lanes take. The `.sync` qualifier requires convergence.
**Source**: PTX ISA, setmaxnreg.{inc/dec}.sync.aligned.u32 instruction

## P5: Consumer Warps Starving Due to Insufficient Prefetching
**Symptom**: Consumer warps frequently stall waiting for data, indicating the producer cannot keep up with the consumption rate.
**Detection**: Ncu shows consumer warps spending most time on `consumer_wait` / barrier wait. Low overall compute utilization.
**Fix**: Increase the number of producer warps (e.g., 2 warps producing, 6 consuming). Increase the number of pipeline stages to provide more buffering. Use vectorized loads (128-bit) in the producer to maximize memory bandwidth utilization per producer thread.
**Source**: Programming Guide, Section 4.11.1.3 (Producer-Consumer Pattern Through Warp Specialization)

## P6: Incorrect Buffer Indexing in Circular Pipeline
**Symptom**: Producer overwrites data that the consumer has not yet processed, or consumer reads stale data from a future batch.
**Detection**: Intermittent incorrect results. Data corruption that appears only with specific batch counts.
**Fix**: Use `stage = (stage + 1) % num_stages` consistently for both producer and consumer. Ensure the producer is always `num_stages` batches ahead, not more. The pipeline synchronization (acquire/commit/wait/release) enforces ordering only if the stage indices are correct.
**Source**: Programming Guide, Section 4.10.7 (Producer-Consumer Pattern using Pipelines)

## P7: Warp specialization with cuda::pipeline can zero out global load sector utilization (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld dropped to 0%) if the producer warp's memcpy_async pattern doesn't match the original coalesced access pattern (discovered in verification)

**Symptom**: Warp specialization with cuda::pipeline can zero out global load sector utilization (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld dropped to 0%) if the producer warp's memcpy_async pattern doesn't match the original coalesced access pattern.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: mbarrier-based warp specialization adds significant instruction overhead (557K→1 (discovered in verification)

**Symptom**: mbarrier-based warp specialization adds significant instruction overhead (557K→1.9M instructions) and register pressure (16→18 regs/thread), and warp active percentage collapsed from 75% to 22%, indicating most warps spend their time waiting on barriers rather than doing useful work
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: The optimized kernel actually executes ~4 (discovered in verification)

**Symptom**: The optimized kernel actually executes ~4.5% more instructions (24.2M vs 23.2M) due to the inline PTX overhead of elect.sync, which could hurt in compute-bound scenarios.
**Source**: Level 3 sandbox verification (2026-04-06)

## P10: Shared memory per block nearly doubled (9216 → 17408 bytes), reducing the occupancy limit from shared memory (14 → 9 blocks); for kernels already tight on shared memory this trade-off could backfire (discovered in verification)

**Symptom**: Shared memory per block nearly doubled (9216 → 17408 bytes), reducing the occupancy limit from shared memory (14 → 9 blocks); for kernels already tight on shared memory this trade-off could backfire.
**Source**: Level 3 sandbox verification (2026-04-06)
