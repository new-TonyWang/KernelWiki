# Instruction Level Parallelism -- Pitfalls

## P1: Over-Unrolling Causes Register Spilling
**Symptom**: Performance degrades despite increased ILP. Ncu shows high local memory (LMEM) traffic.
**Detection**: `--ptxas-options=-v` reports register usage exceeding the architecture limit per-SM divided by target occupancy. Ncu `l1tex__t_sectors_pipe_lsu_mem_local_op_*` counters are non-zero.
**Fix**: Reduce the unroll factor. Use `#pragma unroll N` with a specific N instead of full unrolling. Monitor register count with `--ptxas-options=-v` and keep it below the spill threshold.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure); Section 11.2 (Hiding Register Dependencies)

## P2: ILP Without Independent Instructions
**Symptom**: Unrolling a loop that has cross-iteration dependencies does not improve throughput; it just increases code size.
**Detection**: No improvement in instruction throughput after unrolling. Ncu shows the same warp stall reasons (wait on dependency).
**Fix**: Ensure loop iterations are actually independent. For reduction loops, manually break the dependency chain with multiple accumulators. For loops with loop-carried dependencies, software pipelining (load-ahead) may help partially.
**Source**: Best Practices Guide, Section 11.2 (Hiding Register Dependencies)

## P3: Excessive ILP Reduces Occupancy Below Useful Threshold
**Symptom**: High ILP per thread but overall throughput is low because very few warps are resident.
**Detection**: Ncu occupancy metric shows < 25% achieved occupancy, and the kernel is not memory-bound.
**Fix**: Reduce the ILP factor to free registers. Even moderate ILP (2-4x) combined with moderate occupancy (50%) often outperforms extreme ILP (8x+) with very low occupancy. Experiment to find the sweet spot.
**Source**: Best Practices Guide, Section 11.3 (Thread and Block Heuristics)

## P4: Ignoring Memory-Level Parallelism When Focusing on Compute ILP
**Symptom**: Kernel is memory-bound but developer focuses on arithmetic ILP optimization.
**Detection**: Ncu shows compute utilization is low while memory throughput is near peak.
**Fix**: For memory-bound kernels, focus on memory-level parallelism (MLP) rather than compute ILP. Issue multiple independent loads per thread so that memory requests can pipeline. The same per-thread multi-element pattern helps for both MLP and compute ILP.
**Source**: Programming Guide, Section 3.2.2.2 (Hardware Multithreading)

## P5: Forgetting That the Compiler Already Unrolls Small Loops
**Symptom**: Adding `#pragma unroll` to a loop that is already fully unrolled by the compiler yields no improvement but adds compile time.
**Detection**: Compare PTX/SASS output with and without the pragma; if identical, the compiler already unrolled.
**Fix**: Only use `#pragma unroll` when the compiler's default heuristic is insufficient (e.g., large trip counts, or when you want partial unrolling with a specific factor).
**Source**: Programming Guide, Section 5.4.9.1 (#pragma unroll)

## P6: Register usage doubled from 16 to 32 per thread, cutting occupancy_limit_registers from 16 to 8 — on more register-heavy kernels this could become the occupancy bottleneck and negate the ILP benefit (discovered in verification)

**Symptom**: Register usage doubled from 16 to 32 per thread, cutting occupancy_limit_registers from 16 to 8 — on more register-heavy kernels this could become the occupancy bottleneck and negate the ILP benefit.
**Source**: Level 3 sandbox verification (2026-04-05)

## P7: The naive per-thread multi-element pattern (thread k loads elements [k*N  (discovered in verification)

**Symptom**: The naive per-thread multi-element pattern (thread k loads elements [k*N .. k*N+N-1]) breaks coalescing; must use a strided access pattern (thread k loads elements [k, k+blockDim*gridDim, k+2*blockDim*gridDim, ...]) so that adjacent threads still access adjacent addresses within each load phase
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Software pipelining increases register usage and instruction count; on bandwidth-bound kernels with low compute intensity, these costs can negate any latency-hiding benefit (discovered in verification)

**Symptom**: Software pipelining increases register usage and instruction count; on bandwidth-bound kernels with low compute intensity, these costs can negate any latency-hiding benefit.
**Source**: Level 3 sandbox verification (2026-04-06)
