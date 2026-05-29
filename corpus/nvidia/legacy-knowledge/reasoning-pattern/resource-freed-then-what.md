# Reasoning Pattern: Resource Freed Then What

## Core Question
When an optimization eliminates a bottleneck or frees a resource, ask yourself: "Now that this resource is freed, what new opportunity does that create?"

## Thinking Framework

### Step 1: Identify what was freed or reduced
- A memory access was eliminated (e.g., no longer need to write intermediate to HBM)
- A compute unit was freed (e.g., producer warps are idle after issuing TMA)
- Register pressure was reduced (e.g., TMA handles index computation, freeing registers)
- SMEM capacity was freed (e.g., eliminated a buffer that's no longer needed)
- A latency was hidden (e.g., pipelining hides copy latency)

### Step 2: Check cascading opportunities
- Can we use the freed resource to increase the problem size we handle?
  - Larger tiles, deeper pipelines, more accumulators in flight
- Can we use the freed resource to do additional work?
  - Fuse another operation, overlap a different computation
- Can we trade the freed resource for a different kind of improvement?
  - Convert freed SMEM into more pipeline stages
  - Convert freed registers into larger tile sizes
  - Convert hidden latency into higher throughput

### Step 3: Evaluate the second-order effects
- Does the new opportunity shift the bottleneck somewhere else?
- Are there constraints that prevent exploiting the freed resource?
- What is the marginal value of this resource given the current bottleneck?

## Examples from Literature

### Example 1: FlashAttention -- Eliminating HBM Access Enables Larger Tiles
**From:** FlashAttention (Dao et al.) / FlashAttention-2 Hopper case study (Colfax)

**What was freed:** By computing attention tile-by-tile with online softmax, the full N x N attention matrix never needs to be materialized in HBM. This eliminates O(N^2) memory traffic.

**Cascading opportunity:** Since HBM bandwidth is no longer the bottleneck, the kernel can now use larger tile sizes (Br, Bc) to increase arithmetic intensity. The freed memory bandwidth budget means we can invest it in loading larger Q, K, V tiles per iteration. Larger tiles also mean fewer iterations of the outer loop, reducing per-iteration overhead (synchronization, softmax statistics updates).

**Second-order effect:** With larger tiles, register pressure increases (need to hold larger accumulators). This shifts the bottleneck to registers, which is then addressed by warp specialization with setmaxnreg.

### Example 2: TMA Frees Registers -- Enables Deeper Pipelines
**From:** CUTLASS Pipelining Tutorial (Colfax)

**What was freed:** On Hopper, TMA handles all address calculation and boundary predication in hardware. This frees the registers that previously held loop indices, predicate masks, and address calculations (significant on Ampere where cp.async still needs register-held addresses).

**Cascading opportunity:** The freed registers can be re-invested in:
- More pipeline stages (each stage needs register state for pipeline tracking)
- Larger tile sizes (larger WGMMA atoms, wider N dimension)
- Holding multiple accumulators simultaneously (critical for fused kernels like FlashAttention)

**Second-order effect:** With more pipeline stages and larger tiles, SMEM consumption increases. The optimization chain is: TMA frees registers -> larger tiles/more stages -> more SMEM needed -> must balance against SMEM capacity limit.

### Example 3: FlashAttention-4 -- Software Exp Frees SFU Bottleneck
**From:** FlashAttention-4 (Colfax)

**What was freed:** On Blackwell, the SFU (special function unit) for exponential is a bottleneck: tensor cores are 512x faster than SFU. FlashAttention-4 offloads part of the exp computation to FMA units via polynomial approximation, effectively doubling exp throughput.

**Cascading opportunity:** With the SFU bottleneck partially relieved, the kernel can now better overlap softmax with MMA. The pingpong schedule between warpgroups becomes more effective because the softmax phase completes faster, keeping tensor cores busy for a larger fraction of time.

**Second-order effect:** Using FMA for software exp means those FMA units are unavailable for other elementwise work. But since FMA units are abundant and tensor cores handle the heavy computation, this trade is favorable.

## When This Pattern Applies
- After successfully eliminating a memory bottleneck (coalescing, caching, tiling)
- After reducing register pressure (TMA, setmaxnreg, moving data to TMEM)
- After hiding a latency (pipelining, async operations)
- After fusing operations (fewer kernel launches = freed launch overhead budget)
- Whenever a profiling run shows that the "old" bottleneck is no longer the top issue
