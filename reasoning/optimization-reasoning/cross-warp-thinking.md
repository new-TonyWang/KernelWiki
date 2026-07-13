# Reasoning Pattern: Cross-Warp Thinking

## Core Question
When optimizing at the single-block level, ask yourself: "Am I missing opportunities by not thinking about cooperation across blocks or across the cluster?"

## Thinking Framework

### Step 1: Identify what each block does independently
- What data does each block load from global memory?
- Is there overlap between the data loaded by neighboring blocks?
- Are blocks producing partial results that need to be combined?

### Step 2: Check for cross-block cooperation opportunities
- **Shared data**: Can multiple blocks share the same loaded data via multicasting or caching?
- **Shared computation**: Can blocks cooperate on a larger computation unit (e.g., 2-CTA MMA)?
- **Communication**: Can blocks exchange intermediate results via distributed shared memory (DSMEM)?
- **Scheduling**: Can blocks coordinate to improve cache behavior or load balancing?

### Step 3: Evaluate the cooperation mechanism
- Thread block clusters: guarantee co-scheduling on same GPC, enable DSMEM
- TMA multicast: load data once, deliver to multiple CTAs in cluster
- Distributed shared memory: CTAs in cluster can read/write each other's SMEM
- Cluster-level barriers: synchronize across CTAs in a cluster
- Trade: cluster cooperation adds synchronization overhead; only beneficial when data sharing is significant

## Examples from Literature

### Example 1: TMA Multicast -- Don't Load the Same Tile Twice
**From:** CUTLASS GEMM with Thread Block Clusters on Blackwell (Colfax)

**Single-block view:** Each CTA in a GEMM grid loads its own bMxbK tile of A and bNxbK tile of B from global memory. CTAs in the same column of the output grid load the same A tile; CTAs in the same row load the same B tile.

**Cross-block insight:** With a 2x2 cluster, 4 CTAs share 2 A tiles and 2 B tiles. Without multicasting, each CTA loads 2 tiles = 8 total loads. With TMA multicast, each tile is loaded once and delivered to all CTAs that need it = 4 total loads. This halves global memory traffic.

**Implementation:** Each CTA constructs a bitmask indicating which cluster members share its operand tile. For operand A, the mask includes all CTAs in the same row of the cluster. For B, all CTAs in the same column. The TMA instruction uses this `ctaMask` to multicast.

**Example for 4x4 cluster, CTA 0:**
- `tma_bitmask_a = 0x1111` (CTAs 0, 4, 8, 12 -- same row)
- `tma_bitmask_b = 0x000f` (CTAs 0, 1, 2, 3 -- same column)
- Each CTA loads only its 1/4 slice of the tile; multicast delivers it to all participants

### Example 2: 2-CTA MMA -- Two Blocks as One Compute Unit
**From:** CUTLASS GEMM with Thread Block Clusters, FlashAttention-4 (Colfax)

**Single-block view:** Each CTA computes a bMxbN output tile using its own tensor core resources. The MMA atom size is limited by single-SM resources.

**Cross-block insight:** Blackwell's 2-CTA MMA (`tcgen05.mma` with cta_group::2) allows a pair of CTAs to cooperate on a single MMA spanning both SMs' TMEM and tensor cores. The effective atom size doubles to 256x256x16 (from 128x256x16). This reduces per-CTA shared memory traffic because operand B is shared across the pair.

**Benefit in attention backward:** SMEM bandwidth is the bottleneck. 2-CTA MMA roughly halves the per-CTA operand B traffic. Additionally, the dQ reduction that would require atomics across CTAs now happens within the CTA pair via DSMEM exchange.

### Example 3: DSMEM Exchange for Cross-CTA Data Sharing
**From:** FlashAttention-4 backward pass (Colfax)

**Single-block view:** In attention backward, each CTA computes dS (gradient of attention scores). For the dQ GEMM, the reduction axis is split across the 2-CTA pair, meaning each CTA holds half the dS data.

**Cross-block insight:** Instead of writing partial dS to global memory and re-reading it, the two CTAs exchange their halves of dS via distributed shared memory (DSMEM). Each CTA writes its half to SMEM, and the partner CTA reads it via `ld.shared::cluster`. This repacks dS so each CTA has the full reduction data for its output rows, enabling the dQ MMA without global memory round-trip.

**Key design decision:** The pipeline is reordered to overlap DSMEM exchange latency with other computation. Compute dP for the current tile while exchanging dS and computing dQ for the previous tile.

### Example 4: Cluster-Level Synchronization -- Not Just cluster_sync
**From:** CUTLASS Blackwell examples (Colfax)

**Naive approach:** Use `cute::cluster_sync()` after every pipeline stage to ensure all CTAs are done before any issues new TMA loads.

**Cross-block insight:** `cluster_sync()` is overkill because not all CTAs share all data. CTA 0 only needs to wait for CTAs that share its A tile (same row) and its B tile (same column), not ALL CTAs in the cluster.

**Solution:** Use targeted mbarrier synchronization. The post-MMA barrier's bitmask is the bitwise OR of the A and B multicast bitmasks. For CTA 0 in a 4x4 cluster: `mma_bitmask = 0x1111 | 0x000f = 0x111f`. This allows CTAs that don't share data to run ahead independently, reducing synchronization overhead.

## When This Pattern Applies
- Multiple blocks load the same data (e.g., shared A/B tiles in GEMM, shared KV in attention)
- The per-block problem is small relative to the hardware (under-utilizing SMs)
- Inter-block reduction is needed (e.g., Split-K, dQ accumulation)
- SMEM or register pressure per block is the bottleneck (spreading across blocks helps)
- Architecture supports clusters (Hopper+) with DSMEM and TMA multicast
- Moving from Hopper to Blackwell: 2-CTA MMA is a new cross-block cooperation mode
