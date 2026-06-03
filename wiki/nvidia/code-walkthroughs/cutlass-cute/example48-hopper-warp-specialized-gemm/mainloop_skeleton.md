---
id: code-cutlass-cute-mainloop_skeleton
type: code-walkthrough
vendor: nvidia
title: Mainloop_Skeleton
upstream_repo: NVIDIA/cutlass-cute
---
# Example 48 Mainloop Skeleton (cutlass commit `f74fea9c`)

This is the compact mental model for `48_hopper_warp_specialized_gemm.cu`: one CUTLASS Hopper GEMM that combines TMA, WGMMA, CTA clusters, and a warp-specialized producer / consumer mainloop.

## Template Shape

The key compile-time setup is:

```cpp
using TileShape    = Shape<_128, _128, _32>;
using ClusterShape = Shape<_4,   _2,   _1>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    StageCountAuto,
    KernelScheduleAuto
  >::CollectiveOp;
```

Read this as:

```text
one CTA tile:     128 x 128 x 32
one CTA cluster:   4 x   2 x  1 CTAs
cluster covers:   512 x 256 x 32
```

`CollectiveBuilder` uses these parameters to choose the TMA copy atoms, WGMMA atom, shared-memory layouts, pipeline shape, and schedule policy.

## ClusterShape and TMA Multicast

For GEMM:

```text
C[m,n] += A[m,k] * B[k,n]
```

With `ClusterShape<_4,_2,_1>`, the cluster has 4 CTAs along M and 2 CTAs along N:

```text
        N0        N1
M0   CTA(0,0)  CTA(0,1)
M1   CTA(1,0)  CTA(1,1)
M2   CTA(2,0)  CTA(2,1)
M3   CTA(3,0)  CTA(3,1)
```

That creates two reuse directions:

- A tile reuse: CTAs with the same M and different N share `A[m,k]`; fanout is `ClusterShape.N = 2`.
- B tile reuse: CTAs with the same N and different M share `B[k,n]`; fanout is `ClusterShape.M = 4`.

CUTLASS therefore selects the multicast TMA form:

```text
SM90_TMA_LOAD_MULTICAST
cp.async.bulk.tensor.*.shared::cluster.global.tile...
```

One TMA transaction can read a global tile once and distribute it to multiple CTAs' shared memory inside the hardware cluster.

## Producer / Consumer Mainloop

The mainloop is a staged shared-memory pipeline:

```cpp
// producer role
pipeline.producer_acquire(stage);
cute::copy(tma_atom_a, gA, sA);        // TMA load A tile
cute::copy(tma_atom_b, gB, sB);        // TMA load B tile
pipeline.producer_commit(stage);       // mbarrier arrives for consumers

// consumer role
pipeline.consumer_wait(stage);         // wait for TMA stage to become visible
warpgroup_fence_operand(accum);        // wgmma.fence
cute::gemm(tiled_mma, sA, sB, accum);  // repeated wgmma.mma_async issues
warpgroup_commit_batch();              // wgmma.commit_group
warpgroup_wait<0>();                   // wgmma.wait_group 0
pipeline.consumer_release(stage);      // stage can be reused by producer
```

The producer side is about feeding shared memory with TMA. The consumer side is about issuing WGMMA from shared memory. `PipelineTmaAsync` and mbarriers connect the two sides.

## WGMMA Atom Used

For the TF32 example path, the selected atom is:

```cpp
MMA_Atom<MMA_64x128x8_F32TF32TF32_SS_TN<1,1>>
```

Meaning:

```text
one WGMMA instruction computes: 64 x 128 x 8
CTA tile is:                    128 x 128 x 32
repeat pattern:                   2 x   1 x  4 atom issues
```

For atom-name decoding, see `../wgmma-atom-decoding/wgmma_skeleton.md`.

## Schedule Meaning

In this example, `KernelScheduleAuto` resolves to the cooperative warp-specialized path for the large aligned configuration:

```text
producer role: TMA global -> shared
consumer role: WGMMA shared -> accumulator
cluster role: multicast and cooperative scheduling across CTAs
```

This is why the source file cannot be understood cleanly by reading "TMA" and "warp-specialization" independently. The example's point is the interaction:

```text
ClusterShape enables multicast
multicast feeds the producer side efficiently
PipelineTmaAsync connects producer and consumer
WGMMA consumes the staged smem operands
KernelTmaWarpSpecializedCooperative ties the policy together
```

## Cross-references

- Directory overview: `README.md`
- WGMMA atom decoding: `../wgmma-atom-decoding/wgmma_skeleton.md`
- TMA hardware feature: `wiki/nvidia/hardware/tma/skill.md`
- Warp-specialization algorithm: `wiki/nvidia/techniques/warp-specialization.md`
- Aligned GEMM probe: `sources/experience/api-probes/gemm.md`
- Upstream source: `{{CUTLASS_REPO_REF}}/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu`
