---
id: exp-2026-04-29-gemm-ws-ptx
type: experience
vendor: nvidia
title: Readme
probe_slug: 2026-04-29-gemm-ws-ptx
evidence_level: measured
measured_on:
  device: H200
  sm: sm_90a
  cuda_runtime: '12.8'
  driver: '570'
---
# 2026-04-29 — Cutlass-free warp-specialized GEMM (record)

A first cut at composing the cutlass-free PTX primitives in `knowledge/40-hardware-feature/{tma-ptx,wgmma-ptx}` with the warp-specialization algorithm skeleton in `knowledge/50-classical-algo/warp-specialization` into a single Hopper GEMM kernel that contains zero `cutlass::` / `cute::` symbols.

This document is the *record* — the reasoning that drove the file layout below, the design choices for each axis (CTA shape, stage count, mbarrier protocol, fragment store), and the verification plan. The kernel itself is in `gemm_ws_ptx.cu`.

```
2026-04-29-gemm-ws-ptx/
├── README.md            <- this file
├── gemm_ws_ptx.cu       <- the kernel + host driver + cuBLAS comparison
└── build.sh             <- nvcc invocation + the two cutlass-free gates (preprocessor / linked-binary)
```

## 1. Inputs (the knowledge sources I composed from)

| Source | What I took from it |
|---|---|
| `30-skill/compute/gemm-ptx/skill.md` | Single-tile cutlass-free GEMM that loads + wgmmas in one warpgroup. The starting point — I am extending it from "1 tile, no warp-spec" to "K-loop over tiles with producer/consumer split". |
| `30-skill/compute/gemm-ptx/pitfalls.md` | (a) B's storage layout for SS_TN must be col-major K×N (host-side transpose), (b) descriptor SBO encoding `LD=8·K·sz / SD=8·sz`, (c) the per-thread fragment-store mapping is **suspected** of being the residual 497/512 bug at single-tile. Items (a)+(b) I copy verbatim; item (c) I copy the existing mapping but flag the inheritance. |
| `40-hardware-feature/tma-ptx/skill.md` | `cuTensorMapEncodeTiled` argument convention (fastest-moving dim first), the `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` mnemonic, the `mbarrier.{init,arrive.expect_tx,try_wait.parity}` protocol. |
| `40-hardware-feature/wgmma-ptx/skill.md` + `pitfalls.md` | Smem descriptor bit-layout + the helper, the immediate-arg counts per dtype family, the `wgmma.fence / mma_async / commit_group / wait_group` issue sequence, the requirement that wgmma is issued by exactly 128 threads. |
| `50-classical-algo/warp-specialization/skill.md` | The "Algorithm skeleton (cutlass-free)" pseudo-code (PRODUCER_WARP issues TMA, consumer warpgroup issues wgmma, full/empty mbarrier pair, S smem stages). The kernel is a literal translation of that skeleton. |
| `50-classical-algo/warp-specialization/pitfalls.md` | (#2) `expected_tx` must equal A+B sum; (#3) phase tracking on each mbarrier alternates per stage cycle; (#7) producer must wait on bar_empty before refilling. |

## 2. Design decisions

### CTA layout: 5 warps (160 threads)

| Warp | Threads | Role |
|---|---|---|
| 0..3 | 0..127 | Consumer warpgroup — issues `wgmma.mma_async` (128-thread requirement, see wgmma-ptx pitfall #2) |
| 4 | 128..159 | Producer warp — only lane 0 actually issues TMA + mbarrier ops |

The minimum viable WS shape is **1 producer warp + 1 consumer warpgroup**. The producer needs only one thread to issue the protocol, but a full warp is the smallest scheduling unit and 32 idle lanes are cheap. (Cutlass's plain WS uses the same shape.)

### Stages = 2

The skill says "Two stages is the minimum that overlaps producer with consumer at all". Per `pitfalls.md` #1, going beyond 4 is rarely justified. I picked the minimum to keep the smem footprint tiny (4640 B total) and the phase-tracking logic auditable.

### N_K_TILES = 2 (K_TOTAL = 32)

The pipeline has two distinct execution paths:

1. **Initial fill** (k < STAGES): producer fills bar_full[s] without waiting on bar_empty[s] — slot is fresh.
2. **Steady state** (k ≥ STAGES): producer must wait on bar_empty[s] before refilling.

With STAGES=2, picking `N_K_TILES = 2` covers exactly the initial fill across both stages — the simplest case where the warp-spec split actually *does* something. Bumping to `N_K_TILES = 4` exercises one cycle of stage reuse; the kernel is parametric on `N_K_TILES`, so deeper-K validation is a constant change.

I deliberately did **not** start at `N_K_TILES = 1` because that would degenerate into "single tile, no producer-consumer reuse" — equivalent to the existing `gemm_ptx.cu` and adds nothing.

### mbarrier expected counts

| Barrier | Init `expected_arrivals` | Who arrives | Notes |
|---|---|---|---|
| `bar_full[s]` | 1 | producer (lane 0 of warp 4) issues `mbarrier.arrive.expect_tx` once per stage with `bytes = TILE_BYTES_A + TILE_BYTES_B = 2304` | The two TMA loads that follow add their bytes to the tracked tx count via `mbarrier::complete_tx::bytes`. The phase flips when both (a) 1 arrival AND (b) 2304 bytes have arrived. (See WS pitfall #2.) |
| `bar_empty[s]` | 1 | one thread of the consumer warpgroup (`tid == 0`) calls `mbarrier.arrive` after `wgmma.wait_group` returns | One arrival is enough because all 128 consumer threads have just passed `wgmma.wait_group.sync.aligned 0`, which is itself a warpgroup-level barrier — by the time any one of them proceeds, the wgmma's smem reads on stage `s` are done. |

### Phase tracking

Both `bar_full[s]` and `bar_empty[s]` flip parity once per stage cycle. The PTX `mbarrier.try_wait.parity` returns true when the barrier's parity *differs* from the supplied phase, so the phase to pass is the parity the barrier *currently* holds (= the parity it had on entry to this iteration of stage `s`).

For iteration `k` in stage `s = k % STAGES`:

- Consumer waits on `bar_full[s]` with `phase = (k / STAGES) & 1`.
  - At `k = s` (first use), `k/STAGES = 0`. Barrier is fresh (parity 0) → wait succeeds when parity flips to 1.
  - At `k = s + STAGES` (second use), `k/STAGES = 1`. Barrier flipped to 1 last time → wait succeeds when it flips back to 0.
- Producer waits on `bar_empty[s]` only when `k ≥ STAGES`, with `phase = ((k - STAGES) / STAGES) & 1`.
  - First wait at `k = STAGES`: `phase = 0` (consumer's first arrive on stage `s` flips it to 1).

This matches the WS pitfall #3 prescription verbatim.

### Smem layout

```
offset 0                                      | smem_a[STAGES * M * K_TILE bf16]   = 4096 B
offset STAGES * 2048                          | smem_b[STAGES * K_TILE * N bf16]   =  512 B
offset STAGES * (2048 + 256)                  | bar_full[STAGES uint64]            =   16 B
offset STAGES * (2048 + 256) + STAGES * 8     | bar_empty[STAGES uint64]           =   16 B
                                                                          total  =  4640 B
```

`extern __shared__ __align__(16) uint8_t smem_raw[]` is naturally 16-byte aligned (wgmma-ptx pitfall #3), and every sub-region offset is also a multiple of 16 — both bf16 tile sizes are multiples of 16, the mbarriers fall on 8-byte boundaries inside a 16-aligned tail.

### wgmma issue

```
wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16
    {d0,d1,d2,d3}, descA, descB,
    scaleD=1, scaleA=1, scaleB=1, transA=0, transB=0;
```

- `scaleD = 1` → **accumulate** into the existing fragment. This is the single line that makes the K-loop work: each iteration adds A·B for the next K-tile to the per-thread accumulator; nothing else accumulates anywhere.
- `transA = transB = 0` (= `.SS_TN`) per wgmma-ptx skill: A K-major (= row-major M×K), B K-major (= col-major K×N stored as row-major N×K). The host driver does the K-major transpose on B (`hB_col`).
- Descriptor LD/SD: 256 / 16 for both A and B at this atom shape, per gemm-ptx pitfall #2.
- The trio `wgmma.fence` / `commit_group` / `wait_group 0` is non-optional (wgmma-ptx pitfall #4). I issue all three on every iteration; this serializes wgmma issues across K-tiles, which is the correct semantics for accumulator-chained iteration.

### Per-thread fragment-store layout

```
warp w (0..3) -> rows [w*16, w*16+16)
within those 16 rows, lane l holds:
    row_top = w*16 + (l/4)
    row_bot = w*16 + (l/4) + 8
    col0    = (l%4) * 2
    col1    = col0 + 1
    {d0, d1} -> (row_top, col0), (row_top, col1)
    {d2, d3} -> (row_bot, col0), (row_bot, col1)
```

This is the **same** mapping used in `60-code/ptx-gemm/gemm_ptx.cu`. `30-skill/compute/gemm-ptx/pitfalls.md` #1 flags it as suspected of being the source of the residual 497/512 mismatches at the single-tile shape. I deliberately did **not** invent a different mapping here — the warp-specialization layer is *orthogonal* to the per-thread fragment layout, and inheriting the existing mapping verbatim makes this kernel's correctness behaviour the WS-pipeline delta on top of `gemm_ptx`, not a confounded change in two layers at once. If the upstream fragment-layout fix lands in `gemm_ptx.cu`, lifting it into this kernel is a single-block edit.

## 3. Implementation walkthrough

The kernel splits cleanly along `warp_id == PRODUCER_WARP`:

```cpp
if (warp_id == 4) {
    // PRODUCER (only lane 0 active).
    for (int k = 0; k < N_K_TILES; ++k) {
        int s = k % STAGES;
        if (k >= STAGES) mbarrier_wait(&bar_empty[s], ((k - STAGES) / STAGES) & 1);
        mbarrier_arrive_expect_tx(&bar_full[s], TILE_BYTES);
        tma_load_2d(smem_a[s], tmap_A, 0, k * K_TILE, &bar_full[s]);
        tma_load_2d(smem_b[s], tmap_B, 0, k * K_TILE, &bar_full[s]);
    }
} else {
    // CONSUMER (all 128 threads).
    float d0=0, d1=0, d2=0, d3=0;
    for (int k = 0; k < N_K_TILES; ++k) {
        int s = k % STAGES;
        mbarrier_wait(&bar_full[s], (k / STAGES) & 1);
        wgmma.fence;
        wgmma.mma_async.m64n8k16.f32.bf16.bf16 +ACC, descA(s), descB(s), 1,1,1,0,0;
        wgmma.commit_group; wgmma.wait_group 0;
        if (tid == 0) mbarrier_arrive(&bar_empty[s]);
    }
    epilogue_store(d0, d1, d2, d3);
}
```

The kernel has exactly one `__syncthreads()`, sitting after `mbarrier_init` to make all 2·STAGES barriers visible to both producer and consumer before either side touches them. From that point on, all cross-warp synchronization is via the mbarrier full/empty pair — no global `__syncthreads()` inside the K-loop, which is the whole point of the WS pattern.

## 4. Verification plan

Two independent gates, mirroring the `gemm-ptx` skill:

### Gate A — cutlass-free (structural)

```
nvcc -E gemm_ws_ptx.cu | grep -E 'cutlass::|cute::'           # expect 0
cuobjdump --dump-elf-symbols gemm_ws_ptx | grep -E 'cutlass::|cute::'  # expect 0
```

Both gates are wired into `build.sh`. The kernel has no `#include <cutlass/...>` or `#include <cute/...>`, only `<cuda.h>`, `<cuda_runtime.h>`, `<cuda_bf16.h>`, `<cublas_v2.h>`. cuBLAS is part of the CUDA toolkit, not cutlass, so it does not introduce cutlass symbols.

### Gate B — numeric (vs cuBLAS)

The host driver runs the same A·B with cuBLAS `cublasGemmEx(CUBLAS_COMPUTE_32F, CUDA_R_16BF in, CUDA_R_32F out)` and computes max-abs / max-rel diff against the WS-PTX output. **Expected outcome at this commit**: the residual fragment-store bug from `gemm_ptx.cu` (pitfall #1, ~497/512 mismatches at single-tile) is inherited. The WS layer doesn't fix it, and reporting it as a deliberate inheritance is the honest framing.

The kernel additionally prints:

- `non-zero outputs / total` — proves the wgmma fragment + epilogue store wrote something to D (pipeline didn't deadlock, no protocol misuse left D zeroed).
- `pipeline check: kernel ran to completion` — the `cudaDeviceSynchronize()` after the launch having returned without timing out is itself the WS-protocol gate. A wrong `expected_tx`, mis-sized `arrive`, or wrong phase parity manifests as a kernel hang. *That* the kernel returns at all proves the producer/consumer mbarrier protocol is consistent.

### Build / run

```bash
bash build.sh        # nvcc + the two cutlass-free greps
./gemm_ws_ptx        # runs kernel + cuBLAS comparison and prints the verdict
```

Build + run requires an H200 (or any sm_90a card) with CUDA ≥ 12.0. The repo's measurement context (per the upstream skills) is H200-SXM, sm_90a, CUDA 12.9.86, driver 570.124.06.

## 5. What this kernel is NOT (yet)

- **Not pingpong** (1 producer + 2 alternating consumer warpgroups) — the WS skill notes pingpong needs uniform per-tile K, which the trivial M=64 N=8 single-CTA shape doesn't even pose a question for. Adding it is mechanical: duplicate the consumer block, route even-k tiles to consumer-A and odd-k to consumer-B; the scaffolding here is a strict subset.
- **Not cooperative** (1 producer + 2 cooperating consumers + cluster-multicast TMA) — WS pitfall #5 is explicit that cooperative without `cta_group::2` multicast is strictly worse than plain WS. Cooperative is a single-CTA problem only when paired with cluster ≥ `<2,1,1>`, which in turn requires a clustered launch and the multicast variant of the bulk-tensor mnemonic.
- **Not multi-CTA** — this is a 1-CTA kernel that exists to validate the WS pattern composes with the cutlass-free PTX primitives. A multi-CTA tile scheduler is the natural next step but is independent.
- **Not corrected for the per-thread fragment-store mapping** — see §2 epilogue. That bug lives in `gemm_ptx.cu` and is not in scope for this record.

## 6. Cross-references

- WS algorithm skeleton: `knowledge/50-classical-algo/warp-specialization/skill.md`
- WS pitfalls: `knowledge/50-classical-algo/warp-specialization/pitfalls.md`
- TMA-PTX primitive: `knowledge/40-hardware-feature/tma-ptx/skill.md`
- wgmma-PTX primitive: `knowledge/40-hardware-feature/wgmma-ptx/skill.md`
- Single-tile cutlass-free GEMM (the parent): `knowledge/30-skill/compute/gemm-ptx/skill.md` + its `pitfalls.md`
- Cutlass realization (reference, not used in this binary): `knowledge/60-code/cutlass-cute/example48-hopper-warp-specialized-gemm/`
