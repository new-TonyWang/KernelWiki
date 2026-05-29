# Hardware Microbenchmark Protocol (skeleton)

> **Version**: MVP skeleton. Companion to `api-probing.md`. Covers three
> categories of **active** microbenchmark probes that run real kernels to
> surface timing/throughput facts the passive introspection APIs cannot.
>
> **Authoritative design source**: `docs/design/知识库的问题以及改进方案/方案分册/07-runtime-introspection-方案.md` §R.4.5
>
> If this file and booklet 07 §R.4.5 disagree, booklet 07 wins.

The purpose of a hardware microbenchmark is to turn *"the driver says SM count = 132"* into *"a `fma.rn.f32` costs 0.5 cycles throughput and 4 cycles latency on this exact H200, at this exact clock, with these exact drivers."* Microbench results are the primary source for any `verified.md` claim about **per-instruction cost** or **per-level memory latency**.

## Three probe kinds

| Kind | Purpose | MVP scope |
| --- | --- | --- |
| **Compute latency** | Per-instruction throughput/latency for CUDA core instructions (`fma` / `shfl` / `vote` / `ld.shared` / `ld.global` / `atom`) | **In scope**. Two data sources co-exist: (a) canonical per-instruction numbers from `kp_introspect microbench compute-latency` (pre-built CLI), (b) skill-specific probes written by the KB-gen agent for each skill following this protocol. Both land as records under `80-experience/hw-probes/`; (a) under `80-experience/hw-probes/canonical/`, (b) under `80-experience/hw-probes/<instruction-slug>/`. |
| **Memory pointer chasing** | End-to-end read latency per memory level (SMEM same-bank / SMEM full-width / L1 / L2 / HBM) | **In scope**. Same dual-source arrangement: (a) canonical curves from `kp_introspect microbench memory-latency`, (b) agent-written skill-specific probes for coalescing / bank-conflict. |
| **Feature sweep (whitepaper-driven)** | Agent reads an architecture whitepaper, extracts a feature ledger, designs one probe per feature (TMA / DSMEM cluster sweep / wgmma / setmaxnreg) | **Deferred** — scaffolded only; execution waits for F0 source corpus + 40-hardware-feature layer. |

## Shared rules

Every microbench probe must obey:

1. **`benchmark-protocol.md`** — warmup 5, repeat 20, CUDA events, correctness first.
2. **`api-probing.md` Step 1** — grep upstream **before** writing any kernel.
3. **`api-probing.md` Step 3–4** — same compile / run / record loop.
4. **`templates/frontmatter/experience.yaml`** — same frontmatter schema.

Additional microbench-specific constraints (enforced by `lint_knowledge.py`):

1. **Isolate one effect**. A compute-latency probe must not be confounded by memory stalls; a memory-latency probe must not measure compute. If you cannot isolate, split into two probes.
2. **Defeat predictors**. Pointer chasing uses randomized node order. Compute loops use `volatile` / inline `asm` to prevent dead-code elimination.
3. **Read the clock correctly**. Prefer `%clock64` for cycle-level measurement; fall back to `cudaEventElapsedTime × observed_sm_clock_mhz` (from `kp_introspect device-dynamic`) only when `%clock64` is unreliable.
4. **Sample ≥ 4096 per data point** — not per kernel launch. A single launch collects thousands of samples via an unrolled inner loop.
5. **Record the clock**. Every probe records `observed_clock_mhz`; if `clock_policy: unknown` the probe is still valid but `evidence_level` stays `inferred`.

## Protocol — Compute latency probe

```
Step 1. Pick the instruction (e.g. shfl.sync.bfly under warp-reduce mask).
Step 2. Grep upstream:
          rg 'shfl\.sync\.bfly' 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/ptx-isa/
Step 3. (Optional) Sanity-check against the canonical baseline:
          kp_introspect microbench compute-latency --instructions shfl.sync.bfly --samples 4096 \
              --output 80-experience/hw-probes/canonical/shfl-sync-bfly.json
        Canonical data is general-purpose; the skill-specific probe below
        exercises the exact pattern this skill uses (e.g. with a specific mask,
        under warp-reduce order, with specific dtype).
Step 4. Write probe .cu:
          - kernel body = unrolled loop of 128 dependent instructions
          - single thread, single block, minimal setup
          - uint64_t s = clock64(); <128 insn>; uint64_t e = clock64(); out[tid] = e - s;
          - use `volatile` / inline asm to block DCE
Step 5. nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o probe probe.cu
Step 6. Run N=4096 outer trials; collect all (e - s); divide each by 128
          → per-instruction throughput in cycles.
Step 7. Repeat with pipeline-empty mode (single instruction between bar.sync)
          → per-instruction latency in cycles.
Step 8. Write the probe record to
          80-experience/hw-probes/<instruction-slug>/<YYYY-MM-DD>-<task-slug>.md
        using templates/frontmatter/experience.yaml. Then in the target
        skill.md, append a bullet under `## Measured Characteristics` that
        links to the probe record. **Do not write to skill verified.md**
        (dropped in MVP). **Do not embed the table inline in skill.md**
        (the probe record is the single source of truth).
```

## Protocol — Memory pointer chasing probe

```
Step 1. Pick a level: smem-same-bank | smem-full-width | l1 | l2 | hbm
Step 2. Pick working-set bytes so the list fits exactly this level:
          - smem     → 32 KB - 192 KB (bounded by opt-in shmem)
          - L1       → 24 KB - 48 KB (< L1 capacity)
          - L2       → 40 MB (> L1, < L2 capacity)
          - HBM      → 2 GB (> L2 capacity)
Step 3. (Optional) Sanity-check against canonical:
          kp_introspect microbench memory-latency --levels <level> --samples 4096 \
              --output 80-experience/hw-probes/canonical/memory-<level>.json
Step 4. Allocate a linked list; randomize next-pointer order.
Step 5. Kernel body: block 0, thread 0 walks list for N hops.
          uint64_t s = clock64(); walk(N); uint64_t e = clock64(); out[0] = e - s;
Step 6. nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o probe probe.cu
Step 7. Run 4096 trials → median / p10 / p90 of cycles-per-hop.
Step 8. Write the probe record to
          80-experience/hw-probes/<level-slug>/<YYYY-MM-DD>-<task-slug>.md
        and link it from the target skill.md's ## Measured Characteristics
        section. **Not in skill verified.md** (dropped in MVP).
```

## Protocol — Feature sweep (deferred execution)

See booklet 07 §R.4.5.3 for the full six-step workflow. MVP only scaffolds this path; actual execution waits for:

- F0 source corpus formalization (so the whitepaper location is canonical), and
- 40-hardware-feature layer (so the sweep results have a facts-layer home).

When R11/R12 become available, this file will be extended with the whitepaper → ledger → grid probe workflow. Until then, if a task requires DSMEM / TMA / wgmma / setmaxnreg microbench, stop and record `status: blocked`.

## Agent decision: which protocol to use

| Task asks for... | Use... |
| --- | --- |
| Semantics of a single API (does `__shfl_xor_sync` support lane masking on Hopper?) | `api-probing.md` |
| Cycle-level cost of an instruction (how fast is `fma.rn.f32`?) | This file — compute latency protocol |
| Memory-level latency curve (SMEM vs L1 vs L2 vs HBM) | This file — pointer-chasing protocol |
| Bandwidth or latency of a feature the whitepaper announces (TMA, DSMEM cluster, wgmma) | Record `status: blocked` with reference to `70-reasoning/hardware-microbench.md` §deferred — MVP does not execute feature sweeps |

## Out of scope (MVP)

- DSMEM cluster-size sweeps (tensor-core / Hopper territory).
- TMA bandwidth characterization (tensor-core territory).
- wgmma throughput per tile/dtype (tensor-core territory).
- Whitepaper-driven feature-sweep execution end-to-end.
- NVML-based clock_policy verification (until M5).
