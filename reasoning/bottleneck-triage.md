# Bottleneck → Next Skill (simplified)

> **Version**: MVP skeleton. No NCU metric auto-selection; no limit_factor
> driven routing. Those depend on `kp_introspect` being fully wired and are
> deferred to M5+.

After a benchmark run, this document tells the agent which skill to pull next when the current implementation does not meet the success criteria.

## Inputs

- The latest `verified.md` table (from the skill being built or the probe being run).
- `kp_introspect bundle` output (`artifacts.introspection`).
- Optional: `ncu` output for the same shape (not required in MVP).

## Decision tree

```
Q1. Did correctness fail?
    YES → Stop. Do not pick a new skill until correctness is restored.
          Record the failure in pitfalls.md.
    NO  → Continue.

Q1b. Does wall-clock exceed on-GPU kernel time by a fixed host-side offset?
    Symptom: nsys per-launch duration > NCU kernel duration by ≈1.7 μs on H200,
    OR NCU Compute SOL < 10 % on a tiny-grid kernel, OR you are launching
    many small kernels and per-launch overhead dominates cumulative time.
    YES → pull 90-system-level/launch-overhead/ first; kernel-internal skills
          will not move wall-clock until the launch floor is addressed (or the
          workload is batched / graph-captured).
    NO  → Continue.

Q2. Is `kernel.dynamic.limit_factor` known from the introspection bundle?
    register-bound → pull wiki/nvidia/foundations/memory/register-pressure/
                   + wiki/nvidia/foundations/compute/compiler-hints/
                   (reduce regs/thread via __launch_bounds__, -maxrregcount,
                    or by splitting loops)
    shmem-bound    → pull wiki/nvidia/foundations/memory/bank-conflict/
                   + check vectorized-access / layout-transform
                   (reduce shmem per block or the bank-conflict factor)
    thread-bound   → DEFERRED in MVP — report and ask human.
    unknown        → Continue to Q3.

Q3. Compare median latency against baseline_ms:
    ratio >= 0.95 → Chain converged. Finalize verified.md and stop.
    0.5 <= ratio < 0.95 → Apply Q4 micro-diagnosis.
    ratio < 0.5 → Fundamental approach wrong. Re-check the pattern INDEX.md
                  (library-fallback may be the right answer) and stop.

Q4. Micro-diagnosis by inspection (no NCU required):
    - Reads/writes memory with stride > 1 per thread
      → pull wiki/nvidia/foundations/memory/coalescing/
    - Per-element scalar load/store when dtype * 4 fits a vector
      → pull wiki/nvidia/foundations/memory/vectorized-access/
    - Thread 0 (or lane 0) does a serial reduction
      → pull wiki/nvidia/foundations/compute/warp-primitives/
    - Small tight loop inside kernel without #pragma unroll / ILP
      → pull wiki/nvidia/foundations/compute/ilp/
    - Heavy use of sinf/cosf/expf/logf on fp32 with relaxed precision ok
      → pull wiki/nvidia/foundations/compute/fast-math/
    - Narrow reduction into smem with heavy bank contention
      → pull wiki/nvidia/foundations/memory/bank-conflict/
    - Data race or inconsistent cross-thread reads
      → pull wiki/nvidia/foundations/sync/memory-ordering/

Q4b. Bytes-in-flight analysis (Little's Law, from GTC25-S72683):
    If Q4 skills have been applied but BW utilization (from ncu or wall-clock
    vs theoretical peak) is still < 85%:

    Estimate bytes-in-flight per SM:
      BiF = loads/thread × bytes/load × threads/block × blocks/SM

    Target: H200 needs ~64 KiB/SM for >90% BW utilization.
    If BiF < target:
      - Already using unroll + vectorized loads (register prefetch)?
        → consider wiki/nvidia/foundations/memory/async-copy/ (LDGSTS + cuda::pipeline)
        → async copies skip the register file, freeing regs for compute
        → BUT: only helps compute-heavy or iterative kernels (GTC25-S72683
          showed trivial a*b got NO benefit; sqrt-heavy got 1.3× uplift)
      - Still not enough? Check stage count:
        stages_needed = ceil(target_BiF / (2 × bytes/load × threads/block × blocks/SM))
        → H200 with 256 threads/block at 100% occ: 2 stages minimum

    See optimization flow chart:
    ![Optimization guidelines](/data1/tongyu/workspace/gtc_videos/cuda_fundamentals/gtc25-s72683_work/frames/interval_0215.jpg)

Q5. After pulling a skill and modifying the kernel, rerun the benchmark
    and go back to Q1.
```

## Loop cap

**Max 3 iterations** per task. After three rounds, if the success criteria still are not met, record `status: stuck_at_<ratio>` in the output and surface the case in the done-report for human review. Do not spin forever.

## What goes into `pitfalls.md` each iteration

Each iteration that did not fully converge contributes one entry to `pitfalls.md` of the skill being built, with:
- observed symptom (e.g. `ratio 0.72 vs torch.sum, dtype=fp32, shape=4096`)
- hypothesis (e.g. `non-coalesced loads`)
- skill pulled (e.g. `coalescing`)
- outcome (e.g. `ratio 0.94 after vectorized `float4` loads`)

## Out of scope (skeleton version)

- NCU metric auto-selection by bottleneck hypothesis (full I3/I4 of booklet 05).
- Automatic skill suggestion via the `limit_factor` decision tree in booklet 07 §R.8.6.
- Multi-skill composition (some tasks need to apply 2 skills simultaneously — MVP picks one at a time).
