# Benchmark Protocol (N2, skeleton)

> **Version**: MVP skeleton. Four non-negotiable pillars. NVML-backed clock
> policy verification is deferred to M5; until then `clock_policy: unknown`
> is allowed and auto-downgrades `evidence_level` to `inferred`.

The purpose of this protocol is to ensure any two benchmark records in this KB can be compared directly. "Faster" means nothing without warmup, repeat, baseline, and clock policy being identical — or at least explicit.

## Pillar 1 — Correctness before performance

Every benchmark computes a **reference result** and verifies the kernel matches it **before** recording any latency.

- Reference is one of: `torch.<op>`, a hand-written scalar CPU version, or a library call (`cub::Device<Op>`, `cuBLAS*`).
- Tolerance defaults (override in `verified.md` when needed):
  - `fp32` → `max_abs_err <= 1e-5`
  - `fp16` / `bf16` → `max_abs_err <= 1e-3`
  - `int32` → exact match
- On mismatch: abort benchmark, record `status: failed`, observed error, and the inputs that triggered it. **No fast-but-wrong records.**

## Pillar 2 — Warmup and repeat

Use CUDA events for timing. Never use wall-clock or CPU timers for per-launch latency.

- **Warmup**: 5 launches, discarded.
- **Measure**: 20 launches, record median (primary), p10, and p90.
- **One measurement per launch** — no inside-loop timing.
- **Synchronize explicitly** between launches when the kernel is async w.r.t. the host.

Canonical harness (C++):

```cpp
cudaEvent_t e_start, e_end;
cudaEventCreate(&e_start);
cudaEventCreate(&e_end);

for (int i = 0; i < 5; ++i) run_kernel();           // warmup
cudaDeviceSynchronize();

std::vector<float> times_ms;
for (int i = 0; i < 20; ++i) {
    cudaEventRecord(e_start);
    run_kernel();
    cudaEventRecord(e_end);
    cudaEventSynchronize(e_end);
    float ms; cudaEventElapsedTime(&ms, e_start, e_end);
    times_ms.push_back(ms);
}
std::sort(times_ms.begin(), times_ms.end());
float median = times_ms[10];
float p10    = times_ms[2];
float p90    = times_ms[18];
```

Python/torch equivalent is allowed when the kernel is invoked via torch bindings:

```python
for _ in range(5): run_kernel(); torch.cuda.synchronize()
starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
ends   = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
for i in range(20):
    starts[i].record()
    run_kernel()
    ends[i].record()
torch.cuda.synchronize()
times_ms = sorted(starts[i].elapsed_time(ends[i]) for i in range(20))
```

## Pillar 3 — Clock policy

Record whether the GPU clock was locked or free-running.

- `locked`: `nvidia-smi --lock-gpu-clocks=<freq>` was issued and still effective during the run.
- `auto`: default free-running clocks.
- `stable-auto`: free-running but std/mean < 2% across 5 samples 200 ms apart.
- `unknown`: MVP default when NVML is not available. **Auto-downgrades `evidence_level` to `inferred`.**

MVP rule: set `clock_policy: unknown` unless you explicitly locked clocks. Do not guess `stable-auto` — M5 will add a real detector.

## Pillar 4 — Baseline is mandatory

Every measurement records at least one baseline in the **same benchmark table**.

Allowed baselines (pick whichever matches the layer):
- `torch.<op>` — default for Python-level tasks.
- `cub::Device<Op>` / `cub::Block<Op>` — default for CUB-style tasks.
- `cuBLAS` / `cuBLASLt` / `cuDNN` call — for library-comparable kernels.
- A named hand-written baseline when no library path exists (e.g. `naive_reduce_v0`).

Never report a speedup without naming the baseline.

## Mandatory output schema

Every probe record under `80-experience/api-probes/` or `80-experience/hw-probes/` MUST include a table with exactly these columns (MVP: `verified.md` is dropped; benchmark data never lives in `skill.md` directly, it lives in the probe record and `skill.md` links to it):

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |

Where:
- `ratio = baseline_ms / latency_ms_median` (>1 means the kernel is faster than baseline).
- `reproduce_cmd` is a single shell command the reader can copy-paste to re-run.
- Missing any column → `lint_knowledge.py` rejects the file.

## Anti-patterns (lint will reject)

1. Reporting median only, without p10/p90.
2. Recording latency without specifying a baseline.
3. Using `cudaDeviceSynchronize()` as a timer.
4. Timing a loop body that contains multiple kernel launches.
5. Writing `clock_policy: locked` without running the lock command.
6. Comparing against a baseline on different hardware or different dtype.
