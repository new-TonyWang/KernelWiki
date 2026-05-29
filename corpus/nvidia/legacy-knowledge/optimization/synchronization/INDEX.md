# Synchronization Optimization Techniques

| Technique | When to Consider | Key Indicator |
|-----------|-----------------|---------------|
| [Atomic Reduction](atomic-reduction/) | Reduction (sum/min/max) across many threads; need hierarchical reduction to avoid contention | NCU: `l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum` high with low throughput |
| [Barrier Optimization](barrier-optimization/) | Threads have independent work between signal and wait; split-phase barriers possible | NCU: `smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active` high |
| [Cooperative Groups](cooperative-groups/) | Need sub-block or cross-block synchronization; tiled_partition for sub-warp groups | NCU: `smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active` from coarse sync |
| [Thread Scopes](thread-scopes/) | Atomics/fences wider than necessary; can narrow scope for performance | NCU: `l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum` with wide-scope overhead |
| [Memory Sync Domains](memory-sync-domains/) | Compute and communication kernels concurrent; fences stalled by NVLink/PCIe writes | NCU: `smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active` high |
