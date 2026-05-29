# Scan (Cumulative) Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Warp-Level Inclusive Scan with Shuffle-Up | Parallel scan within a warp in 5 steps | __shfl_up_sync; no shared memory needed |
| Block-Level Scan (Blelloch) | Up-sweep + down-sweep in shared memory | Work-efficient O(N) algorithm for block-sized data |
| Decoupled Lookback | Inter-block scan without a second kernel pass | Each block publishes partial, looks back for prefix |
| Kogge-Stone Scan | Low-latency scan with more total work | O(N log N) work but fewer steps than Blelloch |
| Multi-Dimensional Scan | Scan along one axis of a multi-dimensional tensor | Thread mapping aligned to scan dimension |
| Segmented Scan | Independent scans over variable-length segments | Flag array marks segment boundaries |
| Stream Compaction via Scan | Filter elements using prefix sum of predicate | Exclusive scan of predicate gives output indices |
