# Memory Optimization Techniques

| Technique | When to Consider | Key Indicator |
|-----------|-----------------|---------------|
| [Coalescing](coalescing/) | Any kernel reading/writing global memory; thread index maps to data index | NCU: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` much higher than ideal |
| [Vectorized Access](vectorized-access/) | Kernel is memory-bandwidth bound; each thread processes multiple contiguous elements | NCU: `dram__bytes.sum.per_second` below peak; `sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` < 100% |
| [Shared Memory Cache](shared-memory-cache/) | Multiple threads reuse the same global data; temporal locality exists | NCU: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` high due to repeated loads |
| [Bank Conflict Avoidance](bank-conflict-avoidance/) | Column-access patterns in shared memory; tile dimension matches bank count (32) | NCU: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` > 0 |
| [Data Prefetch](data-prefetch/) | CC 8.0+; copying global to shared where load+store wastes register bandwidth | NCU: high `smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct` with register spills |
| [Cache Load Hints](cache-load-hints/) | Read-only data paths; want texture cache or specific L1/L2 caching behavior | NCU: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` low |
| [L2 Cache Control](l2-cache-control/) | CC 8.0+; kernel repeatedly accesses a hot data region that fits in L2 | NCU: `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` high for hot data |
| [Register Pressure](register-pressure/) | Kernel uses too many registers, limiting occupancy or causing local memory spills | NCU: `launch__registers_per_thread` high; `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` > 0 |
| [Layout Transform](layout-transform/) | Strided access patterns that prevent coalescing; need matrix transpose | NCU: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` >> theoretical minimum |
| [Host-Device Transfer](host-device-transfer/) | Frequent data movement between CPU and GPU; need maximum PCIe/NVLink bandwidth | NCU: transfer time dominates in timeline; `nvprof` shows low overlap |
| [Unified Memory](unified-memory/) | Simplified memory management needed; data shared between CPU and GPU at different times | NCU: page fault counts via `gpu__dram_page_fault.sum` |
| [NUMA Binding](numa-binding/) | Multi-socket Linux system; GPU performance degraded by OS page migration | `numactl --hardware` shows cross-socket traffic |
