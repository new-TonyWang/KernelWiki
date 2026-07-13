# Latency Optimization Techniques

| Technique | When to Consider | Key Indicator |
|-----------|-----------------|---------------|
| [Occupancy Tuning](occupancy-tuning/) | Optimal block size unknown; need to balance register/SMEM usage vs active warps | NCU: `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved occupancy) low |
| [Warp Divergence](warp-divergence/) | Control flow depends on thread ID; branch condition splits warp into divergent paths | NCU: `smsp__thread_inst_executed_per_inst_executed.ratio` < 32 |
| [Stream Concurrency](stream-concurrency/) | Data can be chunked; want overlap of compute with transfer; multiple independent kernels | NCU timeline: sequential execution where overlap is possible |
| [CUDA Graphs](cuda-graphs/) | Same kernel sequence launched repeatedly (training loop, inference pipeline) | NCU: kernel launch overhead visible as gaps in timeline |
| [Kernel Launch Overhead](kernel-launch-overhead/) | Many small kernels; launch overhead dominates execution time | NCU: `gpu__time_duration.sum` << total wall-clock time |
| [Programmatic Dependent Launch](programmatic-dependent-launch/) | Two sequential kernels with independent work phases that can partially overlap | NCU: idle GPU time between dependent kernel pairs |
| [Dynamic Parallelism](dynamic-parallelism/) | Data-dependent work generation; recursive/irregular algorithms; avoid host round-trip | NCU: host-device sync gaps in timeline for work dispatch |
| [Context Management](context-management/) | Multiple contexts on same GPU cause time-slicing; want single primary context | `nvidia-smi` shows multiple processes; context switch overhead |
| [Green Contexts](green-contexts/) | Latency-critical kernel starved by concurrent compute-heavy kernel; CC 9.0+ | NCU: SM utilization uneven across concurrent kernels |
| [GPU Library Usage](gpu-library-usage/) | Standard GEMM/FFT/sparse operations; cuBLAS/cuFFT/cuSPARSE available | Manual kernel underperforms vendor library on same operation |
| [Work Stealing](work-stealing/) | CC 10.0+ (Blackwell); thread block execution times vary; need dynamic load balancing | NCU: tail-effect with some SMs idle while others still working |
