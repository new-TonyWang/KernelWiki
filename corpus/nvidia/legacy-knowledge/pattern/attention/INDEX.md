# Attention Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Online Softmax with Tiled Rescaling | Avoid materializing NxN attention matrix | O(N) memory via running max/sum statistics |
| Overlapping GEMM and Softmax | Hide softmax latency behind tensor core work | Inter/intra-warpgroup pipelining |
| FlashAttention Tiling | Tile-by-tile Q*K^T*V computation | SRAM-resident intermediate results |
| Causal Masking | Skip tiles above the diagonal | Early-exit for fully-masked tiles |
| Multi-Query / Grouped-Query Attention | Shared K/V heads across Q heads | Reduced memory bandwidth for K/V |
| Backward Pass (FlashAttention) | Recompute attention on-the-fly in backward | Avoid storing full attention matrix for gradient |
| Software Exp Emulation (FA4) | Distribute exp across MUFU + FMA | Nearly doubled effective exp throughput on Blackwell |
