// AC-11 ptx-gemm helper. Currently a stub; the gemm_ptx.cu implementation is
// self-contained (descriptor + mbarrier + TMA-issue + wgmma-issue inline-asm
// helpers all defined as __device__ helpers in the .cu). When scaling up to
// multi-tile mainloop iteration, this header will host shared layout / pipeline
// helpers.
#pragma once
