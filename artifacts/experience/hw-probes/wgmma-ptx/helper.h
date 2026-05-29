// wgmma-ptx helper. Currently a stub — the wgmma_hello.cu implementation is
// self-contained (descriptor construction inlined as a __device__ helper). When
// scaling up to m64n128k8 tf32 + multi-iteration, this header will host the
// shared descriptor / fragment-layout helpers.
//
// Kept as a placeholder so the canonical artifact-bundle layout
// (<probe>.cu + helper.h + build.sh + run.sh + device.json + profiles/*.csv)
// is satisfied; future expansions of the cutlass-free wgmma family will populate it.
#pragma once
