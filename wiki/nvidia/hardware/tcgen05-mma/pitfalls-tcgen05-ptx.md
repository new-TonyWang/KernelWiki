---
id: pitfall-tcgen05-ptx
type: pitfall
vendor: nvidia
title: Pitfalls
---
# tcgen05 Pitfalls

**Note**: All pitfalls listed here are derived from spec analysis and cutlass source inspection. None have been verified on hardware (H200 is sm_90a; Blackwell requires sm_100+).

## 1. Not available on H200 (sm_90a)

**Problem**: Attempting to use `tcgen05.*` PTX instructions on an H200 will fail at compile time (`ptxas` rejects the instruction for `sm_90a`) or at runtime if the PTX is somehow loaded.

**Trigger**: Using `-arch=sm_90a` or `-gencode=arch=compute_90a,code=sm_90a` with code that contains tcgen05 inline PTX.

**Mitigation**: Guard tcgen05 code with `#if __CUDA_ARCH__ >= 1000` or use `-arch=sm_100a`. This KB entry exists for forward planning only.

## 2. Opaque tcgen05 register file

**Problem**: Unlike wgmma (which uses smem descriptors) or mma.sync (which uses standard registers), tcgen05 introduces a separate "tcgen05 register file" that is not directly addressable from C/C++. Data must be loaded via `tcgen05.cp` and read back via `tcgen05.ld`. Attempting to pass standard register values directly to the tcgen05 MMA instruction will fail.

**Trigger**: Trying to use the same inline-PTX patterns from wgmma-ptx with tcgen05 without switching to the tcgen05.cp/ld flow.

**Mitigation**: Follow the cutlass SM100 copy atom pattern: `tcgen05.cp` to load from smem, then issue the MMA referencing the tcgen05 register address, then `tcgen05.ld` to read results back into standard registers.

## 3. CTA group sizing

**Problem**: `tcgen05.cp.cta_group::1` and `tcgen05.cp.cta_group::2` encode different cooperative-group sizes. Using the wrong CTA group size for the kernel's thread configuration will cause incorrect data distribution or hangs.

**Trigger**: Mismatching the `cta_group` parameter with the kernel's actual cluster/CTA configuration.

**Mitigation**: Match the `cta_group` parameter to the number of CTAs participating in the cooperative copy. Consult the cutlass SM100 collective mainloop for reference configurations.
