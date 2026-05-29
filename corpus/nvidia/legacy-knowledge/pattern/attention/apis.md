# Attention -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__expf(x)` | Math Intrinsics | Fast variant of expf(x) |
| `__hmax(a,b)` | Math Intrinsics | Maximum |
| `__hmax2(a,b)` | Math Intrinsics | Packed maximum |
| `expf(x)` | Math Intrinsics | Base-e exponential e^x |
| `fmaxf(x,y)` | Math Intrinsics | Maximum (NaN-safe) |
| `fminf(x,y)` | Math Intrinsics | Minimum (NaN-safe) |
| `h2exp(__half2)` | Math Intrinsics | Fast variant of hexp(__half) |
| `h2exp(__nv_bfloat162)` | Math Intrinsics | Fast variant of hexp(__nv_bfloat16) |
| `hexp(__half)` | Math Intrinsics | Base-e exponential |
| `hexp(__nv_bfloat16)` | Math Intrinsics | Base-e exponential |
| `tanh.approx .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision approximate tanh |
| `tanh.approx.f32` | PTX ISA | Fast approximate tanh (activation function) |
| `tcgen05.mma[.kind][.shape]` | PTX ISA | 5th-gen TensorCore MMA (supports M=64/128/256 via CTA groups) |
| `wgmma.mma_async.sync.aligned.shape.dtype.atype.btype` | PTX ISA | Warpgroup-level async MMA (4 warps = 128 threads) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fmax(x,y)` | Math Intrinsics | Maximum (NaN-safe) |
| `fmin(x,y)` | Math Intrinsics | Minimum (NaN-safe) |
