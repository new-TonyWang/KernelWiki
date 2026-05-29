# Atomic Reduction -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `atom[.scope].add.vec.f32` | PTX ISA | Vectorized atomic FP32 add (.v2/.v4) |
| `atom[.sem][.scope].add.noftz.f16/bf16` | PTX ISA | Atomic FP16/BF16 add |
| `atom[.sem][.scope][.space].op.type` | PTX ISA | Atomic read-modify-write (.add/.min/.max/.inc/.dec/.and/.or/.xor/.exch/.cas) |
| `cp.reduce.async.bulk.tensor[.op]` | PTX ISA | TMA tensor async copy with reduction |
| `cp.reduce.async.bulk[.op]` | PTX ISA | Async bulk copy with reduction |
| `red[.sem][.scope][.space].op.type` | PTX ISA | Reduction (no return value); same ops as atom |
| `atomicAdd()` | Runtime API | Atomic addition |
| `atomicCAS()` | Runtime API | Atomic compare-and-swap |
| `cuda::atomic` | Runtime API | Scoped atomic operations (thread/block/device/system scope) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `atom.cas.b128 / atom.exch.b128` | PTX ISA | 128-bit atomic CAS and exchange |
| `multimem.cp.reduce.async.bulk` | PTX ISA | Multi-memory bulk async copy+reduce |
| `multimem.ld_reduce[.op][.type]` | PTX ISA | Load+reduce from multiple memory locations |
| `multimem.red[.op][.type]` | PTX ISA | Multi-memory reduction |
| `multimem.st[.type]` | PTX ISA | Multi-memory store |
| `red.async[.sem].scope.space.op.type` | PTX ISA | Asynchronous reduction |
| `atomicAnd()` / `atomicOr()` / `atomicXor()` | Runtime API | Atomic bitwise operations |
| `atomicExch()` | Runtime API | Atomic exchange |
| `atomicMin()` / `atomicMax()` | Runtime API | Atomic min/max |
| `atomicSub()` | Runtime API | Atomic subtraction |
| `cudaDeviceGetHostAtomicCapabilities` | Runtime API | Queries atomic operations supported between device and host |
| `cudaDeviceGetP2PAtomicCapabilities` | Runtime API | Queries atomic operations supported between two devices |
