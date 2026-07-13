# Instruction Level Parallelism -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fma.rn.f32` / `fma.rn.f64` | PTX ISA | Fused multiply-add; enables software pipelining of dependent chains |
| `#pragma unroll` | CUDA C++ | Compiler directive for loop unrolling to expose ILP |
| `mov[.type]` (.b128) | PTX ISA | Register packing / wider data movement (pack/unpack .b32, .b64, .b128) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `add[.type]` | PTX ISA | Integer add; .u16x2/.s16x2 SIMD variants (sm_90+); .u8x4/.s8x4 (sm_120f+) |
| `lop3.b32` | PTX ISA | Arbitrary 3-input logical operation (256 possible ops via LUT) |
| `mad[.mode][.type]` | PTX ISA | Multiply-add (fused mul+add, integer) |
| `max[.type]` | PTX ISA | Integer max; SIMD .u16x2/.s16x2 (sm_90+), .relu variants; .u8x4/.s8x4 (sm_120f+) |
| `min[.type]` | PTX ISA | Integer min; SIMD .u16x2/.s16x2 (sm_90+), .relu variants; .u8x4/.s8x4 (sm_120f+) |
| `mul[.mode][.type]` | PTX ISA | Integer multiply (.hi/.lo/.wide) |
| `sub[.type]` | PTX ISA | Integer subtract; .u8x4/.s8x4 SIMD variants (sm_120f+) |
