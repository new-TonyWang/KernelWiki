# Operator Fusion -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fma.rnd.f64` | PTX ISA | FP64 fused multiply-add |
| `fma.rnd[.ftz][.sat].f32` | PTX ISA | FP32 fused multiply-add; .f32x2 variant (sm_100+, **Blackwell**) |
| `fma.rnd[.ftz][.sat][.relu].f16 / .f16x2` | PTX ISA | FP16 fused multiply-add; relu/oob variants |
| `fma.rnd[.relu].bf16 / .bf16x2` | PTX ISA | BF16 fused multiply-add; .oob variant (sm_90+) |
| `fma.rnd[.sat].f32.{f16/bf16}` | PTX ISA | Mixed-precision fma: f16/bf16 inputs -> f32 output |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `lop3.b32` | PTX ISA | Arbitrary 3-input logical operation (256 possible ops via LUT) |
| `mad[.mode][.type]` | PTX ISA | Multiply-add (fused mul+add, integer) |
| `mad[.rnd][.ftz][.sat].f32` | PTX ISA | FP32 multiply-add (equiv to fma on sm_20+) |
