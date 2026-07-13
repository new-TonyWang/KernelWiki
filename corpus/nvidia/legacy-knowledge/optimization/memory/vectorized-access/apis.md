# Vectorized Access -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `atom[.scope].add.vec.f32` | PTX ISA | Vectorized atomic FP32 add (.v2/.v4) |
| `ld.global[.vec][.type]` | PTX ISA | Global memory load; supports .v2/.v4 vector variants |
| `st.global[.vec][.type]` | PTX ISA | Global memory store; supports .v2/.v4 |
