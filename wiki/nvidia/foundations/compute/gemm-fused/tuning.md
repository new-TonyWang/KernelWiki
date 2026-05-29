---
id: skill-gemm-fused-tuning
type: skill
vendor: nvidia
title: Tuning
tags:
- cuda-cpp
applies_to:
- general
source:
- path: spec
  anchor: CUDA Programming Guide
evidence_level: spec
---
# Fused GEMM (Epilogue/Prologue) Tuning Parameters

## Compile-time parameters

All aligned-GEMM tuning knobs apply (see `wiki/nvidia/foundations/compute/gemm/aligned/tuning.md`). Additional knobs for fusion:

| Knob | cutlass template param | Default | Range | Effect |
|---|---|---|---|---|
| Epilogue operation | `EpilogueScheduleAuto` + `EVT` (Epilogue Visitor Tree) | `LinCombEltAct<Identity>` | `LinCombEltAct<ReLu>`, `LinCombEltAct<GELU>`, custom EVT | Fused epilogue operation applied to the GEMM output before writeback. |
| Prologue fusion | Mainloop schedule variant | none | mixed-dtype dequant (`MainloopSm90TmaGmmaWarpSpecializedMixedInput`) | Fused prologue operation applied to inputs before GEMM. |
| Epilogue smem usage | `EpilogueTileType` | auto | auto | smem budget for epilogue staging. Auto-computed by cutlass. |

## Epilogue visitor tree (EVT)

cutlass's epilogue visitor tree allows composing arbitrary post-GEMM operations as a tree of elementwise nodes:

```
D = EVT(alpha * acc + beta * C)
```

Common compositions:
- `LinCombEltAct<ReLu>`: D = ReLU(alpha * acc + beta * C)
- `LinCombEltAct<GELU>`: D = GELU(alpha * acc + beta * C)
- `LinCombBiasEltAct<ReLu>`: D = ReLU(alpha * acc + beta * C + bias)
- Quantize: D = quantize(alpha * acc + beta * C)

## Prologue fusion (mixed-dtype)

For mixed-dtype GEMM (e.g., INT4 weights × FP16 activations), the prologue fuses a dequantization step:
- Input B is stored as INT4 with per-group scale/zero-point
- The prologue converts B to FP16 before the wgmma instruction
- This uses `MainloopSm90TmaGmmaWarpSpecializedMixedInput` (the RS variant with B from registers)

## Correctness gate

When testing fused kernels, the reference must replicate the fusion. See BitLesson `BL-20260428-fused-correctness-gate-must-match-fusion`.

## Measured sweep reference

Fused ReLU epilogue: `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md`
