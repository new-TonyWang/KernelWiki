---
id: skill-triton-ascend-hint-mode
title: "Triton Ascend Hint Mode: Parameter Space Configuration"
type: skill
vendor: ascend
tags:
- triton-ascend
- autotune
evidence_level: spec
applies_to:
- ascend910b
source:
- path: local
  anchor: AscendOpGenAgent/skills
architectures:
- ascend910b
languages:
- triton-ascend
techniques:
- autotune
---
# Hint Mode: Parameter Space Configuration Guide

## Hint Syntax

```python
# @hint: param in [val1, val2, ...]         -> type='choice'
# @hint: param in range(min, max, step=N)   -> type='range'
# @hint: param = value                      -> type='fixed'
# @hint: param in pow2(min_pow, max_pow)    -> type='power_of_2'
# @hint: param in pow of 2                  -> type='power_of_2'
```

Compatible formats:
```python
# @range_hint("param", start=min, end=max)
# @elemwise_hint("param", [val1, val2])
```

## SPACE_CONFIG Template

```python
SPACE_CONFIG = {
    'param1': {'type': 'choice', 'values': [val1, val2, ...]},
    'param2': {'type': 'range', 'min': min_val, 'max': max_val, 'step': step_val},
    'param3': {'type': 'power_of_2', 'min_pow': min_exp, 'max_pow': max_exp},
}
```

## BLOCK_SIZE Selection

- Small param range (e.g. M in [128, 256]): use [32, 64, 128]
- Large param range (e.g. M in range(128, 8192)): use [64, 128, 256, 512] with autotune

## Boundary Handling

- **Option A (mask)**: supports arbitrary shapes
- **Option B (no mask)**: requires shape divisible by BLOCK_SIZE, better performance
