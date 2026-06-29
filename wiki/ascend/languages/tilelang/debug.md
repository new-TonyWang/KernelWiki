---
id: skill-tilelang-ascend-debug
title: "TileLang Ascend Debugging Guide"
type: skill
vendor: ascend
tags:
- tilelang
evidence_level: spec
applies_to:
- ascend910b
source:
- path: local
  anchor: AscendOpGenAgent/skills
architectures:
- ascend910b
languages:
- tilelang
---
## Debugging Guide

### T.printf

Device-side formatted output:

```python
def printf(format_str: str, *args)
```

Format specifiers: `%d/%i` (int), `%f` (float), `%x` (hex), `%s` (string), `%p` (pointer, prefer `%x`).

```python
T.printf("fmt %s %d\n", "string", 0x123)
```

### T.dump_tensor

Dumps tensor contents with metadata:

```python
def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple=())
```

- Supports ub_buffer, l1_buffer, l0c_buffer, global_buffer
- `desc`: user-defined tag (e.g. line number)
- `shape_info`: optional shape for formatted output

```python
T.dump_tensor(A_L1, 111, 64, (8, 8))
```

Output includes CANN version, kernel type, operator details, memory info, data type, and location.
