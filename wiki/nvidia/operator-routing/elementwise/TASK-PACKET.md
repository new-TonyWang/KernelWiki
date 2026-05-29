---
title: Elementwise Task Packet Template
pattern_class: cuda-core
op: elementwise
id: routing-elementwise-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: elementwise
---
# Elementwise -- Task Packet Input Contract

This document refines the generic task packet contract (`reasoning/task-packet.md`) for elementwise operations. When a downstream kernel-writing agent receives an elementwise task, the task YAML must include all fields listed below. If any required field is missing, the agent refuses to start.

---

## Required fields (elementwise-specific)

```yaml
# -- Standard task packet fields (see reasoning/task-packet.md) --
task_id: <date>-<slug>                   # e.g., 2026-04-20-fused-bias-gelu
task_type: write-kernel                  # or: optimize-kernel
target_path: <output directory>          # relative to project root
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# -- Elementwise-specific fields --
op: elementwise                          # fixed for this pattern

shape:
  # One of the following forms:
  N: 1048576                             # flat 1D: number of elements
  # OR:
  B: 32                                  # batch dimension
  M: 512                                 # rows (or seq_len)
  N: 1024                                # columns (or hidden_dim)

dtype: float32                           # input dtype: float16, bfloat16, float32, float64
output_dtype: float16                    # output dtype (if different from input; omit if same)

elementwise_ops:                         # ordered list of ops to fuse into one kernel
  - add                                  # binary: out = x + y
  - relu                                 # unary: out = max(0, x)
  # Supported op names:
  #   unary:  relu, gelu, silu, swish, sigmoid, tanh, abs, neg, exp,
  #           log, sqrt, rsqrt, cast
  #   binary: add, sub, mul, div, max, min
  #   fused:  scale_add (out = x * scale + y),
  #           bias_act  (out = act(x + bias))

in_place: false                          # true = write output back to the input buffer
                                         # (valid only for unary ops or when input == output dtype)

baseline:
  method: "torch.relu"                   # or: "torch.compile", "thrust::transform", "custom"
  expected_bandwidth_pct: 85             # percent of peak HBM bandwidth the baseline achieves
                                         # (helps the agent know whether beating it is feasible)
```

---

## Optional fields

```yaml
num_inputs: 2                            # number of input tensors (default: inferred from ops)
alignment: 16                            # byte alignment of input pointers (default: 16 for cuda)
contiguous: true                         # whether input tensors are contiguous (default: true)
                                         # if false, strides must be provided
strides:                                 # only when contiguous: false
  input_0: [1024, 1]                     # row-major strides for input 0
  input_1: [1, 512]                      # column-major strides for input 1

grid_strategy: "full"                    # "full" = one thread per element (default)
                                         # "grid-stride" = fixed grid with stride loop

vectorize: true                          # hint: try float4/half8 vectorized loads
                                         # (only valid when dtype * 4 <= 16 bytes and contiguous)

stream: null                             # CUDA stream to use (null = default stream)
```

---

## Example task packets

### Example 1: simple vector add (baseline comparison)

```yaml
task_id: 2026-04-20-vecadd-fp32
task_type: write-kernel
target_path: kernels/elementwise/vecadd/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: elementwise
shape:
  N: 1048576
dtype: float32
elementwise_ops:
  - add
in_place: false
baseline:
  method: "torch.add"
  expected_bandwidth_pct: 90
```

### Example 2: fused bias + gelu

```yaml
task_id: 2026-04-20-fused-bias-gelu
task_type: write-kernel
target_path: kernels/elementwise/bias_gelu/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: elementwise
shape:
  B: 32
  M: 512
  N: 1024
dtype: float16
elementwise_ops:
  - add       # x + bias
  - gelu      # gelu(x + bias)
in_place: false
baseline:
  method: "torch.compile"
  expected_bandwidth_pct: 80
vectorize: true
```

### Example 3: in-place relu

```yaml
task_id: 2026-04-20-inplace-relu
task_type: write-kernel
target_path: kernels/elementwise/inplace_relu/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: elementwise
shape:
  N: 4194304
dtype: float32
elementwise_ops:
  - relu
in_place: true
baseline:
  method: "torch.relu_"
  expected_bandwidth_pct: 92
grid_strategy: "grid-stride"
```

### Example 4: fp32 to fp16 cast with scale

```yaml
task_id: 2026-04-20-cast-scale-fp16
task_type: write-kernel
target_path: kernels/elementwise/cast_scale/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: elementwise
shape:
  B: 16
  M: 2048
  N: 2048
dtype: float32
output_dtype: float16
elementwise_ops:
  - mul       # x * scale
  - cast      # fp32 -> fp16
in_place: false
baseline:
  method: "torch.compile"
  expected_bandwidth_pct: 75
vectorize: true
```

---

## Agent workflow after receiving this packet

1. Consult `wiki/nvidia/operator-routing/cuda-core/elementwise/INDEX.md` -- run through the decision tree. If the library fallback is sufficient, report that and stop.

2. If a custom kernel is needed, write the kernel following the canonical structure from INDEX.md Step 1.

3. Apply skills from `wiki/nvidia/operator-routing/cuda-core/elementwise/ROUTING.md` in priority order, guided by `reasoning/bottleneck-triage.md`.

4. Measure effective bandwidth and compare against the baseline specified in the task packet.
