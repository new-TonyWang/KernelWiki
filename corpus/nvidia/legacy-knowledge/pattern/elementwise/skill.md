# Elementwise Pattern -- Skills

## When to Apply
- Operations where output[i] depends only on input[i] (or inputs[i] for multi-input)
- Activation functions: ReLU, GELU, SiLU, tanh, sigmoid, etc.
- Binary operations: add, multiply, subtract, max, min
- Unary operations: abs, neg, exp, log, sqrt, rsqrt
- Type conversions, scaling, clamping

## Core Characteristic
Elementwise kernels are almost always **memory-bandwidth bound**: arithmetic intensity is O(1) operations per element loaded. The optimization strategy is fundamentally about maximizing memory throughput, not compute throughput.

## Skill 1: Vectorized Memory Access
- Process multiple elements per thread using vector types
- `float4` = 4 floats = 16 bytes per load/store (128-bit transaction)
- `__half2` = 2 halves = 4 bytes, enabling packed operations
- Requires input/output arrays to be aligned to vector width

### Pattern for vectorized elementwise:
```cuda
// Process 4 elements per thread
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
if (idx + 3 < N) {
    float4 in = reinterpret_cast<float4*>(input)[idx / 4];
    float4 out;
    out.x = activation(in.x);
    out.y = activation(in.y);
    out.z = activation(in.z);
    out.w = activation(in.w);
    reinterpret_cast<float4*>(output)[idx / 4] = out;
}
```

### For half-precision, use packed operations:
```cuda
__half2 h2 = __halves2half2(input[2*i], input[2*i+1]);
h2 = __hmul2(h2, scale_h2);  // Packed multiply: 2 ops in 1 instruction
```

## Skill 2: Kernel Fusion
- The biggest optimization for elementwise ops is to not launch them at all
- Fuse elementwise operations into the epilogue of a preceding GEMM or reduction kernel
- CUTLASS Epilogue Visitor Tree (EVT): compose arbitrary elementwise DAGs as GEMM epilogue
- cuBLAS epilogue fusion: `CUBLASLT_EPILOGUE_RELU`, `GELU`, `BIAS`, etc.

### When to fuse vs keep separate:
- **Fuse** when: elementwise follows GEMM/conv and data is still in registers/SMEM
- **Fuse** when: multiple elementwise ops can be combined into one kernel
- **Keep separate** when: elementwise is on standalone data not produced by a compute kernel
- **Keep separate** when: fusion would cause excessive register pressure in the host kernel

## Skill 3: CUTLASS Epilogue Visitor Tree (EVT) for Custom Fusion
- Compose tree of operations: `Sm90EVT<Sm90Compute<Op, ...>, Children...>`
- Leaf nodes: `Sm90AccFetch` (GEMM output), `Sm90SrcFetch` (C matrix), `Sm90ScalarBroadcast` (scalar)
- Additional nodes: `Sm90ColLoad`/`Sm90RowLoad` for bias vectors, `Sm90AuxStore` for saving intermediates
- Tree visitors recursively evaluate children, apply node operation, return result
- Enables fusing arbitrary elementwise graphs into GEMM epilogue without writing kernel code

### Example: fuse bias + GELU into GEMM:
```cpp
using EVTOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::GELU,
    ElementD, ElementCompute, ElementC, ElementScalar>;
// Pass EVTOp to CollectiveBuilder for automatic fusion
```

## Skill 4: Fast Math Intrinsics for Common Activations
- `__expf(x)`: fast exp, ~2 ULP error, much faster than `expf(x)`
- `__tanhf(x)`: fast tanh approximation
- `__sinf(x)`, `__cosf(x)`: fast trig
- `__frsqrt_rn(x)`: fast reciprocal square root
- Packed half-precision: `__hfma2_relu(a,b,c)` does FMA + ReLU in one instruction
- Compile with `--use_fast_math` to automatically substitute fast variants

### GELU approximation (tanh version):
```cuda
float gelu(float x) {
    return 0.5f * x * (1.0f + __tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
```

## Skill 5: Grid-Stride Loop for Arbitrary Sizes
- Launch a fixed grid size, each thread processes multiple elements
- Handles any input size without grid size calculation
- Better for kernel fusion: same kernel handles varying batch sizes

### Pattern:
```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    output[i] = func(input[i]);
}
```

## Skill 6: Coalesced Access Patterns
- Consecutive threads should access consecutive memory addresses
- For multi-dimensional tensors: ensure the innermost dimension maps to threadIdx.x
- For strided access patterns: consider transposing data or using shared memory as staging

## Skill 7: cuBLAS for Simple Elementwise Operations
- `cublas<t>axpy`: y = alpha*x + y (fused scale + add)
- `cublas<t>scal`: x = alpha*x (in-place scale)
- `cublas<t>dgmm`: C = A * diag(x) (diagonal matrix multiply = column/row scaling)
- `cublas<t>geam`: C = alpha*op(A) + beta*op(B) (matrix add with optional transpose)
- For mixed precision: `cublasAxpyEx`, `cublasScalEx`

## Cross-References
- optimization/compute/fast-math -- fast intrinsic functions
- optimization/compute/half-precision-math -- packed half operations
- optimization/compute/operator-fusion -- fusion strategies
- optimization/memory/vectorized-access -- vector load/store patterns
- optimization/memory/coalescing -- memory access patterns
