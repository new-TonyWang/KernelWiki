#!/usr/bin/env python3
"""Generate wgmma_zoo.cu — a multi-config Hopper wgmma harness.

Writes one kernel per config. Each kernel issues a single wgmma instance, with
all-ones inputs so every output equals K (independent of fragment-store layout).
The host counts how many output cells equal K = correctness gate.

Configs covered:
  - SS_TN bf16 fp32-acc, M=64, K=16, N in {8, 16, 32, 64, 128, 256}
  - SS_TN fp16 fp32-acc, M=64, K=16, N=64
  - SS_TN tf32 fp32-acc, M=64, K=8,  N=64
  - SS_TN s8   s32-acc,  M=64, K=32, N=64
  - SS_NT bf16 fp32-acc, M=64, K=16, N=64       (B-major flipped)
  - RS_TN bf16 fp32-acc, M=64, K=16, N=64       (A in registers)

PTX accumulator dtype f32 -> per-thread fragment = M*N/128 floats.
            f16 -> half of that count (we don't use this here).
            s32 -> per-thread fragment = M*N/128 int32 values.
"""

from dataclasses import dataclass


@dataclass
class Cfg:
    name: str
    M: int
    N: int
    K: int
    dtype_a: str   # "bf16" / "f16" / "tf32" / "s8"
    dtype_b: str
    dtype_acc: str # "f32" or "s32"
    src_a: str     # "SS" or "RS"
    layout: str    # "TN" / "NT" / "NN" / "TT"


CONFIGS = [
    # N-shape sweep at bf16 SS_TN
    Cfg("ss_tn_bf16_n8",   64,   8, 16, "bf16", "bf16", "f32", "SS", "TN"),
    Cfg("ss_tn_bf16_n16",  64,  16, 16, "bf16", "bf16", "f32", "SS", "TN"),
    Cfg("ss_tn_bf16_n32",  64,  32, 16, "bf16", "bf16", "f32", "SS", "TN"),
    Cfg("ss_tn_bf16_n64",  64,  64, 16, "bf16", "bf16", "f32", "SS", "TN"),
    Cfg("ss_tn_bf16_n128", 64, 128, 16, "bf16", "bf16", "f32", "SS", "TN"),
    Cfg("ss_tn_bf16_n256", 64, 256, 16, "bf16", "bf16", "f32", "SS", "TN"),

    # Dtype variants at N=64
    Cfg("ss_tn_fp16_n64",  64,  64, 16, "f16",  "f16",  "f32", "SS", "TN"),
    Cfg("ss_tn_tf32_n64",  64,  64,  8, "tf32", "tf32", "f32", "SS", "TN"),
    Cfg("ss_tn_s8_n64",    64,  64, 32, "s8",   "s8",   "s32", "SS", "TN"),

    # Layout: B-major flipped
    Cfg("ss_nt_bf16_n64",  64,  64, 16, "bf16", "bf16", "f32", "SS", "NT"),

    # A in register (RS variant)
    Cfg("rs_tn_bf16_n64",  64,  64, 16, "bf16", "bf16", "f32", "RS", "TN"),
]


def acc_count(c: Cfg) -> int:
    """Per-thread accumulator regs (warpgroup = 128 threads)."""
    return c.M * c.N // 128


def acc_ctype(c: Cfg) -> str:
    return "float" if c.dtype_acc == "f32" else "int32_t"


def acc_constraint(c: Cfg) -> str:
    # PTX inline-asm operand constraint character.
    # f32 accumulator -> "+f"; s32 accumulator -> "+r" (32-bit reg).
    return "+f" if c.dtype_acc == "f32" else "+r"


def dtype_bytes(d: str) -> int:
    return {"bf16": 2, "f16": 2, "tf32": 4, "s8": 1, "f32": 4, "s32": 4}[d]


def ptx_dtype_suffix(c: Cfg) -> str:
    """Returns ".acc.A.B" suffix for the wgmma instruction mnemonic."""
    return f".{c.dtype_acc}.{c.dtype_a}.{c.dtype_b}"


def reg_list(prefix_offset: int, n: int) -> str:
    """Returns "{%0,%1,...,%n-1}" with offset prefix_offset."""
    return "{" + ",".join(f"%{prefix_offset + i}" for i in range(n)) + "}"


def emit_kernel(c: Cfg) -> str:
    n_acc = acc_count(c)
    actype = acc_ctype(c)
    constraint = acc_constraint(c)
    a_bytes = c.M * c.K * dtype_bytes(c.dtype_a)
    b_bytes = c.K * c.N * dtype_bytes(c.dtype_b)

    # Layout flag immediates: TN: transA=0, transB=0; NT: transA=1, transB=1; NN: 0,1; TT: 1,0.
    # (PTX: transX=1 means MN-major, transX=0 means K-major.)
    trans_map = {"TN": (0, 0), "NT": (1, 1), "NN": (0, 1), "TT": (1, 0)}
    transA, transB = trans_map[c.layout]

    # Build accumulator declaration + zero-init.
    accum_decl_lines = []
    for i in range(n_acc):
        accum_decl_lines.append(f"    {actype} d{i} = 0;")
    accum_decl = "\n".join(accum_decl_lines)

    # Build the operand lists for inline PTX.
    # Output regs %0..%(n_acc-1), then descA at %n_acc, descB at %n_acc+1.
    out_regs_str = reg_list(0, n_acc)

    # For SS variant: instruction = "wgmma.mma_async.sync.aligned.m64nNkK.acc.A.B {regs}, descA, descB, scD, scA, scB, transA, transB;"
    # For RS variant: A operand becomes a register list. For bf16 m64xK16, A is 4 32-bit regs / thread.
    if c.src_a == "SS":
        # Per PTX ISA SM90:
        #   .s32.s8.s8 / .s32.u8.u8  → scaleD only (1 imm)
        #   .f32.tf32.tf32          → scaleD, scaleA, scaleB (3 imm; A/B fixed K-major, no trans)
        #   .f32.bf16.bf16 / .f32.f16.f16 / .f16.f16.f16 → scaleD, scaleA, scaleB, transA, transB (5 imm)
        if c.dtype_acc == "s32":
            tail = "1"
        elif c.dtype_a == "tf32":
            tail = "1, 1, 1"
        else:
            tail = f"1, 1, 1, {transA}, {transB}"
        ptx = (
            f"wgmma.mma_async.sync.aligned.m{c.M}n{c.N}k{c.K}{ptx_dtype_suffix(c)} "
            f"{out_regs_str}, %{n_acc}, %{n_acc + 1}, {tail};"
        )
        op_constraints = ", ".join(f'"{constraint}"(d{i})' for i in range(n_acc))
        in_constraints = '"l"(descA), "l"(descB)'
        operand_setup = f"""    uint64_t descA = make_smem_desc(smem_a, /*ld=*/256, /*sd=*/16, 0);
    uint64_t descB = make_smem_desc(smem_b, /*ld=*/256, /*sd=*/16, 0);"""
    else:
        # RS: A is in register; for bf16 m64xK16, each thread holds 4 32-bit regs (= 8 bf16 elements
        # = a 1-row-per-warp slice of the 64xK A matrix). We build a_regs[0..3] from smem reads.
        # Outputs: %0..%(n_acc-1), then %n_acc..%(n_acc+3) for A regs, then %(n_acc+4) for descB.
        n_a_regs = (c.M * c.K * dtype_bytes(c.dtype_a) // 128) // 4    # bytes per thread / 4 = 32-bit regs per thread
        # For bf16 m64xK16: 64*16*2 / 128 / 4 = 4 regs per thread. Good.
        a_regs_list = "{" + ",".join(f"%{n_acc + i}" for i in range(n_a_regs)) + "}"
        ptx = (
            f"wgmma.mma_async.sync.aligned.m{c.M}n{c.N}k{c.K}{ptx_dtype_suffix(c)} "
            f"{out_regs_str}, {a_regs_list}, %{n_acc + n_a_regs}, 1, 1, 1, {transB};"
        )
        op_constraints = ", ".join(f'"{constraint}"(d{i})' for i in range(n_acc))
        in_constraints = (
            ", ".join(f'"r"(a_reg{i})' for i in range(n_a_regs))
            + f', "l"(descB)'
        )
        # A reg load: each thread reads 4 bf16x2 packed regs from the row of A indexed by lane.
        # Simplification: read from smem_a[(tid * 4) ... (tid * 4 + 3)] as uint32_t each.
        # Caller does cooperative load from gmem to smem first; then we reinterpret a slice.
        operand_setup = f"""    // Build descB (B in shared) and A register vector (A in registers).
    uint64_t descB = make_smem_desc(smem_b, /*ld=*/256, /*sd=*/16, 0);
    const uint32_t* a_pack = reinterpret_cast<const uint32_t*>(smem_a);
    uint32_t a_reg0 = a_pack[(tid * 4) + 0];
    uint32_t a_reg1 = a_pack[(tid * 4) + 1];
    uint32_t a_reg2 = a_pack[(tid * 4) + 2];
    uint32_t a_reg3 = a_pack[(tid * 4) + 3];"""

    # Output writeback: each thread writes its n_acc accumulator regs to D at offset tid * n_acc.
    writeback_lines = []
    for i in range(n_acc):
        writeback_lines.append(f"    D[tid * {n_acc} + {i}] = d{i};")
    writeback = "\n".join(writeback_lines)

    # Each kernel takes A, B, D pointers and shape constants are baked in.
    a_ctype = "__nv_bfloat16" if c.dtype_a == "bf16" else (
        "__half" if c.dtype_a == "f16" else (
        "float" if c.dtype_a == "tf32" else "int8_t"))
    b_ctype = a_ctype  # symmetric for our configs
    d_ctype = "float" if c.dtype_acc == "f32" else "int32_t"

    return f"""
__global__ void wgmma_kernel_{c.name}(
    const {a_ctype}* __restrict__ A,
    const {b_ctype}* __restrict__ B,
    {d_ctype}* __restrict__ D)
{{
    constexpr int M = {c.M};
    constexpr int N = {c.N};
    constexpr int K = {c.K};
    constexpr int A_BYTES = M * K * {dtype_bytes(c.dtype_a)};
    constexpr int B_BYTES = K * N * {dtype_bytes(c.dtype_b)};

    extern __shared__ uint8_t smem[];
    {a_ctype}* smem_a = reinterpret_cast<{a_ctype}*>(smem);
    {b_ctype}* smem_b = reinterpret_cast<{b_ctype}*>(smem + A_BYTES);

    int tid = threadIdx.x;

    // Cooperative load of A (M*K elements) and B (K*N elements) from gmem.
    for (int i = tid; i < M * K; i += 128) smem_a[i] = A[i];
    for (int i = tid; i < K * N; i += 128) smem_b[i] = B[i];
    __syncthreads();

{accum_decl}

{operand_setup}

    constexpr int N_INNER = 1024;   // serialized wgmma issues per launch (accumulator-chained)

    asm volatile("wgmma.fence.sync.aligned;\\n");
    #pragma unroll 1
    for (int it = 0; it < N_INNER; ++it) {{
        asm volatile(
            "{ptx}\\n"
            : {op_constraints}
            : {in_constraints});
    }}
    asm volatile("wgmma.commit_group.sync.aligned;\\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\\n");

{writeback}
}}
"""


def emit_dispatch_table() -> str:
    """Emit a runtime dispatch table that runs each config and reports results."""
    cases = []
    for c in CONFIGS:
        a_ctype = "__nv_bfloat16" if c.dtype_a == "bf16" else (
            "__half" if c.dtype_a == "f16" else (
            "float" if c.dtype_a == "tf32" else "int8_t"))
        d_ctype = "float" if c.dtype_acc == "f32" else "int32_t"
        a_bytes = c.M * c.K * dtype_bytes(c.dtype_a)
        b_bytes = c.K * c.N * dtype_bytes(c.dtype_b)
        d_bytes = c.M * c.N * dtype_bytes(c.dtype_acc)
        smem = a_bytes + b_bytes
        cases.append(f"""
    run_config<{a_ctype}, {d_ctype}>(
        "{c.name}", {c.M}, {c.N}, {c.K},
        {a_bytes}, {b_bytes}, {d_bytes}, {smem},
        wgmma_kernel_{c.name},
        warmup, timed);""")
    return "\n".join(cases)


def emit_file() -> str:
    parts = [
        '''// Auto-generated by gen_wgmma_zoo.py — do not edit by hand.
//
// Cutlass-free Hopper wgmma multi-config harness. One kernel per config; each
// runs all-ones inputs and verifies every output equals K (layout-agnostic
// correctness gate). Wall-clock per config is captured for relative latency.
//
// Build (cutlass-free):
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \\
//        wgmma_zoo.cu -o wgmma_zoo

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \\
    fprintf(stderr, "CUDA error %s at %s:%d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \\
    std::exit(1); } } while (0)

__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr,
    uint32_t leading_dim_bytes,
    uint32_t stride_dim_bytes,
    uint32_t swizzle_mode = 0)
{
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_int >> 4)) & 0x3FFF;
    desc |= ((uint64_t)(leading_dim_bytes >> 4) & 0xFFFF) << 14;
    desc |= ((uint64_t)(stride_dim_bytes  >> 4) & 0xFFFF) << 30;
    desc |= ((uint64_t)(swizzle_mode & 0x3)) << 62;
    return desc;
}
'''
    ]
    for c in CONFIGS:
        parts.append(emit_kernel(c))

    parts.append('''
template <typename A_T, typename D_T>
void run_config(
    const char* name, int M, int N, int K,
    size_t a_bytes, size_t b_bytes, size_t d_bytes, size_t smem_bytes,
    void (*kernel)(const A_T*, const A_T*, D_T*),
    int warmup, int timed)
{
    A_T *dA, *dB; D_T *dD;
    CUDA_CHECK(cudaMalloc(&dA, a_bytes));
    CUDA_CHECK(cudaMalloc(&dB, b_bytes));
    CUDA_CHECK(cudaMalloc(&dD, d_bytes));

    // Fill A, B with one-valued elements (per dtype). Expected D[i,j] = K.
    int n_a = a_bytes / sizeof(A_T);
    int n_b = b_bytes / sizeof(A_T);
    A_T* hA = new A_T[n_a];
    A_T* hB = new A_T[n_b];
    for (int i = 0; i < n_a; ++i) {
        if constexpr (std::is_same_v<A_T, __nv_bfloat16>) hA[i] = __float2bfloat16(1.0f);
        else if constexpr (std::is_same_v<A_T, __half>)   hA[i] = __float2half(1.0f);
        else if constexpr (std::is_same_v<A_T, float>)    hA[i] = 1.0f;
        else hA[i] = (A_T)1;
    }
    for (int i = 0; i < n_b; ++i) {
        if constexpr (std::is_same_v<A_T, __nv_bfloat16>) hB[i] = __float2bfloat16(1.0f);
        else if constexpr (std::is_same_v<A_T, __half>)   hB[i] = __float2half(1.0f);
        else if constexpr (std::is_same_v<A_T, float>)    hB[i] = 1.0f;
        else hB[i] = (A_T)1;
    }
    CUDA_CHECK(cudaMemcpy(dA, hA, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, b_bytes, cudaMemcpyHostToDevice));

    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(smem_bytes));

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(dD, 0, d_bytes));
        kernel<<<1, 128, smem_bytes>>>(dA, dB, dD);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed.
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    ms.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        cudaEventRecord(e0);
        kernel<<<1, 128, smem_bytes>>>(dA, dB, dD);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t = 0;
        cudaEventElapsedTime(&t, e0, e1);
        ms.push_back(t);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(ms.begin(), ms.end());
    float median_ms = ms[ms.size() / 2];

    // The kernel issues N_INNER serialized wgmma per launch (accumulator-chained),
    // so each output cell ends at K * N_INNER. The constant must mirror gen_wgmma_zoo.py.
    constexpr int N_INNER = 1024;
    int expected = K * N_INNER;

    int n_d = d_bytes / sizeof(D_T);
    D_T* hD = new D_T[n_d];
    CUDA_CHECK(cudaMemcpy(hD, dD, d_bytes, cudaMemcpyDeviceToHost));
    int matched = 0, written = 0;
    for (int i = 0; i < n_d; ++i) {
        if (hD[i] != (D_T)0) ++written;
        if (hD[i] == (D_T)expected) ++matched;
    }

    // FLOPs across all serialized wgmma issuances per launch:
    //   N_INNER * 2 * M * N * K  (each wgmma issues M*N MAC = 2*M*N*K FLOPs).
    double flops = double(N_INNER) * 2.0 * M * N * K;
    double tflops = flops / (median_ms * 1e-3) / 1e12;

    printf("%s,%d,%d,%d,%zu,%.5f,%d,%d,%d,%.3f\\n",
           name, M, N, K, smem_bytes, median_ms, written, matched, n_d, tflops);

    delete[] hA; delete[] hB; delete[] hD;
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

int main() {
    int warmup = 5, timed = 20;
    printf("config,M,N,K,smem_bytes,median_ms,written,matched,total,tflops\\n");

''')
    parts.append(emit_dispatch_table())
    parts.append('''
    return 0;
}
''')
    return "".join(parts)


if __name__ == "__main__":
    print(emit_file())
