---
title: Bank Conflict Avoidance -- Pitfalls
parent_skill: bank-conflict
id: pitfall-bank-conflict
type: pitfall
vendor: nvidia
---
## Pitfall 1: Invisible conflicts in column-wise 2D array access

**Symptom**: a kernel using `__shared__ float tile[32][32]` shows unexpectedly low shared-memory throughput or high cycle counts in NCU, even though every load/store appears to use a simple index expression.

**Root cause**: C++ arrays are row-major.  When consecutive threads (varying `threadIdx.x`) access `tile[threadIdx.x][fixed_col]`, the stride between adjacent threads' addresses is 32 elements = 128 bytes.  Since `(128 / 4) % 32 = 0`, all 32 threads hit the same bank, causing a 32-way conflict.  This is easy to miss because the indexing looks simple and correct.

**Detection**: in NCU, check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`.  Non-zero values indicate bank conflicts.

**Fix**: add +1 padding to the inner dimension: `__shared__ float tile[32][33];`.  This changes the stride to 33, and `33 % 32 = 1`, eliminating the conflict entirely.

## Pitfall 2: Padding wastes shared memory at large tile sizes

**Symptom**: after adding +1 padding to a large 2D shared array (e.g., `tile[128][129]` instead of `tile[128][128]`), the kernel fails to launch because the shared memory allocation exceeds the per-SM limit (228 KB on H200 with opt-in dynamic shared memory).

**Root cause**: padding adds one element per row.  For an array of R rows and C columns of `float`, padding adds `R * 4` bytes.  For large R this can push the total beyond the shared memory budget, especially when multiple such arrays are declared.

**Fix**: consider index swizzling (`CU_TENSOR_MAP_SWIZZLE_128B` or manual XOR-based swizzle) instead of padding.  Swizzling remaps addresses without using extra memory.  Alternatively, reduce tile size to stay within budget.

## Pitfall 3: Confusing broadcast with conflict

**Symptom**: a developer adds padding or swizzle to an access pattern where all threads read the **same** shared-memory address, expecting a speedup. No improvement is observed.

**Root cause**: when all threads in a warp read the same address in the same bank, the hardware broadcasts the value to all threads in a single transaction.  This is NOT a bank conflict.  Conflicts only occur when multiple threads access **different addresses** in the **same bank**.

**Detection**: the NCU bank-conflict counters will already show zero for broadcast patterns.  If conflicts are zero, do not apply padding -- the problem is elsewhere.

**Fix**: profile before optimizing.  Only apply bank-conflict fixes when the bank-conflict counters show non-zero values.

## Pitfall 4: 64-bit types double the conflict factor

**Symptom**: a kernel accessing `__shared__ double arr[32]` with stride-1 (`arr[threadIdx.x]`) shows 2-way bank conflicts even though the pattern looks conflict-free.

**Root cause**: each `double` occupies 8 bytes = 2 consecutive 32-bit words.  Thread 0 reads banks 0-1, thread 1 reads banks 2-3, ... thread 15 reads banks 30-31, thread 16 wraps around to banks 0-1.  So threads 0 and 16 conflict on bank 0 (and bank 1), creating a 2-way conflict.

**Detection**: NCU bank-conflict counters show non-zero values.

**Fix**: for 64-bit types, the padding trick requires adding 1 element (8 bytes) per row, which shifts each row by 2 banks.  Alternatively, redesign the access pattern to avoid stride-16 conflicts (e.g., use a different tile layout).

## Pitfall 5: Padding trick does not help stride-2 conflicts much

**Symptom**: a kernel with stride-2 shared-memory access shows 2-way bank conflicts; developer applies +1 padding but sees little or no improvement.

**Root cause**: on H200, 2-way bank conflicts have negligible overhead (measured 1.03x vs conflict-free in our probe).  The padding adds complexity and wastes memory for no practical benefit.

**Detection**: measure before and after.  If the before measurement shows <1.1x overhead from conflicts, padding is not worth the added complexity.

**Fix**: only optimize bank conflicts that are high-degree (16-way or 32-way) where the measured penalty is significant.  Low-degree conflicts on H200 are effectively free.

## Pitfall 6: Dynamic shared memory base address alignment

**Symptom**: a kernel using `extern __shared__` dynamic shared memory with manually computed offsets introduces bank conflicts that were not present with statically declared shared arrays.

**Root cause**: when multiple arrays are packed into a single `extern __shared__` region, incorrect offset alignment can cause the second array's base address to start at a bank offset that produces conflicts.  For example, if array A uses 128 bytes (banks 0-31) and array B starts immediately after at byte 128, B's first element is at bank 0 -- fine.  But if A uses 130 bytes and B starts at byte 130, B's first element is at bank 0 but offset by 2 bytes, causing misalignment and potential conflicts.

**Fix**: align each sub-array's base address to 128 bytes (32 banks x 4 bytes) to ensure the bank mapping is predictable.  Use the standard pattern:

```cuda
extern __shared__ char smem_raw[];
float* arrA = (float*)smem_raw;
// Align next array to 128-byte boundary
float* arrB = (float*)(smem_raw + ((sizeA + 127) & ~127));
```
