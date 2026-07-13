# Programmatic Dependent Launch -- Pitfalls

## P1: Forgetting to Call cudaTriggerProgrammaticLaunchCompletion in All Blocks

**Symptom:** Secondary kernel never starts or starts only after the primary kernel fully completes.

**Detection:** Nsight Systems shows no overlap between primary and secondary kernels despite PDL attribute being set.

**Fix:** Ensure every block in the primary kernel calls `cudaTriggerProgrammaticLaunchCompletion()`. A common pattern is to place it after the shared data production phase.

> Source: PG 3.1.4 -- "The first kernel (the primary kernel) needs to call a special function to indicate that it is done with everything that the subsequent dependent kernels will need."

---

## P2: Secondary Kernel Accessing Dependent Data Before cudaGridDependencySynchronize

**Symptom:** Race condition; secondary kernel reads stale or partially written data from the primary kernel.

**Detection:** Intermittent correctness failures. Results change between runs.

**Fix:** Ensure `cudaGridDependencySynchronize()` is called in the secondary kernel before any access to data produced by the primary kernel.

```cpp
__global__ void secondary() {
    // BAD: accessing shared_data before sync
    // float val = shared_data[tid];

    // Independent work first
    float local = init();

    // GOOD: sync before accessing dependent data
    cudaGridDependencySynchronize();
    float val = shared_data[tid];
}
```

> Source: PG 3.1.4 -- "cudaGridDependencySynchronize() blocks until direct grid dependencies complete."

---

## P3: Missing PDL Launch Attribute on Secondary Kernel

**Symptom:** Secondary kernel does not overlap with primary; it behaves as a normal sequentially launched kernel.

**Detection:** No overlap visible in Nsight Systems timeline. PDL device calls are present but have no effect.

**Fix:** Set `cudaLaunchAttributeProgrammaticStreamSerialization` with `programmaticStreamSerializationAllowed = 1` when launching the secondary kernel.

> Source: PG 3.1.4 -- "The second kernel needs to be launched with a special attribute."

---

## P4: Expecting Full Overlap Instead of Partial

**Symptom:** Performance improvement from PDL is less than expected.

**Detection:** Nsight Systems shows overlap is only a fraction of the secondary kernel's duration.

**Fix:** PDL provides partial overlap -- only the independent phases of each kernel can overlap. The overlap depends on how much independent work exists in each kernel's tail (primary) and head (secondary). Restructure kernels to maximize independent work at these phases.

> Source: PG 3.1.4 -- "the degree of overlap which can be achieved is dependent on the specific structure of the kernels."
