# Context Management -- Pitfalls

## P1: Multiple Processes Creating Separate Contexts on Same GPU

**Symptom:** GPU time-slices between contexts; effective throughput drops significantly. Each context consumes additional GPU memory.

**Detection:** `nvidia-smi` shows multiple processes on the same GPU. Context switch overhead visible in Nsight Systems.

**Fix:** Use MPS (Multi-Process Service) to share a single context across processes, or redesign to use a single process with multiple threads.

> Source: BP 11.6 -- "Creating additional contexts incurs memory overhead for per-context data and time overhead for context switching."

---

## P2: Library Creating a New Context Instead of Using Primary

**Symptom:** Unexpected memory increase and performance degradation when loading a CUDA library that creates its own context.

**Detection:** `cuCtxCreate` calls in the library's initialization path. Multiple contexts visible on the same GPU.

**Fix:** Ensure libraries use `cuDevicePrimaryCtxRetain` or the Runtime API. File bug reports for libraries that create unnecessary contexts.

> Source: BP 11.6 -- "the CUDA Driver API provides methods to access and manage a special context on each GPU called the primary context."

---

## P3: First CUDA Call Incurring Unexpected Latency

**Symptom:** First kernel launch or memory allocation takes much longer than subsequent ones due to lazy context initialization.

**Detection:** Nsight Systems shows a large initialization block at the first CUDA API call.

**Fix:** Call `cudaInitDevice` or `cudaSetDevice` explicitly at application startup to front-load initialization.

```cpp
// Initialize eagerly at startup
cudaSetDevice(0);
// Or more explicit:
cudaInitDevice(0, 0, 0);
```

> Source: PG 4.7 + Runtime API -- lazy initialization defers work to first use.

---

## P4: Using cuCtxCreate When cuDevicePrimaryCtxRetain Suffices

**Symptom:** Each call to `cuCtxCreate` produces a new context, leading to accumulation of contexts and associated overhead.

**Detection:** Multiple `cuCtxCreate` calls in the application without corresponding `cuCtxDestroy`.

**Fix:** Replace `cuCtxCreate` with `cuDevicePrimaryCtxRetain` to reuse the single primary context.

> Source: BP 11.6 -- "These are the same contexts used implicitly by the CUDA Runtime when there is not already a current context for a thread."

## P5: NCU kernel-level profiling will not capture context management improvements since the overhead is on the host/driver side; only end-to-end timing reveals the benefit (discovered in verification)

**Symptom**: NCU kernel-level profiling will not capture context management improvements since the overhead is on the host/driver side; only end-to-end timing reveals the benefit.
**Source**: Level 3 sandbox verification (2026-04-06)

## P6: Any benchmark that uses warmup iterations (standard practice) will mask context-initialization latency, making cudaInitDevice appear to have zero effect even when it legitimately moves overhead earlier (discovered in verification)

**Symptom**: Any benchmark that uses warmup iterations (standard practice) will mask context-initialization latency, making cudaInitDevice appear to have zero effect even when it legitimately moves overhead earlier.
**Source**: Level 3 sandbox verification (2026-04-06)
