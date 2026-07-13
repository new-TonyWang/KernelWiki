# Context Management -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 11.6 (Multiple Contexts)"
  - "Programming Guide 3.3 (Driver API)"
  - "Programming Guide 4.6 (Green Contexts)"
cross_ref:
  - optimization/latency/green-contexts
  - optimization/latency/stream-concurrency
related_apis:
  - cudaSetDevice
  - cudaInitDevice
  - cudaSetDeviceFlags
  - cudaDeviceGetExecutionCtx
unlocks:
  - Lower memory overhead per GPU
  - Elimination of context-switch latency
conflicts_with:
  - None
```

---

## S1: Use the Primary Context Instead of Creating Multiple Contexts

**When to Use:** Always within a single CUDA application. Multiple contexts on the same GPU cause time-slicing and memory overhead.

**How to Apply:**
1. Use the CUDA Runtime API, which automatically uses the primary context.
2. If using the Driver API, call `cuDevicePrimaryCtxRetain` instead of `cuCtxCreate`.
3. Share the primary context across all threads and libraries in the process.

**Code Template:**
```cpp
// Runtime API: automatic primary context (preferred)
cudaSetDevice(0);
myKernel<<<grid, block>>>(...);

// Driver API: explicitly use primary context
CUcontext ctx;
cuDevicePrimaryCtxRetain(&ctx, dev);
cuCtxPushCurrent(ctx);
kernel<<<...>>>(...);
cuCtxPopCurrent(&ctx);
cuDevicePrimaryCtxRelease(dev);
```

> Source: BP 11.6 -- "it is best to avoid multiple contexts per GPU within the same CUDA application."

---

## S2: Use Exclusive Process Mode for Single-Context Guarantee

**When to Use:** In production deployments where you want to ensure only one context exists per GPU, preventing accidental multi-context scenarios from multi-process access.

**How to Apply:**
1. Configure the GPU with `nvidia-smi -i <gpu> -c EXCLUSIVE_PROCESS`.
2. Only one process can create a context on this GPU at a time.
3. Within that process, the primary context is shared by all threads.

**Code Template:**
```bash
# Set exclusive process mode
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Reset to default
nvidia-smi -i 0 -c DEFAULT
```

> Source: BP 11.6 -- "NVIDIA-SMI can be used to configure a GPU for exclusive process mode, which limits the number of contexts per GPU to one."

---

## S3: Use cudaInitDevice to Control Initialization Timing

**When to Use:** When you need to control when CUDA context initialization happens, avoiding lazy-init latency on the first API call.

**How to Apply:**
1. Call `cudaInitDevice` at application startup to eagerly initialize the device.
2. This avoids unexpected latency on the first kernel launch or memory allocation.

**Code Template:**
```cpp
// Eagerly initialize device 0 at startup
cudaInitDevice(0, 0, 0);

// First kernel launch will not incur initialization overhead
myKernel<<<grid, block>>>(...);
```

> Source: Runtime API -- cudaInitDevice provides explicit device initialization.

---

## Cascading Opportunities

- After consolidating to a single context, `green-contexts` can be used for resource partitioning within that context.
- Single-context setups maximize concurrency opportunities for `stream-concurrency`.

## Conflicts

- None. Context consolidation is always beneficial.

## Principles

1. **One context per GPU:** Multiple contexts time-slice and waste memory. Libraries should share the primary context (BP 11.6).
2. **Context switching is expensive:** It causes pipeline drains and memory overhead for per-context data (BP 11.6).
3. **Runtime API manages contexts automatically:** Use it unless Driver API features are needed (BP 11.6).

## Open Questions

- Q1: How does green context creation interact with primary context resource limits?
- Q2: What is the memory overhead per additional CUDA context on Hopper/Blackwell?
