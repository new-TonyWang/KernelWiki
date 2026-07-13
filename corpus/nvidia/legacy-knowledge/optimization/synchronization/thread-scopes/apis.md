# Thread Scopes -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `barrier.cluster.arrive[.sem][.aligned]` | PTX ISA | Cluster-level barrier arrive (.release/.relaxed) |
| `barrier.cluster.wait[.acquire][.aligned]` | PTX ISA | Cluster-level barrier wait |
| `fence[.sem].scope` | PTX ISA | Thread fence (.sc/.acq_rel/.acquire/.release at .cta/.cluster/.gpu/.sys) |
| `membar.cta / .gl / .sys` | PTX ISA | Memory barrier at CTA/global/system level (old-style) |
| `__threadfence()` | Runtime API | Memory fence (device scope) |
| `__threadfence_block()` | Runtime API | Memory fence (block scope) |
| `__threadfence_system()` | Runtime API | Memory fence (system scope, including host) |
| `cuda::atomic` | Runtime API | Scoped atomic operations (thread/block/device/system scope) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceGetHostAtomicCapabilities` | Runtime API | Queries atomic operations supported between device and host |
