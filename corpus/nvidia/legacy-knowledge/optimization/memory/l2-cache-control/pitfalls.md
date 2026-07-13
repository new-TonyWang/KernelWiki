# L2 Cache Control -- Pitfalls

## P1: L2 Thrashing from Oversized Persistence Window
**Symptom**: Performance drops ~10% when persistent data region exceeds L2 set-aside size with hitRatio=1.0.
**Detection**: Profile shows L2 miss rate increases. Nsight Compute L2 hit rate metric degrades.
**Fix**: Tune `hitRatio` to `set_aside_size / actual_data_size` so only a fitting fraction is persisted. Or reduce `num_bytes` to the set-aside size.
**Source**: Best Practices Guide, Section 10.2.2.2 (Tuning the Access Window Hit-Ratio)

## P2: Stale Persisting Lines Reducing L2 for Subsequent Kernels
**Symptom**: Kernel B runs slower after Kernel A used L2 persistence, even though Kernel B has its own data.
**Detection**: L2 hit rate for Kernel B is lower than baseline. Persisting lines from Kernel A still occupy L2.
**Fix**: Call `cudaCtxResetPersistingL2Cache()` between kernel sequences. Set window `num_bytes=0` to disable the policy.
**Source**: Programming Guide, Section 4.13.5 (Reset L2 Access to Normal)

## P3: MIG Mode Disables L2 Set-Aside
**Symptom**: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` silently does nothing.
**Detection**: Check if GPU is in MIG mode: `nvidia-smi -i 0 -q | grep MIG`.
**Fix**: L2 set-aside is not available in MIG mode. Use cache load hints within kernel code instead.
**Source**: Programming Guide, Section 4.13.1 (L2 Cache Set-Aside for Persisting Accesses)

## P4: Concurrent Streams Competing for L2 Set-Aside
**Symptom**: Two concurrent kernels with persisting windows evict each other's data.
**Detection**: Both kernels show lower L2 hit rates than when run alone.
**Fix**: Reduce hitRatio for each stream (e.g., 0.5 for each of 2 streams) so their combined persisting footprint fits the set-aside.
**Source**: Programming Guide, Section 4.13.6 (Manage Utilization of L2 Set-Aside Cache)

## P5: Window Size Exceeds Device Maximum
**Symptom**: L2 persistence has no effect; window is silently clamped.
**Detection**: Check `cudaDeviceProp::accessPolicyMaxWindowSize`. If `num_bytes` exceeds it, behavior is undefined.
**Fix**: Clamp `num_bytes` to `min(prop.accessPolicyMaxWindowSize, desired_size)`.
**Source**: Programming Guide, Section 4.13.7 (Query L2 Cache Properties)

## P6: Setting L2 persisting policy on data that is already well-cached can backfire by reserving L2 capacity and evicting other useful lines, dramatically increasing DRAM traffic and hurting performance (discovered in verification)

**Symptom**: Setting L2 persisting policy on data that is already well-cached can backfire by reserving L2 capacity and evicting other useful lines, dramatically increasing DRAM traffic and hurting performance.
**Source**: Level 3 sandbox verification (2026-04-05)

## P7: When a kernel already operates near peak memory bandwidth, L2 set-aside with hitRatio can improve cache metrics (useful for multi-kernel contention scenarios) but may slightly hurt single-kernel latency due to policy enforcement overhead and altered access patterns (discovered in verification)

**Symptom**: When a kernel already operates near peak memory bandwidth, L2 set-aside with hitRatio can improve cache metrics (useful for multi-kernel contention scenarios) but may slightly hurt single-kernel latency due to policy enforcement overhead and altered access patterns.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: This skill is only observable in multi-kernel sequences where an earlier kernel sets persisting L2 policy; a single-kernel benchmark cannot demonstrate the benefit of cache reset (discovered in verification)

**Symptom**: This skill is only observable in multi-kernel sequences where an earlier kernel sets persisting L2 policy; a single-kernel benchmark cannot demonstrate the benefit of cache reset.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: L2 access policy windows are silently ignored when the data size exceeds the hardware's reservable L2 partition (typically a fraction of total L2); the API succeeds but no persistence actually occurs (discovered in verification)

**Symptom**: L2 access policy windows are silently ignored when the data size exceeds the hardware's reservable L2 partition (typically a fraction of total L2); the API succeeds but no persistence actually occurs.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: This skill is purely about runtime configuration queries; verifying it as a performance optimization is a category error since the kernel code itself doesn't change (discovered in verification)

**Symptom**: This skill is purely about runtime configuration queries; verifying it as a performance optimization is a category error since the kernel code itself doesn't change.
**Source**: Level 3 sandbox verification (2026-04-05)
