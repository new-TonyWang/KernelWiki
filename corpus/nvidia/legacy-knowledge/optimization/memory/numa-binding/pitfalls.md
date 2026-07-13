# NUMA Binding -- Pitfalls

## P1: Automatic NUMA Balancing Silently Degrading GPU Performance
**Symptom**: GPU application performance is inconsistent or lower than expected. Memory is migrated by the OS between NUMA nodes.
**Detection**: Check `cat /proc/sys/kernel/numa_balancing`. Value 1 means auto-balancing is active. `numastat` shows cross-node allocations.
**Fix**: Disable automatic NUMA balancing: `echo 0 > /proc/sys/kernel/numa_balancing` (requires root). Or use `numactl --membind` per application.
**Source**: Best Practices Guide, Section 10.4 (NUMA Best Practices)

## P2: Memory Allocated on Wrong NUMA Node
**Symptom**: PCIe transfer bandwidth is lower than expected. Memory is on a remote NUMA node relative to the GPU.
**Detection**: Use `numastat -p <pid>` to see per-node allocation. Cross-node allocation is visible.
**Fix**: Query GPU's NUMA ID with `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` and allocate on that node with `numa_alloc_onnode()` or EGM APIs.
**Source**: Programming Guide, Section 4.17.1.2 (Socket Identifiers)

## P3: Using cgroups to Limit Devices on EGM Systems
**Symptom**: EGM routing is blocked; memory access falls back to slower paths. Performance drops significantly.
**Detection**: System uses cgroups for device isolation on a Grace Hopper or similar EGM platform.
**Fix**: Use `CUDA_VISIBLE_DEVICES` environment variable instead of cgroups for device isolation on EGM systems.
**Source**: Programming Guide, Section 4.17.1.1 (EGM Platforms)

## P4: EGM Allocation Without Proper Access Permissions
**Symptom**: GPU kernel crashes or returns errors when accessing EGM memory. `cuMemSetAccess` was not called.
**Detection**: CUDA error on kernel launch or memory access. Missing `cuMemSetAccess` call after `cuMemMap`.
**Fix**: Always call `cuMemSetAccess` with appropriate access descriptors for both the host NUMA node and the GPU device after mapping virtual memory.
**Source**: Programming Guide, Section 4.17.2.2.1 (Using VMM APIs)

## P5: Not Checking NUMA Support on the Platform
**Symptom**: NUMA-specific APIs fail or return unexpected values on systems without NUMA (single-socket, Tegra).
**Detection**: `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` returns -1 or an error on non-NUMA systems.
**Fix**: Check for NUMA support before using NUMA-specific APIs. Fall back to standard allocation on single-socket systems.
**Source**: Programming Guide, Section 4.17.1.2 (Socket Identifiers)

## P6: Wall-clock timing of CUDA extensions can be dominated by first-invocation overhead (JIT, context setup), making round-0 baselines unreliable; always use warm-start measurements or multiple iterations to isolate actual optimization effects (discovered in verification)

**Symptom**: Wall-clock timing of CUDA extensions can be dominated by first-invocation overhead (JIT, context setup), making round-0 baselines unreliable; always use warm-start measurements or multiple iterations to isolate actual optimization effects.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: NUMA binding only matters for host-to-device transfer latency on multi-socket systems; it cannot improve GPU kernel execution time, so benchmarks must measure PCIe transfer time specifically, not kernel runtime (discovered in verification)

**Symptom**: NUMA binding only matters for host-to-device transfer latency on multi-socket systems; it cannot improve GPU kernel execution time, so benchmarks must measure PCIe transfer time specifically, not kernel runtime.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: cudaMemPoolCreate with cudaMemLocationTypeHostNuma may silently fall back to default placement if the system has a single NUMA node or the driver doesn't honor the hint, giving zero benefit with added API complexity (discovered in verification)

**Symptom**: cudaMemPoolCreate with cudaMemLocationTypeHostNuma may silently fall back to default placement if the system has a single NUMA node or the driver doesn't honor the hint, giving zero benefit with added API complexity.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: NUMA binding is a host-side optimization; verifying it requires measuring end-to-end pipeline time including host allocation, H2D/D2H transfers, and CPU preprocessing — pure kernel profiling will never show a difference (discovered in verification)

**Symptom**: NUMA binding is a host-side optimization; verifying it requires measuring end-to-end pipeline time including host allocation, H2D/D2H transfers, and CPU preprocessing — pure kernel profiling will never show a difference.
**Source**: Level 3 sandbox verification (2026-04-05)
