---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.4 (NUMA Best Practices)
  - Programming Guide, Section 4.17 (Extended GPU Memory)
cross_ref:
  - Programming Guide, Section 4.17.1.2 (Socket Identifiers)
  - Programming Guide, Section 4.17.2.2 (Single-Node Multi-GPU)
related_apis:
  - cudaMemAdvise, cudaMemPrefetchAsync, cudaDeviceEnablePeerAccess, cudaMemLocation, cudaMemPoolCreate, CU_MEM_LOCATION_TYPE_HOST_NUMA
related_experience: []
unlocks:
  - "memory/host-device-transfer: NUMA-aware allocation ensures transfers use the fastest path"
  - "memory/unified-memory: NUMA binding optimizes unified memory placement on multi-socket systems"
conflicts_with: []
---

# NUMA Binding

## Skill 1: Disable Automatic NUMA Balancing for GPU Workloads

### When to Use
- Running on Linux with automatic NUMA balancing enabled
- GPU application performance is degraded by OS page migration

### How to Apply
1. Identify CPU NUMA nodes associated with GPUs
2. Use `numactl --membind` to bind memory allocations to specific NUMA nodes
3. On IBM POWER9 nodes, CPUs are NUMA nodes 0 and 8

### Code Template
```bash
# Bind memory to NUMA nodes 0 and 8 (IBM POWER9 example)
numactl --membind=0,8 ./my_gpu_app

# Check current NUMA policy
numactl --show

# Disable automatic NUMA balancing system-wide (requires root)
echo 0 > /proc/sys/kernel/numa_balancing
```

### Source
Best Practices Guide, Section 10.4 (NUMA Best Practices)

## Skill 2: Query GPU Host NUMA ID for Memory Placement

### When to Use
- Multi-GPU system where each GPU is attached to a different NUMA node
- Want to allocate host memory closest to the GPU for minimum transfer latency

### How to Apply
1. Query `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` for each GPU
2. Allocate host memory on the same NUMA node as the target GPU
3. Use `numactl` or `numa_alloc_onnode()` for NUMA-local allocation

### Code Template
```cuda
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);
printf("GPU %d is attached to NUMA node %d\n", deviceOrdinal, numaId);

// Linux: allocate on specific NUMA node
#include <numa.h>
void* host_buf = numa_alloc_onnode(size, numaId);
cudaHostRegister(host_buf, size, cudaHostRegisterDefault);
```

### Source
Programming Guide, Section 4.17.1.2 (Socket Identifiers)

## Skill 3: Use Extended GPU Memory (EGM) with VMM APIs

### When to Use
- Grace Hopper or similar NVLink-C2C integrated systems
- Want GPU threads to access all system memory at NVLink speed
- Multi-GPU systems with NUMA-aware allocation

### How to Apply
1. Query NUMA ID with `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)`
2. Create physical allocation with `CU_MEM_LOCATION_TYPE_HOST_NUMA`
3. Reserve virtual address, map, and set access permissions

### Code Template
```cuda
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);

CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;

size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ((size + granularity - 1) / granularity) * granularity;

CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);

CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);

// Set access for both host NUMA node and GPU
CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = deviceOrdinal;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, padded_size, accessDesc, 2);
```

### Source
Programming Guide, Section 4.17.2.2.1 (Using VMM APIs)

## Skill 4: Use CUDA Memory Pool with NUMA Location

### When to Use
- Stream-ordered allocation with NUMA-aware placement
- Simpler API than VMM for NUMA memory management

### How to Apply
1. Create a memory pool with `cudaMemLocationTypeHostNuma` location type
2. Set NUMA node ID
3. Allocate from pool with `cudaMallocFromPoolAsync`
4. Grant peer access with `cudaMemPoolSetAccess`

### Code Template
```cuda
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, homeDevice);

cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;

cudaMemPool_t memPool;
cudaMemPoolCreate(&memPool, &props);

// Grant access to peer GPU
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = peerDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);

float* d_data;
cudaMallocFromPoolAsync(&d_data, N * sizeof(float), memPool, stream);
```

### Source
Programming Guide, Section 4.17.2.2.2 (Using CUDA Memory Pool)

## Skill 5: Bind Process to NUMA-Local CPUs and Memory

### When to Use
- Multi-socket server where GPU is attached to one CPU socket
- Host-side data processing should occur on the same socket as the GPU

### How to Apply
1. Identify which CPU cores and NUMA nodes are local to the target GPU
2. Use `numactl --cpunodebind=N --membind=N` to bind the process
3. Or use `sched_setaffinity()` and `set_mempolicy()` programmatically

### Code Template
```bash
# Find GPU-to-NUMA mapping
nvidia-smi topo --matrix

# Run application bound to NUMA node 0 (GPU 0's NUMA node)
numactl --cpunodebind=0 --membind=0 ./my_app

# For multi-GPU, use MPI rank binding
mpirun -np 4 --bind-to numa ./my_app
```

### Source
Best Practices Guide, Section 10.4 (NUMA Best Practices)

## Cascading Opportunities (unlocks)
After NUMA binding:
1. Check memory/host-device-transfer -- NUMA-local memory ensures fastest PCIe/NVLink transfers
2. Check memory/unified-memory -- NUMA-aware placement hints improve managed memory performance

## Conflicts
- None significant; NUMA binding is purely beneficial

## Principles
- **P1**: Automatic NUMA balancing can degrade GPU application performance by migrating pages to remote NUMA nodes. Disable it for GPU workloads.
- **P2**: Memory allocated on a remote NUMA node may have 2-3x higher latency for PCIe transfers than NUMA-local memory.
- **P3**: EGM on Grace Hopper routes all memory access over high-bandwidth NVLink-C2C, but NUMA-local allocation is still preferred for minimum latency.
- **P4**: Do not use cgroups to limit visible devices on EGM systems; use CUDA_VISIBLE_DEVICES instead, as cgroups can block EGM routing.

## Open Questions (for Level 3 verification)
- Q1: What is the measured bandwidth difference between NUMA-local and NUMA-remote PCIe transfers on a dual-socket x86 server?
- Q2: On Grace Hopper, does EGM provide full NVLink bandwidth for remote socket memory access?
- Q3: How does automatic NUMA balancing interact with cudaMemPrefetchAsync?
