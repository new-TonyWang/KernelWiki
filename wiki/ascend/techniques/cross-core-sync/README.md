---
id: skill-ascend-cross-core-sync
title: "AscendC Cross-Core Synchronization and WorkspaceQueue"
type: skill
vendor: ascend
tags:
- ascendc
- cube-unit
- vector-unit
evidence_level: spec
applies_to:
- ascend910b
source:
- path: local
  anchor: AscendOpGenAgent/skills
architectures:
- ascend910b
languages:
- ascendc
hardware_features:
- cube-unit
- vector-unit
techniques:
- pipeline-stages
- double-buffering
---
## Cross-Core Synchronization

### Sync Mode Decision

| Pattern | Sync Mode | AscendC Implementation |
|:---|:---|:---|
| `T.set_cross_flag` inside n_tile loop | Per-tile sync (WorkspaceQueue) | Ring buffer + per-tile Acquire/Release |
| `T.set_cross_flag` outside n_tile loop | Bulk sync | Single CrossCoreSetFlag/WaitFlag |

### WorkspaceQueue (Ring Buffer)

AIC → AIV data transfer via workspace GM ring buffer with `CrossCoreSetFlag/WaitFlag`:

```cpp
template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    __aicore__ inline void InitFreeSlots() {
        for (uint32_t i = 0; i < DEPTH; ++i)
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
    }

    // Producer (AIC): wait for free slot, write via Fixpipe
    __aicore__ inline GlobalTensor<T> ProducerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(vecNotifyCubeId_);
        return workspace_[head_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ProducerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeNotifyVecId_);
        head_++;
    }

    // Consumer (AIV): wait for data ready, read via MTE2
    __aicore__ inline GlobalTensor<T> ConsumerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(cubeNotifyVecId_);
        return workspace_[tail_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ConsumerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
        tail_++;
    }
};
```

### Bulk Sync (No Ring Buffer)

When Cube must finish all tiles before Vector starts (e.g. two-pass quantization):

```cpp
if ASCEND_IS_AIC {
    for (int by = 0; by < nTiles; by++)
        mm_.ComputeBlock(aBlock, bBlock, wsBlock, H_K, N);
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
}
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);
    // Pass 1: scan all tiles for stats
    // Pass 2: process with stats
}
```

### Comparison

| Feature | WorkspaceQueue | Bulk Sync |
|:---|:---|:---|
| Signal count | Per tile | Once after all tiles |
| Workspace size | DEPTH x tile size | Full output size |
| Vector start | After first tile ready | After all tiles done |
| Use case | Per-tile independent ops | Needs global statistics |

### CrossCore Flag Rules

- `T.set_cross_flag("FIX", idx)` → `CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + idx)`, called once
- All AIV sub-blocks share the same flag (use `GetSubBlockIdx()` for data offset)
- Bulk sync uses single flag broadcast, NOT per-subblock flags
