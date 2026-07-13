# Bank Conflict Avoidance -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `ld.shared[.vec][.type]` | PTX ISA | Shared memory load |
| `st.shared[.vec][.type]` | PTX ISA | Shared memory store |
| `cudaDeviceGetSharedMemConfig` | Runtime API | Returns the shared memory configuration for the current device |
| `cudaDeviceSetSharedMemConfig` | Runtime API | Sets the shared memory bank size for the current device (4-byte or 8-byte) |
| `cudaFuncSetSharedMemConfig` | Runtime API | Sets the shared memory bank size for a function (4-byte or 8-byte banks) |
| `cudaSharedMemConfig` | Runtime API | Shared memory bank size (Default, FourByte, EightByte) |
| `cudaSharedMemoryMode` | Runtime API | Shared memory mode (Default, 8ByteBankSize, 16ByteBankSize) |
