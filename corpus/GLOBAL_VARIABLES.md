# Global Placeholder Variables

These variables appear in `MANIFEST.yaml` tier-2 entries and are resolved
by `scripts/localize.py` using `corpus/localize.yaml`.

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `CUDA_REPO_ROOT` | Root directory containing CUDA-related repo clones | `/home/user/workspace/cuda_repo` |
| `CUTLASS_REPO_REF` | Path to NVIDIA/cutlass clone | `{{CUDA_REPO_ROOT}}/cutlass` |
| `CUDA_SAMPLES_REPO_REF` | Path to NVIDIA/cuda-samples clone | `{{CUDA_REPO_ROOT}}/cuda-samples` |
| `CCCL_REPO_REF` | Path to NVIDIA/cccl clone | `{{CUDA_REPO_ROOT}}/cccl` |
| `FLASH_ATTENTION_REPO_REF` | Path to flash-attention clone | `{{CUDA_REPO_ROOT}}/flash-attention` |
