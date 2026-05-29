# External Repository Links

This directory contains symlinks to local clones of external source repositories.
These are NOT committed to git — each developer creates their own.

## Setup

```bash
python3 scripts/localize.py init-config  # Create corpus/localize.yaml from example
python3 scripts/localize.py link         # Create symlinks here
python3 scripts/localize.py check        # Verify paths
```

## Expected symlinks

- `cutlass` → local NVIDIA/cutlass clone
- `cuda-samples` → local NVIDIA/cuda-samples clone
- `flash-attention` → local flash-attention clone
