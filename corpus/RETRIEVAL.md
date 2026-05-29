# Source Corpus Retrieval Contract

Agents retrieve source material through the two-tier corpus system:

## Tier 1: In-Git Corpus
Files committed under `corpus/nvidia/`. Always available.
```bash
python3 scripts/source_corpus_cli.py search "keyword" --scope cuda-official
```

## Tier 2: External Repos
Referenced via `{{PLACEHOLDER}}` variables, resolved by `corpus/localize.yaml`.
Requires user setup via `scripts/localize.py init-config`.
```bash
python3 scripts/source_corpus_cli.py search "keyword" --scope source-code/cutlass
```
