# Findings: Library-call content isolation

## Open Questions / To Resolve
- Which directories and Markdown files are included by the query/index pipeline?
- What exact patterns in this knowledge base represent “库调用”?
- Which exclusion mechanism prevents content from being retrieved by `query`?

## Repository/query indexing rules
- `scripts/query.py::load_all_pages()` loads Markdown only from top-level `sources/` and `wiki/`.
- `scripts/generate-indices.py::collect_all_pages()` also loads Markdown only from `sources/` and `wiki/` and writes derived indices under `queries/`.
- `scripts/grep_wiki.py` default scopes are `wiki` and `sources`; `nonquery/` is not searched by default.
- Therefore a top-level `nonquery/library-calls/` archive is outside the normal queryable corpus.

## Library-call inventory and handling
- Moved all eight `wiki/nvidia/operator-routing/*/library-fallback.md` pages into `nonquery/library-calls/operator-routing/<op>/library-fallback.md`.
- Archived additional removed/generalized wiki snippets in `nonquery/library-calls/extracted-wiki-snippets.md`.
- Archived exact source snippets containing high-level library-call tokens in `nonquery/library-calls/extracted-source-snippets.md` before generalizing them in queryable `sources/` pages.
- Rewrote operator-routing INDEX/TASK-PACKET/ROUTING pages to be custom-kernel focused and removed links/IDs/titles for moved fallback pages.

## Verification evidence
- `python3 scripts/generate-indices.py` now reports `Collected 2482 pages from sources/ and wiki/`; the moved `nonquery/` archive is not included.
- `python3 scripts/validate.py` passes: 2482 files, 2265 source IDs, 117 asset bundles, 6 candidate ledgers, 0 errors.
- Strict scan over queryable roots found no exact high-level library-call tokens matching: `torch.*`, `thrust::...`, `cub::Device...`, `cublas...`, `CUBLAS_...`, `cudnn...`, `cuDNN`, `flashinfer.<module>.<api>`, or `library-fallback`.
- Exact `scripts/query.py --paths-only` checks return no paths for: `library-fallback`, `routing-elementwise-library-fallback`, `torch.sum`, `torch.compile`, `thrust::transform`, `cub::DeviceReduce::Sum`, `cublasGemmEx`, `cublasLtMatrixTransform`, `cudnnPoolingForward`, and `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`.
