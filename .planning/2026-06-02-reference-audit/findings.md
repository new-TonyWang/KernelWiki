# Findings: KernelWiki reference audit


## Baseline audit results
- `python3 scripts/validate.py` passes: 2490 files, 2265 source IDs, 117 bundles, 0 errors.
- Custom path scan found 152 `sources/experience/.../artifacts` references, 184 frontmatter `artifacts:` path issues, and 27 frontmatter `source[].path` issues.
- The validator currently misses these because legacy `source:` and `artifacts:` fields are not schema-validated as local paths.
- Detailed report written to `docs/source-artifact-reference-audit.md`.

## Post-fix verification
- `python3 scripts/validate.py` passes after path rewrites and validator guardrail changes.
- Custom audit excluding historical prose reports:
  - source-artifact-mix: 0
  - frontmatter source[].path issues: 0
  - active frontmatter artifacts path issues: 0
- Remaining old kp-mvp and absolute path findings are mostly historical docs/templates/verbatim upstream PR text, not active source/artifact frontmatter.
