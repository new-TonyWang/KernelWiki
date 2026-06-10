# Task Plan: KernelWiki source/artifact reference audit

## Goal
全方位排查 KernelWiki 中 source 与 artifact 分离不彻底导致的引用问题：不存在路径、旧 kp-mvp `80-experience/.../artifacts` 路径、绝对路径、source_refs/artifact_dir 不一致、wiki body 中 `source` 标题下误放 artifact 路径等。

## Phases
- [complete] Phase 1: Baseline inventory and identify relevant schema/path conventions.
- [complete] Phase 2: Scan all Markdown/YAML files for path-like references and classify external corpus vs local artifact vs missing.
- [complete] Phase 3: Specifically audit `sources/experience`, `artifacts/experience`, and `wiki/nvidia/**` migrated pages.
- [complete] Phase 4: Produce consolidated findings and recommended fixes.

## Decisions
- Do not perform bulk migration fixes unless explicitly requested; this turn focuses on audit/reporting.
- Treat files under `artifacts/experience/**` as artifact bundles, not source corpus paths.
- Treat external repo references as valid only when represented as logical corpus source (`source_id + repo-relative path`) or resolvable via corpus localize/manifest.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
