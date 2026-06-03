# Source/Artifact Reference Audit and Remediation

Date: 2026-06-02

## Summary

This audit targeted the kp-mvp migration split described in `docs/kernel-wiki-merge-plan.md`: source pages should live under `sources/experience/*.md`, while runnable code/build/profile/device assets should live under `artifacts/experience/**` with bundle provenance.

The initial scan found the following major classes of problems:

- Wiki and source pages referred to migrated artifact files through the old experience source tree path shape.
- Many frontmatter `artifacts:` fields still had an extra legacy path segment between the bundle root and file name.
- Several frontmatter `source[].path` entries pointed at runnable artifacts rather than source pages or corpus/spec references.
- Several renamed pages still used pre-flattening paths, especially TMA/WGMMA PTX skill paths and GEMM PTX paths.

## Remediation performed

- Rewrote migrated artifact file references to the canonical `artifacts/experience/**` bundle layout.
- Rewrote profile references to the actual flat CSV/log locations used by the migrated bundles.
- Converted source frontmatter entries that pointed at local runnable artifacts to their corresponding `sources/experience/*.md` pages.
- Fixed common renamed wiki paths, including TMA PTX, WGMMA PTX, GEMM PTX, and kernel-record page references.
- Fixed `artifacts:` frontmatter entries in active `sources/experience` and `wiki/nvidia` pages so they point to existing files or removed non-artifact placeholders.
- Added validator checks for legacy `source:` and `artifacts:` fields so this class of broken path can no longer pass silently.

## Post-fix status

- `python3 scripts/validate.py` passes.
- Frontmatter `source[].path` issues found by the custom audit: 0.
- Active frontmatter `artifacts:` path issues found by the custom audit: 0.
- Active source/artifact mix references in migrated runtime pages: resolved.

Remaining path-like findings are mostly intentionally non-concrete placeholders in planning/audit prose, historical migration examples, or verbatim upstream PR/debug text. These are not active frontmatter references and are not part of the source/artifact split contract.

## Follow-up guardrail

`scripts/validate.py` now validates:

- `source[].path` local references when they point at repo paths.
- `artifacts.*` local references when they point at repo paths.
- the rule that source paths must not point into migrated artifact file locations.
- the rule that artifact fields must not point into source-page locations.
