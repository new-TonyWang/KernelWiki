# KernelWiki Merge & Multi-Vendor Extension Plan

## Goal Description

Merge **kernel-kb-mvp** (~200 files, atomic-level knowledge: API definitions, skills, probes, operator routing, code walkthroughs) and **legacy-knowledge** (48 skills + hardware/reasoning/patterns) into **KernelWiki** (2,265 files, synthesized knowledge base). Extend KernelWiki to support multi-vendor GPU architectures via vendor-scoped directories, 7 new page types, a two-tier source corpus with localization, and updated validation/query tooling. The merged repository becomes the single canonical GPU kernel optimization knowledge base.

**Scope**: NVIDIA is the only active vendor for this merge. Multi-vendor schema (vendors.yaml, vendor field, by-vendor index) is ready but no stub directories are created for planned vendors (Huawei, Biren).

**Key resolved decisions**:
- Corpus storage: **Git LFS** for the ~204MB in-git corpus
- Path migration: **Breaking change** — all `wiki/*` paths become `wiki/nvidia/*` in one atomic commit
- Probe/experience location: **Sources layer only** — `sources/experience/` with artifacts in `artifacts/experience/`
- `sources` vs `source_refs`: **Coexist permanently** — `sources` = wiki/source page IDs, `source_refs` = external repo file references
- Vendor scope: **Schema ready, no stubs** — `vendors.yaml` includes planned vendors; no `wiki/huawei/` or `wiki/biren/` created
- Language: **Mixed English/Chinese preserved** in the draft; implementation code and docs follow English canonical

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Validation passes with minimal errors after full migration
  - Positive Tests (expected to PASS):
    - `python3 scripts/validate.py` completes with 0 errors on the fully migrated repo
    - Tier-2 corpus warnings (missing `localize.yaml`) are reported as warnings, not errors
  - Negative Tests (expected to FAIL):
    - A page with `type: skill` missing `evidence_level` is rejected
    - A page under `wiki/nvidia/` with `vendor: huawei` is rejected (vendor-path mismatch)
  - AC-1.1: New page types validated against new schemas
    - Positive: A `wiki-skill` page with all required fields passes validation
    - Negative: A `wiki-api-definition` page missing `func_name` or `signature` is rejected

- AC-2: Existing 2,265 pages content-preserved after path migration
  - Positive Tests (expected to PASS):
    - Normalized body-hash comparison (allowing declared path/link rewrites) matches for every existing page before and after migration
    - Frontmatter `id` fields remain unchanged for all existing pages
  - Negative Tests (expected to FAIL):
    - A page whose body content was altered beyond path rewrites is flagged by the comparison tool
    - A page whose `id` was changed is flagged as a validation error

- AC-3: All kernel-kb-mvp content present in new locations
  - Positive Tests (expected to PASS):
    - Migration inventory TSV covers every `.md` file under kernel-kb-mvp `knowledge/` with status `migrated`, `skipped`, or `deleted`
    - Every `migrated` entry has a target file that exists in the repo
  - Negative Tests (expected to FAIL):
    - A kernel-kb-mvp source file not present in the inventory causes the migration check to fail
    - A `migrated` entry pointing to a nonexistent target is flagged

- AC-4: `--vendor nvidia` query returns all NVIDIA pages
  - Positive Tests (expected to PASS):
    - `python3 scripts/query.py --vendor nvidia --compact` returns pages from `wiki/nvidia/`
    - `python3 scripts/query.py --vendor nvidia --type skill` returns only skill pages under `wiki/nvidia/foundations/`
  - Negative Tests (expected to FAIL):
    - `python3 scripts/query.py --vendor huawei` returns 0 results (no Huawei pages exist)
    - A page under `wiki/nvidia/` not returned by `--vendor nvidia` is a bug

- AC-5: Cross-references resolve — no broken `related:`, `sources:`, or `source_refs:` links
  - Positive Tests (expected to PASS):
    - `scripts/validate.py` reports 0 broken reference errors across all frontmatter fields
    - `source_refs` entries with `source_id` matching a `corpus/MANIFEST.yaml` entry pass
  - Negative Tests (expected to FAIL):
    - A `related:` entry pointing to a nonexistent page ID is rejected
    - A `source_refs` entry with unknown `source_id` not in MANIFEST.yaml is rejected
  - AC-5.1: All body markdown links resolve after path migration
    - Positive: Relative links from `wiki/nvidia/hardware/tcgen05-mma.md` to `wiki/nvidia/techniques/warp-specialization.md` resolve
    - Negative: A body link using the old path `../../wiki/hardware/tcgen05-mma.md` fails link validation

- AC-6: Agent smoke-test passes — one task YAML runs to completion
  - Positive Tests (expected to PASS):
    - `tasks/build-*.yaml` or similar task runs with updated paths and completes without path-not-found errors
  - Negative Tests (expected to FAIL):
    - A task YAML referencing old `knowledge/30-skill/...` paths fails to resolve

- AC-7: Legacy experience files deleted with deletion ledger
  - Positive Tests (expected to PASS):
    - Deletion ledger exists with one entry per deleted file (path, reason: superseded/not-migrated/dependency-lost)
    - Ledger file count equals actual deleted file count
    - No legacy experience files remain in `corpus/nvidia/legacy-knowledge/experience/`
  - Negative Tests (expected to FAIL):
    - A file deleted without a ledger entry causes validation to fail
    - A file in the deletion ledger that still exists on disk causes validation to fail

- AC-8: `queries/by-vendor.md` auto-generated correctly
  - Positive Tests (expected to PASS):
    - `python3 scripts/generate-indices.py` produces `queries/by-vendor.md` with NVIDIA page counts per type
    - Page counts match actual file counts under `wiki/nvidia/`
  - Negative Tests (expected to FAIL):
    - `by-vendor.md` listing pages for a vendor with zero wiki pages is a generation error

- AC-9: SKILL.md updated — description includes foundational skills, operator routing, API definitions
  - Positive Tests (expected to PASS):
    - SKILL.md description mentions "foundational CUDA optimization skills", "operator routing", "API definitions"
    - SKILL.md no longer contains "Blackwell/Hopper-specific" exclusion language
  - Negative Tests (expected to FAIL):
    - A grep for "Blackwell/Hopper-specific" or "Do NOT use for.*Blackwell" finds matches in SKILL.md

- AC-10: `scripts/localize.py check` passes with example config
  - Positive Tests (expected to PASS):
    - Copying `corpus/localize.yaml.example` to `corpus/localize.yaml` and running `scripts/localize.py check` reports status for each variable
    - Tier-1 in-git corpus paths report OK
  - Negative Tests (expected to FAIL):
    - Running `scripts/localize.py check` without any `localize.yaml` reports clear warnings for missing tier-2 paths (not crashes)

- AC-11: Tier-1 corpus search works without localize.yaml
  - Positive Tests (expected to PASS):
    - `python3 scripts/source_corpus_cli.py search "__shfl_sync" --scope cuda-official` returns results from in-git corpus
  - Negative Tests (expected to FAIL):
    - Search for tier-2 content without localize.yaml configured fails gracefully with a clear error message

- AC-12: Tier-2 corpus search works WITH localize.yaml configured
  - Positive Tests (expected to PASS):
    - With `localize.yaml` configured pointing to a local CUTLASS repo, `search "wgmma" --scope source-code/cutlass` returns results
  - Negative Tests (expected to FAIL):
    - With `localize.yaml` pointing to a nonexistent path, the search command fails with a clear path-not-found error

- AC-13: MANIFEST.yaml entries all resolve
  - Positive Tests (expected to PASS):
    - Tier-1 entries resolve to committed files under `corpus/nvidia/`
    - Tier-2 entries have valid `remote.url` and `default_ref` fields
  - Negative Tests (expected to FAIL):
    - A MANIFEST entry with missing `remote.url` for an external source is rejected by validation

- AC-14: `corpus/.external/` and `corpus/localize.yaml` gitignored
  - Positive Tests (expected to PASS):
    - `git status` does not show `corpus/.external/` or `corpus/localize.yaml` as untracked after creating them
  - Negative Tests (expected to FAIL):
    - Removing the gitignore entries causes `git status` to show these files

- AC-15: No absolute machine paths in committed Markdown/YAML
  - Positive Tests (expected to PASS):
    - `grep -r "/data1/" sources/ wiki/ corpus/ artifacts/ --include="*.md" --include="*.yaml"` returns 0 results (excluding PROVENANCE.yaml `upstream_path` historical fields)
  - Negative Tests (expected to FAIL):
    - A committed page containing `/home/tongyu/workspace/cuda_repo/cutlass/...` is flagged

- AC-16: No duplicate page IDs across sources/ and wiki/
  - Positive Tests (expected to PASS):
    - `scripts/validate.py` checks for ID uniqueness across all pages and reports 0 duplicates
  - Negative Tests (expected to FAIL):
    - Two pages with `id: skill-warp-primitives` in different locations are rejected

- AC-17: Migration dry-run is deterministic
  - Positive Tests (expected to PASS):
    - Two consecutive `scripts/migrate_kb_mvp.py --dry-run` runs produce identical TSV output
  - Negative Tests (expected to FAIL):
    - Output with non-deterministic ordering or timestamps differs between runs

- AC-18: Every migrated artifact bundle has PROVENANCE.yaml with sha256
  - Positive Tests (expected to PASS):
    - Every directory under `artifacts/experience/` contains a PROVENANCE.yaml with sha256 per file
    - sha256 values match actual file checksums
  - Negative Tests (expected to FAIL):
    - An artifact file without a sha256 entry in PROVENANCE.yaml is rejected by validation

- AC-19: source_refs validated against MANIFEST.yaml source_ids
  - Positive Tests (expected to PASS):
    - A page with `source_refs: [{source_id: source-code/cutlass, path: examples/...}]` passes when MANIFEST.yaml has `source-code/cutlass`
  - Negative Tests (expected to FAIL):
    - A page with `source_refs: [{source_id: source-code/unknown-repo}]` is rejected
    - A `source_refs` entry using an absolute path instead of repo-relative is rejected

- AC-20: Git LFS configured for corpus files
  - Positive Tests (expected to PASS):
    - `git lfs ls-files` lists files under `corpus/nvidia/cuda-official/`, `corpus/nvidia/blogs/`, `corpus/nvidia/whitepapers/`
    - `.gitattributes` has LFS tracking rules for corpus patterns
  - Negative Tests (expected to FAIL):
    - Corpus files committed without LFS tracking appear as regular blobs in `git lfs ls-files`

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes all content from kernel-kb-mvp and legacy-knowledge fully migrated with rewritten frontmatter, all 7 new page types with schemas, complete source corpus two-tier system with localization tooling, Git LFS for corpus, updated validation supporting new types and vendor fields, updated query tools with `--vendor` filter, all indices regenerated including `by-vendor.md`, updated SKILL.md/CLAUDE.md/README.md, migration script with deterministic dry-run and inventory, deletion ledger for legacy experience files, body-link validation after path migration, and an agent smoke-test passing end-to-end.

### Lower Bound (Minimum Acceptable Scope)

The implementation moves existing wiki/ content under wiki/nvidia/, adds the 7 new page type schemas to schemas.yaml, migrates kernel-kb-mvp content with rewritten frontmatter, sets up corpus/ with MANIFEST.yaml and localize.yaml.example, configures Git LFS for corpus, runs validation with 0 errors, regenerates indices, and updates SKILL.md. Localization tooling (localize.py) and agent integration may be minimal stubs that pass basic checks.

### Allowed Choices

- **Can use**: Python 3 for all scripts (consistent with existing tooling); PyYAML for frontmatter parsing; Git LFS for large file storage; existing PROVENANCE.yaml format for artifact bundles; ripgrep via subprocess for body-link validation
- **Cannot use**: External databases or services; JavaScript/Node.js tooling; CI-specific runners (all validation must be runnable locally); modification of existing page IDs (only paths change)
- **Fixed per draft**: Directory structure under wiki/nvidia/ follows the exact layout in Section 2.2; page type schemas follow Section 5; file migration mapping follows Section 6; MANIFEST.yaml format follows Section 13

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Schema-first**: Extend `data/schemas.yaml` with all 7 new types before any content migration. Run `scripts/validate.py` to confirm no regression on existing pages.
2. **Atomic path move**: Use `git mv` for the `wiki/` → `wiki/nvidia/` move, then bulk-rewrite all relative links in frontmatter and body text with a script. Verify with normalized body-hash comparison.
3. **Migration script pattern**: Parse each kb-mvp `.md` file → determine target path from mapping table → parse YAML frontmatter → add/rewrite fields (type, vendor, id, source paths) → write to target → log to TSV.
4. **Git LFS setup**: Add `.gitattributes` rules for `corpus/nvidia/**` before committing corpus files. This ensures LFS tracks them from the first commit.
5. **Incremental validation**: After each milestone, run full validation to catch issues early.

### Relevant References

- `data/schemas.yaml` — Current page type schemas to extend
- `data/tags.yaml` — Controlled vocabulary to extend with vendor tags
- `scripts/validate.py` — Validation pipeline to extend for new types, vendor field, source_refs, body links
- `scripts/generate-indices.py` — Index generator to extend for by-vendor.md
- `scripts/query.py` — Query tool to extend with --vendor filter
- `artifacts/kernels/flash-attention-4/full/PROVENANCE.yaml` — Example PROVENANCE.yaml format
- `SKILL.md` — Agent entry point to update
- `CLAUDE.md` — Schema documentation to update

## Dependencies and Sequence

### Milestones

1. **M1: Schema & Tooling Foundation**
   - Extend `data/schemas.yaml` with 7 new page types (skill, pitfall, api-definition, operator-routing, algorithm, code-walkthrough, experience) + source-experience
   - Create `data/vendors.yaml` with NVIDIA (active) + planned vendors
   - Extend `data/tags.yaml` with vendor dimension + new source categories
   - Extend `data/aliases.yaml` with new aliases
   - Update `scripts/validate.py` to validate new types, vendor field, vendor-path consistency, source_refs against MANIFEST, no-absolute-paths check, duplicate ID check, body-link validation
   - Update `scripts/query.py` with `--vendor` filter
   - Update `scripts/generate-indices.py` to produce `queries/by-vendor.md`
   - Validate: existing 2,265 pages still pass with extended schemas (no regression)

2. **M2: Structural Path Migration**
   - **Decision gate**: DEC-2 resolved (breaking change confirmed)
   - Move `wiki/{hardware,techniques,kernels,patterns,languages,migration}/` → `wiki/nvidia/`
   - Create new directories: `wiki/nvidia/foundations/`, `wiki/nvidia/operator-routing/`, `wiki/nvidia/api-definitions/`, `wiki/nvidia/code-walkthroughs/`
   - Rewrite all relative links in frontmatter (`related:`, `sources:`) and body text
   - Set up corpus/ structure: `corpus/MANIFEST.yaml`, `corpus/localize.yaml.example`, `corpus/GLOBAL_VARIABLES.md`, `corpus/RETRIEVAL.md`, `corpus/.external/`
   - Configure Git LFS: add `.gitattributes` for `corpus/nvidia/**` patterns
   - Copy tier-1 in-git corpus via Git LFS: `corpus/nvidia/cuda-official/`, `corpus/nvidia/blogs/`, `corpus/nvidia/whitepapers/`
   - Copy `reasoning/`, `templates/`, `agent/`, `tasks/` from kernel-kb-mvp
   - Create `sources/experience/` directory
   - Update `.gitignore` for `corpus/localize.yaml`, `corpus/.external/`
   - Validate: normalized body-hash match for all existing pages, 0 validation errors

3. **M3: Content Migration**
   - **Decision gate**: DEC-3, DEC-6 resolved (probe model, language)
   - Develop `scripts/migrate_kb_mvp.py` with dry-run, deterministic TSV inventory
   - Migrate all kb-mvp content per Section 6 mapping:
     - `30-skill/` → `wiki/nvidia/foundations/`
     - `40-hardware-feature/` → `wiki/nvidia/hardware/{topic}/`
     - `50-classical-algo/` → `wiki/nvidia/techniques/{topic}/`
     - `60-code/` → `wiki/nvidia/code-walkthroughs/`
     - `10-api-raw/` → `wiki/nvidia/api-definitions/`
     - `20-pattern/` → `wiki/nvidia/operator-routing/`
     - `80-experience/` → `sources/experience/` + `artifacts/experience/`
   - Rewrite all `source:` paths (absolute → source_id + relative)
   - Generate PROVENANCE.yaml for all migrated artifact bundles
   - Generate deletion ledger for legacy experience files, then delete
   - Migrate legacy-knowledge: reasoning-pattern → `reasoning/optimization-reasoning/`, hardware → `wiki/nvidia/hardware/foundation/`
   - Validate: migration inventory complete, 0 validation errors, no absolute paths

4. **M4: Source Corpus & Localization Tooling**
   - **Decision gate**: DEC-1 resolved (Git LFS confirmed)
   - Develop `scripts/localize.py` (init-config, check, link, resolve-vars, doctor)
   - Migrate `scripts/source_corpus/` CLI from kb-mvp, adapt corpus root
   - Rewrite MANIFEST.yaml local_path prefixes for new structure
   - Test: `localize.py check` passes with example config
   - Test: Tier-1 search works without localize.yaml
   - Test: Tier-2 search works with localize.yaml
   - Validate: MANIFEST.yaml entries resolve, source_refs validate

5. **M5: Index, Documentation & Smoke Test**
   - Regenerate all `queries/` indices including new `by-vendor.md`
   - Update `index.md` — add foundations, operator-routing, api-definitions sections
   - Update `SKILL.md` — expanded description per Section 7 of draft
   - Update `CLAUDE.md` — new schema reference, new page types, vendor field
   - Update `references/primer.md`, `references/schema.md`, `references/examples.md`
   - Update `README.md` — new install, content inventory, Git LFS setup
   - Agent smoke-test: run one task YAML end-to-end
   - Final full validation pass

### Dependency Graph

```
M1 (Schema/Tooling) → M2 (Path Migration) → M3 (Content Migration) → M5 (Docs/Smoke)
                                                                   ↗
                       M4 (Corpus/Localization) ──────────────────
```

M1 must complete before M2 (schemas needed for validation after path move).
M2 must complete before M3 (target directories must exist).
M4 can proceed in parallel with M3 once M2 corpus/ structure is set up.
M5 depends on both M3 and M4 completing.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| t1 | Analyze kb-mvp repo structure, catalog all files, validate migration mapping completeness | AC-3 | analyze | - |
| t2 | Analyze legacy-knowledge repo, identify files to migrate vs skip vs delete | AC-7, AC-3 | analyze | - |
| t3 | Extend `data/schemas.yaml` with 7 new wiki page types + source-experience | AC-1.1 | coding | t1 |
| t4 | Create `data/vendors.yaml` with NVIDIA active + planned vendors | AC-4 | coding | - |
| t5 | Extend `data/tags.yaml` with vendor dimension, new source categories, new technique tags | AC-1 | coding | t4 |
| t6 | Extend `data/aliases.yaml` with new canonical terms | AC-1 | coding | t5 |
| t7 | Update `scripts/validate.py`: new type validation, vendor-path check, source_refs check, duplicate ID check, no-absolute-paths check | AC-1, AC-5, AC-15, AC-16, AC-19 | coding | t3, t4, t5 |
| t8 | Update `scripts/validate.py`: body-link validation for markdown relative links | AC-5.1 | coding | t7 |
| t9 | Update `scripts/query.py` with `--vendor` filter | AC-4 | coding | t4 |
| t10 | Update `scripts/generate-indices.py` to produce `queries/by-vendor.md` | AC-8 | coding | t4, t9 |
| t11 | Verify existing 2,265 pages pass validation with extended schemas (no regression) | AC-1 | analyze | t7 |
| t12 | Move `wiki/*` → `wiki/nvidia/*` with `git mv`, create new empty directories | AC-2 | coding | t11 |
| t13 | Bulk-rewrite relative links in frontmatter and body text for moved wiki pages | AC-2, AC-5.1 | coding | t12 |
| t14 | Validate path migration: normalized body-hash comparison for all existing pages | AC-2 | analyze | t13 |
| t15 | Set up `corpus/` directory structure: MANIFEST.yaml, localize.yaml.example, .external/, .gitignore | AC-13, AC-14 | coding | t12 |
| t16 | Configure Git LFS: add `.gitattributes` rules for corpus patterns | AC-20 | coding | t15 |
| t17 | Copy tier-1 in-git corpus via Git LFS into `corpus/nvidia/` | AC-11, AC-20 | coding | t16 |
| t18 | Copy `reasoning/`, `templates/`, `agent/`, `tasks/` from kernel-kb-mvp | AC-3 | coding | t12 |
| t19 | Create `sources/experience/` directory and structure | AC-3 | coding | t12 |
| t20 | Develop `scripts/migrate_kb_mvp.py` with dry-run, TSV inventory, frontmatter rewriting | AC-3, AC-17 | coding | t3, t12 |
| t21 | Run migration: 30-skill/ → wiki/nvidia/foundations/ | AC-3, AC-1.1 | coding | t20 |
| t22 | Run migration: 40-hardware-feature/ → wiki/nvidia/hardware/{topic}/ | AC-3 | coding | t20 |
| t23 | Run migration: 50-classical-algo/ → wiki/nvidia/techniques/{topic}/ | AC-3 | coding | t20 |
| t24 | Run migration: 60-code/ → wiki/nvidia/code-walkthroughs/ | AC-3 | coding | t20 |
| t25 | Run migration: 10-api-raw/ → wiki/nvidia/api-definitions/ | AC-3 | coding | t20 |
| t26 | Run migration: 20-pattern/ → wiki/nvidia/operator-routing/ | AC-3 | coding | t20 |
| t27 | Run migration: 80-experience/ → sources/experience/ + artifacts/experience/ with PROVENANCE.yaml | AC-3, AC-18 | coding | t19, t20 |
| t28 | Rewrite all `source:` paths in migrated pages (absolute → source_id + relative) | AC-15, AC-19 | coding | t21, t22, t23, t24, t25, t26, t27 |
| t29 | Generate deletion ledger for legacy experience files | AC-7 | coding | t2 |
| t30 | Delete legacy experience files per ledger | AC-7 | coding | t29 |
| t31 | Migrate legacy-knowledge: reasoning-pattern → reasoning/optimization-reasoning/ | AC-3 | coding | t2, t12 |
| t32 | Migrate legacy-knowledge: hardware → wiki/nvidia/hardware/foundation/ | AC-3 | coding | t2, t12 |
| t33 | Validate full migration: inventory TSV complete, 0 errors, no absolute paths, no duplicate IDs | AC-1, AC-3, AC-15, AC-16 | analyze | t28, t30, t31, t32 |
| t34 | Develop `scripts/localize.py` (init-config, check, link, resolve-vars, doctor) | AC-10 | coding | t15 |
| t35 | Migrate `scripts/source_corpus/` CLI from kb-mvp, adapt for new corpus root | AC-11, AC-12 | coding | t15, t17 |
| t36 | Test: localize.py check passes; tier-1 search works; tier-2 search works with config | AC-10, AC-11, AC-12 | analyze | t34, t35 |
| t37 | Regenerate all `queries/` indices including `by-vendor.md` | AC-8 | coding | t33 |
| t38 | Update `SKILL.md` per Section 7 of draft | AC-9 | coding | t33 |
| t39 | Update `CLAUDE.md` with new schema reference, page types, vendor field | AC-9 | coding | t33 |
| t40 | Update `index.md`, `references/primer.md`, `references/schema.md`, `references/examples.md` | AC-9 | coding | t33 |
| t41 | Update `README.md` with new install, content inventory, Git LFS setup | AC-9 | coding | t33 |
| t42 | Agent smoke-test: run one task YAML end-to-end | AC-6 | analyze | t37, t38, t39, t40 |
| t43 | Final full validation pass | AC-1 | analyze | t42 |

## Claude-Codex Deliberation

### Agreements

- Multi-vendor directory scoping under `wiki/{vendor}/` is the right long-term architecture
- Schema extensions must land before any content migration to enable validation at each step
- Absolute machine paths must never appear in committed files
- A deterministic dry-run migration log is essential for auditability
- `source_refs` and `sources` serve distinct purposes and can coexist — `sources` for page IDs, `source_refs` for external repo file references
- Probes are source evidence (not synthesized wiki knowledge) and belong in `sources/experience/`
- PROVENANCE.yaml with sha256 per file is mandatory for all migrated artifact bundles
- Acceptance criteria for validation, cross-reference resolution, duplicate IDs, and body-link validation are appropriate
- Migration order must be schema/tooling first, then structural move, then content import

### Resolved Disagreements

- **Migration order**: Claude initially proposed path migration first; Codex argued schema/tool readiness must precede the mass move. **Resolution**: Adopted Codex's position — M1 (Schema/Tooling) must complete before M2 (Path Migration). This ensures validation catches issues immediately after each structural change.

- **AC-2 body-hash semantics**: Claude proposed raw body-hash comparison; Codex noted that body links are rewritten during migration so raw hashes would always fail. **Resolution**: Adopted Codex's "normalized body hash" — comparison canonicalizes declared path/link rewrites before hashing to confirm only allowed changes occurred.

- **AC-8 deletion count**: Claude initially hardcoded "193 legacy experience files"; Codex argued the count should be derived from the generated deletion manifest, not hardcoded. **Resolution**: Adopted — deletion ledger determines the count. If inventory changes, the criterion adapts.

- **PROVENANCE upstream_path**: Claude initially allowed machine-local absolute paths in upstream_path; Codex insisted upstream_path must be repo-relative. **Resolution**: Adopted — upstream_path uses the original kb-mvp repo-relative path (e.g., `80-experience/hw-probes/tma-ptx/artifacts/tma_hello.cu`), not machine absolute paths.

- **Seven new page types (schema sprawl concern)**: Codex raised concern about excessive types without justified required fields. **Resolution**: Accepted the 7 types as specified in the draft, noting that validation tiers (Strict/Standard/Light) limit the overhead — light types (pitfall, operator-routing, code-walkthrough) have minimal required fields.

- **source_refs.path semantics**: Codex required precise definition. **Resolution**: For local sources (experience artifacts): repo-relative from KernelWiki root. For external sources: path relative to the external source root identified by source_id. `commit` field must be a full SHA or immutable tag, not a branch name.

- **source/source_refs cross-contamination**: Codex required validators reject wrong types in wrong fields. **Resolution**: Adopted — `sources` only accepts wiki/source page IDs, `source_refs` only accepts MANIFEST-backed source_id references. Validator rejects page IDs in source_refs and source_ids in sources.

### Convergence Status

- Final Status: `converged`
- Convergence rounds: 2
- All REQUIRED_CHANGES from both rounds addressed
- Remaining items are user decisions (all resolved in Phase 6)

## Pending User Decisions

- DEC-1: Corpus storage strategy
  - Claude Position: Git LFS recommended — balances accessibility with repo health
  - Codex Position: Must be decided before implementation; direct commit too risky for 204MB
  - Tradeoff Summary: Git LFS adds setup complexity but prevents permanent repo bloat
  - Decision Status: **Git LFS** (user confirmed)

- DEC-2: Path backward compatibility for wiki/* → wiki/nvidia/*
  - Claude Position: Breaking change acceptable for internal repo
  - Codex Position: Redirect/fallback needed unless explicitly breaking
  - Tradeoff Summary: Breaking change is simpler but requires updating all consumers; redirects add file count and maintenance
  - Decision Status: **Breaking change** (user confirmed)

- DEC-3: Probe/experience record location model
  - Claude Position: Sources-only (sources/experience/) for clean layer separation
  - Codex Position: Agreed that probes as evidence in source layer is better model
  - Tradeoff Summary: Sources-only is cleaner but requires wiki pages to cross-reference probes via source_refs
  - Decision Status: **Sources only** (user confirmed)

- DEC-4: source_refs as transitional or permanent field
  - Claude Position: Coexist permanently — different semantic purposes
  - Codex Position: Coexistence is acceptable if validators enforce correct usage
  - Tradeoff Summary: Permanent coexistence is simpler than eventual unification; requires clear documentation
  - Decision Status: **Coexist permanently** (user confirmed)

- DEC-5: Vendor scope for first merge
  - Claude Position: Schema ready with no stubs — cleanest approach
  - Codex Position: N/A — open question
  - Tradeoff Summary: Stub directories add clutter; schema readiness sufficient for future expansion
  - Decision Status: **Schema ready, no stubs** (user confirmed)

- DEC-6: Section 13 Chinese content handling
  - Claude Position: English canonical per repo policy
  - Codex Position: N/A — open question
  - Tradeoff Summary: Mixed language reflects the draft's original contribution; translation can happen later
  - Decision Status: **Keep mixed as-is** (user confirmed)

- DEC-7: AC-1 validation error tolerance
  - Claude Position: Directional — aim for 0, document any exceptions
  - Codex Position: N/A — open question
  - Tradeoff Summary: Hard 0 may block on edge cases; directional allows documented exceptions
  - Decision Status: **Directional (minimize to 0)** (user confirmed)

- DEC-8: Legacy experience deletion count
  - Claude Position: Derive from inventory
  - Codex Position: Count from manifest, not hardcoded
  - Tradeoff Summary: Inventory-derived count adapts if file set changes
  - Decision Status: **Derive from inventory** (user confirmed)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Key Technical Details

#### source_refs Schema
```yaml
source_refs:
  - source_id: string          # Must match a corpus/MANIFEST.yaml entry
    path: string               # Repo-relative path within the source
    anchor: string             # Optional: description of what this reference provides
    dependency_mode: string    # Optional: "reference-only" | "code-dependency"
    commit: string             # Optional: full SHA or immutable tag (overrides MANIFEST default_ref)
```

#### Validation Tiers for New Types
| Tier | Types | Required Fields |
|------|-------|-----------------|
| Strict | skill, api-definition, experience, algorithm | Full frontmatter including evidence_level, measured_on (if measured), artifacts, source with anchor |
| Standard | hardware, technique, kernel, pattern, language, migration | Existing KernelWiki standard (confidence, sources, reproducibility) |
| Light | pitfall, operator-routing, code-walkthrough | Minimal (title, type, vendor required; body free-form) |
| Immutable | source-pr, source-blog, source-doc, source-contest, source-experience | Strict schema, auto-generated |

#### Git LFS Configuration
```gitattributes
corpus/nvidia/cuda-official/** filter=lfs diff=lfs merge=lfs -text
corpus/nvidia/blogs/** filter=lfs diff=lfs merge=lfs -text
corpus/nvidia/whitepapers/** filter=lfs diff=lfs merge=lfs -text
```

#### Migration Script Output Format
```
# migration_inventory.tsv
input_path\toutput_path\tpage_id\tpage_type\tvendor\tstatus\treason
knowledge/30-skill/compute/warp-primitives/skill.md\twiki/nvidia/foundations/compute/warp-primitives.md\tskill-warp-primitives\tskill\tnvidia\tmigrated\t
knowledge/80-experience/hw-probes/tma-ptx/\tsources/experience/hw-probes/tma-ptx.md\texp-tma-ptx-hello\texperience\tnvidia\tmigrated\t
```

--- Original Design Draft Start ---

# KernelWiki Merge Plan: Unified Multi-Vendor GPU Knowledge Base

> **Author**: Claude Opus 4.6 + tongyu
> **Date**: 2026-05-29
> **Status**: Draft — pending review
> **Scope**: Merge kernel-kb-mvp + legacy-knowledge into KernelWiki; extend to multi-vendor

---

## 1. Current State of Three Knowledge Bases

### 1.1 KernelWiki (target, 2265 files)

| Layer | Content | Count |
|-------|---------|-------|
| `sources/prs/` | Merged PR summaries (vllm/sglang/flashinfer/cutlass/pytorch/deepgemm) | 2179 |
| `sources/blogs/` | Community blog summaries | 20 |
| `sources/docs/` | Official doc summaries | 11 |
| `sources/contests/` | Competition pages | 7 |
| `wiki/hardware/` | Hardware feature synthesis pages | 7 |
| `wiki/techniques/` | Optimization technique pages | 15 |
| `wiki/kernels/` | Kernel case study pages | 12 |
| `wiki/patterns/` | Problem-solution pattern pages | 7 |
| `wiki/languages/` | DSL guide pages | 4 |
| `wiki/migration/` | Architecture migration guides | 2 |
| `queries/` | Auto-generated cross-reference indices | 6 |
| `artifacts/` | Verbatim/extracted/derived code bundles | 89 |

**Strengths**: Rich PR coverage, robust validation (0 errors), cross-referencing, controlled vocabulary, version-claim tracking.
**Scope**: Blackwell (SM100) + Hopper (SM90) only. Synthesis-level knowledge (wiki layer).

### 1.2 kernel-kb-mvp (source, ~200 files)

| Layer | Content | Count |
|-------|---------|-------|
| `10-api-raw/` | CUDA Runtime API definitions | ~15 |
| `20-pattern/` | Operator decision-tree four-packs (cuda-core + tensor-core) | 8 ops |
| `30-skill/` | Single-kernel optimization skills | ~25 skills |
| `40-hardware-feature/` | HW feature characterization (tma, wgmma, tcgen05, ldmatrix, mma-sync) | 7 |
| `50-classical-algo/` | Classical algorithms (warp-specialization, persistent-kernel) | 2 |
| `60-code/` | Code-repository extraction (cutlass-cute, flash-attention-v3) | 8 topics |
| `70-reasoning/` | Frozen meta-skills (api-probing, hw-microbench, benchmark-protocol, etc.) | 5 |
| `80-experience/hw-probes/` | Microbench measurement records | ~20 |
| `80-experience/api-probes/` | API semantics probe records | ~5 |
| `agent/` | KB-gen agent (OpenAI path + Claude Code path) | 2 paths |
| `tools/` | lint, introspect, source-corpus CLI, probe-gap-scan | 6 tools |
| `tasks/` | Task YAML packets | ~32 |

**Strengths**: Agent-driven generation, strict evidence grounding, measured data with artifacts, GPU hardware introspection.
**Scope**: H200 (sm_90a) only. Atomic-level knowledge (skill/API/probe).

**CRITICAL: Meta-Repo Architecture**. kernel-kb-mvp is NOT a self-contained repo. Its source corpus has two tiers:

| Tier | Storage | Size | Examples |
|------|---------|------|----------|
| **In-git corpus** | Committed in `05-source-corpus/` | ~204MB, 1733 files | CUDA 13.2 docs (122MB/1037 files), Colfax blogs (48MB/208 files), GPU whitepapers (34MB/55 files), legacy-knowledge (429 files) |
| **External refs** | Local disk, referenced via `{{PLACEHOLDER}}` variables | Unbounded | `{{CUTLASS_REPO_REF}}`, `{{CUDA_SAMPLES_REPO_REF}}`, `{{FLASH_ATTENTION_REPO_REF}}`, `{{CLAUDE_RESEARCH_REF}}` |

The `MANIFEST.yaml` declares both tiers. External refs use placeholder syntax in `local_path`:
```yaml
# In-git: relative path
- source_id: cuda-official/toolkit-docs-13.2
  local_path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2

# External: placeholder variable
- source_id: source-code/cutlass
  local_path: "{{CUTLASS_REPO_REF}}"
```

To make the KB functional on a new machine, users must:
1. Clone the external repos to local disk
2. Create a localization config YAML mapping placeholders to local paths
3. Run `kb_localize.py render` to resolve all `{{VAR}}` references

The `source_corpus` CLI tools handle both tiers transparently via the `registry.py` → `kb_localize` resolution chain. **The merge plan must preserve this two-tier architecture.**

### 1.3 legacy-knowledge (delete experience, keep skills)

| Category | Content | Count |
|----------|---------|-------|
| `optimization/compute/` | Skill packs (warp-primitives, fast-math, tensor-core, etc.) | 8 |
| `optimization/memory/` | Skill packs (coalescing, bank-conflict, vectorized-access, etc.) | 11 |
| `optimization/latency/` | Skill packs (kernel-launch, cuda-graphs, stream-concurrency, etc.) | 9 |
| `optimization/synchronization/` | Skill packs (barrier, atomic-reduction, cooperative-groups, etc.) | 5 |
| `hardware/` | H200 specs, memory hierarchy, warp execution model | 4 |
| `reasoning-pattern/` | Optimization reasoning patterns | 5 |
| `advanced/` | Pipeline design, resource tradeoff, cross-component, branch-elimination | 4+ |
| `pattern/` | Operator patterns (elementwise, reduction, gemm, attention, etc.) | 8 |
| `api-reference/` | API index files | 4 |
| ~~`experience/`~~ | ~~193 measurement files~~ | ~~DELETE~~ |

**Decision**: Delete all 193 experience files (dependency artifacts lost). Keep 48 legacy skills + hardware + reasoning-pattern + advanced + pattern + api-reference.

---

## 2. Unified Directory Design

### 2.1 Core Principle: Topic-First + Type Tag

Abolish the numbered layer hierarchy (10/20/30/...). Replace with a **topic-first** directory organized by vendor, with `type` frontmatter distinguishing page kinds.

### 2.2 Target Structure

```
KernelWiki/                                   # Repository root
├── SKILL.md                                  # Skill entry point (updated)
├── CLAUDE.md                                 # Agent navigation (updated)
├── README.md                                 # (updated)
├── index.md                                  # Curated top-level nav (updated)
│
├── data/                                     # Config & vocabulary (extended)
│   ├── schemas.yaml                          # Page-type schemas (extended)
│   ├── tags.yaml                             # Controlled vocabulary (extended)
│   ├── aliases.yaml                          # Canonical aliases (extended)
│   ├── vendors.yaml                          # NEW: vendor registry
│   ├── version-claims.yaml                   # Version-sensitive claims
│   ├── tool-versions.yaml                    # Tool release snapshots
│   └── refresh-cutoff.yaml                   # Knowledge cutoff
│
├── references/                               # Companion docs (updated)
│   ├── primer.md
│   ├── schema.md
│   └── examples.md
│
├── scripts/                                  # Tools (extended)
│   ├── query.py                              # Extended: --vendor filter
│   ├── get_page.py                           # Extended: vendor-aware
│   ├── grep_wiki.py                          # Unchanged
│   ├── validate.py                           # Extended: new page types
│   ├── generate-indices.py                   # Extended: by-vendor.md
│   ├── generate-pr-pages.py                  # Unchanged
│   ├── lint_knowledge.py                     # NEW: migrated from kernel-kb-mvp
│   ├── source_corpus/                        # NEW: migrated from kernel-kb-mvp
│   │   ├── cli.py
│   │   ├── service.py
│   │   ├── registry.py
│   │   ├── reader.py
│   │   └── provenance.py
│   └── ...                                   # Other existing scripts
│
├── sources/                                  # Layer 1: Raw upstream data (unchanged)
│   ├── prs/{repo}/PR-*.md                    # 2179 PR pages
│   ├── blogs/*.md                            # 20 blog pages
│   ├── docs/*.md                             # 11 doc pages
│   └── contests/                             # 7 contest pages
│
├── wiki/                                     # Layer 2: ALL synthesized knowledge
│   ├── nvidia/                               # ━━━ Vendor: NVIDIA ━━━
│   │   ├── hardware/                         #   Hardware features
│   │   │   ├── tcgen05-mma.md                #     type: hardware (existing wiki)
│   │   │   ├── tcgen05-mma/                  #     Detail pages for tcgen05
│   │   │   │   ├── skill-tcgen05-ptx.md      #       type: skill (from kb-mvp 40-hw-feat)
│   │   │   │   └── pitfalls.md               #       type: pitfall
│   │   │   ├── tmem.md                       #     type: hardware (existing wiki)
│   │   │   ├── tma.md                        #     type: hardware (existing wiki)
│   │   │   ├── tma/
│   │   │   │   ├── skill-tma-ptx.md          #       type: skill (from kb-mvp 40-hw-feat)
│   │   │   │   ├── pitfalls.md
│   │   │   │   └── probe-tma-throughput.md   #       type: experience
│   │   │   ├── wgmma.md                      #     NEW overview (from kb-mvp 40-hw-feat)
│   │   │   ├── wgmma/
│   │   │   │   ├── skill-wgmma.md
│   │   │   │   ├── skill-wgmma-ptx.md
│   │   │   │   ├── pitfalls.md
│   │   │   │   └── probe-wgmma-ptx.md
│   │   │   └── ...
│   │   │
│   │   ├── techniques/                       #   Optimization techniques
│   │   │   ├── warp-specialization.md        #     type: technique (existing wiki)
│   │   │   ├── warp-specialization/
│   │   │   │   └── algo-warp-specialization.md  # type: algorithm (from kb-mvp 50-classical)
│   │   │   ├── persistent-kernels.md         #     type: technique (existing wiki)
│   │   │   ├── persistent-kernels/
│   │   │   │   └── algo-persistent-kernel.md
│   │   │   ├── swizzling.md
│   │   │   └── ...
│   │   │
│   │   ├── kernels/                          #   Kernel case studies
│   │   │   ├── flash-attention-4.md          #     type: kernel (existing wiki)
│   │   │   ├── flash-attention-4/
│   │   │   │   ├── code-flash-attention-v3.md  # type: code-walkthrough (from kb-mvp 60-code)
│   │   │   │   └── code-cutlass-fmha.md
│   │   │   ├── deepgemm.md
│   │   │   └── ...
│   │   │
│   │   ├── patterns/                         #   Problem-solution patterns
│   │   │   ├── low-sm-utilization.md         #     type: pattern (existing wiki)
│   │   │   └── ...
│   │   │
│   │   ├── languages/                        #   DSL guides
│   │   │   ├── cute-dsl.md                   #     type: language (existing wiki)
│   │   │   └── ...
│   │   │
│   │   ├── migration/                        #   Architecture migration
│   │   │   ├── wgmma-to-tcgen05.md           #     type: migration (existing wiki)
│   │   │   └── ...
│   │   │
│   │   ├── foundations/                       #   ━━━ NEW: foundational CUDA skills ━━━
│   │   │   ├── compute/                      #     (from kb-mvp 30-skill/compute/)
│   │   │   │   ├── warp-primitives.md        #       type: skill (overview role)
│   │   │   │   ├── warp-primitives/
│   │   │   │   │   ├── pitfalls.md           #       type: pitfall
│   │   │   │   │   ├── apis.md               #       type: api-reference
│   │   │   │   │   └── probe-warp-primitives.md  # type: experience
│   │   │   │   ├── fast-math.md
│   │   │   │   ├── fast-math/
│   │   │   │   │   ├── pitfalls.md
│   │   │   │   │   ├── apis.md
│   │   │   │   │   └── probe-fast-math.md
│   │   │   │   ├── compiler-hints.md
│   │   │   │   ├── half-precision-math.md
│   │   │   │   ├── occupancy-tuning.md
│   │   │   │   ├── ilp.md
│   │   │   │   ├── warp-divergence.md
│   │   │   │   └── branch-elimination.md
│   │   │   ├── memory/                       #     (from kb-mvp 30-skill/memory/)
│   │   │   │   ├── coalescing.md
│   │   │   │   ├── coalescing/
│   │   │   │   │   ├── pitfalls.md
│   │   │   │   │   ├── apis.md
│   │   │   │   │   └── probe-coalescing.md
│   │   │   │   ├── bank-conflict.md
│   │   │   │   ├── vectorized-access.md
│   │   │   │   ├── shared-memory-cache.md
│   │   │   │   ├── cache-load-hints.md
│   │   │   │   ├── register-pressure.md
│   │   │   │   ├── async-copy.md
│   │   │   │   ├── layout-transform.md
│   │   │   │   └── l2-access-policy.md
│   │   │   ├── sync/                         #     (from kb-mvp 30-skill/sync/)
│   │   │   │   ├── barrier-optimization.md
│   │   │   │   ├── atomic-reduction.md
│   │   │   │   └── memory-ordering.md
│   │   │   └── latency/                      #     (from legacy-knowledge + kb-mvp 90-system-level)
│   │   │       ├── kernel-launch-overhead.md
│   │   │       ├── cuda-graphs.md
│   │   │       └── stream-concurrency.md
│   │   │
│   │   ├── operator-routing/                 #   ━━━ NEW: operator decision trees ━━━
│   │   │   ├── elementwise/                  #     (from kb-mvp 20-pattern/)
│   │   │   │   ├── INDEX.md
│   │   │   │   ├── ROUTING.md
│   │   │   │   ├── library-fallback.md
│   │   │   │   └── TASK-PACKET.md
│   │   │   ├── reduction/
│   │   │   ├── normalization/
│   │   │   ├── gemm/
│   │   │   └── ...
│   │   │
│   │   ├── api-definitions/                  #   ━━━ NEW: API definitions ━━━
│   │   │   ├── runtime/                      #     (from kb-mvp 10-api-raw/)
│   │   │   │   ├── __shfl_sync.md
│   │   │   │   ├── __all_sync.md
│   │   │   │   └── ...
│   │   │   └── ptx/
│   │   │
│   │   ├── code-walkthroughs/                #   ━━━ NEW: code extraction ━━━
│   │   │   ├── cutlass-cute/                 #     (from kb-mvp 60-code/)
│   │   │   │   ├── gemm-aligned/
│   │   │   │   ├── gemm-tail/
│   │   │   │   ├── persistent-kernel/
│   │   │   │   └── ...
│   │   │   └── flash-attention-v3/
│   │   │
│   │   └── probes/                           #   ━━━ NEW: measurement records ━━━
│   │       ├── hw-probes/                    #     (from kb-mvp 80-experience/hw-probes/)
│   │       │   ├── coalescing/
│   │       │   ├── fast-math/
│   │       │   ├── tma-ptx/
│   │       │   ├── wgmma-ptx/
│   │       │   └── ...
│   │       └── api-probes/                   #     (from kb-mvp 80-experience/api-probes/)
│   │
│   ├── huawei/                               # ━━━ Vendor: Huawei (future) ━━━
│   │   ├── hardware/
│   │   │   └── ascend910b.md
│   │   ├── foundations/
│   │   │   └── ...
│   │   └── ...
│   │
│   └── biren/                                # ━━━ Vendor: Biren (future) ━━━
│       ├── hardware/
│       └── ...
│
├── queries/                                  # Layer 3: Auto-generated indices
│   ├── by-problem.md
│   ├── by-technique.md
│   ├── by-hardware-feature.md
│   ├── by-kernel-type.md
│   ├── by-language.md
│   ├── by-repo.md
│   └── by-vendor.md                          # NEW
│
├── candidates/                               # PR candidate ledgers (unchanged)
│
├── artifacts/                                # Code bundles (unchanged)
│
├── corpus/                                   # ━━━ NEW: upstream source corpus (two-tier) ━━━
│   ├── MANIFEST.yaml                         #   Source registry (from kb-mvp 05-source-corpus/)
│   ├── RETRIEVAL.md                          #   Retrieval contract for agents
│   ├── GLOBAL_VARIABLES.md                   #   Placeholder variable documentation
│   ├── localize.yaml.example                 #   Example localization config
│   ├── nvidia/                               #   ━━ Tier 1: In-git vendor corpus ━━
│   │   ├── cuda-official/                    #     CUDA 13.2 docs (122MB, 1037 files)
│   │   ├── blogs/                            #     Colfax blog archive (48MB, 208 files)
│   │   ├── whitepapers/                      #     GPU arch whitepapers (34MB, 55 files)
│   │   └── legacy-knowledge/                 #     Legacy skills (minus experience/)
│   ├── huawei/                               #   (future)
│   │   └── cann-docs/
│   ├── INDEX/                                #   Provenance indices
│   └── .external/                            #   ━━ Tier 2: Symlinks to local repos ━━
│       ├── README.md                         #     Setup instructions
│       ├── cutlass -> {{CUTLASS_REPO_REF}}   #     Resolved by localize.py
│       ├── cuda-samples -> {{CUDA_SAMPLES_REPO_REF}}
│       ├── flash-attention -> {{FLASH_ATTENTION_REPO_REF}}
│       └── .gitignore                        #     Ignore all symlink targets
│
├── reasoning/                                # ━━━ NEW: frozen meta-skills ━━━
│   ├── api-probing.md                        #   (from kb-mvp 70-reasoning/)
│   ├── hardware-microbench.md
│   ├── benchmark-protocol.md
│   ├── task-packet.md
│   ├── bottleneck-triage.md
│   └── optimization-reasoning/               #   (from legacy reasoning-pattern/)
│       ├── tradeoff-rebalancing.md
│       ├── constraint-relaxation.md
│       ├── idle-resource-detection.md
│       └── ...
│
├── agent/                                    # ━━━ NEW: KB-gen agent ━━━
│   ├── shared/                               #   (from kb-mvp agent/)
│   │   ├── system_prompt.py
│   │   ├── task_schema.py
│   │   ├── tools.py
│   │   └── config.py
│   ├── openai_path/
│   └── claude_code_path/
│
├── tasks/                                    # ━━━ NEW: task YAML packets ━━━
│   ├── build-*.yaml                          #   (from kb-mvp tasks/)
│   ├── probe-*.yaml
│   └── extract-*.yaml
│
└── templates/                                # ━━━ NEW: frontmatter templates ━━━
    └── frontmatter/                          #   (from kb-mvp knowledge/templates/)
        ├── skill.yaml
        ├── api-raw.yaml
        ├── experience.yaml
        ├── hardware-feature.yaml
        └── classical-algo.yaml
```

---

## 2.5 Source Corpus Localization System (Meta-Repo)

### 2.5.1 Problem

kernel-kb-mvp's knowledge pages reference upstream source code repos (CUTLASS, cuda-samples, flash-attention) that are too large to commit. These are referenced via `{{PLACEHOLDER}}` variables in MANIFEST.yaml and throughout `.md` files. Without resolving these placeholders, the KB-gen agent cannot search/read upstream sources, and many `source:` path references in skill/API pages are broken.

### 2.5.2 Two-Tier Corpus Design

> **Update (source/artifact migration detail):** External code repositories must support both normal online remote-Git resolution and offline local-repository overrides. The detailed migration contract for kp-mvp `source:` and `artifacts:` entries is merged in [Section 13](#13-kp-mvp-sourceartifact-migration-contract).

```
corpus/
├── nvidia/                          # Tier 1: IN-GIT (committed, ~204MB)
│   ├── cuda-official/               #   CUDA 13.2 full docs
│   ├── blogs/                       #   Colfax blog snapshots
│   └── whitepapers/                 #   Architecture whitepapers
│
├── .external/                       # Tier 2: LOCAL-ONLY (not in git)
│   ├── cutlass/                     #   → /path/to/NVIDIA/cutlass
│   ├── cuda-samples/                #   → /path/to/NVIDIA/cuda-samples
│   ├── flash-attention/             #   → /path/to/flash-attention
│   └── claude-research/             #   → /path/to/research notes
│
├── MANIFEST.yaml                    # Registry: declares BOTH tiers
├── GLOBAL_VARIABLES.md              # Documents all {{PLACEHOLDER}} vars
├── RETRIEVAL.md                     # Agent retrieval contract
└── localize.yaml.example            # Template for user's local config
```

### 2.5.3 MANIFEST.yaml: Unified Registry

```yaml
# Tier 1 entries: local_path is relative, no placeholder
- source_id: cuda-official/toolkit-docs-13.2
  local_path: nvidia/cuda-official/cuda-toolkit-documentation-13.2
  tier: in-git

# Tier 2 entries: local_path uses placeholder
- source_id: source-code/cutlass
  local_path: "{{CUTLASS_REPO_REF}}"
  upstream_url: https://github.com/NVIDIA/cutlass
  tier: external

# Future: Huawei tier 2
- source_id: source-code/cann-samples
  local_path: "{{CANN_SAMPLES_REF}}"
  upstream_url: https://gitee.com/ascend/samples
  tier: external
  platform: huawei
```

### 2.5.4 Localization Config (`localize.yaml`)

Each user creates `corpus/localize.yaml` (gitignored) from the example:

```yaml
# corpus/localize.yaml — user-specific, NOT committed
vars:
  CUDA_REPO_ROOT: /home/tongyu/workspace/cuda_repo

derived:
  CUTLASS_REPO_REF: "{{CUDA_REPO_ROOT}}/cutlass"
  CUDA_SAMPLES_REPO_REF: "{{CUDA_REPO_ROOT}}/cuda-samples"
  CCCL_REPO_REF: "{{CUDA_REPO_ROOT}}/cccl"
  FLASH_ATTENTION_REPO_REF: "{{CUDA_REPO_ROOT}}/flash-attention"
  CLAUDE_RESEARCH_REF: /path/to/claude_research

  # Future: Huawei
  # CANN_TOOLKIT_ROOT: /usr/local/Ascend/ascend-toolkit/latest
  # CANN_SAMPLES_REF: /home/tongyu/workspace/ascend-samples

env:
  inherit: true    # Also read from environment variables
```

### 2.5.5 Localization Tool (`scripts/localize.py`)

Migrated from `kb_localize.py`, adapted for new paths:

```bash
# First-time setup: create config from example
python3 scripts/localize.py init-config

# Check: verify all placeholders have values and paths exist
python3 scripts/localize.py check
# Output:
#   OK    CUTLASS_REPO_REF → /home/tongyu/workspace/cuda_repo/cutlass (exists)
#   OK    FLASH_ATTENTION_REPO_REF → .../flash-attention (exists, commit bbda031f)
#   WARN  CLAUDE_RESEARCH_REF → not configured (3 pages affected)

# Create .external/ symlinks (optional convenience)
python3 scripts/localize.py link
# Creates corpus/.external/cutlass → /home/tongyu/.../cutlass etc.

# Resolve: inline-render all {{VAR}} in a dry-run
python3 scripts/localize.py resolve-vars --dry-run

# Doctor: full health check (placeholders + path existence + MANIFEST consistency)
python3 scripts/localize.py doctor
```

### 2.5.6 Source Corpus CLI Integration

The `source_corpus` tools (search, read, list, resolve) handle both tiers transparently:

```python
# In scripts/source_corpus/registry.py
def resolve_corpus_path(path: str) -> Path:
    """Resolve a corpus path, handling both tiers.

    - Tier 1 (relative): corpus/nvidia/cuda-official/... → absolute path
    - Tier 2 (placeholder): {{CUTLASS_REPO_REF}}/... → localized path
    """
    if PLACEHOLDER_RE.search(path):
        return localizer.resolve(path)  # Uses localize.yaml
    return CORPUS_ROOT / path           # Relative to corpus/
```

Agent tools work identically regardless of tier:
```bash
# Search in-git corpus (tier 1)
python3 scripts/source_corpus_cli.py search "__shfl_xor_sync" --scope cuda-official

# Search external repo (tier 2) — requires localize.yaml configured
python3 scripts/source_corpus_cli.py search "wgmma" --scope source-code/cutlass

# Provenance tracing works across both tiers
python3 scripts/source_corpus_cli.py provenance-walk wiki/nvidia/hardware/tma/skill-tma-ptx.md
```

### 2.5.7 Git Strategy for Corpus

```
corpus/
├── .gitignore                  # Contains: localize.yaml, .external/
├── localize.yaml.example       # Committed: template for users
├── MANIFEST.yaml               # Committed: full registry
├── nvidia/cuda-official/       # Committed (122MB) — consider Git LFS
├── nvidia/blogs/               # Committed (48MB)
├── nvidia/whitepapers/         # Committed (34MB)
└── .external/                  # NOT committed (symlinks to local repos)
```

**Decision needed (D1)**: The in-git corpus is ~204MB. Options:
1. **Commit directly** — simplest, but bloats clone. Acceptable if repo is internal-only.
2. **Git LFS** — tracks large files efficiently. Best if repo will be shared.
3. **Git submodule** — `corpus/nvidia/` as separate repo. Most complex but cleanest separation.
4. **Gitignore + bootstrap script** — corpus not in git at all; `scripts/bootstrap_corpus.py` downloads/copies from a known location. Lightest repo, but requires setup step.

### 2.5.8 First-Time Setup Flow (Post-Merge)

```bash
# 1. Clone KernelWiki
git clone ... KernelWiki && cd KernelWiki

# 2. Install Python deps
pip install -r requirements.txt

# 3. Configure local paths (one-time)
python3 scripts/localize.py init-config
# Edit corpus/localize.yaml with your local paths

# 4. Verify setup
python3 scripts/localize.py doctor
# Should show: all tier-1 present, tier-2 resolved or warned

# 5. (Optional) Create convenience symlinks
python3 scripts/localize.py link

# 6. Smoke test
python3 scripts/query.py "warp specialization" --type skill
python3 scripts/source_corpus_cli.py search "tcgen05" --scope cuda-official
```

### 2.5.9 Multi-Vendor Localization Extension

When adding Huawei Ascend, the same system scales:

```yaml
# corpus/localize.yaml — extended for Huawei
vars:
  CUDA_REPO_ROOT: /home/tongyu/workspace/cuda_repo
  CANN_TOOLKIT_ROOT: /usr/local/Ascend/ascend-toolkit/latest

derived:
  CUTLASS_REPO_REF: "{{CUDA_REPO_ROOT}}/cutlass"
  CANN_SAMPLES_REF: /home/tongyu/workspace/ascend-samples
  MINDSPORE_REPO_REF: /home/tongyu/workspace/mindspore
```

MANIFEST.yaml gains Huawei entries:
```yaml
- source_id: huawei/cann-docs
  local_path: huawei/cann-docs           # tier: in-git (if committed)
  platform: huawei

- source_id: source-code/cann-samples
  local_path: "{{CANN_SAMPLES_REF}}"     # tier: external
  platform: huawei
```

---

## 3. Page Type Taxonomy (Unified)

### 3.1 Existing Types (from KernelWiki, unchanged)

| Type | id prefix | Location | Schema |
|------|-----------|----------|--------|
| `hardware` | `hw-` | `wiki/{vendor}/hardware/*.md` | wiki-hardware |
| `technique` | `technique-` | `wiki/{vendor}/techniques/*.md` | wiki-technique |
| `kernel` | `kernel-` | `wiki/{vendor}/kernels/*.md` | wiki-kernel |
| `pattern` | `pattern-` | `wiki/{vendor}/patterns/*.md` | wiki-pattern |
| `language` | `lang-` | `wiki/{vendor}/languages/*.md` | wiki-language |
| `migration` | `migration-` | `wiki/{vendor}/migration/*.md` | wiki-migration |
| `source-pr` | `pr-` | `sources/prs/` | source-pr |
| `source-blog` | `blog-` | `sources/blogs/` | source-blog |
| `source-doc` | `doc-` | `sources/docs/` | source-doc |
| `source-contest` | `contest-` | `sources/contests/` | source-contest |

### 3.2 New Types (from kernel-kb-mvp, to be added to schemas.yaml)

| Type | id prefix | Location | Origin | Schema |
|------|-----------|----------|--------|--------|
| `skill` | `skill-` | `wiki/{vendor}/foundations/**/*.md` | kb-mvp `30-skill/` | wiki-skill |
| `pitfall` | `pitfall-` | `wiki/{vendor}/**/{topic}/pitfalls.md` | kb-mvp `pitfalls.md` | wiki-pitfall |
| `api-definition` | `api-` | `wiki/{vendor}/api-definitions/**/*.md` | kb-mvp `10-api-raw/` | wiki-api-definition |
| `operator-routing` | `routing-` | `wiki/{vendor}/operator-routing/*/*.md` | kb-mvp `20-pattern/` | wiki-operator-routing |
| `algorithm` | `algo-` | `wiki/{vendor}/techniques/*/*.md` | kb-mvp `50-classical-algo/` | wiki-algorithm |
| `code-walkthrough` | `code-` | `wiki/{vendor}/code-walkthroughs/**/*.md` | kb-mvp `60-code/` | wiki-code-walkthrough |
| `experience` | `exp-` | `wiki/{vendor}/probes/**/*.md` | kb-mvp `80-experience/` | wiki-experience |

### 3.3 Validation Tiers

| Tier | Types | Validation Level |
|------|-------|-----------------|
| **Strict** | `skill`, `api-definition`, `experience`, `algorithm` | Full frontmatter (evidence_level, measured_on, artifacts, source with anchor) |
| **Standard** | `hardware`, `technique`, `kernel`, `pattern`, `language`, `migration` | KernelWiki standard (confidence, sources, reproducibility) |
| **Light** | `pitfall`, `operator-routing`, `code-walkthrough` | Minimal (title, type, vendor required; body free-form) |
| **Immutable** | `source-pr`, `source-blog`, `source-doc`, `source-contest` | Strict schema, auto-generated |

---

## 4. Multi-Vendor Design

### 4.1 Vendor Registry (`data/vendors.yaml`)

```yaml
vendors:
  - id: nvidia
    display_name: NVIDIA
    architectures: [sm100, sm100a, sm90, sm90a, sm120]
    compute_stack: [cuda, ptx, cute-dsl, triton, cutlass]
    status: active

  - id: huawei
    display_name: Huawei Ascend
    architectures: [ascend910b, ascend910c, ascend310p]
    compute_stack: [cann, mindspore, ascendc]
    status: planned

  - id: biren
    display_name: Biren
    architectures: [br100, br104]
    compute_stack: [supa, birensdk]
    status: planned
```

### 4.2 Frontmatter Extension

All page types gain an **optional** `vendor` field:

```yaml
vendor: nvidia          # Required for pages under wiki/{vendor}/
                        # Inferred from path if omitted
                        # Validated: must match path prefix
```

For pages that are **vendor-agnostic** (e.g., general optimization reasoning, cross-vendor patterns):

```yaml
vendor: generic         # Explicit opt-out of vendor scoping
```

### 4.3 Tags Extension (`data/tags.yaml`)

```yaml
# Add to existing tags.yaml:
vendors:
  - nvidia
  - huawei
  - biren
  - generic

# Huawei-specific tags (added when huawei/ is populated):
huawei_architectures:
  - ascend910b
  - ascend910c
  - ascend310p

huawei_hardware_features:
  - cube-unit           # Ascend tensor core equivalent
  - aiv                 # AI Vector unit
  - aic                 # AI Core
  - hccl                # Huawei Collective Communication Library

# Biren-specific tags (added when biren/ is populated):
biren_architectures:
  - br100
  - br104
```

### 4.4 Query Extension

```bash
# Existing queries still work as before:
python3 scripts/query.py "warp specialization"
python3 scripts/query.py --tag tcgen05 --type kernel

# New vendor filter:
python3 scripts/query.py --vendor nvidia "memory coalescing"
python3 scripts/query.py --vendor huawei --type skill
python3 scripts/query.py --vendor all --type hardware    # Cross-vendor comparison

# New index:
# queries/by-vendor.md — all vendors with page counts per type
```

### 4.5 Cross-Vendor References

Pages can reference across vendors via standard `related:` or `sources:` fields:

```yaml
# In wiki/huawei/techniques/tensor-compute.md
related:
  - hw-tcgen05-mma     # NVIDIA equivalent
  - skill-wgmma        # NVIDIA wgmma skill for comparison
```

---

## 5. Schema Definitions for New Page Types

### 5.1 wiki-skill (replaces kb-mvp `skill.yaml`)

```yaml
wiki-skill:
  required:
    - id
    - title
    - type
    - vendor
    - tags
    - evidence_level         # spec | measured | inferred | anecdotal
    - applies_to             # operator classes this skill applies to
    - source                 # [{path, anchor, excerpt?}]
  optional:
    - architectures
    - confidence
    - requires_sm
    - requires_features
    - measured_on            # Required if evidence_level == measured
    - cuda_version_tested
    - artifacts              # {code, build, introspection, profile}
    - related
    - related_apis
    - related_skills
    - reproducibility
    - version_sensitive
  constraints:
    type: skill
    id_prefix: skill-
    measured_requires_artifacts: true  # evidence_level=measured → artifacts required
```

### 5.2 wiki-api-definition (replaces kb-mvp `api-raw.yaml`)

```yaml
wiki-api-definition:
  required:
    - id
    - title
    - type
    - vendor
    - func_name
    - namespace              # runtime | driver | cublas | cub | ptx | math
    - header
    - signature
    - source
  optional:
    - since_cuda
    - status
    - has_end_to_end_example
    - parameters
    - return
    - preconditions
    - error_modes
    - probed_by              # link to experience page
  constraints:
    type: api-definition
    id_prefix: api-
```

### 5.3 wiki-experience (replaces kb-mvp `experience.yaml`)

```yaml
wiki-experience:
  required:
    - id
    - title
    - type
    - vendor
    - probe_slug
    - evidence_level
    - measured_on             # {device, sm, cuda_runtime, driver}
    - source
  optional:
    - api
    - namespace
    - status
    - kind
    - trigger
    - clock_policy
    - artifacts
    - conclusions
    - referenced_in_corpus
    - back_filled_into
    - open_questions
  constraints:
    type: experience
    id_prefix: exp-
```

### 5.4 wiki-operator-routing (new, lightweight)

```yaml
wiki-operator-routing:
  required:
    - id
    - title
    - type
    - vendor
    - operator               # elementwise | reduction | gemm | normalization | ...
  optional:
    - tags
    - related
    - sources
  constraints:
    type: operator-routing
    id_prefix: routing-
```

### 5.5 wiki-algorithm (replaces kb-mvp `classical-algo.yaml`)

```yaml
wiki-algorithm:
  required:
    - id
    - title
    - type
    - vendor
    - tags
    - evidence_level
    - source
  optional:
    - architectures
    - applies_to
    - requires_sm
    - measured_on
    - artifacts
    - related
    - reproducibility
    - confidence
  constraints:
    type: algorithm
    id_prefix: algo-
```

### 5.6 wiki-code-walkthrough (new, light validation)

```yaml
wiki-code-walkthrough:
  required:
    - id
    - title
    - type
    - vendor
    - upstream_repo           # e.g., NVIDIA/cutlass
  optional:
    - tags
    - related
    - source
    - artifacts
  constraints:
    type: code-walkthrough
    id_prefix: code-
```

---

## 6. File Migration Map

### 6.1 kernel-kb-mvp → KernelWiki

| Source (kernel-kb-mvp) | Target (KernelWiki) | Action |
|------------------------|---------------------|--------|
| `knowledge/10-api-raw/runtime/*.md` | `wiki/nvidia/api-definitions/runtime/*.md` | Copy + rewrite frontmatter (add `type: api-definition`, `vendor: nvidia`, `id: api-*`) |
| `knowledge/20-pattern/cuda-core/{op}/` | `wiki/nvidia/operator-routing/{op}/` | Copy + add `type: operator-routing`, `vendor: nvidia` |
| `knowledge/20-pattern/tensor-core/{op}/` | `wiki/nvidia/operator-routing/{op}/` | Copy + merge with cuda-core if same op |
| `knowledge/30-skill/{family}/{name}/skill.md` | `wiki/nvidia/foundations/{family}/{name}.md` | Copy + rewrite frontmatter (type→skill, add vendor) |
| `knowledge/30-skill/{family}/{name}/pitfalls.md` | `wiki/nvidia/foundations/{family}/{name}/pitfalls.md` | Copy + add `type: pitfall` |
| `knowledge/30-skill/{family}/{name}/apis.md` | `wiki/nvidia/foundations/{family}/{name}/apis.md` | Copy + add `type: api-reference` |
| `knowledge/40-hardware-feature/{feat}/skill.md` | `wiki/nvidia/hardware/{mapped-topic}/skill-{feat}.md` | Copy into matching hardware topic dir |
| `knowledge/40-hardware-feature/{feat}/pitfalls.md` | `wiki/nvidia/hardware/{mapped-topic}/pitfalls-{feat}.md` | Same |
| `knowledge/50-classical-algo/{algo}/skill.md` | `wiki/nvidia/techniques/{algo}/algo-{algo}.md` | Copy into matching technique topic dir |
| `knowledge/60-code/{repo}/{topic}/` | `wiki/nvidia/code-walkthroughs/{repo}/{topic}/` | Copy + add `type: code-walkthrough` |
| `knowledge/70-reasoning/*.md` | `reasoning/*.md` | Copy (frozen, read-only) |
| `knowledge/80-experience/hw-probes/{slug}/` | `wiki/nvidia/probes/hw-probes/{slug}/` | Copy + rewrite frontmatter |
| `knowledge/80-experience/api-probes/` | `wiki/nvidia/probes/api-probes/` | Copy + rewrite frontmatter |
| `knowledge/00-foundation/` | `wiki/nvidia/hardware/foundation/` | Copy (reference data) |
| `knowledge/templates/frontmatter/` | `templates/frontmatter/` | Copy (for agent-gen compatibility) |
| `knowledge/05-source-corpus/cuda-official/` | `corpus/nvidia/cuda-official/` | Copy tier-1 in-git corpus (122MB) |
| `knowledge/05-source-corpus/blogs/` | `corpus/nvidia/blogs/` | Copy tier-1 in-git corpus (48MB) |
| `knowledge/05-source-corpus/whitepapers/` | `corpus/nvidia/whitepapers/` | Copy tier-1 in-git corpus (34MB) |
| `knowledge/05-source-corpus/legacy-knowledge/` | `corpus/nvidia/legacy-knowledge/` | Copy minus `experience/` (193 files deleted) |
| `knowledge/05-source-corpus/MANIFEST.yaml` | `corpus/MANIFEST.yaml` | Copy + rewrite `local_path` prefixes (`05-source-corpus/` → `nvidia/`) |
| `knowledge/05-source-corpus/RETRIEVAL.md` | `corpus/RETRIEVAL.md` | Copy + update path references |
| `knowledge/05-source-corpus/INDEX/` | `corpus/INDEX/` | Copy provenance indices |
| `knowledge/GLOBAL_VARIABLES.md` | `corpus/GLOBAL_VARIABLES.md` | Copy + extend for multi-vendor |
| *Tier-2 external refs* | `corpus/.external/` (gitignored) | NOT copied. User runs `scripts/localize.py init-config` to configure local paths, then `scripts/localize.py link` to create symlinks |
| `tools/kb_localize.py` | `scripts/localize.py` | Copy + adapt: config path → `corpus/localize.yaml`, resolve root → `corpus/` |
| `tools/source_corpus/` | `scripts/source_corpus/` | Copy + adapt: corpus root → `corpus/`, placeholder resolution via `localize.py` |
| `knowledge/AGENTS.md` | `reasoning/AGENTS.md` | Copy + update paths |
| `agent/` | `agent/` | Copy |
| `tools/lint_knowledge.py` | `scripts/lint_knowledge.py` | Copy + adapt paths |
| `tools/kp_introspect.py` | `scripts/kp_introspect.py` | Copy (optional, for GPU introspection) |
| `tasks/` | `tasks/` | Copy + update `target_path` and `upstream_scope` references |

### 6.2 legacy-knowledge → KernelWiki

| Source (legacy-knowledge) | Target (KernelWiki) | Action |
|---------------------------|---------------------|--------|
| `optimization/{cat}/{name}/skill.md` | Already in kb-mvp `30-skill/` | **Skip** — kb-mvp version supersedes |
| `hardware/*.md` | `wiki/nvidia/hardware/foundation/` | Copy if not already covered |
| `reasoning-pattern/*.md` | `reasoning/optimization-reasoning/` | Copy |
| `advanced/` | `wiki/nvidia/foundations/advanced/` | Copy (pipeline-design, resource-tradeoff, etc.) |
| `pattern/{op}/` | Superseded by kb-mvp `20-pattern/` | **Skip** |
| `api-reference/*.md` | Superseded by kb-mvp `10-api-raw/` | **Skip** |
| `experience/` (193 files) | — | **DELETE** |

### 6.3 Hardware Feature Topic Mapping

Map kb-mvp `40-hardware-feature/` to existing KernelWiki `wiki/hardware/` topics:

| kb-mvp source | KernelWiki target | Notes |
|---------------|-------------------|-------|
| `40-hardware-feature/tma/` | `wiki/nvidia/hardware/tma/` | Nest under existing `tma.md` |
| `40-hardware-feature/tma-ptx/` | `wiki/nvidia/hardware/tma/` | Merge into same topic |
| `40-hardware-feature/wgmma/` | `wiki/nvidia/hardware/wgmma.md` + `wgmma/` | New overview + detail dir |
| `40-hardware-feature/wgmma-ptx/` | `wiki/nvidia/hardware/wgmma/` | Merge into wgmma topic |
| `40-hardware-feature/tcgen05-ptx/` | `wiki/nvidia/hardware/tcgen05-mma/` | Nest under existing |
| `40-hardware-feature/mma-sync-ptx/` | `wiki/nvidia/hardware/mma-sync/` | New topic |
| `40-hardware-feature/ldmatrix-ptx/` | `wiki/nvidia/hardware/ldmatrix/` | New topic |

---

## 7. SKILL.md Update

```yaml
---
name: KernelWiki
description: >-
  Use when the user asks about optimizing NVIDIA GPU kernels (Blackwell SM100,
  Hopper SM90, or general CUDA), foundational CUDA optimization skills
  (coalescing, warp primitives, shared memory, etc.), hardware feature
  characterization (tcgen05, TMA, wgmma), operator routing (elementwise,
  reduction, GEMM), API definitions, or wants PR references from
  CUTLASS/SGLang/vLLM/FlashInfer/PyTorch. Future: Huawei Ascend NPU, Biren GPU.
  Do NOT use for host-side framework integration or distributed systems
  (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--vendor nvidia --tag foo --type skill] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---
```

Key changes:
- Remove "Blackwell/Hopper-specific" constraint → general NVIDIA GPU
- Add "foundational CUDA optimization skills" to trigger list
- Add "operator routing", "API definitions" to trigger list
- Add "Future: Huawei Ascend NPU, Biren GPU"
- Add `--vendor` to argument-hint

---

## 8. Implementation Phases

### Phase 1: Structural Migration (Week 1)

1. Create `wiki/nvidia/` and move existing `wiki/{hardware,techniques,kernels,patterns,languages,migration}/` under it
2. Create `wiki/nvidia/foundations/`, `wiki/nvidia/operator-routing/`, `wiki/nvidia/api-definitions/`, `wiki/nvidia/code-walkthroughs/`, `wiki/nvidia/probes/`
3. **Set up corpus/ two-tier structure:**
   - Copy tier-1 in-git corpus: `05-source-corpus/{cuda-official,blogs,whitepapers}` → `corpus/nvidia/`
   - Copy `MANIFEST.yaml`, `RETRIEVAL.md`, `GLOBAL_VARIABLES.md`, `INDEX/` → `corpus/`
   - Rewrite MANIFEST.yaml `local_path` prefixes (`05-source-corpus/X` → `nvidia/X`)
   - Create `corpus/localize.yaml.example` from GLOBAL_VARIABLES.md
   - Create `corpus/.external/` with `.gitignore` (ignore all contents)
   - Create `corpus/.external/README.md` with setup instructions
4. **Set up localization tooling:**
   - Migrate `kb_localize.py` → `scripts/localize.py` (adapt config path to `corpus/localize.yaml`)
   - Migrate `tools/source_corpus/` → `scripts/source_corpus/` (adapt corpus root)
   - Test: `scripts/localize.py check` passes with a sample `localize.yaml`
5. Copy `reasoning/`, `templates/`, `agent/`, `tasks/` from kernel-kb-mvp
6. Delete legacy-knowledge `experience/` (193 files) from corpus copy
7. Copy legacy-knowledge `reasoning-pattern/` → `reasoning/optimization-reasoning/`

### Phase 2: Content Migration (Week 2)

1. Run migration script to copy + rewrite frontmatter for all kb-mvp pages:
   - `30-skill/` → `wiki/nvidia/foundations/`
   - `40-hardware-feature/` → `wiki/nvidia/hardware/{topic}/`
   - `50-classical-algo/` → `wiki/nvidia/techniques/{topic}/`
   - `60-code/` → `wiki/nvidia/code-walkthroughs/`
   - `10-api-raw/` → `wiki/nvidia/api-definitions/`
   - `20-pattern/` → `wiki/nvidia/operator-routing/`
   - `80-experience/` → `wiki/nvidia/probes/`
2. **Rewrite all `source:` path references** in migrated pages:
   - `05-source-corpus/cuda-official/...` → `corpus/nvidia/cuda-official/...` (tier-1)
   - `{{CUTLASS_REPO_REF}}/...` → keep as `{{CUTLASS_REPO_REF}}/...` (tier-2, resolved at runtime)
   - `knowledge/30-skill/...` → `wiki/nvidia/foundations/...` (internal cross-ref)
3. Update all internal cross-references (related links, deep_refs)

### Phase 3: Schema & Tooling Update (Week 3)

1. Extend `data/schemas.yaml` with new page types (Section 5)
2. Extend `data/tags.yaml` with vendor dimension + new tags
3. Create `data/vendors.yaml`
4. Update `scripts/query.py` — add `--vendor` filter
5. Update `scripts/validate.py` — validate new page types + tier-2 path warnings
6. Update `scripts/generate-indices.py` — add `queries/by-vendor.md`
7. Integrate `scripts/lint_knowledge.py` with new directory layout
8. Integrate `scripts/source_corpus/` — verify search works across both tiers
9. Run full validation: target 0 errors (tier-2 paths emit warnings, not errors, if localize.yaml missing)

### Phase 4: Agent & Task Update (Week 4)

1. Update `agent/shared/config.py` — point to new directory layout
2. Update `agent/shared/system_prompt.py` — reference new paths
3. Update `agent/shared/tools.py` — adapt source_corpus paths
4. Update all `tasks/*.yaml` — rewrite `target_path` references
5. Update `reasoning/AGENTS.md` — new layer table, path references
6. Smoke-test: run one agent task end-to-end in new structure

### Phase 5: Index & Documentation (Week 4)

1. Regenerate all `queries/` indices
2. Update `index.md` — add foundations, operator-routing, api-definitions sections
3. Update `SKILL.md` — expanded description
4. Update `CLAUDE.md` — new schema reference
5. Update `references/primer.md` — add foundations + vendor sections
6. Update `README.md` — new install + content inventory

---

## 9. Migration Script Specification

Create `scripts/migrate_kb_mvp.py`:

```
Usage: python3 scripts/migrate_kb_mvp.py --source /path/to/kernel-kb-mvp --dry-run

Actions:
  1. Scan source knowledge/ tree
  2. For each .md file, determine target path via mapping table (Section 6)
  3. Parse YAML frontmatter
  4. Rewrite frontmatter:
     - Add `type: <mapped-type>`
     - Add `vendor: nvidia`
     - Add `id: <prefix>-<slug>`  (generate from filename + parent)
     - Map `evidence_level` → keep as-is (compatible)
     - Map `source:` paths → rewrite to new corpus/ paths
     - Map `artifacts:` paths → rewrite to new relative paths
  5. Copy .cu / .sh / .json artifacts alongside
  6. Write migration log (TSV): source_path, target_path, action, status

Flags:
  --dry-run       Show plan without writing
  --source PATH   kernel-kb-mvp root
  --legacy PATH   legacy-knowledge root (optional, for reasoning-pattern)
  --force         Overwrite existing target files
```

---

## 10. Validation Acceptance Criteria

| # | Criterion | Tool |
|---|-----------|------|
| AC-1 | 0 validation errors after migration | `scripts/validate.py` |
| AC-2 | All existing 2265 source/wiki pages unchanged or only path-adjusted | `git diff --stat` |
| AC-3 | All kb-mvp content present in new location | migration log check |
| AC-4 | `--vendor nvidia` query returns all NVIDIA pages | `scripts/query.py` |
| AC-5 | New page types validated against new schemas | `scripts/validate.py` |
| AC-6 | All cross-references resolve (no broken `related:` / `sources:`) | `scripts/validate.py` |
| AC-7 | Agent smoke-test: one task YAML runs to completion | manual |
| AC-8 | Legacy experience/ files deleted | `find ... -count` |
| AC-9 | `queries/by-vendor.md` auto-generated correctly | `scripts/generate-indices.py` |
| AC-10 | SKILL.md description no longer says "Blackwell/Hopper-specific" | manual |
| AC-11 | `scripts/localize.py check` passes with example config | `scripts/localize.py` |
| AC-12 | Tier-1 corpus search works without localize.yaml | `scripts/source_corpus_cli.py search "shfl" --scope cuda-official` |
| AC-13 | Tier-2 corpus search works WITH localize.yaml configured | `scripts/source_corpus_cli.py search "wgmma" --scope source-code/cutlass` |
| AC-14 | MANIFEST.yaml `local_path` entries all resolve (tier-1 to files, tier-2 to placeholders) | `scripts/localize.py doctor` |
| AC-15 | `corpus/.external/` is gitignored; `corpus/localize.yaml` is gitignored | `git status` |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Path rewrite breaks internal links | Broken cross-references | Migration script rewrites all `source:` and `related:` paths; validate.py catches broken links |
| Duplicate content (legacy skill vs kb-mvp skill) | Confusion, bloat | kb-mvp version always wins; legacy-only skills copied with `confidence: inferred` |
| New page types confuse existing KernelWiki consumers | Agent queries return unexpected results | `--type` filter still works; new types are additive |
| Vendor dimension complicates queries | Users must specify `--vendor` | Default to `nvidia` when omitted (backward compat); `--vendor all` for cross-vendor |
| Agent tools (source_corpus CLI) path assumptions | Agent fails after migration | Phase 4 explicitly updates all config paths + smoke-test |
| Large diff obscures real changes in PR review | Hard to review | Phase 1 (structure) separate commit from Phase 2 (content); each phase one PR |
| Tier-2 external repos not configured | Agent search fails for CUTLASS/flash-attention sources | `localize.py doctor` warns clearly; tier-1 queries (cuda-official, blogs) always work without config; degrade gracefully |
| In-git corpus bloats repo (204MB) | Slow clone, large git history | Decision D1: evaluate Git LFS vs submodule vs bootstrap script before Phase 1 |
| Placeholder `{{VAR}}` syntax conflicts with Jinja/other templating | Rendering errors | Keep `{{UPPERCASE_SNAKE}}` convention (no lowercase); localize.py is the only resolver |

---

## 12. Post-Merge: Adding a New Vendor

When adding Huawei Ascend support:

1. Add vendor entry to `data/vendors.yaml`
2. Add architecture tags to `data/tags.yaml` under `huawei_architectures`
3. Create `wiki/huawei/` with at minimum `hardware/` and `foundations/`
4. Create `corpus/huawei/` with CANN docs and upstream references
5. Add Huawei-specific task YAML templates under `tasks/`
6. Update `reasoning/AGENTS.md` to add Ascend constraints (target hardware, compile command, etc.)
7. Agent can now generate pages under `wiki/huawei/` using same pipeline
8. Run `scripts/generate-indices.py` to add Huawei entries to `queries/by-vendor.md`
9. Update SKILL.md description to include Huawei

No structural changes required — the vendor dimension is a first-class directory partition.

---

## 13. kp-mvp Source/Artifact Migration Contract

### 目标

将 kp-mvp 中形如：

```yaml
source:
  - path: /data1/.../cutlass/examples/88_hopper_fmha/88_hopper_fmha.cu
    anchor: Hopper FMHA example — used as algorithmic reference only, not as code dependency
  - path: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu
    anchor: wgmma PTX inline-asm pattern reused for attention Q@K^T and P@V tiles
artifacts:
  code: 80-experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu
  build: 80-experience/api-probes/attention/artifacts/build.sh
  introspection: 80-experience/api-probes/attention/artifacts/device.json
  profile: 80-experience/api-probes/attention/artifacts/profiles/tma-wgmma-ncu.csv
```

迁移成 KernelWiki 的两层模型：

1. **外部代码仓库引用**：进入 `corpus/` 的逻辑 source registry，支持远程 Git 和本地仓库两种解析方式。
2. **kp-mvp 自产实验资产**：进入 `artifacts/experience/`，用 `PROVENANCE.yaml` 记录来源、文件角色、sha256 和原始 kp-mvp 路径。

知识页只引用稳定 ID、repo-relative path 和 `artifact_dir`，不提交机器相关绝对路径。

---

### 设计原则

#### 1. 不提交绝对路径

禁止在迁移后的 source/wiki 页面中保留：

```text
/data1/tongyu/workspace/cuda_repo/cutlass/...
```

应改为：

```yaml
source_id: source-code/cutlass
path: examples/88_hopper_fmha/88_hopper_fmha.cu
```

实际机器如何找到 `source-code/cutlass`，由 `corpus/localize.yaml` 或远程 Git resolver 决定。

#### 2. 默认远程 Git，离线机器本地覆盖

大多数联网机器直接 clone/fetch 远程仓库；无法联网的机器使用本地已存在仓库。

解析优先级：

1. `corpus/localize.yaml` 中配置了 `local_path`：使用本地仓库。
2. 否则使用 `corpus/MANIFEST.yaml` 中的 `remote.url` clone 到 `corpus/.external/`。
3. 如果无网络且没有 local override：明确报错，提示配置本地路径。

#### 3. source 和 artifact 分离

- 外部 repo 文件，例如 CUTLASS `88_hopper_fmha.cu`：只作为 corpus source reference，不复制进 KernelWiki。
- kp-mvp 自产代码、build 脚本、device introspection、profile：复制进 `artifacts/experience/...`。
- wiki/source 页面只写 `artifact_dir`，细粒度文件清单放进 bundle 的 `PROVENANCE.yaml`。

---

### 新增目录结构

建议新增：

```text
corpus/
├── MANIFEST.yaml
├── README.md
├── localize.yaml.example
├── localize.yaml              # git-ignored, 每台机器自己配置
└── .external/                 # git-ignored, 默认远程 clone 位置

artifacts/experience/
├── hw-probes/
│   ├── tma-ptx/
│   │   ├── tma_hello.cu
│   │   ├── build.sh
│   │   ├── device.json
│   │   ├── profiles/...
│   │   └── PROVENANCE.yaml
│   └── wgmma-ptx/
│       ├── wgmma_hello.cu
│       ├── build.sh
│       ├── device.json
│       ├── profiles/...
│       └── PROVENANCE.yaml
└── api-probes/
    └── attention-tma-wgmma/
        ├── flash_attn_tma_wgmma.cu
        ├── build.sh
        ├── device.json
        ├── profiles/tma-wgmma-ncu.csv
        └── PROVENANCE.yaml

sources/experience/
├── hw-probes/
│   ├── tma-ptx.md
│   └── wgmma-ptx.md
└── api-probes/
    └── attention-tma-wgmma.md
```

---

### corpus 远程/本地双模式

#### `corpus/MANIFEST.yaml`

```yaml
sources:
  - source_id: source-code/cutlass
    kind: git-repo
    description: NVIDIA CUTLASS repository used for CuTe/CUTLASS kernel references.
    remote:
      url: https://github.com/NVIDIA/cutlass.git
      default_ref: main
    required: true

  - source_id: source-code/cuda-samples
    kind: git-repo
    remote:
      url: https://github.com/NVIDIA/cuda-samples.git
      default_ref: master
    required: false

  - source_id: source-code/flash-attention
    kind: git-repo
    remote:
      url: https://github.com/Dao-AILab/flash-attention.git
      default_ref: main
    required: false
```

#### `corpus/localize.yaml.example`

```yaml
## Copy to corpus/localize.yaml on offline machines or machines with local mirrors.
## corpus/localize.yaml must be git-ignored.

sources:
  source-code/cutlass:
    local_path: /path/to/local/cutlass

  source-code/cuda-samples:
    local_path: /path/to/local/cuda-samples

  source-code/flash-attention:
    local_path: /path/to/local/flash-attention
```

#### `.gitignore`

```gitignore
corpus/localize.yaml
corpus/.external/
```

#### Resolver 行为

建议实现 `scripts/resolve_corpus_source.py`：

```bash
python3 scripts/resolve_corpus_source.py \
  source-code/cutlass \
  examples/88_hopper_fmha/88_hopper_fmha.cu
```

输出本机可访问路径，例如：

```text
/data/repos/cutlass/examples/88_hopper_fmha/88_hopper_fmha.cu
```

或：

```text
corpus/.external/source-code__cutlass/examples/88_hopper_fmha/88_hopper_fmha.cu
```

---

### kp-mvp 示例迁移映射

#### 1. 外部 CUTLASS 文件

原始：

```yaml
- path: /data1/tongyu/workspace/cuda_repo/cutlass/examples/88_hopper_fmha/88_hopper_fmha.cu
  anchor: Hopper FMHA example — used as algorithmic reference only, not as code dependency
```

迁移后：

```yaml
source_refs:
  - source_id: source-code/cutlass
    path: examples/88_hopper_fmha/88_hopper_fmha.cu
    anchor: Hopper FMHA example — used as algorithmic reference only, not as code dependency
    dependency_mode: reference-only
```

#### 2. kp-mvp TMA/WGMMA probe 源码

原始：

```yaml
- path: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu
  anchor: wgmma PTX inline-asm pattern reused for attention Q@K^T and P@V tiles
- path: 80-experience/hw-probes/tma-ptx/artifacts/tma_hello.cu
  anchor: TMA load PTX pattern reused for Q/K/V tile loads
```

迁移后：

```yaml
source_refs:
  - source_id: experience-wgmma-ptx-hello
    path: artifacts/experience/hw-probes/wgmma-ptx/wgmma_hello.cu
    anchor: wgmma PTX inline-asm pattern reused for attention Q@K^T and P@V tiles
  - source_id: experience-tma-ptx-hello
    path: artifacts/experience/hw-probes/tma-ptx/tma_hello.cu
    anchor: TMA load PTX pattern reused for Q/K/V tile loads
```

其中 `experience-wgmma-ptx-hello` 和 `experience-tma-ptx-hello` 是 `sources/experience/...` 页面 ID。

#### 3. kp-mvp attention artifact bundle

原始：

```yaml
artifacts:
  code: 80-experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu
  build: 80-experience/api-probes/attention/artifacts/build.sh
  introspection: 80-experience/api-probes/attention/artifacts/device.json
  profile: 80-experience/api-probes/attention/artifacts/profiles/tma-wgmma-ncu.csv
```

迁移后页面只保留：

```yaml
artifact_dir: artifacts/experience/api-probes/attention-tma-wgmma
```

细粒度文件角色放在：

```text
artifacts/experience/api-probes/attention-tma-wgmma/PROVENANCE.yaml
```

---

### source-experience 页面示例

#### `sources/experience/api-probes/attention-tma-wgmma.md`

```markdown
---
id: experience-attention-tma-wgmma
title: TMA + WGMMA Online-Softmax Attention Probe
source_category: api-probe
architectures: [sm90]
tags: [attention, flash-attention, tma, wgmma, online-softmax, cuda-cpp, ptx]
captured_at: 2026-05-29
artifact_dir: artifacts/experience/api-probes/attention-tma-wgmma
source_refs:
  - source_id: source-code/cutlass
    path: examples/88_hopper_fmha/88_hopper_fmha.cu
    anchor: Hopper FMHA example — used as algorithmic reference only, not as code dependency
  - source_id: experience-wgmma-ptx-hello
    path: artifacts/experience/hw-probes/wgmma-ptx/wgmma_hello.cu
    anchor: wgmma PTX inline-asm pattern reused for attention Q@K^T and P@V tiles
  - source_id: experience-tma-ptx-hello
    path: artifacts/experience/hw-probes/tma-ptx/tma_hello.cu
    anchor: TMA load PTX pattern reused for Q/K/V tile loads
---

### Summary

Migrated kp-mvp TMA + WGMMA online-softmax attention probe.

### Artifact bundle

- Primary code: `flash_attn_tma_wgmma.cu`
- Build script: `build.sh`
- Introspection: `device.json`
- Profile: `profiles/tma-wgmma-ncu.csv`
```

---

### PROVENANCE.yaml 示例

```yaml
origin_url: kp-mvp://80-experience/api-probes/attention/artifacts
upstream_repo: kernel-kb-mvp
upstream_sha: <kernel-kb-mvp commit sha>
license: inherits-from-kp-mvp
retrieved_at: 2026-05-29
asset_mode: verbatim
size_cap_truncated: false
files:
  - local_path: flash_attn_tma_wgmma.cu
    role: kernel-source
    mode: verbatim
    upstream_path: 80-experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu
    sha256: <sha256>
  - local_path: build.sh
    role: build-script
    mode: verbatim
    upstream_path: 80-experience/api-probes/attention/artifacts/build.sh
    sha256: <sha256>
  - local_path: device.json
    role: introspection
    mode: verbatim
    upstream_path: 80-experience/api-probes/attention/artifacts/device.json
    sha256: <sha256>
  - local_path: profiles/tma-wgmma-ncu.csv
    role: profile
    mode: verbatim
    upstream_path: 80-experience/api-probes/attention/artifacts/profiles/tma-wgmma-ncu.csv
    sha256: <sha256>
```

推荐新增 artifact file roles：

```yaml
role:
  - kernel-source
  - build-script
  - run-script
  - introspection
  - profile
```

保留已有 roles：`pr-diff`, `upstream-file`, `extracted-block`, `derived-source`, `approach-notes`, `bench-record`。

---

### 需要修改的 schema / validator

#### 1. 新增 page type: `source-experience`

`data/schemas.yaml`：

```yaml
source-experience:
  required:
    - id
    - title
    - source_category
    - architectures
    - tags
    - captured_at
  optional:
    - description
    - artifact_dir
    - source_refs
  constraints:
    source_category: [measured-probe, api-probe, hw-probe]
```

`data/tags.yaml`：

```yaml
source_categories:
  - measured-probe
  - api-probe
  - hw-probe
```

如果使用 `online-softmax` 作为 tag，也需要加入 controlled vocabulary，建议放在 `techniques`。

#### 2. `scripts/validate.py`

需要支持：

- `sources/experience/**/*.md` → `source-experience`
- `artifacts/experience/**/PROVENANCE.yaml` 作为合法 bundle root
- artifact roles：`kernel-source`, `build-script`, `run-script`, `introspection`, `profile`

#### 3. 查询工具

`scripts/query.py` 可将 `--type experience` 映射到 `source-experience`，或保留通用路径检测：`sources/experience` → `source-experience`。

`scripts/get_page.py` 不需要特殊 fallback，只要页面有显式 `artifact_dir` 即可。

---

### 迁移步骤

1. 新增 corpus contract：
   - `corpus/MANIFEST.yaml`
   - `corpus/localize.yaml.example`
   - `corpus/README.md`
   - `.gitignore` 加入 `corpus/localize.yaml` 和 `corpus/.external/`
   - 可选：`scripts/resolve_corpus_source.py`

2. 扩展 schema/validator：
   - 新增 `source-experience`
   - 新增 source categories
   - 新增 artifact roles
   - 识别 `artifacts/experience/**/PROVENANCE.yaml`

3. 迁移 kp-mvp probe bundles：
   - `80-experience/hw-probes/tma-ptx/artifacts` → `artifacts/experience/hw-probes/tma-ptx`
   - `80-experience/hw-probes/wgmma-ptx/artifacts` → `artifacts/experience/hw-probes/wgmma-ptx`
   - `80-experience/api-probes/attention/artifacts` → `artifacts/experience/api-probes/attention-tma-wgmma`

4. 生成 `PROVENANCE.yaml`：
   - `origin_url: kp-mvp://<old-path>`
   - `upstream_repo: kernel-kb-mvp`
   - `upstream_sha: <kp-mvp git sha>`
   - 每个文件记录 `local_path`, `role`, `mode`, `upstream_path`, `sha256`

5. 新增 `sources/experience` 页面：
   - `experience-tma-ptx-hello`
   - `experience-wgmma-ptx-hello`
   - `experience-attention-tma-wgmma`

6. 更新引用方页面：
   - 外部 repo 引用改成 `source_id + path`
   - kp-mvp artifact 引用改成 `artifact_dir`
   - 不再出现旧 `80-experience/...` 路径，除非在 `PROVENANCE.yaml::upstream_path` 中作为历史来源记录。

7. 验证：
   - `python3 scripts/validate.py`
   - `python3 scripts/query.py --type experience --compact`
   - `python3 scripts/get_page.py experience-attention-tma-wgmma --include-code`
   - 离线解析测试：配置 `corpus/localize.yaml` 后运行 resolver `--no-clone`

---

### 验收标准

- Committed files 中不存在 `/data1/.../cutlass` 这类机器绝对路径。
- `source-code/cutlass` 可通过远程 Git 或 `corpus/localize.yaml` 本地路径解析。
- kp-mvp 自产代码、build、device、profile 文件均在 `artifacts/experience/...` 中，并有 sha256 provenance。
- `sources/experience` 页面能通过 `artifact_dir` 找到对应 bundle。
- `scripts/validate.py` 通过。
- 离线机器只需配置 `corpus/localize.yaml`，不需要修改知识库正文。

--- Original Design Draft End ---
