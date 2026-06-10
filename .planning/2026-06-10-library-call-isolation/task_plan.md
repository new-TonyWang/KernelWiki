# Task Plan: Isolate library-call content from queryable knowledge files

## Goal
找到当前知识库中所有包含“库调用”的 Markdown 知识文件，修改相关知识文件，将库调用相关内容全部挪到单独文件夹，并确保这些库调用内容不会被 query 检索到。

## Requirements / Success Criteria
- [x] Inventory all queryable Markdown knowledge files and identify the query/indexing inclusion/exclusion rules.
- [x] Detect Markdown knowledge files that contain library-call content (imports/includes, framework/library API calls, third-party helper usage, or code/prose focused on invoking libraries rather than kernel implementation concepts).
- [x] Move the library-call-related content into a dedicated isolation folder outside the queryable knowledge corpus.
- [x] Rewrite the original knowledge Markdown files so queryable content no longer includes those library-call sections or references that would surface the isolated content.
- [x] Verify with repository search and the query/index pipeline that isolated library-call content is excluded from query retrieval.

## Phases
- [complete] Phase 1: Restore context and map repository/query indexing rules.
- [complete] Phase 2: Scan Markdown knowledge files for library-call content and build an evidence inventory.
- [complete] Phase 3: Create/choose the isolation folder and move extracted library-call content there.
- [complete] Phase 4: Rewrite affected Markdown knowledge files to remove/move library-call content.
- [complete] Phase 5: Verify by text scans, validation scripts, and query/index checks.

## Decisions
- Treat repository files and command output as authoritative.
- Preserve original knowledge where possible by moving rather than deleting library-call content.
- The isolation folder must be excluded from the query corpus/index, not merely hidden by naming.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
