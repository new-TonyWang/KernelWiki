# System-level library-call archive

This directory stores system-level or library-call material that is intentionally kept outside the queryable KernelWiki corpus.

`query.py`, `grep_wiki.py` default scopes, and `generate-indices.py` load Markdown from `sources/` and `wiki/` only. This top-level `system-level/` directory is therefore outside the retrievable knowledge corpus.

Moved content:
- operator-routing fallback/reference call pages under `operator-routing/<operator>/library-fallback.md`

Removed content:
- `extracted-source-snippets.md` and `extracted-wiki-snippets.md` were deleted because they no longer carry useful state after restoring selected source/wiki library references.
