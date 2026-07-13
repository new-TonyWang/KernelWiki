# Progress

- Initialized dedicated planning files for library-call content isolation.
- Mapped query/index scopes: `sources/` + `wiki/` only; selected top-level `nonquery/library-calls/` as archive outside retrieval.
- Moved all operator-routing `library-fallback.md` pages out of `wiki/` into `nonquery/library-calls/operator-routing/`.
- Rewrote affected operator routing, foundations, kernel, source, and generated query-index Markdown to remove exact high-level library-call tokens from queryable roots.
- Archived original removed/generalized snippets under `nonquery/library-calls/extracted-wiki-snippets.md` and `nonquery/library-calls/extracted-source-snippets.md`.
- Regenerated query indices and updated README/SKILL page counts to 2482.
- Verification passed: strict exact-token grep over `sources wiki queries`, exact `query.py --paths-only` checks for moved/API terms, and `python3 scripts/validate.py`.
