"""Service functions for KernelWiki keyword search.

Extracted from query.py for programmatic reuse by MCP server and other tools.
No argparse, no stdout/stderr writes, no sys.exit().
"""

import re
import yaml
from pathlib import Path

from _wiki_root import WIKI_ROOT

_WIKI_ROOT_RESOLVED = WIKI_ROOT.resolve()


def _is_within_root(path):
    """Return True if resolved path is inside WIKI_ROOT."""
    return path.resolve().is_relative_to(_WIKI_ROOT_RESOLVED)


# ---------------------------------------------------------------------------
# Global caches (lazy-loaded)
# ---------------------------------------------------------------------------

_ALIAS_CACHE = None
_VENDOR_REGISTRY = None


# ---------------------------------------------------------------------------
# Alias / vendor loading
# ---------------------------------------------------------------------------

def load_alias_expansions():
    """Return dict mapping lowercased alias -> canonical term from data/aliases.yaml."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    """Return [kw] or [kw, canonical] if the alias map provides a canonical form."""
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_vendor_registry():
    """Load vendor->architecture mapping from data/vendors.yaml."""
    global _VENDOR_REGISTRY
    if _VENDOR_REGISTRY is not None:
        return _VENDOR_REGISTRY
    vpath = WIKI_ROOT / "data" / "vendors.yaml"
    try:
        raw = yaml.safe_load(vpath.read_text(encoding="utf-8")) or {}
    except Exception:
        _VENDOR_REGISTRY = {}
        return _VENDOR_REGISTRY
    reg = {}
    for v in raw.get("vendors", []):
        vid = v.get("id", "")
        for arch in v.get("architectures", []):
            reg[arch.lower()] = vid
        for cs in v.get("compute_stack", []):
            reg[cs.lower()] = vid
    _VENDOR_REGISTRY = reg
    return reg


# ---------------------------------------------------------------------------
# Frontmatter loading
# ---------------------------------------------------------------------------

def load_frontmatter(path):
    """Parse YAML frontmatter from a markdown file.  Returns (fm_dict, body_str) or (None, None)."""
    if not _is_within_root(path):
        return None, None
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages():
    """Load frontmatter + body for every sources/*.md and wiki/*.md file."""
    pages = []
    for subdir in ["sources", "wiki"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            if not _is_within_root(md):
                continue
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append({
                "path": str(md.relative_to(WIKI_ROOT)),
                "fm": fm,
                "body": body or "",
            })
    return pages


# ---------------------------------------------------------------------------
# Page type detection
# ---------------------------------------------------------------------------

def detect_page_type(fm, path):
    """Return a page-type label: source-pr, wiki-kernel, etc."""
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        if parts[1] == "experience":
            return "source-experience"
        if "type" not in fm:
            return f"source-{parts[1].rstrip('s')}"
    if "type" in fm:
        return f"wiki-{fm['type']}"
    return "unknown"


# ---------------------------------------------------------------------------
# Helper: flatten nested frontmatter values
# ---------------------------------------------------------------------------

def flatten_meta_values(value):
    """Yield strings from nested frontmatter structures for search/filtering."""
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        yield str(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from flatten_meta_values(v)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten_meta_values(item)


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def score_keyword_match(fm, body, keywords):
    """Score a page by keyword matches.  title(10) > tags(5) > metadata(4) > body(1-3)."""
    score = 0
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    metadata_text = " ".join(
        s
        for k in (
            "repo", "upstream_repo", "source_refs", "source", "sources",
            "artifacts", "artifact_dir", "api", "namespace", "func_name",
            "operator", "applies_to", "applies_to_ops", "requires_features",
            "related", "related_apis", "related_skills", "probe_slug",
        )
        for s in flatten_meta_values(fm.get(k))
    ).lower()
    body_lower = body.lower()
    for kw in keywords:
        best_variant_score = 0
        for variant in expand_keyword(kw):
            v_l = variant.lower()
            variant_score = 0
            if v_l in title_text:
                variant_score += 10
            if v_l in tag_text:
                variant_score += 5
            if v_l in metadata_text:
                variant_score += 4
            body_hits = body_lower.count(v_l)
            variant_score += min(body_hits, 3)
            if variant_score > best_variant_score:
                best_variant_score = variant_score
        score += best_variant_score
    return score


# ---------------------------------------------------------------------------
# Filtering  (accepts a plain dict, NOT argparse.Namespace)
# ---------------------------------------------------------------------------

# Code extensions for --has-code filtering (shared contract with validate.py
# and get_page.py).
CODE_EXTS = {
    ".cu", ".cuh", ".ptx", ".py", ".cpp", ".h", ".hpp", ".inl",
    ".pyx", ".cxx", ".cc", ".txt",
    ".sh", ".yaml", ".json",
}


def filter_pages(pages, params):
    """Apply filters to narrow the page set.

    ``params`` is a plain dict with optional keys:
        type, tag, vendor, repo, language, architecture, symptom,
        confidence, has_code
    Missing keys default to None / False (no filter).
    """
    f_type = params.get("type")
    f_tag = params.get("tag")
    f_vendor = params.get("vendor")
    f_repo = params.get("repo")
    f_language = params.get("language")
    f_architecture = params.get("architecture")
    f_symptom = params.get("symptom")
    f_confidence = params.get("confidence")
    f_has_code = params.get("has_code", False)

    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if f_type and not ptype.endswith(f_type):
            if ptype != f_type:
                continue

        if f_tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(f_tag)}
            if not any(t.lower() in tag_variants for t in all_tags):
                continue

        if f_vendor and f_vendor != "all":
            fm_vendor = fm.get("vendor", "")
            path_parts = path.split("/")
            path_vendor = path_parts[1] if path_parts[0] == "wiki" and len(path_parts) > 2 else ""
            vendor_values = []
            if isinstance(fm_vendor, (list, tuple, set)):
                vendor_values.extend(str(v) for v in fm_vendor)
            elif fm_vendor:
                vendor_values.append(str(fm_vendor))
            if path_vendor:
                vendor_values.append(path_vendor)

            reg = load_vendor_registry()
            for k in ("architectures", "languages", "tags", "hardware_features", "kernel_types"):
                for value in fm.get(k) or []:
                    inferred_vendor = reg.get(str(value).lower())
                    if inferred_vendor:
                        vendor_values.append(inferred_vendor)

            if f_vendor not in set(vendor_values):
                continue

        if f_repo:
            repo_text = " ".join(
                flatten_meta_values({
                    "repo": fm.get("repo"),
                    "upstream_repo": fm.get("upstream_repo"),
                    "source_refs": fm.get("source_refs"),
                    "source": fm.get("source"),
                })
            ).lower()
            if f_repo.lower() not in repo_text:
                continue

        if f_language:
            langs = {l.lower() for l in (fm.get("languages") or [])}
            tags = {t.lower() for t in (fm.get("tags") or [])}
            lang_variants = {v.lower() for v in expand_keyword(f_language)}
            if not (lang_variants & langs) and not (lang_variants & tags):
                continue

        if f_architecture:
            archs = {a.lower() for a in (fm.get("architectures") or [])}
            arch_variants = {v.lower() for v in expand_keyword(f_architecture)}
            if not (archs & arch_variants):
                continue

        if f_symptom:
            symptoms = set(fm.get("symptoms") or [])
            if f_symptom not in symptoms:
                continue

        if f_confidence:
            if str(fm.get("confidence", "")) != f_confidence:
                continue

        if f_has_code:
            candidate_dirs = []
            ad = fm.get("artifact_dir")
            if ad:
                ad_path = WIKI_ROOT / ad
                if _is_within_root(ad_path):
                    candidate_dirs.append(ad_path)

            explicit_artifact_files = []
            for art in flatten_meta_values(fm.get("artifacts")):
                art_path = WIKI_ROOT / art
                if not _is_within_root(art_path):
                    continue
                if art_path.is_file():
                    explicit_artifact_files.append(art_path)
                    candidate_dirs.append(art_path.parent)
                elif art_path.is_dir():
                    candidate_dirs.append(art_path)

            if p["_ptype"] == "source-blog":
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "blogs" / Path(path).stem / "code")
            elif p["_ptype"] == "source-contest":
                contest_dir_name = Path(path).parent.name
                problem_stem = Path(path).stem
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "contests" / contest_dir_name / problem_stem)

            if p["_ptype"] == "source-pr" and fm.get("repo") and fm.get("pr"):
                repo_short = str(fm["repo"]).split("/")[-1].lower()
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "prs" / repo_short / f"PR-{fm['pr']}")

            has_any = False
            for f in explicit_artifact_files:
                if _is_within_root(f) and f.suffix.lower() in CODE_EXTS:
                    has_any = True
                    break
            for cand in candidate_dirs:
                if has_any:
                    break
                if not cand.is_dir() or not _is_within_root(cand):
                    continue
                for f in cand.rglob("*"):
                    if f.is_file() and _is_within_root(f) and f.suffix.lower() in CODE_EXTS:
                        has_any = True
                        break
                if has_any:
                    break

            if not has_any:
                continue

        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Vendor inference  (accepts a plain dict)
# ---------------------------------------------------------------------------

def infer_vendor(params):
    """Auto-infer vendor from architecture, language, tag, or query keywords.

    ``params`` keys: architecture, language, tag, query (list of strings).
    Returns vendor string or None.
    """
    reg = load_vendor_registry()
    aliases = load_alias_expansions()
    candidates = set()

    if params.get("architecture"):
        for variant in expand_keyword(params["architecture"]):
            v = reg.get(variant.lower())
            if v:
                candidates.add(v)

    if params.get("language"):
        v = reg.get(params["language"].lower())
        if v:
            candidates.add(v)

    if params.get("tag"):
        for variant in expand_keyword(params["tag"]):
            v = reg.get(variant.lower())
            if v:
                candidates.add(v)

    for q in (params.get("query") or []):
        for tok in re.split(r"\s+", q.strip()):
            tok_l = tok.lower()
            v = reg.get(tok_l)
            if v:
                candidates.add(v)
            canonical = aliases.get(tok_l)
            if canonical:
                v = reg.get(canonical.lower())
                if v:
                    candidates.add(v)

    if len(candidates) == 1:
        return candidates.pop()
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_result(page, compact=False):
    """Format a single page as a readable string."""
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")

    if compact:
        return f"  [{ptype}] {pid}: {title}  ({path})"

    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    if "performance_claims" in fm and isinstance(fm["performance_claims"], list):
        for claim in fm["performance_claims"][:2]:
            lines.append(f"- **perf**: {claim.get('value')} {claim.get('metric')} on {claim.get('gpu')} ({claim.get('dtype')}, {claim.get('shape')})")
    if "sources" in fm:
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)
