#!/usr/bin/env python3
"""Migrate kernel-kb-mvp content into KernelWiki.

Reads source files from kernel-kb-mvp/knowledge/, rewrites frontmatter,
and writes to the appropriate KernelWiki target paths.

Usage:
    python3 scripts/migrate_kb_mvp.py --source /path/to/kernel-kb-mvp --dry-run
    python3 scripts/migrate_kb_mvp.py --source /path/to/kernel-kb-mvp
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def extract_frontmatter(filepath):
    """Extract YAML frontmatter and body from a markdown file."""
    content = filepath.read_text(encoding="utf-8")
    match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not match:
        return None, content
    try:
        fm = yaml.safe_load(match.group(1))
        body = content[match.end():]
        return fm, body
    except yaml.YAMLError:
        return None, content


def write_page(target_path, fm, body, dry_run=False):
    """Write a page with YAML frontmatter + body."""
    if dry_run:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---\n"]
    lines.append(yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False))
    lines.append("---\n")
    lines.append(body)
    target_path.write_text("".join(lines), encoding="utf-8")


def slug_from_path(path):
    """Generate a slug from a path (e.g., 'compute/warp-primitives')."""
    return path.stem


def map_skill(src_rel, src_root):
    """Map 30-skill/ files to wiki/nvidia/foundations/."""
    parts = src_rel.parts  # e.g., ('30-skill', 'compute', 'warp-primitives', 'skill.md')
    if len(parts) < 3:
        return None, None, None
    family = parts[1]  # compute, memory, sync
    name = parts[2]    # warp-primitives, coalescing, etc.
    filename = parts[-1]

    if filename == "skill.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "foundations" / family / f"{name}.md"
        page_type = "skill"
        page_id = f"skill-{name}"
    elif filename == "pitfalls.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "foundations" / family / name / "pitfalls.md"
        page_type = "pitfall"
        page_id = f"pitfall-{name}"
    elif filename == "apis.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "foundations" / family / name / "apis.md"
        page_type = "api-definition"
        page_id = f"api-{name}-ref"
    else:
        target = REPO_ROOT / "wiki" / "nvidia" / "foundations" / family / name / filename
        page_type = "skill"
        page_id = f"skill-{name}-{Path(filename).stem}"
    return target, page_type, page_id


def map_hardware_feature(src_rel, src_root):
    """Map 40-hardware-feature/ files to wiki/nvidia/hardware/."""
    parts = src_rel.parts
    if len(parts) < 2:
        return None, None, None
    feat = parts[1]  # tma, wgmma, tcgen05-ptx, etc.

    # Map feature names to existing hardware topics
    topic_map = {
        "tma": "tma",
        "tma-ptx": "tma",
        "wgmma": "wgmma",
        "wgmma-ptx": "wgmma",
        "tcgen05-ptx": "tcgen05-mma",
        "mma-sync-ptx": "mma-sync",
        "ldmatrix-ptx": "ldmatrix",
    }
    topic = topic_map.get(feat, feat)
    filename = parts[-1]

    if filename == "skill.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "hardware" / topic / f"skill-{feat}.md"
        return target, "skill", f"skill-{feat}"
    elif filename == "pitfalls.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "hardware" / topic / f"pitfalls-{feat}.md"
        return target, "pitfall", f"pitfall-{feat}"
    else:
        target = REPO_ROOT / "wiki" / "nvidia" / "hardware" / topic / filename
        return target, "skill", f"skill-{feat}-{Path(filename).stem}"


def map_classical_algo(src_rel, src_root):
    """Map 50-classical-algo/ to wiki/nvidia/techniques/."""
    parts = src_rel.parts
    if len(parts) < 2:
        return None, None, None
    algo = parts[1]
    filename = parts[-1]

    if filename == "skill.md":
        target = REPO_ROOT / "wiki" / "nvidia" / "techniques" / algo / f"algo-{algo}.md"
        return target, "algorithm", f"algo-{algo}"
    else:
        target = REPO_ROOT / "wiki" / "nvidia" / "techniques" / algo / filename
        return target, "algorithm", f"algo-{algo}-{Path(filename).stem}"


def map_code(src_rel, src_root):
    """Map 60-code/ to wiki/nvidia/code-walkthroughs/."""
    parts = src_rel.parts
    if len(parts) < 2:
        return None, None, None
    repo = parts[1]
    rest = Path(*parts[2:]) if len(parts) > 2 else Path(parts[-1])
    target = REPO_ROOT / "wiki" / "nvidia" / "code-walkthroughs" / repo / rest
    slug = f"{repo}-{rest.stem}"
    return target, "code-walkthrough", f"code-{slug}"


def map_api_raw(src_rel, src_root):
    """Map 10-api-raw/ to wiki/nvidia/api-definitions/."""
    parts = src_rel.parts
    if len(parts) < 2:
        return None, None, None
    ns = parts[1] if len(parts) > 2 else "runtime"
    filename = parts[-1]
    target = REPO_ROOT / "wiki" / "nvidia" / "api-definitions" / ns / filename
    name = Path(filename).stem
    return target, "api-definition", f"api-{name}"


def map_pattern(src_rel, src_root):
    """Map 20-pattern/ to wiki/nvidia/operator-routing/."""
    parts = src_rel.parts
    # 20-pattern/{cuda-core|tensor-core}/{op}/...
    if len(parts) < 3:
        return None, None, None
    op = parts[2] if len(parts) > 2 else parts[1]
    rest = Path(*parts[3:]) if len(parts) > 3 else Path(parts[-1])
    target = REPO_ROOT / "wiki" / "nvidia" / "operator-routing" / op / rest
    return target, "operator-routing", f"routing-{op}-{rest.stem}"


def map_experience(src_rel, src_root):
    """Map 80-experience/ to sources/experience/."""
    parts = src_rel.parts
    if len(parts) < 2:
        return None, None, None
    # 80-experience/{hw-probes|api-probes}/{slug}/...
    category = parts[1] if len(parts) > 1 else "hw-probes"
    slug = parts[2] if len(parts) > 2 else parts[-1]
    filename = parts[-1]

    if filename.endswith(".md"):
        target = REPO_ROOT / "sources" / "experience" / category / f"{Path(slug).stem}.md"
        return target, "experience", f"exp-{Path(slug).stem}"
    else:
        # Artifact file
        target = REPO_ROOT / "artifacts" / "experience" / category / slug / filename
        return target, None, None  # Not a page


def rewrite_source_paths(fm, body):
    """Rewrite absolute source paths to source_id + relative format."""
    if not isinstance(fm, dict):
        return fm, body

    # Rewrite source field entries
    if "source" in fm and isinstance(fm["source"], list):
        new_sources = []
        for entry in fm["source"]:
            if isinstance(entry, dict) and "path" in entry:
                path = entry["path"]
                if isinstance(path, str) and "05-source-corpus/" in path:
                    path = path.replace("05-source-corpus/", "")
                    entry["path"] = path
            new_sources.append(entry)
        fm["source"] = new_sources

    # Rewrite body text references
    body = re.sub(
        r'05-source-corpus/',
        'corpus/nvidia/',
        body
    )

    return fm, body


def migrate_file(src_path, src_root, inventory, dry_run=False):
    """Migrate a single file. Returns (target_path, status, reason)."""
    src_rel = src_path.relative_to(src_root / "knowledge")

    # Determine mapping based on top-level directory
    top_dir = src_rel.parts[0]
    mapper = {
        "30-skill": map_skill,
        "40-hardware-feature": map_hardware_feature,
        "50-classical-algo": map_classical_algo,
        "60-code": map_code,
        "10-api-raw": map_api_raw,
        "20-pattern": map_pattern,
        "80-experience": map_experience,
    }.get(top_dir)

    if mapper is None:
        return None, "skipped", f"no mapping for {top_dir}"

    target, page_type, page_id = mapper(src_rel, src_root)
    if target is None:
        return None, "skipped", "mapping returned None"

    # For non-.md files (artifacts), just copy
    if not src_path.suffix == ".md":
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, target)
        return str(target.relative_to(REPO_ROOT)), "migrated", "artifact copy"

    # Parse frontmatter
    fm, body = extract_frontmatter(src_path)
    if fm is None:
        fm = {}

    # Add/rewrite frontmatter
    if page_id:
        fm["id"] = page_id
    if page_type:
        fm["type"] = page_type
    fm["vendor"] = "nvidia"
    if "title" not in fm:
        fm["title"] = src_rel.stem.replace("-", " ").title()

    # Rewrite source paths
    fm, body = rewrite_source_paths(fm, body)

    # Write
    write_page(target, fm, body, dry_run=dry_run)
    return str(target.relative_to(REPO_ROOT)), "migrated", ""


def main():
    parser = argparse.ArgumentParser(description="Migrate kernel-kb-mvp into KernelWiki")
    parser.add_argument("--source", required=True, help="Path to kernel-kb-mvp repo")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files")
    parser.add_argument("--output-tsv", default="migration_inventory.tsv", help="TSV inventory output")
    args = parser.parse_args()

    src_root = Path(args.source).resolve()
    knowledge = src_root / "knowledge"
    if not knowledge.exists():
        print(f"Error: {knowledge} not found")
        sys.exit(1)

    inventory = []
    migrated = 0
    skipped = 0

    # Process all .md files in knowledge/
    for top_dir in sorted(["10-api-raw", "20-pattern", "30-skill", "40-hardware-feature",
                           "50-classical-algo", "60-code", "80-experience"]):
        dir_path = knowledge / top_dir
        if not dir_path.exists():
            continue
        for src_file in sorted(dir_path.rglob("*.md")):
            src_rel = src_file.relative_to(knowledge)
            target, status, reason = migrate_file(src_file, src_root, inventory, dry_run=args.dry_run)

            inventory.append({
                "input_path": str(src_rel),
                "output_path": target or "",
                "page_id": "",
                "page_type": "",
                "vendor": "nvidia",
                "status": status,
                "reason": reason,
            })

            if status == "migrated":
                migrated += 1
            else:
                skipped += 1

    # Also process artifact files in 80-experience
    exp_dir = knowledge / "80-experience"
    if exp_dir.exists():
        for src_file in sorted(exp_dir.rglob("*")):
            if src_file.is_file() and src_file.suffix != ".md":
                src_rel = src_file.relative_to(knowledge)
                target, status, reason = migrate_file(src_file, src_root, inventory, dry_run=args.dry_run)
                inventory.append({
                    "input_path": str(src_rel),
                    "output_path": target or "",
                    "status": status,
                    "reason": reason,
                })
                if status == "migrated":
                    migrated += 1

    # Write TSV inventory
    tsv_path = REPO_ROOT / args.output_tsv
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("input_path\toutput_path\tstatus\treason\n")
        for entry in sorted(inventory, key=lambda x: x["input_path"]):
            f.write(f"{entry['input_path']}\t{entry['output_path']}\t{entry['status']}\t{entry['reason']}\n")

    print(f"Migration {'dry-run' if args.dry_run else 'complete'}")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Inventory: {tsv_path}")


if __name__ == "__main__":
    main()
