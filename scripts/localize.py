#!/usr/bin/env python3
"""Localization tool for the two-tier source corpus.

Manages user-specific configuration for resolving {{PLACEHOLDER}} variables
in MANIFEST.yaml tier-2 entries.

Usage:
    python3 scripts/localize.py init-config   # Create localize.yaml from example
    python3 scripts/localize.py check         # Verify all paths
    python3 scripts/localize.py link          # Create .external/ symlinks
    python3 scripts/localize.py resolve-vars  # Show resolved variable values
    python3 scripts/localize.py doctor        # Full health check
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
CONFIG_PATH = CORPUS_DIR / "localize.yaml"
EXAMPLE_PATH = CORPUS_DIR / "localize.yaml.example"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.yaml"
EXTERNAL_DIR = CORPUS_DIR / ".external"

PLACEHOLDER_RE = re.compile(r'\{\{(\w+)\}\}')


def load_config():
    """Load localize.yaml and resolve derived variables."""
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    vars_ = dict(config.get("vars", {}) or {})
    derived = dict(config.get("derived", {}) or {})

    # Resolve environment variables if inherit is true
    env_config = config.get("env", {}) or {}
    if env_config.get("inherit", False):
        for key in list(vars_) + list(derived):
            env_val = os.environ.get(key)
            if env_val and key not in vars_ and key not in derived:
                vars_[key] = env_val

    # Resolve derived variables (may reference vars)
    resolved = dict(vars_)
    for key, val in derived.items():
        if isinstance(val, str):
            for m in PLACEHOLDER_RE.finditer(val):
                ref = m.group(1)
                if ref in resolved:
                    val = val.replace(f"{{{{{ref}}}}}", str(resolved[ref]))
        resolved[key] = val

    return resolved


def resolve_path(path_str, variables):
    """Resolve a path string, replacing {{PLACEHOLDER}} with values."""
    if not variables:
        return path_str
    result = path_str
    for m in PLACEHOLDER_RE.finditer(path_str):
        var = m.group(1)
        if var in variables:
            result = result.replace(f"{{{{{var}}}}}", str(variables[var]))
    return result


def load_manifest():
    """Load MANIFEST.yaml entries."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def cmd_init_config(args):
    """Create localize.yaml from the example template."""
    if CONFIG_PATH.exists() and not args.force:
        print(f"Config already exists: {CONFIG_PATH}")
        print("Use --force to overwrite.")
        return 1
    if not EXAMPLE_PATH.exists():
        print(f"Example not found: {EXAMPLE_PATH}")
        return 1
    import shutil
    shutil.copy2(EXAMPLE_PATH, CONFIG_PATH)
    print(f"Created {CONFIG_PATH} from example.")
    print(f"Edit it to set your local paths, then run: python3 {__file__} check")
    return 0


def cmd_check(args):
    """Check that all placeholders have values and paths exist."""
    variables = load_config()
    manifest = load_manifest()
    has_config = variables is not None

    if not has_config:
        print(f"WARN  No localize.yaml found at {CONFIG_PATH}")
        print(f"      Tier-2 external repos will not be accessible.")
        print(f"      Run: python3 {__file__} init-config")
        variables = {}

    ok = 0
    warn = 0
    for entry in manifest:
        sid = entry.get("source_id", "")
        lpath = entry.get("local_path", "")

        if PLACEHOLDER_RE.search(lpath):
            # Tier-2: external reference
            resolved = resolve_path(lpath, variables)
            if PLACEHOLDER_RE.search(resolved):
                unresolved = PLACEHOLDER_RE.findall(resolved)
                print(f"WARN  {sid} → unresolved placeholders: {unresolved}")
                warn += 1
            elif Path(resolved).exists():
                print(f"OK    {sid} → {resolved} (exists)")
                ok += 1
            else:
                print(f"WARN  {sid} → {resolved} (path not found)")
                warn += 1
        else:
            # Tier-1: in-git
            full_path = CORPUS_DIR / lpath
            if full_path.exists():
                print(f"OK    {sid} → {full_path.relative_to(REPO_ROOT)} (in-git)")
                ok += 1
            else:
                print(f"WARN  {sid} → {lpath} (tier-1 path missing)")
                warn += 1

    print(f"\n{ok} OK, {warn} warnings")
    return 0 if warn == 0 else 0  # Warnings don't fail


def cmd_link(args):
    """Create symlinks in .external/ for tier-2 repos."""
    variables = load_config()
    if not variables:
        print("No localize.yaml found. Run init-config first.")
        return 1

    manifest = load_manifest()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    created = 0

    for entry in manifest:
        lpath = entry.get("local_path", "")
        if not PLACEHOLDER_RE.search(lpath):
            continue
        sid = entry.get("source_id", "")
        resolved = resolve_path(lpath, variables)
        if PLACEHOLDER_RE.search(resolved):
            continue

        link_name = sid.split("/")[-1]
        link_path = EXTERNAL_DIR / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            link_path.symlink_to(resolved)
            print(f"Linked {link_name} → {resolved}")
            created += 1
        except OSError as e:
            print(f"FAIL  {link_name}: {e}")

    print(f"\nCreated {created} symlinks in {EXTERNAL_DIR.relative_to(REPO_ROOT)}")
    return 0


def cmd_resolve_vars(args):
    """Show all resolved variable values."""
    variables = load_config()
    if not variables:
        print("No localize.yaml found.")
        return 1
    for k, v in sorted(variables.items()):
        exists = Path(v).exists() if not PLACEHOLDER_RE.search(str(v)) else False
        status = "exists" if exists else "not found"
        print(f"  {k} = {v}  ({status})")
    return 0


def cmd_doctor(args):
    """Full health check: placeholders, paths, MANIFEST consistency."""
    print("=== Localization Doctor ===\n")
    print("--- Config ---")
    variables = load_config()
    if variables:
        print(f"  Config: {CONFIG_PATH} (loaded)")
        cmd_resolve_vars(args)
    else:
        print(f"  Config: not found (tier-2 unavailable)")

    print("\n--- MANIFEST ---")
    ret = cmd_check(args)

    print("\n--- Symlinks ---")
    if EXTERNAL_DIR.exists():
        for item in sorted(EXTERNAL_DIR.iterdir()):
            if item.is_symlink():
                target = item.resolve()
                exists = target.exists()
                print(f"  {item.name} → {target} ({'OK' if exists else 'BROKEN'})")
    else:
        print("  .external/ not created yet")

    return ret


def main():
    parser = argparse.ArgumentParser(description="Manage source corpus localization")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init-config", help="Create localize.yaml from example")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")

    sub.add_parser("check", help="Verify placeholder resolution and paths")
    sub.add_parser("link", help="Create .external/ symlinks")
    sub.add_parser("resolve-vars", help="Show resolved variables")
    sub.add_parser("doctor", help="Full health check")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    cmds = {
        "init-config": cmd_init_config,
        "check": cmd_check,
        "link": cmd_link,
        "resolve-vars": cmd_resolve_vars,
        "doctor": cmd_doctor,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
