#!/usr/bin/env python3
"""Codebase Cleanup Script

Safely removes deprecated/obsolete files from the AI-Media-Indexer project.
Run with --dry-run to preview changes before execution.

Usage:
    python scripts/cleanup.py --dry-run  # Preview only
    python scripts/cleanup.py --execute  # Actually delete files
"""

import argparse
import shutil
from pathlib import Path

# Deprecated files to remove
OBSOLETE_FILES = [
    # CLI tools replaced by Web UI and API
    "agent_cli.py",
    "agent_main.py",
    "search_cli.py",
    # Dangerous reset scripts replaced by documented procedures
    "nuke.bat",
    # Deprecated test files (if any)
    "test_*.tmp",
]

# Directories to clean (cache patterns)
CACHE_PATTERNS = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.mypy_cache",
    "**/*.egg-info",
]

# Directories that should never be deleted
PROTECTED_DIRS = [
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "qdrant_data",
]


def find_obsolete_files(root: Path) -> list[Path]:
    """Find all obsolete files to be removed."""
    found = []

    for pattern in OBSOLETE_FILES:
        if "*" in pattern:
            found.extend(root.glob(pattern))
        else:
            filepath = root / pattern
            if filepath.exists():
                found.append(filepath)

    return found


def find_cache_dirs(root: Path) -> list[Path]:
    """Find all cache directories/files to be removed."""
    found = []

    for pattern in CACHE_PATTERNS:
        for match in root.glob(pattern):
            # Skip protected directories
            if any(protected in str(match) for protected in PROTECTED_DIRS):
                continue
            found.append(match)

    return found


def remove_path(path: Path, dry_run: bool = True) -> bool:
    """Remove a file or directory. Returns True if successful."""
    try:
        if dry_run:
            print(f"  [DRY-RUN] Would delete: {path}")
            return True

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        print(f"  [DELETED] {path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Could not delete {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean up obsolete files from the codebase"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run", action="store_true", help="Preview changes without deleting"
    )
    group.add_argument("--execute", action="store_true", help="Actually delete files")
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="Also remove __pycache__ directories",
    )

    args = parser.parse_args()
    dry_run = args.dry_run

    # Find project root (directory containing this script's parent)
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    print(f"\n{'=' * 60}")
    print("AI-Media-Indexer Cleanup Script")
    print(f"{'=' * 60}")
    print(f"Project root: {root}")
    print(
        f"Mode: {'DRY-RUN (no changes)' if dry_run else 'EXECUTE (will delete files)'}"
    )
    print(f"{'=' * 60}\n")

    # Find obsolete files
    obsolete = find_obsolete_files(root)
    print(f"Obsolete files found: {len(obsolete)}")

    deleted_count = 0
    for path in obsolete:
        if remove_path(path, dry_run):
            deleted_count += 1

    # Optionally clean cache
    if args.include_cache:
        print("\nCache directories/files:")
        cache_items = find_cache_dirs(root)
        print(f"Cache items found: {len(cache_items)}")

        for path in cache_items:
            if remove_path(path, dry_run):
                deleted_count += 1

    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"DRY-RUN complete. {deleted_count} items would be deleted.")
        print("Run with --execute to actually delete files.")
    else:
        print(f"Cleanup complete. {deleted_count} items deleted.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
