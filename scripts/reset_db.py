#!/usr/bin/env python3
"""Database Reset Script.

Safely resets the application databases (Qdrant, SQLite jobs.db).
Replaces the dangerous nuke.bat with a documented, safe procedure.

Usage:
    python scripts/reset_db.py --confirm  # Reset all databases
    python scripts/reset_db.py --jobs     # Reset only jobs.db
    python scripts/reset_db.py --qdrant   # Reset only Qdrant collections
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def reset_jobs_db() -> bool:
    """Delete the SQLite jobs database."""
    jobs_path = Path("jobs.db")
    if jobs_path.exists():
        try:
            jobs_path.unlink()
            print("[OK] Deleted jobs.db")
            return True
        except Exception as e:
            print(f"[ERROR] Could not delete jobs.db: {e}")
            return False
    else:
        print("[SKIP] jobs.db does not exist")
        return True


def reset_qdrant() -> bool:
    """Delete all Qdrant collections."""
    try:
        from core.storage.db import VectorDB

        db = VectorDB()
        collections = [
            db.MEDIA_COLLECTION,
            db.FACES_COLLECTION,
            db.VOICE_COLLECTION,
        ]

        for coll in collections:
            try:
                db.client.delete_collection(coll)
                print(f"[OK] Deleted collection: {coll}")
            except Exception:
                print(f"[SKIP] Collection does not exist: {coll}")

        return True
    except Exception as e:
        print(f"[ERROR] Could not connect to Qdrant: {e}")
        return False


def reset_cache() -> bool:
    """Delete cache directories."""
    import shutil

    cache_dirs = [
        Path("cache"),
        Path(".cache"),
        Path("thumbnails"),
    ]

    for cache_dir in cache_dirs:
        # SAFETY: Explicitly skip logs directory to preserve history
        if "logs" in str(cache_dir).lower():
            print(f"[SAFETY] Skipping potential log dir: {cache_dir}")
            continue

        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"[OK] Deleted cache: {cache_dir}")
            except Exception as e:
                print(f"[ERROR] Could not delete {cache_dir}: {e}")

    return True


def main() -> None:
    """Main execution entry point for the database reset utility.

    Parses command-line arguments and coordinates the deletion of
    requested databases and cache directories.
    """
    parser = argparse.ArgumentParser(description="Reset application databases")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Reset all databases (requires confirmation)",
    )
    parser.add_argument(
        "--jobs", action="store_true", help="Reset only jobs.db"
    )
    parser.add_argument(
        "--qdrant", action="store_true", help="Reset only Qdrant collections"
    )
    parser.add_argument(
        "--cache", action="store_true", help="Reset only cache directories"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    if not any([args.confirm, args.jobs, args.qdrant, args.cache]):
        parser.print_help()
        print("\n[ERROR] At least one reset option is required.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("AI-Media-Indexer Database Reset")
    print("=" * 60 + "\n")

    # Confirmation prompt
    if not args.yes:
        targets = []
        if args.confirm or args.jobs:
            targets.append("jobs.db")
        if args.confirm or args.qdrant:
            targets.append("Qdrant collections")
        if args.confirm or args.cache:
            targets.append("cache directories")

        print(f"This will DELETE: {', '.join(targets)}")
        response = input("\nType 'yes' to confirm: ")
        if response.lower() != "yes":
            print("[CANCELLED] No changes made.")
            sys.exit(0)

    # Execute resets
    success = True

    if args.confirm or args.jobs:
        success &= reset_jobs_db()

    if args.confirm or args.qdrant:
        success &= reset_qdrant()

    if args.confirm or args.cache:
        success &= reset_cache()

    print("\n" + "=" * 60)
    if success:
        print("[COMPLETE] Database reset successful.")
    else:
        print("[WARNING] Some operations failed. Check output above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
