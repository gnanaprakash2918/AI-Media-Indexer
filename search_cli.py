import sys
from pathlib import Path

from core.retrieval.search import SearchEngine
from core.storage.db import VectorDB


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python search_cli.py "your search query"')
        sys.exit(1)

    query = sys.argv[1]

    db = VectorDB(backend="docker")
    engine = SearchEngine(db)

    try:
        results = engine.search(query)

        print("\n--- VISUAL MATCHES ---")
        if not results["visual_matches"]:
            print("No visual matches found.")

        for r in results["visual_matches"]:
            # Extract just the filename from the full path for cleaner display
            filename = Path(r["file"]).name if r.get("file") else "Unknown"
            print(f"[{r['score']}] {filename} ({r['time']}) - {r['content'][:80]}...")

        print("\n--- DIALOGUE MATCHES ---")
        if not results["dialogue_matches"]:
            print("No dialogue matches found.")

        for r in results["dialogue_matches"]:
            filename = Path(r["file"]).name if r.get("file") else "Unknown"
            print(f"[{r['score']}] {filename} ({r['time']}) - {r['content'][:80]}...")

    finally:
        db.close()


if __name__ == "__main__":
    main()
