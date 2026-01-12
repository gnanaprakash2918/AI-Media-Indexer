"""Search Weight Tuner.

Optimizes the boosting weights for hybrid search by running synthetic queries
and measuring Mean Reciprocal Rank (MRR).

Usage:
    python tools/tune_search.py
"""

import itertools
import os
import sys
from typing import NamedTuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.storage.db import VectorDB
from core.utils.logger import log

# Synthetic Dataset (Query -> Expected Content/Keywords in Result)
# Ideally this should target specific files if known, but for generic tuning
# we check if the top results contain expected keywords in their payload.
DATASET = [
    {
        "query": "Prakash bowling",
        "expected_in_payload": ["bowling", "Prakash"],
        "min_rank": 5,
    },
    {"query": "red car", "expected_in_payload": ["red", "car"], "min_rank": 5},
    {
        "query": "someone eating",
        "expected_in_payload": ["eating", "food"],
        "min_rank": 10,
    },
]


class WeightConfig(NamedTuple):
    face_match: float
    speaker_match: float
    entity_match: float
    text_match: float
    scene_match: float
    action_match: float


def calculate_mrr(results: list, expected_keywords: list[str]) -> float:
    """Calculate Reciprocal Rank for a single query result set."""
    for rank, hit in enumerate(results, 1):
        # Check if hit is relevant
        payload_str = str(hit).lower()
        if all(k.lower() in payload_str for k in expected_keywords):
            return 1.0 / rank
    return 0.0


def main():
    print("🚀 Starting Search Tuner...")

    try:
        db = VectorDB()
    except Exception as e:
        print(f"❌ Failed to connect to DB: {e}")
        print("Ensure Qdrant is running.")
        sys.exit(1)

    # Grid Search Space
    # We explore a few variations around the default
    face_weights = [0.2, 0.4]
    entity_weights = [0.1, 0.2]
    action_weights = [0.05, 0.1]

    # Defaults for others
    base_config = {
        "speaker_match": 0.15,
        "text_match": 0.08,
        "scene_match": 0.08,
    }

    best_mrr = -1.0
    best_config = None

    combinations = list(itertools.product(face_weights, entity_weights, action_weights))
    print(f"🔍 Testing {len(combinations)} configurations...")

    for fw, ew, aw in combinations:
        current_weights = base_config.copy()
        current_weights.update(
            {"face_match": fw, "entity_match": ew, "action_match": aw}
        )

        total_mrr = 0.0

        for case in DATASET:
            query = case["query"]
            expected = case["expected_in_payload"]

            try:
                results = db.search_frames_hybrid(
                    query=query, limit=case["min_rank"], weights=current_weights
                )
                mrr = calculate_mrr(results, expected)
                total_mrr += mrr
            except Exception as e:
                log(f"Search failed for '{query}': {e}")

        avg_mrr = total_mrr / len(DATASET)
        print(f"  Config (F={fw}, E={ew}, A={aw}) -> MRR: {avg_mrr:.4f}")

        if avg_mrr > best_mrr:
            best_mrr = avg_mrr
            best_config = current_weights

    print("\n🏆 Optimization Complete!")
    print(f"Best MRR: {best_mrr:.4f}")
    print("Best Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    # Recommendation
    print("\nRecommended Action:")
    print("Update core/storage/db.py default weights with these values.")


if __name__ == "__main__":
    main()
