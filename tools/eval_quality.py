"""Search Quality Evaluation using LLM-as-a-Judge.

Runs test queries, evaluates results using an LLM, and reports accuracy score.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from llm.factory import LLMFactory

TEST_SET = [
    {
        "query": "person bowling",
        "expected_keywords": ["bowl", "lane", "strike", "ball"],
    },
    {"query": "red car", "expected_keywords": ["car", "red", "vehicle"]},
    {"query": "person speaking", "expected_keywords": ["speak", "talk", "dialogue"]},
    {
        "query": "outdoor scene",
        "expected_keywords": ["outdoor", "outside", "sky", "tree"],
    },
    {
        "query": "group of people",
        "expected_keywords": ["people", "group", "crowd", "together"],
    },
]

JUDGE_PROMPT = """You are a search quality evaluator. Given a search query and the returned result description, rate the relevance on a scale of 0-10.

Query: {query}
Result Description: {description}

Rate relevance 0-10 where:
- 0-2: Completely irrelevant
- 3-4: Weakly related
- 5-6: Moderately relevant
- 7-8: Strongly relevant
- 9-10: Perfect match

Respond with ONLY this JSON format:
{{"score": <number>, "reason": "<brief explanation>"}}
"""


class QualityEvaluator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.llm: Optional[object] = None

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        try:
            resp = await self.client.get(
                f"{self.base_url}/search", params={"q": query, "limit": limit}
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            print(f"  Search error: {e}")
            return []

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = LLMFactory.create_llm(provider=settings.llm_provider)  # type: ignore

    async def judge_result(self, query: str, result: dict) -> dict:
        self._ensure_llm()

        description = (
            result.get("action")
            or result.get("description")
            or result.get("text")
            or ""
        )
        if not description:
            return {"score": 0, "reason": "No description available"}

        prompt = JUDGE_PROMPT.format(query=query, description=description)

        try:
            response = await self.llm.generate(prompt)

            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            judgment = json.loads(response)
            return {
                "score": float(judgment.get("score", 0)),
                "reason": judgment.get("reason", ""),
            }
        except Exception as e:
            keywords = result.get("expected_keywords", [])
            desc_lower = description.lower()
            matches = sum(1 for kw in keywords if kw in desc_lower)
            fallback_score = min(10, matches * 2.5)
            return {
                "score": fallback_score,
                "reason": f"LLM failed ({e}), keyword fallback",
            }

    async def evaluate_test_set(self, test_set: list[dict] = None) -> dict:
        tests = test_set or TEST_SET
        results = []
        total_score = 0.0

        print("\n" + "=" * 60)
        print("SEARCH QUALITY EVALUATION (LLM-as-a-Judge)")
        print("=" * 60)

        for i, test in enumerate(tests, 1):
            query = test["query"]
            print(f'\n[{i}/{len(tests)}] Query: "{query}"')

            search_results = await self.search(query, limit=3)

            if not search_results:
                print("  ❌ No results found")
                results.append(
                    {
                        "query": query,
                        "score": 0,
                        "reason": "No results",
                    }
                )
                continue

            top_result = search_results[0]
            judgment = await self.judge_result(query, top_result)
            score = judgment["score"]
            reason = judgment["reason"]

            emoji = "✅" if score >= 7 else "⚠️" if score >= 4 else "❌"
            print(f"  {emoji} Score: {score}/10 | {reason}")
            print(
                f"     Result: {(top_result.get('action') or top_result.get('description', ''))[:80]}..."
            )

            results.append(
                {
                    "query": query,
                    "score": score,
                    "reason": reason,
                    "result_preview": (
                        top_result.get("action") or top_result.get("description", "")
                    )[:100],
                }
            )
            total_score += score

        avg_score = total_score / len(tests) if tests else 0

        print("\n" + "=" * 60)
        print(f"FINAL ACCURACY SCORE: {avg_score:.1f} / 10")
        print("=" * 60)

        if avg_score >= 8:
            print("🏆 EXCELLENT - Production Ready")
        elif avg_score >= 6:
            print("✅ GOOD - Minor improvements needed")
        elif avg_score >= 4:
            print("⚠️  FAIR - Significant improvements needed")
        else:
            print("❌ POOR - Major overhaul required")

        return {
            "average_score": round(avg_score, 2),
            "total_tests": len(tests),
            "results": results,
        }

    async def close(self):
        await self.client.aclose()


async def main():
    print("\nInitializing Quality Evaluator...")
    evaluator = QualityEvaluator()

    try:
        report = await evaluator.evaluate_test_set()

        output_path = Path(__file__).parent / "quality_report.json"
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📊 Report saved to: {output_path}")

    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
