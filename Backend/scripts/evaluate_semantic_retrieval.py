from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.standard_vector_index import SemanticCandidate, StandardVectorIndex  # noqa: E402

LABELED_QUERIES = [
    ("tiles for lining an irrigation canal", "IS 3367: 1993"),
    ("hand made burnt clay tiles for a terrace", "IS 2690 (Part 2): 1992"),
    ("machine made burnt clay tiles for terracing", "IS 2690 (Part 1): 1993"),
    ("hollow clay filler tiles for a roof", "IS 3951 (Part 1): 1975"),
    ("structural hollow clay tiles for flooring", "IS 3951 (Part 2): 1975"),
    ("clay floor tiles", "IS 1478: 1992"),
    ("Mangalore pattern clay roofing tiles", "IS 654: 1992"),
    ("clay tiles for a roof ridge", "IS 1464: 1992"),
    ("handmade country roofing tiles", "IS 13317: 1992"),
]

AMBIGUOUS_QUERIES = [
    "tiles for roofing",
    "clay tiles",
]


async def main() -> None:
    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    index = StandardVectorIndex(
        qdrant.client,
        BgeM3EmbeddingService(settings.embedding_model),
        settings.qdrant_collection,
    )
    try:
        async with database.session_factory() as session:
            reciprocal_ranks: list[float] = []
            hit_at_1 = 0
            hit_at_3 = 0
            for query, expected in LABELED_QUERIES:
                candidates = await index.semantic_search(session, query, limit=5)
                expected_rank = _expected_rank(candidates, expected)
                reciprocal_ranks.append(1 / expected_rank if expected_rank else 0.0)
                hit_at_1 += int(expected_rank == 1)
                hit_at_3 += int(expected_rank is not None and expected_rank <= 3)
                _print_labeled_result(query, expected, candidates, expected_rank)

            query_count = len(LABELED_QUERIES)
            print("Aggregate:")
            print(f"Queries: {query_count}")
            print(f"Hit@1: {hit_at_1} / {query_count}")
            print(f"Hit@3: {hit_at_3} / {query_count}")
            print(f"MRR: {sum(reciprocal_ranks) / query_count:.3f}")
            print()

            print("Ambiguous inspection queries:")
            for query in AMBIGUOUS_QUERIES:
                candidates = await index.semantic_search(session, query, limit=5)
                print()
                print(f"Query: {query}")
                _print_ranks(candidates, limit=5)
    finally:
        await database.close()
        await qdrant.close()


def _expected_rank(candidates: list[SemanticCandidate], expected: str) -> int | None:
    expected_key = _standard_key(expected)
    for rank, candidate in enumerate(candidates, start=1):
        if _standard_key(candidate.standard_code) == expected_key:
            return rank
    return None


def _standard_key(value: str) -> str:
    return " ".join(value.upper().replace(":", " : ").split())


def _print_labeled_result(
    query: str,
    expected: str,
    candidates: list[SemanticCandidate],
    expected_rank: int | None,
) -> None:
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print()
    _print_ranks(candidates, limit=3)
    print()
    print(f"Expected rank: {expected_rank if expected_rank is not None else 'not found'}")
    print(f"Hit@1: {expected_rank == 1}")
    print(f"Hit@3: {expected_rank is not None and expected_rank <= 3}")
    print()


def _print_ranks(candidates: list[SemanticCandidate], limit: int) -> None:
    for rank in range(1, limit + 1):
        if rank <= len(candidates):
            candidate = candidates[rank - 1]
            print(
                f"Rank {rank}: {candidate.standard_code} | "
                f"{candidate.title} | score={candidate.score:.4f}"
            )
        else:
            print(f"Rank {rank}: <none>")


if __name__ == "__main__":
    asyncio.run(main())
