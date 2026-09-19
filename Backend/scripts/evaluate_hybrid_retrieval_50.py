from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.standards import canonicalize_standard_id  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.bm25_service import Bm25LexicalSearchService  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.hybrid_search_service import HybridCandidate, HybridSearchService  # noqa: E402
from app.services.standard_vector_index import StandardVectorIndex  # noqa: E402
from scripts.evaluate_semantic_retrieval_50 import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DIAGNOSTIC_QUERIES,
    PRIMARY_BENCHMARK,
)

SEMANTIC_BASELINE = {
    "hit_at_1": "35/39",
    "hit_at_3": "39/39",
    "hit_at_5": "39/39",
    "mrr": "0.949",
}


@dataclass(frozen=True)
class HybridEvaluationResult:
    case: object
    candidates: list[HybridCandidate]
    expected_rank: int | None
    expected_candidate: HybridCandidate | None
    latency_ms: float


async def main(dataset_path: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    family_by_code = _family_by_code(dataset)
    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    semantic_index = StandardVectorIndex(
        qdrant.client,
        BgeM3EmbeddingService(settings.embedding_model),
        settings.qdrant_collection,
        dataset_name=str(dataset["dataset_name"]),
    )
    hybrid = HybridSearchService(
        semantic_index,
        Bm25LexicalSearchService(dataset_name=str(dataset["dataset_name"])),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    try:
        async with database.session_factory() as session:
            results: list[HybridEvaluationResult] = []
            print("Hybrid benchmark:")
            for case in PRIMARY_BENCHMARK:
                result = await hybrid.search(session, case.query, limit=5)
                expected_candidate = _expected_candidate(
                    result.candidates,
                    case.expected_standard,
                )
                evaluation_result = HybridEvaluationResult(
                    case=case,
                    candidates=result.candidates,
                    expected_rank=expected_candidate.rank if expected_candidate else None,
                    expected_candidate=expected_candidate,
                    latency_ms=result.timings_ms.total_ms,
                )
                results.append(evaluation_result)
                _print_case(evaluation_result, family_by_code)

            _print_comparison(results)
            _print_aggregate_metrics(results)
            _print_non_hit_at_one(results, family_by_code)
            _print_original_failure_check(results)
            await _run_diagnostics(session, hybrid, family_by_code)
    finally:
        await database.close()
        await qdrant.close()


async def _run_diagnostics(session, hybrid, family_by_code: dict[str, str]) -> None:
    print()
    print("Diagnostic queries (excluded from aggregate metrics):")
    for case in DIAGNOSTIC_QUERIES:
        result = await hybrid.search(session, case.query, limit=5)
        expected_candidate = (
            _expected_candidate(result.candidates, case.expected_standard)
            if case.expected_standard
            else None
        )
        print()
        print(f"Query: {case.query}")
        if case.expected_standard:
            print(f"Expected: {case.expected_standard}")
            print(f"Expected hybrid rank: {_display_rank(expected_candidate)}")
        else:
            print("Expected: exploratory; no forced result")
        print(f"Hybrid latency: {result.timings_ms.total_ms:.1f} ms")
        print(f"Semantic retrieval time: {result.timings_ms.semantic_ms:.1f} ms")
        print(f"BM25 retrieval time: {result.timings_ms.bm25_ms:.1f} ms")
        print(f"RRF fusion time: {result.timings_ms.rrf_ms:.1f} ms")
        _print_candidates(result.candidates, family_by_code)


def _print_case(result: HybridEvaluationResult, family_by_code: dict[str, str]) -> None:
    case = result.case
    print()
    print(f"Query: {case.query}")
    print(f"Expected: {case.expected_standard}")
    print(f"Expected family: {case.family}")
    print(f"Hybrid latency: {result.latency_ms:.1f} ms")
    _print_candidates(result.candidates, family_by_code)
    print(f"Expected hybrid rank: {_display_rank(result.expected_candidate)}")
    print(f"Hit@1: {'PASS' if result.expected_rank == 1 else 'FAIL'}")
    print(
        f"Hit@3: "
        f"{'PASS' if result.expected_rank is not None and result.expected_rank <= 3 else 'FAIL'}"
    )
    print(
        f"Hit@5: "
        f"{'PASS' if result.expected_rank is not None and result.expected_rank <= 5 else 'FAIL'}"
    )


def _print_candidates(candidates: list[HybridCandidate], family_by_code: dict[str, str]) -> None:
    for rank in range(1, 6):
        if rank > len(candidates):
            print(f"Rank {rank}: <none>")
            continue
        candidate = candidates[rank - 1]
        family = family_by_code.get(canonicalize_standard_id(candidate.standard_code), "unknown")
        print(
            f"Rank {rank}: {candidate.standard_code} | {candidate.title} | "
            f"family={family} | rrf_score={candidate.rrf_score:.6f} | "
            f"semantic_rank={_display_optional_int(candidate.semantic_rank)} | "
            f"semantic_score={_display_optional_score(candidate.semantic_score)} | "
            f"bm25_rank={_display_optional_int(candidate.bm25_rank)} | "
            f"bm25_score={_display_optional_score(candidate.bm25_score)}"
        )


def _print_comparison(results: list[HybridEvaluationResult]) -> None:
    metrics = _metrics(results)
    print()
    print("SEMANTIC BASELINE")
    print(f"Hit@1 = {SEMANTIC_BASELINE['hit_at_1']}")
    print(f"Hit@3 = {SEMANTIC_BASELINE['hit_at_3']}")
    print(f"Hit@5 = {SEMANTIC_BASELINE['hit_at_5']}")
    print(f"MRR = {SEMANTIC_BASELINE['mrr']}")
    print()
    print("HYBRID")
    print(f"Hit@1 = {metrics['hit_at_1']}/{metrics['query_count']}")
    print(f"Hit@3 = {metrics['hit_at_3']}/{metrics['query_count']}")
    print(f"Hit@5 = {metrics['hit_at_5']}/{metrics['query_count']}")
    print(f"MRR = {metrics['mrr']:.3f}")


def _print_aggregate_metrics(results: list[HybridEvaluationResult]) -> None:
    print()
    print("Overall hybrid metrics:")
    _print_metrics(results)
    print()
    print("Hybrid metrics by family:")
    by_family: dict[str, list[HybridEvaluationResult]] = defaultdict(list)
    for result in results:
        by_family[result.case.family].append(result)
    for family, family_results in by_family.items():
        print(f"{family}:")
        _print_metrics(family_results)


def _print_metrics(results: list[HybridEvaluationResult]) -> None:
    metrics = _metrics(results)
    print(f"  Queries: {metrics['query_count']}")
    print(f"  Hit@1: {metrics['hit_at_1']}/{metrics['query_count']}")
    print(f"  Hit@3: {metrics['hit_at_3']}/{metrics['query_count']}")
    print(f"  Hit@5: {metrics['hit_at_5']}/{metrics['query_count']}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  Average hybrid latency: {metrics['average_latency_ms']:.1f} ms")


def _metrics(results: list[HybridEvaluationResult]) -> dict[str, float]:
    query_count = len(results)
    hit_at_1 = sum(result.expected_rank == 1 for result in results)
    hit_at_3 = sum(
        result.expected_rank is not None and result.expected_rank <= 3 for result in results
    )
    hit_at_5 = sum(
        result.expected_rank is not None and result.expected_rank <= 5 for result in results
    )
    mrr = sum(1 / result.expected_rank if result.expected_rank else 0 for result in results)
    average_latency_ms = sum(result.latency_ms for result in results) / query_count
    return {
        "query_count": query_count,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": mrr / query_count,
        "average_latency_ms": average_latency_ms,
    }


def _print_non_hit_at_one(
    results: list[HybridEvaluationResult],
    family_by_code: dict[str, str],
) -> None:
    non_hits = [result for result in results if result.expected_rank != 1]
    print()
    print("Failed Hit@1 hybrid queries and cross-domain analysis:")
    if not non_hits:
        print("None")
        return
    for result in non_hits:
        _print_failure(result, family_by_code)


def _print_failure(result: HybridEvaluationResult, family_by_code: dict[str, str]) -> None:
    case = result.case
    rank_one = result.candidates[0] if result.candidates else None
    rank_one_family = (
        family_by_code.get(canonicalize_standard_id(rank_one.standard_code), "unknown")
        if rank_one
        else "none"
    )
    classification = "CROSS_DOMAIN" if rank_one_family != case.family else "WITHIN_FAMILY"
    expected = result.expected_candidate
    print()
    print(f"Classification: {classification}")
    print(f"Query: {case.query}")
    print(f"Expected standard: {case.expected_standard}")
    print(f"Hybrid Rank 1: {rank_one.standard_code if rank_one else '<none>'}")
    print(f"Expected hybrid rank: {_display_rank(expected)}")
    print(f"Semantic rank: {_display_optional_int(expected.semantic_rank if expected else None)}")
    print(f"BM25 rank: {_display_optional_int(expected.bm25_rank if expected else None)}")
    print(f"RRF rank: {_display_rank(expected)}")
    print(
        f"Semantic score: {_display_optional_score(expected.semantic_score if expected else None)}"
    )
    print(f"BM25 score: {_display_optional_score(expected.bm25_score if expected else None)}")
    print(f"Expected family: {case.family}")
    print(f"Rank-1 family: {rank_one_family}")


def _print_original_failure_check(results: list[HybridEvaluationResult]) -> None:
    watched_queries = {
        "clay tiles for a roof ridge",
        "UPVC casing pipe for a borewell",
        "product grade C hexagon head bolt M12",
        "single door non return valve for a water pipeline",
    }
    print()
    print("Original semantic Hit@1 failures:")
    for result in results:
        if result.case.query not in watched_queries:
            continue
        status = "IMPROVED" if result.expected_rank == 1 else "UNCHANGED_OR_WORSE"
        print(
            f"{status}: {result.case.query} | expected={result.case.expected_standard} | "
            f"hybrid_rank={_display_rank(result.expected_candidate)}"
        )


def _expected_candidate(
    candidates: list[HybridCandidate],
    expected_standard: str,
) -> HybridCandidate | None:
    expected_code = canonicalize_standard_id(expected_standard)
    return next(
        (
            candidate
            for candidate in candidates
            if canonicalize_standard_id(candidate.standard_code) == expected_code
        ),
        None,
    )


def _family_by_code(dataset: dict) -> dict[str, str]:
    return {
        canonicalize_standard_id(record["standard_code"]): record["family"]
        for record in dataset["records"]
    }


def _display_rank(candidate: HybridCandidate | None) -> str:
    return str(candidate.rank) if candidate else "not found in Top 5"


def _display_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "not retrieved"


def _display_optional_score(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "not retrieved"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 + RRF hybrid retrieval on the 50-standard pilot."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.dataset))
