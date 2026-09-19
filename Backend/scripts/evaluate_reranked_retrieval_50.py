from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

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
from app.services.reranker_service import (  # noqa: E402
    CrossEncoderRerankerService,
    RerankedCandidate,
)
from app.services.standard_vector_index import StandardVectorIndex  # noqa: E402
from scripts.evaluate_semantic_retrieval_50 import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    PRIMARY_BENCHMARK,
    _validate_benchmark,
)

RERANK_K_VALUES = [20, 10, 5]
RETURN_K = 5

SEMANTIC_BASELINE = {
    "hit_at_1": "35/39",
    "hit_at_3": "39/39",
    "hit_at_5": "39/39",
    "mrr": "0.949",
}
HYBRID_BASELINE = {
    "hit_at_1": "36/39",
    "hit_at_3": "39/39",
    "hit_at_5": "39/39",
    "mrr": "0.953",
}


@dataclass(frozen=True)
class LabeledCase:
    family: str
    query: str
    expected_standard: str


@dataclass(frozen=True)
class DiagnosticCase:
    query: str


@dataclass(frozen=True)
class CaseTimings:
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    hybrid_ms: float
    reranker_ms: float
    total_ms: float


@dataclass(frozen=True)
class EvaluationResult:
    case: object
    rerank_k: int
    hybrid_candidates: list[HybridCandidate]
    candidates: list[RerankedCandidate]
    expected_rrf_candidate: HybridCandidate | None
    expected_reranked_candidate: RerankedCandidate | None
    timings_ms: CaseTimings

    @property
    def expected_rank(self) -> int | None:
        return self.expected_reranked_candidate.rank if self.expected_reranked_candidate else None

    @property
    def expected_rrf_rank(self) -> int | None:
        return self.expected_rrf_candidate.rank if self.expected_rrf_candidate else None

    @property
    def expected_in_rrf_pool(self) -> bool:
        return self.expected_rrf_candidate is not None


HOLDOUT_CASES = [
    LabeledCase(
        "pipe_water_drainage",
        "I need UPVC fittings for drainage pipes in a building.",
        "IS 14735: 1999",
    ),
    LabeledCase(
        "thermal_insulation",
        "I need insulation for hot pipes.",
        "IS 9842: 1994",
    ),
    LabeledCase(
        "cement",
        "I need cement that gains strength quickly.",
        "IS 8041: 1990",
    ),
    LabeledCase("fastener", "I need washers for bolts.", "IS 2016: 1967"),
    LabeledCase(
        "water_valve",
        "I need a stop valve for a water pipeline.",
        "IS 9338: 1984",
    ),
]

UNSEEN_HOLDOUT_CASES = [
    LabeledCase(
        "pipe_water_drainage",
        "I need GRP pipes for water supply.",
        "IS 12709: 1994",
    ),
    LabeledCase(
        "water_valve",
        "I need a valve to reduce water pressure at home.",
        "IS 9739: 1981",
    ),
    LabeledCase(
        "cement",
        "I need white cement for finishing work.",
        "IS 8042: 1989",
    ),
    LabeledCase(
        "fastener",
        "I need strong bolts for a steel structure.",
        "IS 3757: 1985",
    ),
    LabeledCase(
        "thermal_insulation",
        "I need foam insulation for thermal protection.",
        "IS 12436: 1988",
    ),
]

AMBIGUOUS_DIAGNOSTIC_QUERIES = [
    DiagnosticCase("I need pipes for water supply."),
    DiagnosticCase("I need cement for finishing work."),
    DiagnosticCase("I need cement for leakage repair."),
]


async def main(dataset_path: Path, rerank_k_values: list[int]) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    family_by_code = _family_by_code(dataset)
    _validate_benchmark(family_by_code)

    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    hybrid = HybridSearchService(
        StandardVectorIndex(
            qdrant.client,
            BgeM3EmbeddingService(settings.embedding_model),
            settings.qdrant_collection,
            dataset_name=str(dataset["dataset_name"]),
        ),
        Bm25LexicalSearchService(dataset_name=str(dataset["dataset_name"])),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    reranker = CrossEncoderRerankerService(
        settings.reranker_model,
        batch_size=settings.rerank_batch_size,
    )

    try:
        async with database.session_factory() as session:
            print("Reranker latency optimization benchmark")
            print(f"Dataset: {dataset_path}")
            print(f"Reranker model: {settings.reranker_model}")
            print(f"Rerank batch size: {settings.rerank_batch_size}")
            print(f"Return K: {RETURN_K}")
            print()
            _print_frozen_baselines()

            warmup = await _evaluate_case(
                session,
                hybrid,
                reranker,
                PRIMARY_BENCHMARK[0],
                rerank_k=rerank_k_values[0],
            )
            print()
            print("Warm-up request, excluded from metrics:")
            _print_single_latency(warmup)

            results_by_config: dict[int, list[EvaluationResult]] = {}
            for rerank_k in rerank_k_values:
                print()
                print(f"Running main 39-query benchmark for RERANK_K={rerank_k}...")
                results = [
                    await _evaluate_case(
                        session,
                        hybrid,
                        reranker,
                        case,
                        rerank_k=rerank_k,
                    )
                    for case in PRIMARY_BENCHMARK
                ]
                results_by_config[rerank_k] = results

            _print_summary_table(results_by_config)
            for rerank_k, results in results_by_config.items():
                _print_accuracy_safeguards(rerank_k, results, family_by_code)

            recommended_rerank_k = _recommend_config(results_by_config)
            print()
            print(
                "Recommended production configuration: "
                f"RERANK_K={recommended_rerank_k}, "
                f"RETURN_K={RETURN_K}, "
                f"RERANK_BATCH_SIZE={settings.rerank_batch_size}"
            )

            await _run_labeled_section(
                "Frozen natural holdout queries, evaluation-only",
                session,
                hybrid,
                reranker,
                HOLDOUT_CASES,
                recommended_rerank_k,
                family_by_code,
            )
            await _run_labeled_section(
                "Newer unseen holdout checks, evaluation-only",
                session,
                hybrid,
                reranker,
                UNSEEN_HOLDOUT_CASES,
                recommended_rerank_k,
                family_by_code,
            )
            await _run_ambiguous_diagnostics(
                session,
                hybrid,
                reranker,
                recommended_rerank_k,
                family_by_code,
            )
    finally:
        await database.close()
        await qdrant.close()


async def _evaluate_case(
    session,
    hybrid: HybridSearchService,
    reranker: CrossEncoderRerankerService,
    case: object,
    *,
    rerank_k: int,
) -> EvaluationResult:
    started_at = perf_counter()
    hybrid_result = await hybrid.search(session, case.query, limit=rerank_k)
    reranker_started_at = perf_counter()
    candidates = await reranker.rerank(
        session,
        case.query,
        hybrid_result.candidates,
        limit=RETURN_K,
    )
    reranker_ms = (perf_counter() - reranker_started_at) * 1000
    timings = hybrid_result.timings_ms
    expected_rrf_candidate = _expected_hybrid_candidate(
        hybrid_result.candidates,
        case.expected_standard,
    )
    expected_reranked_candidate = _expected_reranked_candidate(
        candidates,
        case.expected_standard,
    )
    return EvaluationResult(
        case=case,
        rerank_k=rerank_k,
        hybrid_candidates=hybrid_result.candidates,
        candidates=candidates,
        expected_rrf_candidate=expected_rrf_candidate,
        expected_reranked_candidate=expected_reranked_candidate,
        timings_ms=CaseTimings(
            semantic_ms=timings.semantic_ms,
            bm25_ms=timings.bm25_ms,
            rrf_ms=timings.rrf_ms,
            hybrid_ms=timings.total_ms,
            reranker_ms=reranker_ms,
            total_ms=(perf_counter() - started_at) * 1000,
        ),
    )


async def _run_labeled_section(
    title: str,
    session,
    hybrid: HybridSearchService,
    reranker: CrossEncoderRerankerService,
    cases: list[LabeledCase],
    rerank_k: int,
    family_by_code: dict[str, str],
) -> None:
    print()
    print(f"{title} using RERANK_K={rerank_k}:")
    results = [
        await _evaluate_case(session, hybrid, reranker, case, rerank_k=rerank_k) for case in cases
    ]
    _print_metrics(_metrics(results))
    for result in results:
        _print_case(result, family_by_code)


async def _run_ambiguous_diagnostics(
    session,
    hybrid: HybridSearchService,
    reranker: CrossEncoderRerankerService,
    rerank_k: int,
    family_by_code: dict[str, str],
) -> None:
    print()
    print(f"General ambiguous diagnostics using RERANK_K={rerank_k}, excluded from metrics:")
    for case in AMBIGUOUS_DIAGNOSTIC_QUERIES:
        hybrid_result = await hybrid.search(session, case.query, limit=rerank_k)
        candidates = await reranker.rerank(
            session,
            case.query,
            hybrid_result.candidates,
            limit=RETURN_K,
        )
        print()
        print(f"Query: {case.query}")
        _print_candidates(candidates, family_by_code)


def _print_frozen_baselines() -> None:
    print("Frozen baselines, not recomputed here:")
    print(
        "Semantic: "
        f"Hit@1={SEMANTIC_BASELINE['hit_at_1']} | "
        f"Hit@3={SEMANTIC_BASELINE['hit_at_3']} | "
        f"Hit@5={SEMANTIC_BASELINE['hit_at_5']} | "
        f"MRR={SEMANTIC_BASELINE['mrr']}"
    )
    print(
        "Hybrid: "
        f"Hit@1={HYBRID_BASELINE['hit_at_1']} | "
        f"Hit@3={HYBRID_BASELINE['hit_at_3']} | "
        f"Hit@5={HYBRID_BASELINE['hit_at_5']} | "
        f"MRR={HYBRID_BASELINE['mrr']}"
    )


def _print_summary_table(results_by_config: dict[int, list[EvaluationResult]]) -> None:
    print()
    print("Reranker configuration summary:")
    print("Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Avg rerank ms | Median | P95")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for rerank_k, results in results_by_config.items():
        metrics = _metrics(results)
        print(
            f"RERANK_K={rerank_k} | "
            f"{metrics['hit_at_1']}/{metrics['query_count']} | "
            f"{metrics['hit_at_3']}/{metrics['query_count']} | "
            f"{metrics['hit_at_5']}/{metrics['query_count']} | "
            f"{metrics['mrr']:.3f} | "
            f"{metrics['average_reranker_ms']:.1f} | "
            f"{metrics['median_reranker_ms']:.1f} | "
            f"{metrics['p95_reranker_ms']:.1f}"
        )
        print(
            "Latency averages: "
            f"semantic={metrics['average_semantic_ms']:.1f} ms, "
            f"bm25={metrics['average_bm25_ms']:.1f} ms, "
            f"rrf={metrics['average_rrf_ms']:.1f} ms, "
            f"hybrid={metrics['average_hybrid_ms']:.1f} ms, "
            f"total={metrics['average_total_ms']:.1f} ms"
        )


def _print_accuracy_safeguards(
    rerank_k: int,
    results: list[EvaluationResult],
    family_by_code: dict[str, str],
) -> None:
    print()
    print(f"Accuracy safeguard details for RERANK_K={rerank_k}:")
    failures = [result for result in results if result.expected_rank != 1]
    if not failures:
        print("All expected standards were Rank 1.")
        return
    for result in failures:
        case = result.case
        classification = _failure_classification(result)
        print()
        print(f"Classification: {classification}")
        print(f"Query: {case.query}")
        print(f"Expected standard: {case.expected_standard}")
        print(f"Expected reranked rank: {_display_rank(result.expected_rank)}")
        print(f"Expected present in RRF candidate pool: {result.expected_in_rrf_pool}")
        print(f"Expected RRF rank: {_display_rank(result.expected_rrf_rank)}")
        print(f"Reranker rank change: {_rank_change(result)}")
        _print_candidates(result.candidates, family_by_code)


def _print_case(result: EvaluationResult, family_by_code: dict[str, str]) -> None:
    case = result.case
    print()
    print(f"Query: {case.query}")
    print(f"Expected: {case.expected_standard}")
    print(f"Expected family: {case.family}")
    print(f"Expected reranked rank: {_display_rank(result.expected_rank)}")
    print(f"Expected present in RRF candidate pool: {result.expected_in_rrf_pool}")
    print(f"Expected RRF rank: {_display_rank(result.expected_rrf_rank)}")
    print(f"Reranker rank change: {_rank_change(result)}")
    _print_candidates(result.candidates, family_by_code)


def _print_candidates(candidates: list[RerankedCandidate], family_by_code: dict[str, str]) -> None:
    for rank in range(1, RETURN_K + 1):
        if rank > len(candidates):
            print(f"Rank {rank}: <none>")
            continue
        candidate = candidates[rank - 1]
        family = family_by_code.get(canonicalize_standard_id(candidate.standard_code), "unknown")
        print(
            f"Rank {rank}: {candidate.standard_code} | {candidate.title} | "
            f"family={family} | reranker_score={candidate.reranker_score:.4f} | "
            f"rrf_rank={candidate.rrf_rank} | rrf_score={candidate.rrf_score:.6f} | "
            f"semantic_rank={_display_optional_int(candidate.semantic_rank)} | "
            f"semantic_score={_display_optional_score(candidate.semantic_score)} | "
            f"bm25_rank={_display_optional_int(candidate.bm25_rank)} | "
            f"bm25_score={_display_optional_score(candidate.bm25_score)}"
        )


def _print_single_latency(result: EvaluationResult) -> None:
    timings = result.timings_ms
    print(f"Query: {result.case.query}")
    print(f"semantic={timings.semantic_ms:.1f} ms")
    print(f"bm25={timings.bm25_ms:.1f} ms")
    print(f"rrf={timings.rrf_ms:.1f} ms")
    print(f"hybrid={timings.hybrid_ms:.1f} ms")
    print(f"reranker={timings.reranker_ms:.1f} ms")
    print(f"total={timings.total_ms:.1f} ms")


def _print_metrics(metrics: dict[str, float]) -> None:
    print(f"  Queries: {metrics['query_count']}")
    print(f"  Hit@1: {metrics['hit_at_1']}/{metrics['query_count']}")
    print(f"  Hit@3: {metrics['hit_at_3']}/{metrics['query_count']}")
    print(f"  Hit@5: {metrics['hit_at_5']}/{metrics['query_count']}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  Average semantic latency: {metrics['average_semantic_ms']:.1f} ms")
    print(f"  Average BM25 latency: {metrics['average_bm25_ms']:.1f} ms")
    print(f"  Average RRF latency: {metrics['average_rrf_ms']:.1f} ms")
    print(f"  Average hybrid latency: {metrics['average_hybrid_ms']:.1f} ms")
    print(f"  Average reranker latency: {metrics['average_reranker_ms']:.1f} ms")
    print(f"  Average total latency: {metrics['average_total_ms']:.1f} ms")
    print(f"  Median reranker latency: {metrics['median_reranker_ms']:.1f} ms")
    print(f"  P95 reranker latency: {metrics['p95_reranker_ms']:.1f} ms")


def _metrics(results: list[EvaluationResult]) -> dict[str, float]:
    query_count = len(results)
    hit_at_1 = sum(result.expected_rank == 1 for result in results)
    hit_at_3 = sum(
        result.expected_rank is not None and result.expected_rank <= 3 for result in results
    )
    hit_at_5 = sum(
        result.expected_rank is not None and result.expected_rank <= 5 for result in results
    )
    mrr = sum(1 / result.expected_rank if result.expected_rank else 0 for result in results)
    reranker_latencies = [result.timings_ms.reranker_ms for result in results]
    return {
        "query_count": query_count,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": mrr / query_count,
        "average_semantic_ms": _average(result.timings_ms.semantic_ms for result in results),
        "average_bm25_ms": _average(result.timings_ms.bm25_ms for result in results),
        "average_rrf_ms": _average(result.timings_ms.rrf_ms for result in results),
        "average_hybrid_ms": _average(result.timings_ms.hybrid_ms for result in results),
        "average_reranker_ms": _average(reranker_latencies),
        "average_total_ms": _average(result.timings_ms.total_ms for result in results),
        "median_reranker_ms": median(reranker_latencies),
        "p95_reranker_ms": _percentile(reranker_latencies, 95),
    }


def _recommend_config(results_by_config: dict[int, list[EvaluationResult]]) -> int:
    metric_by_config = {
        rerank_k: _metrics(results) for rerank_k, results in results_by_config.items()
    }
    best_hit_at_3 = max(metrics["hit_at_3"] for metrics in metric_by_config.values())
    best_mrr = max(metrics["mrr"] for metrics in metric_by_config.values())
    viable = [
        rerank_k
        for rerank_k, metrics in metric_by_config.items()
        if metrics["hit_at_3"] == best_hit_at_3 and metrics["mrr"] >= best_mrr - 0.01
    ]
    return min(viable)


def _failure_classification(result: EvaluationResult) -> str:
    if not result.expected_in_rrf_pool:
        return "CANDIDATE_RECALL_FAILURE"
    if result.expected_rank != 1:
        return "RERANKING_FAILURE"
    return "PASS"


def _rank_change(result: EvaluationResult) -> str:
    rrf_rank = result.expected_rrf_rank
    reranked_rank = result.expected_rank
    if rrf_rank is None:
        return "not applicable; absent from RRF pool"
    if reranked_rank is None:
        return "WORSENED; dropped from returned Top 5"
    if reranked_rank < rrf_rank:
        return f"IMPROVED from RRF rank {rrf_rank} to reranked rank {reranked_rank}"
    if reranked_rank > rrf_rank:
        return f"WORSENED from RRF rank {rrf_rank} to reranked rank {reranked_rank}"
    return f"UNCHANGED at rank {reranked_rank}"


def _expected_hybrid_candidate(
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


def _expected_reranked_candidate(
    candidates: list[RerankedCandidate],
    expected_standard: str,
) -> RerankedCandidate | None:
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


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _display_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "not found"


def _display_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "not retrieved"


def _display_optional_score(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "not retrieved"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained cross-encoder reranking pool sizes."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--rerank-k",
        type=int,
        nargs="*",
        default=RERANK_K_VALUES,
        help="Rerank candidate pool sizes to benchmark.",
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.dataset, arguments.rerank_k))
