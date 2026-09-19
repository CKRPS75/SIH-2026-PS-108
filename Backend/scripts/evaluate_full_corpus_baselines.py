from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.standards import canonicalize_standard_id  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.bm25_service import Bm25LexicalSearchService  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.hybrid_search_service import HybridSearchService  # noqa: E402
from app.services.parsed_standards_corpus import (  # noqa: E402
    FULL_CORPUS_DATASET_NAME,
)
from app.services.product_aware_search_service import ProductAwareSearchService  # noqa: E402
from app.services.reranker_service import CrossEncoderRerankerService  # noqa: E402
from app.services.standard_vector_index import StandardVectorIndex  # noqa: E402
from scripts.evaluate_semantic_retrieval_50 import PRIMARY_BENCHMARK  # noqa: E402

RESULTS_DIR = Path("data/evaluation/results")
FULL_EVAL_PATH = Path("data/evaluation/full_corpus_eval.jsonl")
PRODUCT_HOLDOUT_PATH = Path("data/evaluation/product_aware_holdout.jsonl")
KNOWN_FLY_ASH_QUERY = "Portland pozzolana cement made using fly ash"
KNOWN_FLY_ASH_EXPECTED = "IS 1489 (Part 1): 1991"


@dataclass(frozen=True)
class EvalCase:
    query_id: str
    query: str
    expected_standard: str
    product: str | None = None
    description: str | None = None
    family: str | None = None
    standard_kind: str | None = None


async def main_async() -> None:
    args = _parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    semantic_index = StandardVectorIndex(
        qdrant.client,
        BgeM3EmbeddingService(settings.embedding_model),
        args.collection,
        dataset_name=FULL_CORPUS_DATASET_NAME,
    )
    bm25 = Bm25LexicalSearchService(dataset_name=FULL_CORPUS_DATASET_NAME)
    hybrid = HybridSearchService(
        semantic_index,
        bm25,
        semantic_top_k=args.pool_k,
        bm25_top_k=args.pool_k,
        rrf_k=settings.rrf_k,
    )
    reranker = CrossEncoderRerankerService(
        args.reranker_model,
        batch_size=settings.rerank_batch_size,
    )
    try:
        async with database.session_factory() as session:
            old_cases = _old_regression_cases()
            full_cases = _load_full_eval(FULL_EVAL_PATH) if FULL_EVAL_PATH.exists() else []
            product_holdout = (
                _load_product_holdout(PRODUCT_HOLDOUT_PATH)
                if PRODUCT_HOLDOUT_PATH.exists()
                else []
            )
            if args.only_product_aware:
                product_results = await _evaluate_product_diagnostics(
                    session,
                    hybrid,
                    reranker,
                    product_holdout,
                    rerank_k=max(args.rerank_k),
                )
                _write_json(
                    RESULTS_DIR / "full_corpus_product_aware_pretrained.json",
                    product_results,
                )
                print(
                    "Product-aware holdout summary: "
                    f"{_format_metrics(product_results['metrics'])} "
                    f"cross_product_errors={product_results['cross_product_errors']}"
                )
                return
            if args.only_full_eval:
                await _run_full_eval_only(
                    session,
                    semantic_index,
                    bm25,
                    hybrid,
                    reranker,
                    full_cases,
                    args,
                )
                return

            semantic_results = await _evaluate_semantic(session, semantic_index, old_cases)
            _write_json(
                RESULTS_DIR / "full_corpus_semantic_regression.json",
                semantic_results,
            )
            bm25_results = await _evaluate_bm25(session, bm25, old_cases)
            _write_json(RESULTS_DIR / "full_corpus_bm25_regression.json", bm25_results)
            hybrid_results = await _evaluate_hybrid(session, hybrid, old_cases, limit=20)
            _write_json(RESULTS_DIR / "full_corpus_hybrid_regression.json", hybrid_results)

            reranker_results_by_k = {}
            for rerank_k in args.rerank_k:
                reranker_results = await _evaluate_reranker(
                    session,
                    hybrid,
                    reranker,
                    old_cases,
                    rerank_k=rerank_k,
                )
                reranker_results_by_k[rerank_k] = reranker_results
                _write_json(
                    RESULTS_DIR / f"full_corpus_pretrained_reranker_k{rerank_k}.json",
                    reranker_results,
                )

            product_results = await _evaluate_product_diagnostics(
                session,
                hybrid,
                reranker,
                product_holdout,
                rerank_k=max(args.rerank_k),
            )
            _write_json(
                RESULTS_DIR / "full_corpus_product_aware_pretrained.json",
                product_results,
            )
            _write_error_analysis(
                RESULTS_DIR / "full_corpus_error_analysis.csv",
                old_cases,
                semantic_results,
                bm25_results,
                hybrid_results,
                reranker_results_by_k[max(args.rerank_k)],
            )

            if full_cases and not args.skip_full_eval:
                await _run_full_eval_only(
                    session,
                    semantic_index,
                    bm25,
                    hybrid,
                    reranker,
                    full_cases,
                    args,
                )

            _print_summary(semantic_results, bm25_results, hybrid_results, reranker_results_by_k)
    finally:
        await database.close()
        await qdrant.close()


async def _evaluate_semantic(session, semantic_index, cases: list[EvalCase]) -> dict[str, Any]:
    results = []
    timings = []
    for case in cases:
        started = perf_counter()
        candidates = await semantic_index.semantic_search(session, case.query, limit=20)
        timings.append((perf_counter() - started) * 1000)
        rows = [
            {
                "rank": rank,
                "standard_code": candidate.standard_code,
                "title": candidate.title,
                "score": candidate.score,
            }
            for rank, candidate in enumerate(candidates, start=1)
        ]
        results.append(_case_result(case, rows, "rank"))
    return _result_payload("semantic", results, timings)


async def _evaluate_bm25(session, bm25, cases: list[EvalCase]) -> dict[str, Any]:
    results = []
    timings = []
    for case in cases:
        started = perf_counter()
        candidates = await bm25.search(session, case.query, limit=20)
        timings.append((perf_counter() - started) * 1000)
        rows = [
            {
                "rank": candidate.bm25_rank,
                "standard_code": candidate.standard_code,
                "title": candidate.title,
                "bm25_score": candidate.bm25_score,
            }
            for candidate in candidates
        ]
        results.append(_case_result(case, rows, "rank"))
    return _result_payload("bm25", results, timings)


async def _evaluate_hybrid(session, hybrid, cases: list[EvalCase], *, limit: int) -> dict[str, Any]:
    results = []
    timings = []
    for case in cases:
        result = await hybrid.search(session, case.query, limit=limit)
        timings.append(result.timings_ms.total_ms)
        rows = [
            {
                "rank": candidate.rank,
                "standard_code": candidate.standard_code,
                "title": candidate.title,
                "rrf_score": candidate.rrf_score,
                "semantic_rank": candidate.semantic_rank,
                "semantic_score": candidate.semantic_score,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
            }
            for candidate in result.candidates
        ]
        results.append(_case_result(case, rows, "rank"))
    return _result_payload("hybrid", results, timings)


async def _evaluate_reranker(
    session,
    hybrid,
    reranker,
    cases: list[EvalCase],
    *,
    rerank_k: int,
) -> dict[str, Any]:
    results = []
    reranker_timings = []
    total_timings = []
    warmup_done = False
    for case in cases:
        started = perf_counter()
        hybrid_result = await hybrid.search(session, case.query, limit=rerank_k)
        reranker_started = perf_counter()
        candidates = await reranker.rerank(
            session,
            case.query,
            hybrid_result.candidates,
            limit=rerank_k,
        )
        reranker_ms = (perf_counter() - reranker_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        if warmup_done:
            reranker_timings.append(reranker_ms)
            total_timings.append(total_ms)
        warmup_done = True
        rows = [
            {
                "rank": candidate.rank,
                "standard_code": candidate.standard_code,
                "title": candidate.title,
                "reranker_score": candidate.reranker_score,
                "rrf_rank": candidate.rrf_rank,
                "rrf_score": candidate.rrf_score,
                "semantic_rank": candidate.semantic_rank,
                "semantic_score": candidate.semantic_score,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
            }
            for candidate in candidates
        ]
        case_result = _case_result(case, rows, "rank")
        case_result["candidate_recall_pool"] = _expected_rank(
            [
                {"standard_code": candidate.standard_code, "rank": candidate.rank}
                for candidate in hybrid_result.candidates
            ],
            case.expected_standard,
            "rank",
        )
        results.append(case_result)
    payload = _result_payload("pretrained_reranker", results, reranker_timings)
    payload["rerank_k"] = rerank_k
    payload["total_latency_ms"] = _latency_summary(total_timings)
    payload["candidate_recall"] = {
        f"at_{rerank_k}": sum(r["candidate_recall_pool"] is not None for r in results)
        / len(results)
        if results
        else 0.0
    }
    return payload


async def _evaluate_product_diagnostics(
    session,
    hybrid,
    reranker,
    product_holdout: list[EvalCase],
    *,
    rerank_k: int,
) -> dict[str, Any]:
    diagnostics = [
        EvalCase(
            query_id="product_diag_valve_steel_pipeline",
            query="",
            product="valve",
            description="for a steel water pipeline",
            expected_standard="",
        ),
        EvalCase(
            query_id="product_diag_valve_reverse_flow",
            query="",
            product="valve",
            description="installed in a steel water pipeline to prevent reverse flow",
            expected_standard="",
        ),
        EvalCase(
            query_id="product_diag_cement_rapid_strength",
            query="",
            product="cement",
            description="required where rapid early strength is needed",
            expected_standard="",
        ),
        EvalCase(
            query_id="product_diag_grp_pipe_potable",
            query="",
            product="GRP pipe",
            description="required to carry potable water",
            expected_standard="",
        ),
        EvalCase(
            query_id="product_diag_thermal_pipe",
            query="",
            product="thermal insulation",
            description="required around a high-temperature process pipe",
            expected_standard="",
        ),
        EvalCase(
            query_id="product_diag_bolt_structural",
            query="",
            product="bolt",
            description="high-strength connection for structural steel work",
            expected_standard="",
        ),
    ]
    cases = diagnostics + product_holdout
    service = ProductAwareSearchService(hybrid, reranker, rerank_k=rerank_k, return_k=5)
    rows = []
    timings = []
    for case in cases:
        started = perf_counter()
        result = await service.search(
            session,
            product=case.product or "",
            description=case.description or "",
            limit=5,
        )
        timings.append((perf_counter() - started) * 1000)
        candidates = [asdict(candidate) for candidate in result.candidates]
        rows.append(
            {
                "query_id": case.query_id,
                "product": case.product,
                "description": case.description,
                "expected_standard": case.expected_standard,
                "canonical_product": result.canonical_product,
                "product_confidence": result.product_confidence,
                "top1_standard": candidates[0]["standard_code"] if candidates else None,
                "top1_compatibility": (
                    candidates[0]["product_compatibility"] if candidates else None
                ),
                "expected_rank": _expected_rank(
                    candidates,
                    case.expected_standard,
                    "rank",
                )
                if case.expected_standard
                else None,
                "candidates": candidates,
            }
        )
    cross_product_errors = sum(
        row["top1_compatibility"] == "incompatible" for row in rows if row["canonical_product"]
    )
    return {
        "method": "product_aware_pretrained",
        "query_count": len(rows),
        "labeled_query_count": sum(bool(row["expected_standard"]) for row in rows),
        "metrics": _metrics(
            [
                {
                    "expected_rank": row["expected_rank"],
                }
                for row in rows
                if row["expected_standard"]
            ]
        ),
        "cross_product_errors": cross_product_errors,
        "latency_ms": _latency_summary(timings[1:]),
        "results": rows,
    }


async def _run_full_eval_only(
    session,
    semantic_index,
    bm25,
    hybrid,
    reranker,
    full_cases: list[EvalCase],
    args: argparse.Namespace,
) -> None:
    if not full_cases:
        print("No full-corpus evaluation file found; skipping full eval.")
        return
    full_eval_results = {
        "semantic": await _evaluate_semantic(session, semantic_index, full_cases),
        "bm25": await _evaluate_bm25(session, bm25, full_cases),
        "hybrid": await _evaluate_hybrid(session, hybrid, full_cases, limit=20),
        "pretrained_reranker": await _evaluate_reranker(
            session,
            hybrid,
            reranker,
            full_cases,
            rerank_k=max(args.rerank_k),
        ),
    }
    _write_json(
        RESULTS_DIR / "full_corpus_eval_baselines.json",
        full_eval_results,
    )
    print("Full-corpus frozen 200-case evaluation summary")
    for label, result in full_eval_results.items():
        print(f"{label}: {_format_metrics(result['metrics'])}")


def _case_result(case: EvalCase, candidates: list[dict[str, Any]], rank_key: str) -> dict[str, Any]:
    expected_rank = _expected_rank(candidates, case.expected_standard, rank_key)
    return {
        **asdict(case),
        "expected_rank": expected_rank,
        "top1_standard": candidates[0]["standard_code"] if candidates else None,
        "candidates": candidates,
    }


def _result_payload(
    method: str,
    results: list[dict[str, Any]],
    timings: list[float],
) -> dict[str, Any]:
    return {
        "method": method,
        "query_count": len(results),
        "metrics": _metrics(results),
        "latency_ms": _latency_summary(timings),
        "results": results,
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    ranks = [result["expected_rank"] for result in results]
    query_count = len(results)
    return {
        "hit_at_1": sum(rank == 1 for rank in ranks),
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
        "hit_at_10": sum(rank is not None and rank <= 10 for rank in ranks),
        "candidate_recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks)
        / query_count,
        "candidate_recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks)
        / query_count,
        "candidate_recall_at_20": sum(rank is not None and rank <= 20 for rank in ranks)
        / query_count,
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / query_count,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": _percentile(ordered, 95),
    }


def _percentile(ordered: list[float], percentile: int) -> float:
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _expected_rank(candidates: list[dict[str, Any]], expected: str, rank_key: str) -> int | None:
    if not expected:
        return None
    expected_code = canonicalize_standard_id(expected)
    for candidate in candidates:
        if canonicalize_standard_id(str(candidate["standard_code"])) == expected_code:
            return int(candidate[rank_key])
    return None


def _write_error_analysis(
    path: Path,
    cases: list[EvalCase],
    semantic: dict[str, Any],
    bm25: dict[str, Any],
    hybrid: dict[str, Any],
    reranked: dict[str, Any],
) -> None:
    by_query = {
        "semantic": {row["query_id"]: row for row in semantic["results"]},
        "bm25": {row["query_id"]: row for row in bm25["results"]},
        "hybrid": {row["query_id"]: row for row in hybrid["results"]},
        "reranked": {row["query_id"]: row for row in reranked["results"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query/product",
                "description",
                "expected_standard",
                "semantic_rank",
                "bm25_rank",
                "rrf_rank",
                "reranked_rank",
                "error_type",
                "top1_standard",
                "notes",
            ],
        )
        writer.writeheader()
        for case in cases:
            sem = by_query["semantic"][case.query_id]
            bm = by_query["bm25"][case.query_id]
            hy = by_query["hybrid"][case.query_id]
            rr = by_query["reranked"][case.query_id]
            writer.writerow(
                {
                    "query_id": case.query_id,
                    "query/product": case.query,
                    "description": "",
                    "expected_standard": case.expected_standard,
                    "semantic_rank": sem["expected_rank"],
                    "bm25_rank": bm["expected_rank"],
                    "rrf_rank": hy["expected_rank"],
                    "reranked_rank": rr["expected_rank"],
                    "error_type": _classify_error(case, hy, rr),
                    "top1_standard": rr["top1_standard"],
                    "notes": "",
                }
            )


def _classify_error(
    case: EvalCase,
    hybrid_row: dict[str, Any],
    reranked_row: dict[str, Any],
) -> str:
    if reranked_row["expected_rank"] == 1:
        return "PASS"
    if hybrid_row["expected_rank"] is None:
        return "RETRIEVAL_MISS"
    if "Part" in case.expected_standard and reranked_row["top1_standard"]:
        return "PART_OR_VARIANT_CONFUSION"
    return "RERANKER_ERROR"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _old_regression_cases() -> list[EvalCase]:
    return [
        EvalCase(
            query_id=f"old39_{index:02d}",
            query=case.query,
            expected_standard=case.expected_standard,
            family=case.family,
        )
        for index, case in enumerate(PRIMARY_BENCHMARK, start=1)
    ]


def _load_full_eval(path: Path) -> list[EvalCase]:
    return [
        EvalCase(
            query_id=record["query_id"],
            query=record["query"],
            expected_standard=record["expected_standard"],
            product=record.get("expected_product"),
            family=record.get("family"),
            standard_kind=record.get("standard_kind"),
        )
        for record in _read_jsonl(path)
    ]


def _load_product_holdout(path: Path) -> list[EvalCase]:
    return [
        EvalCase(
            query_id=record["query_id"],
            query="",
            expected_standard=record.get("expected_standard") or "",
            product=record["product"],
            description=record["description"],
            family=record.get("family"),
            standard_kind=record.get("standard_kind"),
        )
        for record in _read_jsonl(path)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _print_summary(
    semantic: dict[str, Any],
    bm25: dict[str, Any],
    hybrid: dict[str, Any],
    reranker_by_k: dict[int, dict[str, Any]],
) -> None:
    print("Full-corpus old 39-query regression summary")
    for label, result in [("Semantic", semantic), ("BM25", bm25), ("Hybrid", hybrid)]:
        print(f"{label}: {_format_metrics(result['metrics'])}")
    for rerank_k, result in reranker_by_k.items():
        print(
            f"Pretrained reranker K={rerank_k}: {_format_metrics(result['metrics'])} "
            f"reranker_ms={result['latency_ms']}"
        )
    fly_ash = next(
        row
        for row in reranker_by_k[max(reranker_by_k)]["results"]
        if row["query"] == KNOWN_FLY_ASH_QUERY
    )
    print(
        "Known fly-ash PPC reranker rank: "
        f"{fly_ash['expected_rank']} expected={KNOWN_FLY_ASH_EXPECTED}"
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"Hit@1={metrics.get('hit_at_1', 0)} | "
        f"Hit@3={metrics.get('hit_at_3', 0)} | "
        f"Hit@5={metrics.get('hit_at_5', 0)} | "
        f"Hit@10={metrics.get('hit_at_10', 0)} | "
        f"MRR={metrics.get('mrr', 0):.3f} | "
        f"Recall@20={metrics.get('candidate_recall_at_20', 0):.3f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-corpus baselines.")
    parser.add_argument("--collection", default="standardwise_standards_full")
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--reranker-model", default="models/bge-reranker-large")
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--only-full-eval", action="store_true")
    parser.add_argument("--only-product-aware", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
