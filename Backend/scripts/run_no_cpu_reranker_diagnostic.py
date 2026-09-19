from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.db.models import Standard  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.bm25_service import Bm25LexicalSearchService  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.gemini_query_interpreter import (  # noqa: E402
    SemanticQueryIntent,
    interpretation_to_dict,
    normalize_intent,
)
from app.services.hybrid_search_service import HybridSearchService  # noqa: E402
from app.services.parsed_standards_corpus import FULL_CORPUS_DATASET_NAME  # noqa: E402
from app.services.product_aware_search_service import (  # noqa: E402
    ProductAwareSearchResult,
    ProductAwareSearchService,
)
from app.services.reranker_service import CrossEncoderRerankerService  # noqa: E402
from app.services.standard_vector_index import StandardVectorIndex  # noqa: E402

BEFORE = Path("data/evaluation/results/gemini_constraint_diagnostic_before.json")
OUTPUT_JSON = Path("data/evaluation/results/gemini_constraint_diagnostic_no_cpu_reranker.json")
OUTPUT_MD = Path("data/evaluation/results/gemini_constraint_diagnostic_no_cpu_reranker.md")

DESIRED_BY_DESCRIPTION = {
    "UPVC soil/waste/rainwater building pipe": ["IS 13592: 1992"],
    "GRP potable-water pipe": ["IS 12709: 1994"],
    "calcium silicate insulation at 600 C": ["IS 8154: 1993"],
    "preformed fibrous insulation for hot-water pipe": ["IS 9842: 1994"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-CPU-reranker Gemini diagnostics.")
    parser.add_argument("--mode", choices=["replay", "http"], default="replay")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--interpretations-from", default=str(BEFORE))
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--report-title", default="Gemini Constraint Diagnostic: No CPU Reranker")
    parser.add_argument("--timeout-s", type=float, default=25.0)
    args = parser.parse_args()

    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    queries = [item["query"] for item in before]
    if args.mode == "replay":
        source = json.loads(Path(args.interpretations_from).read_text(encoding="utf-8"))
        results = asyncio.run(_run_replay(source, embedding_model=args.embedding_model))
    else:
        results = [
            _run_query(args.base_url, query, timeout_s=args.timeout_s)
            for query in queries
        ]
    summary = _summary(results, before)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": args.mode,
        "base_url": args.base_url,
        "interpretations_from": args.interpretations_from if args.mode == "replay" else None,
        "client_timeout_s": args.timeout_s,
        "summary": summary,
        "results": results,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(payload, before, title=args.report_title), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


def _run_query(base_url: str, query: dict[str, str], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps({**query, "limit": 5}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/search/product-aware",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Query-Interpreter-Mode": "gemini",
        },
        method="POST",
    )
    started_at = datetime.now().isoformat()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "ok": True,
            "elapsed_ms": round(elapsed_ms, 3),
            "started_at": started_at,
            "response": json.loads(raw),
        }
    except TimeoutError as exc:
        return _error_result(query, started_at, started, f"timeout: {exc}")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return _error_result(query, started_at, started, f"url_error: {reason}")
    except Exception as exc:  # noqa: BLE001 - diagnostic output should capture failures
        return _error_result(query, started_at, started, f"{exc.__class__.__name__}: {exc}")


def _error_result(
    query: dict[str, str],
    started_at: str,
    started: float,
    error: str,
) -> dict[str, Any]:
    return {
        "query": query,
        "ok": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "started_at": started_at,
        "error": error,
    }


def _summary(results: list[dict[str, Any]], before: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("ok")]
    latencies = [float(item["elapsed_ms"]) for item in completed]
    timeout_count = len(results) - len(completed)
    before_completed = [item for item in before if item.get("ok")]
    before_latencies = [float(item["elapsed_ms"]) for item in before_completed]
    return {
        "completed_query_count": len(completed),
        "timeout_count": timeout_count,
        "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 95), 3) if latencies else None,
        "before_completed_query_count": len(before_completed),
        "before_timeout_count": len(before) - len(before_completed),
        "before_mean_latency_ms": (
            round(statistics.fmean(before_latencies), 3) if before_latencies else None
        ),
        "before_median_latency_ms": (
            round(statistics.median(before_latencies), 3) if before_latencies else None
        ),
        "before_p95_latency_ms": (
            round(_percentile(before_latencies, 95), 3) if before_latencies else None
        ),
        "top1_changed_count": sum(
            _top_code(before_item) != _top_code(result)
            for before_item, result in zip(before, results, strict=True)
        ),
    }


def _markdown(payload: dict[str, Any], before: list[dict[str, Any]], *, title: str) -> str:
    summary = payload["summary"]
    lines = [
        f"# {title}",
        "",
        f"- Base URL: `{payload['base_url']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Interpretations from: `{payload.get('interpretations_from')}`",
        f"- Client timeout: {payload['client_timeout_s']} s",
        f"- Completed queries: {summary['completed_query_count']}/15",
        f"- Timeout count: {summary['timeout_count']}",
        (
            "- Latency: "
            f"mean {summary['mean_latency_ms']} ms, "
            f"median {summary['median_latency_ms']} ms, "
            f"p95 {summary['p95_latency_ms']} ms"
        ),
        (
            "- Before latency: "
            f"mean {summary['before_mean_latency_ms']} ms, "
            f"median {summary['before_median_latency_ms']} ms, "
            f"p95 {summary['before_p95_latency_ms']} ms"
        ),
        f"- Top-1 changed vs before: {summary['top1_changed_count']}/15",
        "",
    ]
    paired_results = zip(before, payload["results"], strict=True)
    for index, (before_item, item) in enumerate(paired_results, start=1):
        query = item["query"]
        lines.extend([f"## {index}. {query['product']} | {query['description']}", ""])
        if not item.get("ok"):
            lines.extend(
                [
                    f"- ERROR: {item.get('error')}",
                    f"- Latency: {item.get('elapsed_ms')} ms",
                    "",
                ]
            )
            continue
        response = item["response"]
        timings = response.get("timings_ms") or {}
        interpretation = response.get("query_interpretation") or {}
        lines.extend(
            [
                f"- Top 1: {_top_code(item)}",
                f"- Top 3: {', '.join(_top_codes(item, 3))}",
                f"- Desired rank: {response.get('desired_rank')}",
                f"- Before Top 1: {_top_code(before_item)}",
                f"- Before Top 3: {', '.join(_top_codes(before_item, 3))}",
                f"- Gemini interpretation: {_interpretation_summary(interpretation)}",
                f"- Constraint flags: {_flags_summary(response)}",
                (
                    f"- Ambiguity: {response.get('ambiguity')}; "
                    f"missing={response.get('missing_information')}"
                ),
                (
                    "- Reranker: "
                    f"used={timings.get('reranker_used')} "
                    f"success={timings.get('reranker_success')} "
                    f"device={timings.get('reranker_device')} "
                    f"fallback={timings.get('reranker_fallback_reason')} "
                    f"inference_ms={timings.get('reranker_inference_ms')}"
                ),
                f"- Total latency: {item.get('elapsed_ms')} ms",
                "",
            ]
        )
    return "\n".join(lines)


def _interpretation_summary(interpretation: dict[str, Any]) -> str:
    fields = [
        ("product", interpretation.get("normalized_product")),
        ("subtype", interpretation.get("subtype") or interpretation.get("query_subtype")),
        ("material", interpretation.get("material")),
        ("function", interpretation.get("function")),
        ("application", interpretation.get("application")),
        ("excluded_function", interpretation.get("excluded_function")),
        ("excluded_application", interpretation.get("excluded_application")),
        ("grade", interpretation.get("grade")),
        ("temperature_c", interpretation.get("temperature_c")),
    ]
    return "; ".join(f"{name}={value}" for name, value in fields if value not in (None, [], ""))


async def _run_replay(
    source: list[dict[str, Any]],
    *,
    embedding_model: str | None,
) -> list[dict[str, Any]]:
    settings_kwargs = {"reranker_mode": "auto"}
    if embedding_model:
        settings_kwargs["embedding_model"] = embedding_model
    settings = Settings(**settings_kwargs)
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    embedding_service = BgeM3EmbeddingService(settings.embedding_model)
    semantic_index = StandardVectorIndex(
        qdrant_client=qdrant.client,
        embedding_service=embedding_service,
        collection_name=settings.qdrant_full_collection,
        dataset_name=FULL_CORPUS_DATASET_NAME,
    )
    hybrid_service = HybridSearchService(
        semantic_index,
        Bm25LexicalSearchService(dataset_name=FULL_CORPUS_DATASET_NAME),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    reranker = CrossEncoderRerankerService(
        settings.reranker_model,
        batch_size=settings.rerank_batch_size,
        reranker_mode=settings.reranker_mode,
    )
    service = ProductAwareSearchService(
        hybrid_service,
        reranker,
        rerank_k=settings.rerank_k,
        return_k=settings.return_k,
        reranker_timeout_s=settings.product_aware_reranker_timeout_s,
    )
    try:
        results = []
        for item in source:
            query = item["query"]
            interpretation = (item.get("response") or {}).get("query_interpretation")
            if not interpretation:
                results.append(
                    {
                        "query": query,
                        "ok": False,
                        "elapsed_ms": 0.0,
                        "started_at": datetime.now().isoformat(),
                        "error": "missing_preserved_query_interpretation",
                    }
                )
                continue
            intent = normalize_intent(
                SemanticQueryIntent.model_validate(interpretation),
                product=query["product"],
                description=query["description"],
            )
            started_at = datetime.now().isoformat()
            started = time.perf_counter()
            try:
                async with database.session_factory() as session:
                    result = await service.search(
                        session=session,
                        product=query["product"],
                        description=query["description"],
                        limit=settings.return_k,
                        query_intent=intent,
                        gemini_ms=0.0,
                    )
                    standards_by_id = await _candidate_standards_by_id(session, result)
                results.append(
                    {
                        "query": query,
                        "ok": True,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                        "started_at": started_at,
                        "response": _response_payload(result, intent, standards_by_id),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic output should capture failures
                results.append(
                    _error_result(
                        query,
                        started_at,
                        started,
                        f"{exc.__class__.__name__}: {exc}",
                    )
                )
        return results
    finally:
        await qdrant.close()
        await database.close()


def _response_payload(
    result: ProductAwareSearchResult,
    intent: SemanticQueryIntent,
    standards_by_id: dict[str, Standard],
) -> dict[str, Any]:
    desired_codes = DESIRED_BY_DESCRIPTION.get(result.description, [])
    candidate_codes = [candidate.standard_code for candidate in result.candidates]
    return {
        "product": result.product,
        "description": result.description,
        "query": result.query,
        "canonical_product": result.canonical_product,
        "product_confidence": result.product_confidence,
        "query_interpretation": interpretation_to_dict(intent),
        "query_interpreter": {
            "mode": "gemini_replay",
            "gemini_used": True,
            "gemini_success": True,
            "gemini_latency_ms": 0.0,
            "gemini_fallback_reason": "replayed_from_preserved_diagnostic",
        },
        "ambiguity": result.ambiguity,
        "missing_information": result.missing_information,
        "desired_standard_codes": desired_codes,
        "desired_rank": _desired_rank(candidate_codes, desired_codes),
        "candidates": [
            {
                "rank": candidate.rank,
                "standard_id": candidate.standard_id,
                "standard_code": candidate.standard_code,
                "title": candidate.title,
                "reranker_score": candidate.reranker_score,
                "rrf_rank": candidate.rrf_rank,
                "rrf_score": candidate.rrf_score,
                "product_compatibility": candidate.product_compatibility,
                "canonical_product": candidate.canonical_product,
                "standard_kind": candidate.standard_kind,
                "family": candidate.family,
                "constraint_score": candidate.constraint_score,
                "constraint_flags": candidate.constraint_flags or [],
                "raw_cross_encoder_score": candidate.raw_cross_encoder_score,
                "reranker_score_source": candidate.reranker_score_source,
                "final_score": candidate.final_score,
                "semantic_rank": candidate.semantic_rank,
                "semantic_score": candidate.semantic_score,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "trace": _candidate_trace(candidate, standards_by_id.get(candidate.standard_id)),
            }
            for candidate in result.candidates
        ],
        "timings_ms": {
            "gemini_ms": result.timings_ms.gemini_ms,
            "embedding_ms": result.timings_ms.embedding_ms,
            "semantic_retrieval_ms": result.timings_ms.semantic_retrieval_ms,
            "semantic_ms": result.timings_ms.semantic_ms,
            "bm25_ms": result.timings_ms.bm25_ms,
            "rrf_ms": result.timings_ms.rrf_ms,
            "hybrid_ms": result.timings_ms.hybrid_ms,
            "product_gate_ms": result.timings_ms.product_gate_ms,
            "constraint_ms": result.timings_ms.constraint_ms,
            "reranker_ms": result.timings_ms.reranker_ms,
            "reranker_device": result.timings_ms.reranker_device,
            "reranker_model_loaded": result.timings_ms.reranker_model_loaded,
            "reranker_used": result.timings_ms.reranker_used,
            "reranker_timeout": result.timings_ms.reranker_timeout,
            "reranker_success": result.timings_ms.reranker_success,
            "reranker_fallback_reason": result.timings_ms.reranker_fallback_reason,
            "reranker_inference_ms": result.timings_ms.reranker_inference_ms,
            "postprocess_ms": result.timings_ms.postprocess_ms,
            "total_ms": result.timings_ms.total_ms,
        },
    }


async def _candidate_standards_by_id(
    session: Any,
    result: ProductAwareSearchResult,
) -> dict[str, Standard]:
    standard_ids = [candidate.standard_id for candidate in result.candidates]
    if not standard_ids:
        return {}
    rows = await session.execute(select(Standard).where(Standard.id.in_(standard_ids)))
    return {str(standard.id): standard for standard in rows.scalars().all()}


def _candidate_trace(candidate: Any, standard: Standard | None) -> dict[str, Any]:
    flags = candidate.constraint_flags or []
    return {
        "standard_code": candidate.standard_code,
        "canonical_product": standard.canonical_product if standard else None,
        "family": standard.family if standard else None,
        "product_subtype": standard.product_subtype if standard else None,
        "material": standard.material if standard else None,
        "function": standard.function if standard else None,
        "application": standard.application if standard else None,
        "grade": _metadata_value(standard, "grade"),
        "temperature": _metadata_value(standard, "temperature"),
        "semantic_rank": candidate.semantic_rank,
        "semantic_score": candidate.semantic_score,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
        "rrf_rank": candidate.rrf_rank,
        "rrf_score": candidate.rrf_score,
        "product_compatibility": candidate.product_compatibility,
        "product_state": _flag_state(flags, "PRODUCT"),
        "material_state": _flag_state(flags, "MATERIAL"),
        "subtype_state": _flag_state(flags, "SUBTYPE"),
        "function_state": _flag_state(flags, "FUNCTION"),
        "application_state": _flag_state(flags, "APPLICATION"),
        "grade_state": _flag_state(flags, "GRADE"),
        "temperature_state": _flag_state(flags, "TEMPERATURE"),
        "constraint_score": candidate.constraint_score,
        "final_score": candidate.final_score,
        "final_rank": candidate.rank,
    }


def _metadata_value(standard: Standard | None, key: str) -> Any:
    if standard is None:
        return None
    profile = standard.search_profile or {}
    return profile.get(key)


def _flag_state(flags: list[str], prefix: str) -> str | None:
    token = f"{prefix}_"
    for flag in flags:
        if flag.startswith(token):
            return flag.removeprefix(token)
    return None


def _desired_rank(candidate_codes: list[str], desired_codes: list[str]) -> int | None:
    for rank, code in enumerate(candidate_codes, start=1):
        if code in desired_codes:
            return rank
    return None


def _flags_summary(item: dict[str, Any]) -> str:
    parts = []
    for candidate in (item.get("candidates") or [])[:3]:
        flags = ",".join(candidate.get("constraint_flags") or [])
        parts.append(f"{candidate.get('standard_code')}=[{flags}]")
    return "; ".join(parts)


def _top_code(item: dict[str, Any]) -> str | None:
    candidates = ((item.get("response") or {}).get("candidates") or [])
    if not candidates:
        return None
    return candidates[0].get("standard_code")


def _top_codes(item: dict[str, Any], count: int) -> list[str]:
    candidates = ((item.get("response") or {}).get("candidates") or [])
    return [candidate.get("standard_code") for candidate in candidates[:count]]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    main()
