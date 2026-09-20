from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.schemas.search import (
    ProductAwareSearchCandidate,
    ProductAwareSearchResponse,
    ProductAwareSearchTimings,
    QueryInterpreterDiagnostics,
    SemanticQueryInterpretation,
)
from app.services.bm25_service import Bm25LexicalSearchService
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.gemini_query_interpreter import (
    GeminiQueryInterpreter,
    SemanticQueryIntent,
    interpretation_to_dict,
    normalize_intent,
)
from app.services.hybrid_search_service import HybridSearchService
from app.services.parsed_standards_corpus import FULL_CORPUS_DATASET_NAME
from app.services.product_aware_search_service import ProductAwareSearchService
from app.services.reranker_service import CrossEncoderRerankerService, RerankerUnavailable
from app.services.standard_vector_index import StandardVectorIndex, VectorIndexUnavailable


async def run_product_aware_search(
    *,
    session: AsyncSession,
    qdrant_client: Any,
    embedding_service: BgeM3EmbeddingService,
    reranker_service: CrossEncoderRerankerService,
    query_interpreter: GeminiQueryInterpreter,
    settings: Settings,
    product: str,
    description: str,
    limit: int,
    interpreter_mode: str | None = None,
    include_debug_trace: bool = False,
) -> ProductAwareSearchResponse:
    resolved_mode = interpreter_mode or settings.query_interpreter_mode
    if resolved_mode not in {"disabled", "gemini"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_QUERY_INTERPRETER_MODE",
                "message": "X-Query-Interpreter-Mode must be disabled or gemini",
            },
        )

    interpretation = None
    interpreter_diagnostics = QueryInterpreterDiagnostics(mode=resolved_mode)
    if resolved_mode == "gemini":
        interpreted = await query_interpreter.interpret(
            product=product,
            description=description,
        )
        interpretation = interpreted.intent
        interpreter_diagnostics = QueryInterpreterDiagnostics(
            mode=resolved_mode,
            gemini_used=interpreted.gemini_used,
            gemini_success=interpreted.gemini_success,
            gemini_latency_ms=interpreted.gemini_latency_ms,
            gemini_fallback_reason=interpreted.fallback_reason,
        )
        if interpretation is None:
            interpretation = normalize_intent(
                SemanticQueryIntent(normalized_product=product),
                product=product,
                description=description,
            )

    semantic_index = StandardVectorIndex(
        qdrant_client=qdrant_client,
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
    return_k = max(settings.return_k, 10) if include_debug_trace else settings.return_k
    service = ProductAwareSearchService(
        hybrid_service,
        reranker_service,
        rerank_k=settings.rerank_k,
        return_k=return_k,
        reranker_timeout_s=settings.product_aware_reranker_timeout_s,
    )
    try:
        result = await service.search(
            session=session,
            product=product,
            description=description,
            limit=min(limit, return_k),
            query_intent=interpretation,
            gemini_ms=interpreter_diagnostics.gemini_latency_ms,
            include_debug_trace=include_debug_trace,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_SEARCH_QUERY", "message": str(exc)},
        ) from exc
    except VectorIndexUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "PRODUCT_AWARE_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc
    except RerankerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "PRODUCT_AWARE_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc

    return ProductAwareSearchResponse(
        product=result.product,
        description=result.description,
        query=result.query,
        canonical_product=result.canonical_product,
        product_confidence=result.product_confidence,
        query_interpretation=(
            SemanticQueryInterpretation(**interpretation_to_dict(result.query_interpretation))
            if result.query_interpretation
            else None
        ),
        query_interpreter=interpreter_diagnostics,
        ambiguity=result.ambiguity,
        missing_information=result.missing_information,
        candidates=[
            _candidate_to_schema(candidate) for candidate in result.candidates
        ],
        timings_ms=ProductAwareSearchTimings(
            gemini_ms=result.timings_ms.gemini_ms,
            embedding_ms=result.timings_ms.embedding_ms,
            semantic_retrieval_ms=result.timings_ms.semantic_retrieval_ms,
            semantic_ms=result.timings_ms.semantic_ms,
            bm25_ms=result.timings_ms.bm25_ms,
            rrf_ms=result.timings_ms.rrf_ms,
            hybrid_ms=result.timings_ms.hybrid_ms,
            product_gate_ms=result.timings_ms.product_gate_ms,
            constraint_ms=result.timings_ms.constraint_ms,
            reranker_ms=result.timings_ms.reranker_ms,
            reranker_device=result.timings_ms.reranker_device,
            reranker_model_loaded=result.timings_ms.reranker_model_loaded,
            reranker_used=result.timings_ms.reranker_used,
            reranker_timeout=result.timings_ms.reranker_timeout,
            reranker_success=result.timings_ms.reranker_success,
            reranker_fallback_reason=result.timings_ms.reranker_fallback_reason,
            reranker_inference_ms=result.timings_ms.reranker_inference_ms,
            postprocess_ms=result.timings_ms.postprocess_ms,
            total_ms=result.timings_ms.total_ms,
        ),
        debug_trace=result.debug_trace,
    )


def _candidate_to_schema(candidate) -> ProductAwareSearchCandidate:
    return ProductAwareSearchCandidate(
        rank=candidate.rank,
        standard_id=candidate.standard_id,
        standard_code=candidate.standard_code,
        title=candidate.title,
        reranker_score=candidate.reranker_score,
        rrf_rank=candidate.rrf_rank,
        rrf_score=candidate.rrf_score,
        product_compatibility=candidate.product_compatibility,
        canonical_product=candidate.canonical_product,
        standard_kind=candidate.standard_kind,
        family=candidate.family,
        constraint_score=candidate.constraint_score,
        constraint_flags=candidate.constraint_flags or [],
        raw_cross_encoder_score=candidate.raw_cross_encoder_score,
        reranker_score_source=candidate.reranker_score_source,
        final_score=candidate.final_score,
        semantic_rank=candidate.semantic_rank,
        semantic_score=candidate.semantic_score,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
    )
