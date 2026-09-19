from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_embedding_service,
    get_qdrant_client,
    get_query_interpreter,
    get_reranker_service,
    get_session,
    get_settings_from_app,
)
from app.core.config import Settings
from app.schemas.search import (
    HybridSearchCandidate,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchTimings,
    ProductAwareSearchCandidate,
    ProductAwareSearchRequest,
    ProductAwareSearchResponse,
    ProductAwareSearchTimings,
    QueryInterpreterDiagnostics,
    RerankedSearchCandidate,
    RerankedSearchRequest,
    RerankedSearchResponse,
    RerankedSearchTimings,
    SemanticQueryInterpretation,
    SemanticSearchCandidate,
    SemanticSearchRequest,
    SemanticSearchResponse,
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
from app.services.reranker_service import (
    CrossEncoderRerankerService,
    RerankedSearchService,
    RerankerUnavailable,
)
from app.services.standard_vector_index import StandardVectorIndex, VectorIndexUnavailable

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    session: AsyncSession = Depends(get_session),
    qdrant_client: Any = Depends(get_qdrant_client),
    embedding_service: BgeM3EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings_from_app),
) -> SemanticSearchResponse:
    index = StandardVectorIndex(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        collection_name=settings.qdrant_collection,
    )
    try:
        candidates = await index.semantic_search(
            session=session,
            query=request.query,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_SEARCH_QUERY", "message": str(exc)},
        ) from exc
    except VectorIndexUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "SEMANTIC_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc

    return SemanticSearchResponse(
        query=request.query,
        candidates=[
            SemanticSearchCandidate(
                rank=rank,
                standard_id=candidate.standard_id,
                standard_code=candidate.standard_code,
                title=candidate.title,
                score=candidate.score,
            )
            for rank, candidate in enumerate(candidates, start=1)
        ],
    )


@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    session: AsyncSession = Depends(get_session),
    qdrant_client: Any = Depends(get_qdrant_client),
    embedding_service: BgeM3EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings_from_app),
) -> HybridSearchResponse:
    semantic_index = StandardVectorIndex(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        collection_name=settings.qdrant_collection,
    )
    hybrid_service = HybridSearchService(
        semantic_index,
        Bm25LexicalSearchService(),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    try:
        result = await hybrid_service.search(
            session=session,
            query=request.query,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_SEARCH_QUERY", "message": str(exc)},
        ) from exc
    except VectorIndexUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "HYBRID_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc

    return HybridSearchResponse(
        query=result.query,
        candidates=[
            HybridSearchCandidate(
                rank=candidate.rank,
                standard_id=candidate.standard_id,
                standard_code=candidate.standard_code,
                title=candidate.title,
                rrf_score=candidate.rrf_score,
                semantic_rank=candidate.semantic_rank,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
            )
            for candidate in result.candidates
        ],
        timings_ms=HybridSearchTimings(
            semantic_ms=result.timings_ms.semantic_ms,
            bm25_ms=result.timings_ms.bm25_ms,
            rrf_ms=result.timings_ms.rrf_ms,
            total_ms=result.timings_ms.total_ms,
        ),
    )


@router.post("/product-aware", response_model=ProductAwareSearchResponse)
async def product_aware_search(
    request: ProductAwareSearchRequest,
    session: AsyncSession = Depends(get_session),
    qdrant_client: Any = Depends(get_qdrant_client),
    embedding_service: BgeM3EmbeddingService = Depends(get_embedding_service),
    reranker_service: CrossEncoderRerankerService = Depends(get_reranker_service),
    query_interpreter: GeminiQueryInterpreter = Depends(get_query_interpreter),
    settings: Settings = Depends(get_settings_from_app),
    x_query_interpreter_mode: str | None = Header(default=None),
    x_debug_trace: str | None = Header(default=None),
) -> ProductAwareSearchResponse:
    interpreter_mode = x_query_interpreter_mode or settings.query_interpreter_mode
    if interpreter_mode not in {"disabled", "gemini"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_QUERY_INTERPRETER_MODE",
                "message": "X-Query-Interpreter-Mode must be disabled or gemini",
            },
        )
    interpretation = None
    interpreter_diagnostics = QueryInterpreterDiagnostics(mode=interpreter_mode)
    if interpreter_mode == "gemini":
        interpreted = await query_interpreter.interpret(
            product=request.product,
            description=request.description,
        )
        interpretation = interpreted.intent
        interpreter_diagnostics = QueryInterpreterDiagnostics(
            mode=interpreter_mode,
            gemini_used=interpreted.gemini_used,
            gemini_success=interpreted.gemini_success,
            gemini_latency_ms=interpreted.gemini_latency_ms,
            gemini_fallback_reason=interpreted.fallback_reason,
        )
        if interpretation is None:
            interpretation = normalize_intent(
                SemanticQueryIntent(normalized_product=request.product),
                product=request.product,
                description=request.description,
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
    service = ProductAwareSearchService(
        hybrid_service,
        reranker_service,
        rerank_k=settings.rerank_k,
        return_k=(
            max(settings.return_k, 10)
            if _debug_trace_enabled(x_debug_trace)
            else settings.return_k
        ),
        reranker_timeout_s=settings.product_aware_reranker_timeout_s,
    )
    try:
        result = await service.search(
            session=session,
            product=request.product,
            description=request.description,
            limit=min(
                request.limit,
                max(settings.return_k, 10)
                if _debug_trace_enabled(x_debug_trace)
                else settings.return_k,
            ),
            query_intent=interpretation,
            gemini_ms=interpreter_diagnostics.gemini_latency_ms,
            include_debug_trace=_debug_trace_enabled(x_debug_trace),
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
            ProductAwareSearchCandidate(
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
            for candidate in result.candidates
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


def _debug_trace_enabled(header_value: str | None) -> bool:
    return str(header_value or "").casefold() in {"1", "true", "yes", "on"}


@router.post("/reranked", response_model=RerankedSearchResponse)
async def reranked_search(
    request: RerankedSearchRequest,
    session: AsyncSession = Depends(get_session),
    qdrant_client: Any = Depends(get_qdrant_client),
    embedding_service: BgeM3EmbeddingService = Depends(get_embedding_service),
    reranker_service: CrossEncoderRerankerService = Depends(get_reranker_service),
    settings: Settings = Depends(get_settings_from_app),
) -> RerankedSearchResponse:
    semantic_index = StandardVectorIndex(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        collection_name=settings.qdrant_collection,
    )
    hybrid_service = HybridSearchService(
        semantic_index,
        Bm25LexicalSearchService(),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    service = RerankedSearchService(
        hybrid_service,
        reranker_service,
        rerank_k=settings.rerank_k,
        return_k=settings.return_k,
    )
    try:
        result = await service.search(
            session=session,
            query=request.query,
            limit=min(request.limit, settings.return_k),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_SEARCH_QUERY", "message": str(exc)},
        ) from exc
    except VectorIndexUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "RERANKED_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc
    except RerankerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "RERANKED_SEARCH_UNAVAILABLE", "message": str(exc)},
        ) from exc

    return RerankedSearchResponse(
        query=result.query,
        candidates=[
            RerankedSearchCandidate(
                rank=candidate.rank,
                standard_id=candidate.standard_id,
                standard_code=candidate.standard_code,
                title=candidate.title,
                reranker_score=candidate.reranker_score,
                rrf_rank=candidate.rrf_rank,
                rrf_score=candidate.rrf_score,
                semantic_rank=candidate.semantic_rank,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
            )
            for candidate in result.candidates
        ],
        timings_ms=RerankedSearchTimings(
            semantic_ms=result.timings_ms.semantic_ms,
            bm25_ms=result.timings_ms.bm25_ms,
            rrf_ms=result.timings_ms.rrf_ms,
            hybrid_ms=result.timings_ms.hybrid_ms,
            reranker_ms=result.timings_ms.reranker_ms,
            total_ms=result.timings_ms.total_ms,
        ),
    )
