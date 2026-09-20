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
    ProductAwareSearchRequest,
    ProductAwareSearchResponse,
    RerankedSearchCandidate,
    RerankedSearchRequest,
    RerankedSearchResponse,
    RerankedSearchTimings,
    SemanticSearchCandidate,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services.bm25_service import Bm25LexicalSearchService
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.gemini_query_interpreter import (
    GeminiQueryInterpreter,
)
from app.services.hybrid_search_service import HybridSearchService
from app.services.product_aware_orchestrator import run_product_aware_search
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
    return await run_product_aware_search(
        session=session,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        reranker_service=reranker_service,
        query_interpreter=query_interpreter,
        settings=settings,
        product=request.product,
        description=request.description,
        limit=request.limit,
        interpreter_mode=x_query_interpreter_mode,
        include_debug_trace=_debug_trace_enabled(x_debug_trace),
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
