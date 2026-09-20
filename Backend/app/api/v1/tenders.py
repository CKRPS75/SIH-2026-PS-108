from __future__ import annotations

from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_embedding_service,
    get_qdrant_client,
    get_query_interpreter,
    get_reranker_service,
    get_session,
    get_settings_from_app,
    get_tender_extractor,
)
from app.core.config import Settings
from app.db.models import Standard
from app.schemas.tenders import (
    AttributeEvidence,
    ExtractedTenderItem,
    IdentifiedTenderProduct,
    TechnicalStandardDetail,
    TenderAnalysisDiagnostics,
    TenderAnalyzedItem,
    TenderRecommendation,
    TenderRecommendationRanking,
    TenderSearchPayload,
    TenderTextAnalysisRequest,
    TenderTextAnalysisResponse,
)
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.gemini_query_interpreter import GeminiQueryInterpreter
from app.services.gemini_tender_extractor import GeminiTenderExtractor
from app.services.product_aware_orchestrator import run_product_aware_search
from app.services.reranker_service import CrossEncoderRerankerService
from app.services.standard_detail_enrichment import (
    build_readable_technical_details,
    build_source_metadata,
    build_technical_standard_detail,
)

router = APIRouter(prefix="/api/v1/tenders", tags=["tenders"])


@router.post("/analyze-text", response_model=TenderTextAnalysisResponse)
async def analyze_tender_text(
    request: TenderTextAnalysisRequest,
    session: AsyncSession = Depends(get_session),
    qdrant_client=Depends(get_qdrant_client),
    embedding_service: BgeM3EmbeddingService = Depends(get_embedding_service),
    reranker_service: CrossEncoderRerankerService = Depends(get_reranker_service),
    query_interpreter: GeminiQueryInterpreter = Depends(get_query_interpreter),
    tender_extractor: GeminiTenderExtractor = Depends(get_tender_extractor),
    settings: Settings = Depends(get_settings_from_app),
) -> TenderTextAnalysisResponse:
    started_at = perf_counter()
    extraction = await tender_extractor.extract(request.text)
    warnings = list(extraction.warnings)
    analyzed_items: list[TenderAnalyzedItem] = []

    for extracted in extraction.items:
        payload = _search_payload(extracted, request.limit)
        item_warnings = []
        try:
            search_response = await run_product_aware_search(
                session=session,
                qdrant_client=qdrant_client,
                embedding_service=embedding_service,
                reranker_service=reranker_service,
                query_interpreter=query_interpreter,
                settings=settings,
                product=payload.product,
                description=payload.description,
                limit=payload.limit,
                interpreter_mode=settings.query_interpreter_mode,
            )
            details_by_id = await _standard_details_by_id(
                session,
                [candidate.standard_id for candidate in search_response.candidates],
            )
            recommendations = [
                _recommendation(candidate, details_by_id.get(candidate.standard_id))
                for candidate in search_response.candidates
            ]
            timings = search_response.timings_ms
        except Exception as exc:  # noqa: BLE001 - per-item diagnostics, not silent failure
            item_warnings.append(f"search_failed: {exc.__class__.__name__}: {str(exc)[:160]}")
            recommendations = []
            timings = None

        analyzed_items.append(
            TenderAnalyzedItem(
                item_no=extracted.item_no,
                raw_text=extracted.raw_text,
                identified_product=_identified_product(extracted),
                structured_requirement=extracted,
                search_payload=payload,
                recommendations=recommendations,
                search_timings_ms=timings,
                warnings=item_warnings,
            )
        )

    if not extraction.items:
        warnings.append("No structured tender items were available for search")
    has_recommendations = any(item.recommendations for item in analyzed_items)
    status = "ok" if analyzed_items and has_recommendations else "degraded"
    if not extraction.items:
        status = "extraction_failed"

    return TenderTextAnalysisResponse(
        text_length=len(request.text),
        original_tender_text=request.text,
        item_count=len(extraction.items),
        limit=request.limit,
        items=analyzed_items,
        diagnostics=TenderAnalysisDiagnostics(
            status=status,
            warnings=warnings,
            extraction=extraction,
            total_latency_ms=(perf_counter() - started_at) * 1000,
        ),
    )


def _search_payload(item: ExtractedTenderItem, limit: int) -> TenderSearchPayload:
    product = item.normalized_product or item.product.strip() or "procurement item"
    description = item.normalized_description.strip() or item.raw_text.strip()
    return TenderSearchPayload(product=product, description=description, limit=limit)


async def _standard_details_by_id(
    session: AsyncSession,
    standard_ids: list[str],
) -> dict[str, Standard]:
    ids = []
    for standard_id in standard_ids:
        try:
            ids.append(UUID(standard_id))
        except ValueError:
            continue
    if not ids:
        return {}
    result = await session.execute(select(Standard).where(Standard.id.in_(ids)))
    return {str(standard.id): standard for standard in result.scalars().all()}


def _recommendation(candidate, standard: Standard | None) -> TenderRecommendation:
    technical_detail = _technical_detail(standard) if standard else None
    return TenderRecommendation(
        rank=candidate.rank,
        standard_code=candidate.standard_code,
        standard_id=candidate.standard_id,
        title=candidate.title,
        rrf_rank=candidate.rrf_rank,
        rrf_score=candidate.rrf_score,
        semantic_rank=candidate.semantic_rank,
        semantic_score=candidate.semantic_score,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
        product_compatibility=candidate.product_compatibility,
        constraint_score=candidate.constraint_score,
        constraint_flags=candidate.constraint_flags or [],
        final_score=candidate.final_score,
        raw_cross_encoder_score=candidate.raw_cross_encoder_score,
        reranker_score=candidate.reranker_score,
        reranker_score_source=candidate.reranker_score_source,
        match_summary=_match_summary(candidate),
        technical_detail=technical_detail,
        technical_details=build_readable_technical_details(standard) if standard else None,
        ranking=TenderRecommendationRanking(
            final_score=candidate.final_score,
            rrf_rank=candidate.rrf_rank,
            rrf_score=candidate.rrf_score,
            semantic_rank=candidate.semantic_rank,
            semantic_score=candidate.semantic_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            product_compatibility=candidate.product_compatibility,
            constraint_score=candidate.constraint_score,
            constraint_flags=candidate.constraint_flags or [],
            raw_cross_encoder_score=candidate.raw_cross_encoder_score,
            reranker_score=candidate.reranker_score,
            reranker_score_source=candidate.reranker_score_source,
        ),
        source=build_source_metadata(standard) if standard else None,
    )


def _match_summary(candidate) -> list[str]:
    summary = [f"product_compatibility={candidate.product_compatibility}"]
    if candidate.constraint_score is not None:
        summary.append(f"constraint_score={candidate.constraint_score:g}")
    for flag in candidate.constraint_flags[:6]:
        summary.append(flag)
    if candidate.reranker_score_source:
        summary.append(f"score_source={candidate.reranker_score_source}")
    return summary


def _technical_detail(standard: Standard) -> TechnicalStandardDetail:
    return build_technical_standard_detail(standard)


def _identified_product(item: ExtractedTenderItem) -> IdentifiedTenderProduct:
    facts: list[AttributeEvidence] = []
    for field, values in [
        ("material", item.material),
        ("application", item.application),
        ("function", item.function),
        ("installation_context", item.installation_context),
        ("technical_requirement", item.technical_requirements),
    ]:
        facts.extend(
            AttributeEvidence(field=field, value=value, evidence=item.raw_text)
            for value in values[:4]
        )
    facts.extend(item.attribute_evidence[:8])
    return IdentifiedTenderProduct(
        display_name=item.product,
        canonical_product=item.normalized_product,
        material=item.material,
        application=item.application,
        function=item.function,
        installation_context=item.installation_context,
        important_technical_attributes=facts[:12],
    )
