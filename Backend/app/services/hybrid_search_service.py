from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Standard
from app.services.bm25_service import Bm25Candidate, Bm25LexicalSearchService
from app.services.standard_vector_index import SemanticCandidate, StandardVectorIndex

RRF_K = 60


class SemanticSearchService(Protocol):
    async def semantic_search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
    ) -> list[SemanticCandidate]: ...


@dataclass(frozen=True)
class HybridCandidate:
    rank: int
    standard_id: str
    standard_code: str
    title: str
    rrf_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


@dataclass(frozen=True)
class HybridTimings:
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    total_ms: float
    embedding_ms: float | None = None
    semantic_retrieval_ms: float | None = None


@dataclass(frozen=True)
class HybridSearchResult:
    query: str
    candidates: list[HybridCandidate]
    timings_ms: HybridTimings
    semantic_candidates: Sequence[SemanticCandidate] = ()
    bm25_candidates: Sequence[Bm25Candidate] = ()


class HybridSearchService:
    def __init__(
        self,
        semantic_service: SemanticSearchService | StandardVectorIndex,
        bm25_service: Bm25LexicalSearchService,
        *,
        semantic_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = RRF_K,
    ) -> None:
        self._semantic_service = semantic_service
        self._bm25_service = bm25_service
        self._semantic_top_k = semantic_top_k
        self._bm25_top_k = bm25_top_k
        self._rrf_k = rrf_k

    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
    ) -> HybridSearchResult:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")

        started_at = perf_counter()
        semantic_started_at = perf_counter()
        semantic_candidates = await self._semantic_service.semantic_search(
            session,
            query,
            limit=self._semantic_top_k,
        )
        semantic_ms = (perf_counter() - semantic_started_at) * 1000
        semantic_timings = getattr(self._semantic_service, "last_search_timings_ms", {})

        bm25_started_at = perf_counter()
        bm25_candidates = await self._bm25_service.search(
            session,
            query,
            limit=self._bm25_top_k,
        )
        bm25_ms = (perf_counter() - bm25_started_at) * 1000

        rrf_started_at = perf_counter()
        fused_candidates = await reciprocal_rank_fusion(
            session,
            semantic_candidates,
            bm25_candidates,
            limit=limit,
            k=self._rrf_k,
        )
        rrf_ms = (perf_counter() - rrf_started_at) * 1000

        return HybridSearchResult(
            query=query,
            candidates=fused_candidates,
            timings_ms=HybridTimings(
                semantic_ms=semantic_ms,
                bm25_ms=bm25_ms,
                rrf_ms=rrf_ms,
                total_ms=(perf_counter() - started_at) * 1000,
                embedding_ms=semantic_timings.get("embedding_ms"),
                semantic_retrieval_ms=semantic_timings.get("semantic_retrieval_ms"),
            ),
            semantic_candidates=semantic_candidates,
            bm25_candidates=bm25_candidates,
        )


async def reciprocal_rank_fusion(
    session: AsyncSession,
    semantic_candidates: list[SemanticCandidate],
    bm25_candidates: list[Bm25Candidate],
    *,
    limit: int,
    k: int = RRF_K,
) -> list[HybridCandidate]:
    semantic_by_id = {
        candidate.standard_id: (rank, candidate)
        for rank, candidate in enumerate(semantic_candidates, start=1)
    }
    bm25_by_id = {candidate.standard_id: candidate for candidate in bm25_candidates}
    candidate_ids = set(semantic_by_id) | set(bm25_by_id)
    standards_by_id = await _standards_by_id(session, candidate_ids)

    fused = []
    for standard_id in candidate_ids:
        semantic_rank, semantic_candidate = semantic_by_id.get(standard_id, (None, None))
        bm25_candidate = bm25_by_id.get(standard_id)
        bm25_rank = bm25_candidate.bm25_rank if bm25_candidate else None
        rrf_score = _rrf_score(k, semantic_rank, bm25_rank)
        standard = standards_by_id.get(standard_id)
        if standard is None:
            continue
        fused.append(
            HybridCandidate(
                rank=0,
                standard_id=standard_id,
                standard_code=standard.standard_id,
                title=standard.title,
                rrf_score=rrf_score,
                semantic_rank=semantic_rank,
                semantic_score=semantic_candidate.score if semantic_candidate else None,
                bm25_rank=bm25_rank,
                bm25_score=bm25_candidate.bm25_score if bm25_candidate else None,
            )
        )

    fused.sort(
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.semantic_rank if candidate.semantic_rank is not None else 10_000,
            candidate.bm25_rank if candidate.bm25_rank is not None else 10_000,
            candidate.standard_code,
        )
    )
    return [
        HybridCandidate(
            rank=rank,
            standard_id=candidate.standard_id,
            standard_code=candidate.standard_code,
            title=candidate.title,
            rrf_score=candidate.rrf_score,
            semantic_rank=candidate.semantic_rank,
            semantic_score=candidate.semantic_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
        )
        for rank, candidate in enumerate(fused[:limit], start=1)
    ]


def _rrf_score(k: int, semantic_rank: int | None, bm25_rank: int | None) -> float:
    score = 0.0
    if semantic_rank is not None:
        score += 1 / (k + semantic_rank)
    if bm25_rank is not None:
        score += 1 / (k + bm25_rank)
    return score


async def _standards_by_id(
    session: AsyncSession,
    standard_ids: set[str],
) -> dict[str, Standard]:
    if not standard_ids:
        return {}
    standard_uuids = [UUID(standard_id) for standard_id in standard_ids]
    result = await session.execute(select(Standard).where(Standard.id.in_(standard_uuids)))
    return {str(standard.id): standard for standard in result.scalars().all()}
