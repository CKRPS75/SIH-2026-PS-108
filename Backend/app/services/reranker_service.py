from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Standard
from app.services.hybrid_search_service import HybridCandidate, HybridSearchService

VALID_RERANKER_MODES = {"auto", "enabled", "disabled"}
CPU_RERANKER_DISABLED_REASON = "cpu_reranker_disabled"
RERANKER_DISABLED_REASON = "reranker_disabled"


class RerankerUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankedCandidate:
    rank: int
    standard_id: str
    standard_code: str
    title: str
    reranker_score: float
    rrf_rank: int
    rrf_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    raw_cross_encoder_score: float | None = None
    score_source: str = "cross_encoder"


@dataclass(frozen=True)
class RerankedTimings:
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    hybrid_ms: float
    reranker_ms: float
    total_ms: float


@dataclass(frozen=True)
class RerankedSearchResult:
    query: str
    candidates: list[RerankedCandidate]
    timings_ms: RerankedTimings


class CrossEncoderRerankerService:
    def __init__(
        self,
        model_name: str,
        *,
        model: Any | None = None,
        batch_size: int = 8,
        reranker_mode: str = "auto",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        normalized_mode = reranker_mode.strip().lower()
        if normalized_mode not in VALID_RERANKER_MODES:
            raise ValueError("reranker_mode must be auto, enabled, or disabled")
        self.model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._reranker_mode = normalized_mode
        self._predict_lock = Lock()
        self.last_inference_ms: float | None = None
        self.last_success: bool = False
        self.last_fallback_reason: str | None = None

    async def rerank(
        self,
        session: AsyncSession,
        query: str,
        candidates: list[HybridCandidate],
        *,
        limit: int = 5,
    ) -> list[RerankedCandidate]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not candidates:
            return []
        if self.should_skip():
            self.mark_skipped()
            raise RerankerUnavailable(self.skip_reason() or RERANKER_DISABLED_REASON)

        unique_candidates = _deduplicate_candidates(candidates)
        standards_by_id = await _standards_by_id(
            session,
            {candidate.standard_id for candidate in unique_candidates},
        )
        pairs = [
            [query, _candidate_text(standards_by_id[candidate.standard_id])]
            for candidate in unique_candidates
            if candidate.standard_id in standards_by_id
        ]
        scored_candidates = [
            candidate for candidate in unique_candidates if candidate.standard_id in standards_by_id
        ]
        if not pairs:
            return []

        try:
            inference_started_at = perf_counter()
            scores = await self._predict_async(pairs)
            self.last_inference_ms = (perf_counter() - inference_started_at) * 1000
            self.last_success = True
            self.last_fallback_reason = None
        except Exception as exc:  # noqa: BLE001 - normalize model failures for API callers
            self.last_success = False
            self.last_fallback_reason = exc.__class__.__name__
            raise RerankerUnavailable("Cross-encoder reranker is unavailable") from exc

        reranked = [
            RerankedCandidate(
                rank=0,
                standard_id=candidate.standard_id,
                standard_code=standards_by_id[candidate.standard_id].standard_id,
                title=standards_by_id[candidate.standard_id].title,
                reranker_score=score,
                rrf_rank=candidate.rank,
                rrf_score=candidate.rrf_score,
                semantic_rank=candidate.semantic_rank,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
                raw_cross_encoder_score=score,
                score_source="cross_encoder",
            )
            for candidate, score in zip(scored_candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda candidate: (-candidate.reranker_score, candidate.rrf_rank))
        return [
            RerankedCandidate(
                rank=rank,
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
                raw_cross_encoder_score=candidate.raw_cross_encoder_score,
                score_source=candidate.score_source,
            )
            for rank, candidate in enumerate(reranked[:limit], start=1)
        ]

    def _predict(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        model = self._load_model()
        raw_scores = model.predict([list(pair) for pair in pairs], batch_size=self._batch_size)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, int | float):
            return [float(raw_scores)]
        return [_as_float_score(score) for score in raw_scores]

    async def _predict_async(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        return await asyncio.to_thread(self._predict_locked, pairs)

    def _predict_locked(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        with self._predict_lock:
            return self._predict(pairs)

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking"
                ) from exc
            self._model = CrossEncoder(_resolve_model_name(self.model_name))
        return self._model

    def should_skip(self) -> bool:
        if self._model is not None:
            return False
        if self._reranker_mode == "disabled":
            return True
        return self._reranker_mode == "auto" and not _cuda_available()

    def skip_reason(self) -> str | None:
        if self._model is not None:
            return None
        if self._reranker_mode == "disabled":
            return RERANKER_DISABLED_REASON
        if self._reranker_mode == "auto" and not _cuda_available():
            return CPU_RERANKER_DISABLED_REASON
        return None

    def mark_skipped(self) -> None:
        self.last_success = False
        self.last_fallback_reason = self.skip_reason()
        self.last_inference_ms = 0.0

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        model = self._model
        if model is None:
            return _runtime_device()
        raw_model = getattr(model, "model", model)
        device = getattr(raw_model, "device", None)
        if device is not None:
            return str(device)
        try:
            first_parameter = next(raw_model.parameters())
            return str(first_parameter.device)
        except Exception:  # noqa: BLE001 - optional diagnostic only
            return "unknown"


class RerankedSearchService:
    def __init__(
        self,
        hybrid_service: HybridSearchService,
        reranker_service: CrossEncoderRerankerService,
        *,
        rerank_k: int = 10,
        return_k: int = 5,
    ) -> None:
        if rerank_k < 1:
            raise ValueError("rerank_k must be at least 1")
        if return_k < 1:
            raise ValueError("return_k must be at least 1")
        self._hybrid_service = hybrid_service
        self._reranker_service = reranker_service
        self._rerank_k = rerank_k
        self._return_k = return_k

    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: int | None = None,
    ) -> RerankedSearchResult:
        limit = self._return_k if limit is None else min(limit, self._return_k)
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")

        started_at = perf_counter()
        hybrid_result = await self._hybrid_service.search(session, query, limit=self._rerank_k)
        reranker_started_at = perf_counter()
        reranked_candidates = await self._reranker_service.rerank(
            session,
            query,
            hybrid_result.candidates,
            limit=limit,
        )
        reranker_ms = (perf_counter() - reranker_started_at) * 1000
        timings = hybrid_result.timings_ms
        return RerankedSearchResult(
            query=query,
            candidates=reranked_candidates,
            timings_ms=RerankedTimings(
                semantic_ms=timings.semantic_ms,
                bm25_ms=timings.bm25_ms,
                rrf_ms=timings.rrf_ms,
                hybrid_ms=timings.total_ms,
                reranker_ms=reranker_ms,
                total_ms=(perf_counter() - started_at) * 1000,
            ),
        )


def _deduplicate_candidates(candidates: list[HybridCandidate]) -> list[HybridCandidate]:
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if candidate.standard_id in seen:
            continue
        unique.append(candidate)
        seen.add(candidate.standard_id)
    return unique


def _resolve_model_name(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    backend_relative_path = Path(__file__).resolve().parents[2] / model_name
    if backend_relative_path.exists():
        return str(backend_relative_path)
    return model_name


def _cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - environment dependent
        return False
    except Exception:  # noqa: BLE001 - optional diagnostic only
        return False
    return bool(torch.cuda.is_available())


def _runtime_device() -> str:
    return "cuda" if _cuda_available() else "cpu"


async def _standards_by_id(
    session: AsyncSession,
    standard_ids: set[str],
) -> dict[str, Standard]:
    if not standard_ids:
        return {}
    standard_uuids = [UUID(standard_id) for standard_id in standard_ids]
    result = await session.execute(select(Standard).where(Standard.id.in_(standard_uuids)))
    return {str(standard.id): standard for standard in result.scalars().all()}


def _candidate_text(standard: Standard) -> str:
    profile = standard.search_profile or {}
    lines = [
        f"STANDARD: {standard.standard_id}",
    ]
    canonical_product = standard.canonical_product or profile.get("canonical_product")
    if canonical_product:
        lines.append(f"PRODUCT: {canonical_product}")
    standard_kind = standard.standard_kind or profile.get("standard_kind")
    if standard_kind:
        lines.append(f"TYPE: {standard_kind}")
    lines.append(f"TITLE: {standard.title}")
    product_family = _join_profile_values(profile.get("product_family"))
    if product_family:
        lines.append(f"Product family: {product_family}")
    applications = _join_profile_values(profile.get("applications"))
    if applications:
        lines.append(f"Applications: {applications}")
    materials = _join_profile_values(profile.get("materials"))
    if materials:
        lines.append(f"Materials: {materials}")
    if standard.scope_text:
        lines.append(f"SCOPE: {standard.scope_text}")
    return "\n".join(line for line in lines if line)


def _join_profile_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ", ".join(str(item) for item in value if item)
    return str(value)


def _as_float_score(score: Any) -> float:
    if isinstance(score, Sequence) and not isinstance(score, str):
        if not score:
            raise ValueError("reranker returned an empty score sequence")
        score = score[0]
    return float(score)
