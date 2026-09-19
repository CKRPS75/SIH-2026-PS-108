from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Standard
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.parsed_standards_corpus import FULL_CORPUS_DATASET_NAME

try:
    from qdrant_client import models as qdrant_models
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    qdrant_models = None

PILOT_DATASET_NAME = "standardwise_50_pilot_dataset"
PILOT_RECORD_COUNT = 50


class VectorIndexUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticCandidate:
    standard_id: str
    standard_code: str
    title: str
    score: float


@dataclass
class PilotIndexReport:
    collection: str
    postgres_pilot_standards: int = 0
    valid_retrieval_texts: int = 0
    embeddings_generated: int = 0
    vectors_inserted_or_updated: int = 0
    qdrant_pilot_records_verified: int = 0
    expected_records: int = 9
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.postgres_pilot_standards != self.expected_records:
            return "FAIL"
        if self.valid_retrieval_texts != self.expected_records:
            return "FAIL"
        if self.embeddings_generated != self.expected_records:
            return "FAIL"
        if self.qdrant_pilot_records_verified != self.expected_records:
            return "FAIL"
        return "PASS"


class StandardVectorIndex:
    def __init__(
        self,
        qdrant_client: Any,
        embedding_service: BgeM3EmbeddingService,
        collection_name: str,
        dataset_name: str = PILOT_DATASET_NAME,
    ) -> None:
        self._qdrant = qdrant_client
        self._embeddings = embedding_service
        self._collection = collection_name
        self._dataset_name = dataset_name
        self.last_search_timings_ms: dict[str, float] = {}

    async def index_pilot_standards(
        self,
        session: AsyncSession,
        *,
        expected_records: int = PILOT_RECORD_COUNT,
    ) -> PilotIndexReport:
        return await self.index_standards(session, expected_records=expected_records)

    async def index_standards(
        self,
        session: AsyncSession,
        *,
        expected_records: int | None = None,
    ) -> PilotIndexReport:
        self._require_qdrant()
        standards = await fetch_standards_for_dataset(session, self._dataset_name)
        expected = expected_records if expected_records is not None else len(standards)
        report = PilotIndexReport(collection=self._collection, expected_records=expected)
        report.postgres_pilot_standards = len(standards)
        if len(standards) != expected:
            report.errors.append(
                f"expected {expected} PostgreSQL standards, found {len(standards)}"
            )

        missing = [
            standard.standard_id for standard in standards if not _has_retrieval_text(standard)
        ]
        if missing:
            report.errors.append(f"missing retrieval_text for standards: {', '.join(missing)}")
        valid_standards = [standard for standard in standards if _has_retrieval_text(standard)]
        report.valid_retrieval_texts = len(valid_standards)
        if report.errors:
            return report

        vectors = self._embeddings.embed_documents(
            [standard.retrieval_text or "" for standard in valid_standards]
        )
        report.embeddings_generated = len(vectors)
        if len(vectors) != len(valid_standards):
            report.errors.append("embedding count did not match valid pilot standards")
            return report

        vector_size = len(vectors[0]) if vectors else self._embeddings.embedding_dimension()
        await self.ensure_collection(vector_size)
        await self._upsert_standards(valid_standards, vectors)
        report.vectors_inserted_or_updated = len(valid_standards)
        report.qdrant_pilot_records_verified = await self.count_records()
        if report.qdrant_pilot_records_verified != expected:
            report.errors.append(
                "Qdrant record count did not match expected "
                f"{expected}: {report.qdrant_pilot_records_verified}"
            )
        return report

    async def ensure_collection(self, vector_size: int) -> None:
        self._require_qdrant()
        exists = await self._collection_exists()
        if exists:
            return
        await self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config=_vector_params(vector_size),
        )

    async def count_pilot_records(self) -> int:
        return await self.count_records()

    async def count_records(self) -> int:
        self._require_qdrant()
        result = await self._qdrant.count(
            collection_name=self._collection,
            count_filter=_pilot_filter(self._dataset_name),
            exact=True,
        )
        return int(getattr(result, "count", result))

    async def semantic_search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
    ) -> list[SemanticCandidate]:
        self._require_qdrant()
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")

        try:
            embedding_started_at = perf_counter()
            query_vector = self._embeddings.embed_query(query)
            embedding_ms = (perf_counter() - embedding_started_at) * 1000
            retrieval_started_at = perf_counter()
            points = await self._query_points(query_vector, limit)
            semantic_retrieval_ms = (perf_counter() - retrieval_started_at) * 1000
            self.last_search_timings_ms = {
                "embedding_ms": embedding_ms,
                "semantic_retrieval_ms": semantic_retrieval_ms,
            }
        except Exception as exc:  # noqa: BLE001 - normalize external retrieval failures
            raise VectorIndexUnavailable("Semantic vector search is unavailable") from exc

        return await self._resolve_candidates(session, points, limit)

    async def _collection_exists(self) -> bool:
        if hasattr(self._qdrant, "collection_exists"):
            return bool(await self._qdrant.collection_exists(self._collection))
        try:
            await self._qdrant.get_collection(self._collection)
            return True
        except Exception:  # noqa: BLE001 - older Qdrant clients signal absence differently
            return False

    async def _upsert_standards(
        self,
        standards: list[Standard],
        vectors: list[list[float]],
    ) -> None:
        points = [
            _point_struct(
                point_id=str(standard.id),
                vector=vector,
                payload=_payload_for_standard(standard),
            )
            for standard, vector in zip(standards, vectors, strict=True)
        ]
        await self._qdrant.upsert(collection_name=self._collection, points=points)

    async def _query_points(self, query_vector: list[float], limit: int) -> list[Any]:
        if hasattr(self._qdrant, "query_points"):
            result = await self._qdrant.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=_pilot_filter(self._dataset_name),
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result))
        return list(
            await self._qdrant.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=_pilot_filter(self._dataset_name),
                limit=limit,
                with_payload=True,
            )
        )

    async def _resolve_candidates(
        self,
        session: AsyncSession,
        points: list[Any],
        limit: int,
    ) -> list[SemanticCandidate]:
        scored_ids: list[tuple[UUID, float]] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            raw_id = payload.get("standard_id")
            try:
                standard_uuid = UUID(str(raw_id))
            except (TypeError, ValueError):
                continue
            scored_ids.append((standard_uuid, float(getattr(point, "score", 0.0))))

        if not scored_ids:
            return []

        result = await session.execute(
            select(Standard).where(Standard.id.in_([standard_id for standard_id, _ in scored_ids]))
        )
        standards_by_id = {standard.id: standard for standard in result.scalars().all()}
        candidates: list[SemanticCandidate] = []
        for standard_id, score in scored_ids:
            standard = standards_by_id.get(standard_id)
            if standard is None:
                continue
            candidates.append(
                SemanticCandidate(
                    standard_id=str(standard.id),
                    standard_code=standard.standard_id,
                    title=standard.title,
                    score=score,
                )
            )
            if len(candidates) == limit:
                break
        return candidates

    def _require_qdrant(self) -> None:
        if self._qdrant is None:
            raise VectorIndexUnavailable("Qdrant client is not configured")


async def fetch_pilot_standards(
    session: AsyncSession, dataset_name: str = PILOT_DATASET_NAME
) -> list[Standard]:
    return await fetch_standards_for_dataset(session, dataset_name)


async def fetch_standards_for_dataset(
    session: AsyncSession, dataset_name: str = PILOT_DATASET_NAME
) -> list[Standard]:
    result = await session.execute(
        select(Standard)
        .where(Standard.source_dataset == dataset_name)
        .order_by(Standard.canonical_id)
    )
    return list(result.scalars().all())


async def count_pilot_standards(
    session: AsyncSession, dataset_name: str = PILOT_DATASET_NAME
) -> int:
    result = await session.execute(
        select(func.count(Standard.id)).where(Standard.source_dataset == dataset_name)
    )
    return int(result.scalar_one())


async def count_full_corpus_standards(session: AsyncSession) -> int:
    return await count_pilot_standards(session, FULL_CORPUS_DATASET_NAME)


def _has_retrieval_text(standard: Standard) -> bool:
    return bool((standard.retrieval_text or "").strip())


def _payload_for_standard(standard: Standard) -> dict[str, Any]:
    profile = standard.search_profile or {}
    return {
        "standard_id": str(standard.id),
        "standard_code": standard.standard_id,
        "canonical_id": standard.canonical_id,
        "title": standard.title,
        "canonical_product": standard.canonical_product or profile.get("canonical_product"),
        "standard_kind": standard.standard_kind or profile.get("standard_kind"),
        "family": standard.family or profile.get("family"),
        "product_family": list(profile.get("product_family") or []),
        "applications": list(profile.get("applications") or []),
        "source_dataset": standard.source_dataset,
    }


def _vector_params(vector_size: int) -> Any:
    if qdrant_models is None:
        return {"size": vector_size, "distance": "Cosine"}
    return qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)


def _point_struct(point_id: str, vector: list[float], payload: dict[str, Any]) -> Any:
    if qdrant_models is None:
        return {"id": point_id, "vector": vector, "payload": payload}
    return qdrant_models.PointStruct(id=point_id, vector=vector, payload=payload)


def _pilot_filter(dataset_name: str) -> Any:
    if qdrant_models is None:
        return {"source_dataset": dataset_name}
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="source_dataset",
                match=qdrant_models.MatchValue(value=dataset_name),
            )
        ]
    )
