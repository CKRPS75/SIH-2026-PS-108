import asyncio
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.db.base import Base
from app.db.models import Standard
from app.db.session import Database
from app.main import create_app
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.standard_ingestion import ingest_standards_dataset
from app.services.standard_vector_index import StandardVectorIndex, VectorIndexUnavailable

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "standardwise_tile_pilot_dataset.json"
EXPANDED_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "standardwise_50_pilot_dataset.json"
)


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = text.lower().split()
            vector = [
                float(len(words)),
                float(text.lower().count("tile")),
                float(text.lower().count("roof")),
                float(text.lower().count("clay")),
            ]
            if normalize_embeddings:
                magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
                vector = [value / magnitude for value in vector]
            vectors.append(vector)
        return vectors


class QueryPointsResult:
    def __init__(self, points: list[Any]) -> None:
        self.points = points


class CountResult:
    def __init__(self, count: int) -> None:
        self.count = count


class ScoredPoint:
    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score


class FakeQdrantClient:
    def __init__(self, *, fail_search: bool = False) -> None:
        self.collections: set[str] = set()
        self.create_calls = 0
        self.points: dict[str, dict[str, Any]] = {}
        self.fail_search = fail_search

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.create_calls += 1
        self.collections.add(collection_name)

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.collections.add(collection_name)
        for point in points:
            point_id = getattr(point, "id", None) or point["id"]
            vector = getattr(point, "vector", None) or point["vector"]
            payload = getattr(point, "payload", None) or point["payload"]
            self.points[str(point_id)] = {"vector": vector, "payload": payload}

    async def count(self, collection_name: str, count_filter: Any, exact: bool) -> CountResult:
        return CountResult(len(self.points))

    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: Any,
        limit: int,
        with_payload: bool,
    ) -> QueryPointsResult:
        if self.fail_search:
            raise RuntimeError("Qdrant is down")
        scored = [
            ScoredPoint(point["payload"], _dot(query, point["vector"]))
            for point in self.points.values()
        ]
        scored.sort(key=lambda point: point.score, reverse=True)
        return QueryPointsResult(scored[:limit])

    async def close(self) -> None:
        return None


async def _create_database(tmp_path: Path) -> Database:
    database_path = tmp_path / "standards.db"
    database = Database(Settings(database_url=f"sqlite+aiosqlite:///{database_path}"))
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return database


def _embedding_service() -> BgeM3EmbeddingService:
    return BgeM3EmbeddingService(model=FakeEmbeddingModel())


def _dot(left: list[float], right: list[float]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def test_embedding_model_produces_non_empty_fixed_size_vectors() -> None:
    service = _embedding_service()

    vectors = service.embed_documents(["clay floor tiles", "roofing tiles"])
    query_vector = service.embed_query("clay floor tiles")

    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)
    assert len(query_vector) == 4
    assert service.embedding_dimension() == 4


def test_same_input_always_produces_compatible_vector_dimensions() -> None:
    service = _embedding_service()

    first = service.embed_query("Mangalore pattern clay roofing tiles")
    second = service.embed_query("Mangalore pattern clay roofing tiles")

    assert len(first) == len(second) == 4


def test_all_9_pilot_standards_can_be_indexed_idempotently(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        qdrant = FakeQdrantClient()
        try:
            async with database.session_factory() as session:
                await ingest_standards_dataset(str(DATASET_PATH), session)
                index = StandardVectorIndex(
                    qdrant,
                    _embedding_service(),
                    "standards_v1",
                    dataset_name="standardwise_tile_pilot_dataset",
                )

                first = await index.index_pilot_standards(session, expected_records=9)
                second = await index.index_pilot_standards(session, expected_records=9)

            assert first.status == "PASS"
            assert first.postgres_pilot_standards == 9
            assert first.valid_retrieval_texts == 9
            assert first.embeddings_generated == 9
            assert first.vectors_inserted_or_updated == 9
            assert second.status == "PASS"
            assert len(qdrant.points) == 9
            assert second.qdrant_pilot_records_verified == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_all_50_pilot_standards_are_indexed_idempotently(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        qdrant = FakeQdrantClient()
        try:
            async with database.session_factory() as session:
                await ingest_standards_dataset(str(EXPANDED_DATASET_PATH), session)
                index = StandardVectorIndex(qdrant, _embedding_service(), "standards_v1")

                first = await index.index_pilot_standards(session)
                second = await index.index_pilot_standards(session)

            assert first.status == "PASS"
            assert first.postgres_pilot_standards == 50
            assert first.qdrant_pilot_records_verified == 50
            assert second.status == "PASS"
            assert second.qdrant_pilot_records_verified == 50
            assert len(qdrant.points) == 50
            assert qdrant.create_calls == 1
        finally:
            await database.close()

    asyncio.run(run())


def test_semantic_search_returns_limit_and_valid_postgresql_standards(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        qdrant = FakeQdrantClient()
        try:
            async with database.session_factory() as session:
                await ingest_standards_dataset(str(DATASET_PATH), session)
                index = StandardVectorIndex(
                    qdrant,
                    _embedding_service(),
                    "standards_v1",
                    dataset_name="standardwise_tile_pilot_dataset",
                )
                await index.index_pilot_standards(session, expected_records=9)

                candidates = await index.semantic_search(session, "tiles for roofing", limit=5)
                result = await session.execute(select(Standard.id))
                standard_ids = {str(standard_id) for standard_id in result.scalars().all()}

            assert len(candidates) <= 5
            assert candidates
            assert {candidate.standard_id for candidate in candidates} <= standard_ids
        finally:
            await database.close()

    asyncio.run(run())


def test_semantic_search_rejects_empty_query(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                index = StandardVectorIndex(
                    FakeQdrantClient(),
                    _embedding_service(),
                    "standards_v1",
                )
                with pytest.raises(ValueError, match="query must not be empty"):
                    await index.semantic_search(session, "  ", limit=5)
        finally:
            await database.close()

    asyncio.run(run())


def test_unavailable_qdrant_is_handled_cleanly(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                index = StandardVectorIndex(None, _embedding_service(), "standards_v1")
                with pytest.raises(VectorIndexUnavailable):
                    await index.semantic_search(session, "clay tiles", limit=5)
        finally:
            await database.close()

    asyncio.run(run())


def test_semantic_search_endpoint_returns_ranked_candidates(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)
    app.state.qdrant.client = FakeQdrantClient()
    app.state.embedding_service = _embedding_service()

    async def seed_and_index() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with app.state.database.session_factory() as session:
            await ingest_standards_dataset(str(DATASET_PATH), session)
            index = StandardVectorIndex(
                app.state.qdrant.client,
                app.state.embedding_service,
                settings.qdrant_collection,
                dataset_name="standardwise_tile_pilot_dataset",
            )
            await index.index_pilot_standards(session, expected_records=9)

    asyncio.run(seed_and_index())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "tiles for lining an irrigation canal", "limit": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "tiles for lining an irrigation canal"
    assert len(body["candidates"]) <= 3
    assert body["candidates"][0]["rank"] == 1
    assert "score" in body["candidates"][0]


def test_semantic_search_endpoint_reports_qdrant_unavailable(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)
    app.state.qdrant.client = FakeQdrantClient(fail_search=True)
    app.state.embedding_service = _embedding_service()

    async def seed_database() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(seed_database())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "clay tiles", "limit": 3},
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "SEMANTIC_SEARCH_UNAVAILABLE"
