import asyncio
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.base import Base
from app.db.models import Standard
from app.db.session import Database
from app.main import create_app
from app.services.bm25_service import Bm25Candidate, Bm25LexicalSearchService
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.hybrid_search_service import HybridSearchService, reciprocal_rank_fusion
from app.services.standard_ingestion import ingest_standards_dataset
from app.services.standard_vector_index import (
    PILOT_DATASET_NAME,
    SemanticCandidate,
    StandardVectorIndex,
)

EXPANDED_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "standardwise_50_pilot_dataset.json"
)


class FakeSemanticService:
    def __init__(self, candidates: list[SemanticCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    async def semantic_search(
        self,
        session,
        query: str,
        limit: int = 5,
    ) -> list[SemanticCandidate]:
        self.calls.append((query, limit))
        return self.candidates[:limit]


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
            vector = [
                float(len(text.split())),
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
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: dict[str, dict[str, Any]] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config: Any) -> None:
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


async def _seed_standards(session, records: list[tuple[str, str, str]]) -> list[Standard]:
    standards = []
    for standard_code, title, retrieval_text in records:
        canonical_id = canonicalize_standard_id(standard_code)
        standard = Standard(
            id=uuid4(),
            standard_id=standard_code,
            canonical_id=canonical_id,
            base_code=extract_base_code(canonical_id),
            title=title,
            status="ACTIVE",
            source_dataset=PILOT_DATASET_NAME,
            retrieval_text=retrieval_text,
            search_profile={},
        )
        session.add(standard)
        standards.append(standard)
    await session.commit()
    return standards


def _embedding_service() -> BgeM3EmbeddingService:
    return BgeM3EmbeddingService(model=FakeEmbeddingModel())


def _dot(left: list[float], right: list[float]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def test_bm25_dynamically_ranks_corpus_records_and_respects_limit(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "Ridge tile", "clay roof ridge tile"),
                        ("IS 2: 2000", "Roof tile", "clay roof tile"),
                        ("IS 3: 2000", "Casing pipe", "UPVC borewell casing pipe"),
                    ],
                )

                candidates = await Bm25LexicalSearchService().search(
                    session,
                    "ridge clay tile",
                    limit=2,
                )
                standard_ids = {
                    str(standard_id)
                    for standard_id in (await session.execute(select(Standard.id))).scalars()
                }

            assert [candidate.standard_code for candidate in candidates] == [
                "IS 1: 2000",
                "IS 2: 2000",
            ]
            assert len(candidates) == 2
            assert len({candidate.standard_id for candidate in candidates}) == len(candidates)
            assert {candidate.standard_id for candidate in candidates} <= standard_ids
            assert candidates[0].bm25_rank == 1
            assert candidates[0].bm25_score > candidates[1].bm25_score
        finally:
            await database.close()

    asyncio.run(run())


def test_bm25_empty_query_rejected(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                with pytest.raises(ValueError, match="query must not be empty"):
                    await Bm25LexicalSearchService().search(session, "   ", limit=5)
        finally:
            await database.close()

    asyncio.run(run())


def test_rrf_math_retains_single_source_candidates_and_orders_deterministically(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "Semantic only", "semantic only"),
                        ("IS 2: 2000", "Both", "both"),
                        ("IS 3: 2000", "Semantic tail", "semantic tail"),
                        ("IS 4: 2000", "BM25 only", "bm25 only"),
                    ],
                )
                semantic = [
                    SemanticCandidate(str(standards[0].id), "IS 1: 2000", "Semantic only", 0.9),
                    SemanticCandidate(str(standards[1].id), "IS 2: 2000", "Both", 0.8),
                    SemanticCandidate(str(standards[2].id), "IS 3: 2000", "Semantic tail", 0.7),
                ]
                bm25 = [
                    Bm25Candidate(str(standards[1].id), "IS 2: 2000", "Both", 1, 4.0),
                    Bm25Candidate(str(standards[3].id), "IS 4: 2000", "BM25 only", 2, 3.0),
                ]

                first = await reciprocal_rank_fusion(session, semantic, bm25, limit=4, k=60)
                second = await reciprocal_rank_fusion(session, semantic, bm25, limit=4, k=60)

            assert [candidate.standard_code for candidate in first] == [
                "IS 2: 2000",
                "IS 1: 2000",
                "IS 4: 2000",
                "IS 3: 2000",
            ]
            assert [candidate.standard_code for candidate in second] == [
                candidate.standard_code for candidate in first
            ]
            assert first[0].rrf_score == pytest.approx((1 / 62) + (1 / 61))
            assert first[1].rrf_score == pytest.approx(1 / 61)
            assert first[2].rrf_score == pytest.approx(1 / 62)
            assert first[0].semantic_rank == 2
            assert first[0].bm25_rank == 1
        finally:
            await database.close()

    asyncio.run(run())


def test_hybrid_service_uses_semantic_service_and_returns_provenance(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "Ridge tile", "clay roof ridge tile"),
                        ("IS 2: 2000", "Roof tile", "clay roof tile"),
                    ],
                )
                semantic_service = FakeSemanticService(
                    [
                        SemanticCandidate(str(standards[1].id), "IS 2: 2000", "Roof tile", 0.9),
                        SemanticCandidate(str(standards[0].id), "IS 1: 2000", "Ridge tile", 0.8),
                    ]
                )
                hybrid = HybridSearchService(
                    semantic_service,
                    Bm25LexicalSearchService(),
                    semantic_top_k=20,
                    bm25_top_k=20,
                )

                result = await hybrid.search(session, "ridge", limit=2)

            assert semantic_service.calls == [("ridge", 20)]
            assert len(result.candidates) == 2
            assert len({candidate.standard_id for candidate in result.candidates}) == 2
            assert result.candidates[0].standard_code == "IS 1: 2000"
            assert result.candidates[0].title == "Ridge tile"
            assert result.candidates[0].semantic_rank == 2
            assert result.candidates[0].semantic_score == pytest.approx(0.8)
            assert result.candidates[0].bm25_rank == 1
            assert result.candidates[0].bm25_score is not None
            assert result.timings_ms.total_ms >= result.timings_ms.rrf_ms
        finally:
            await database.close()

    asyncio.run(run())


def test_hybrid_endpoint_works_and_semantic_endpoint_still_works(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)
    app.state.qdrant.client = FakeQdrantClient()
    app.state.embedding_service = _embedding_service()

    async def seed_and_index() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with app.state.database.session_factory() as session:
            await ingest_standards_dataset(str(EXPANDED_DATASET_PATH), session)
            index = StandardVectorIndex(
                app.state.qdrant.client,
                app.state.embedding_service,
                settings.qdrant_collection,
            )
            await index.index_pilot_standards(session)

    asyncio.run(seed_and_index())
    with TestClient(app) as client:
        semantic_response = client.post(
            "/api/v1/search/semantic",
            json={"query": "tiles for lining an irrigation canal", "limit": 3},
        )
        hybrid_response = client.post(
            "/api/v1/search/hybrid",
            json={"query": "clay tiles for a roof ridge", "limit": 5},
        )
        empty_response = client.post(
            "/api/v1/search/hybrid",
            json={"query": "   ", "limit": 5},
        )

    assert semantic_response.status_code == 200
    assert hybrid_response.status_code == 200
    body = hybrid_response.json()
    assert body["query"] == "clay tiles for a roof ridge"
    assert len(body["candidates"]) <= 5
    assert body["candidates"][0]["rank"] == 1
    assert "rrf_score" in body["candidates"][0]
    assert "semantic_rank" in body["candidates"][0]
    assert "bm25_rank" in body["candidates"][0]
    assert "timings_ms" in body
    assert empty_response.status_code == 400
    assert empty_response.json()["detail"]["code"] == "INVALID_SEARCH_QUERY"
