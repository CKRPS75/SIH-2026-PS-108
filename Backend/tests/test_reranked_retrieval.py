import asyncio
import math
import sys
import types
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.base import Base
from app.db.models import Standard
from app.db.session import Database
from app.main import create_app
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.hybrid_search_service import (
    HybridCandidate,
    HybridSearchResult,
    HybridTimings,
)
from app.services.reranker_service import CrossEncoderRerankerService, RerankedSearchService
from app.services.standard_ingestion import ingest_standards_dataset
from app.services.standard_vector_index import PILOT_DATASET_NAME, StandardVectorIndex

EXPANDED_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "standardwise_50_pilot_dataset.json"
)


class FakeCrossEncoderModel:
    def __init__(self, scores_by_text: dict[str, float] | None = None) -> None:
        self.scores_by_text = scores_by_text or {}
        self.calls: list[list[list[str]]] = []
        self.batch_sizes: list[int | None] = []

    def predict(self, pairs: list[list[str]], batch_size: int | None = None) -> list[float]:
        self.calls.append([list(pair) for pair in pairs])
        self.batch_sizes.append(batch_size)
        scores = []
        for _, candidate_text in pairs:
            score = 1.0
            for marker, marker_score in self.scores_by_text.items():
                if marker in candidate_text:
                    score = marker_score
                    break
            scores.append(score)
        return scores


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
            scope_text=f"Scope for {title}",
            search_profile={
                "product_family": [title],
                "applications": ["testing"],
                "materials": ["steel"],
            },
        )
        session.add(standard)
        standards.append(standard)
    await session.commit()
    return standards


def _hybrid_candidate(
    standard: Standard,
    code: str,
    title: str,
    *,
    rank: int,
) -> HybridCandidate:
    return HybridCandidate(
        rank=rank,
        standard_id=str(standard.id),
        standard_code=code,
        title=title,
        rrf_score=1 / (60 + rank),
        semantic_rank=rank,
        semantic_score=0.9 - (rank / 10),
        bm25_rank=rank,
        bm25_score=10.0 - rank,
    )


def _embedding_service() -> BgeM3EmbeddingService:
    return BgeM3EmbeddingService(model=FakeEmbeddingModel())


def _dot(left: list[float], right: list[float]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def test_reranker_orders_hybrid_pool_by_score_and_respects_limit(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "Weak match", "weak"),
                        ("IS 2: 2000", "Best match", "best"),
                        ("IS 3: 2000", "Middle match", "middle"),
                    ],
                )
                candidates = [
                    _hybrid_candidate(standards[0], "STALE 1", "Stale weak", rank=1),
                    _hybrid_candidate(standards[1], "STALE 2", "Stale best", rank=2),
                    _hybrid_candidate(standards[2], "STALE 3", "Stale middle", rank=3),
                    _hybrid_candidate(standards[1], "DUPLICATE", "Duplicate", rank=4),
                ]
                model = FakeCrossEncoderModel(
                    {"Best match": 9.0, "Middle match": 5.0, "Weak match": 1.0}
                )
                service = CrossEncoderRerankerService("fake-reranker", model=model)

                result = await service.rerank(session, "need the best match", candidates, limit=2)

            assert [candidate.standard_code for candidate in result] == [
                "IS 2: 2000",
                "IS 3: 2000",
            ]
            assert [candidate.rank for candidate in result] == [1, 2]
            assert [candidate.rrf_rank for candidate in result] == [2, 3]
            assert result[0].reranker_score == pytest.approx(9.0)
            assert len(model.calls) == 1
            assert len(model.calls[0]) == 3
            assert model.batch_sizes == [8]
            assert all(isinstance(pair, list) for pair in model.calls[0])
        finally:
            await database.close()

    asyncio.run(run())


def test_reranker_tie_breaks_by_original_rrf_rank(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "Later RRF", "later"),
                        ("IS 2: 2000", "Earlier RRF", "earlier"),
                    ],
                )
                service = CrossEncoderRerankerService(
                    "fake-reranker",
                    model=FakeCrossEncoderModel({"RRF": 4.0}),
                )
                result = await service.rerank(
                    session,
                    "same score",
                    [
                        _hybrid_candidate(standards[0], "IS 1: 2000", "Later", rank=2),
                        _hybrid_candidate(standards[1], "IS 2: 2000", "Earlier", rank=1),
                    ],
                    limit=2,
                )

            assert [candidate.standard_code for candidate in result] == [
                "IS 2: 2000",
                "IS 1: 2000",
            ]
        finally:
            await database.close()

    asyncio.run(run())


def test_reranker_uses_database_metadata_for_candidate_text_and_response(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [("IS 1: 2000", "Fresh DB title", "fresh retrieval text")],
                )
                model = FakeCrossEncoderModel({"Fresh DB title": 7.0})
                service = CrossEncoderRerankerService("fake-reranker", model=model)

                result = await service.rerank(
                    session,
                    "fresh",
                    [_hybrid_candidate(standards[0], "STALE", "Stale title", rank=1)],
                    limit=1,
                )

            assert result[0].standard_code == "IS 1: 2000"
            assert result[0].title == "Fresh DB title"
            assert result[0].reranker_score == pytest.approx(7.0)
            assert "Fresh DB title" in model.calls[0][0][1]
            assert "Scope for Fresh DB title" in model.calls[0][0][1]
            assert "fresh retrieval text" not in model.calls[0][0][1]
        finally:
            await database.close()

    asyncio.run(run())


def test_reranked_service_calls_hybrid_with_rerank_k_and_returns_timings(tmp_path) -> None:
    class FakeHybridService:
        def __init__(self, candidates: list[HybridCandidate]) -> None:
            self.candidates = candidates
            self.calls: list[tuple[str, int]] = []

        async def search(self, session, query: str, limit: int) -> HybridSearchResult:
            self.calls.append((query, limit))
            return HybridSearchResult(
                query=query,
                candidates=self.candidates,
                timings_ms=HybridTimings(
                    semantic_ms=1.0,
                    bm25_ms=2.0,
                    rrf_ms=3.0,
                    total_ms=6.0,
                ),
            )

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "First", "first"),
                        ("IS 2: 2000", "Second", "second"),
                    ],
                )
                hybrid = FakeHybridService(
                    [
                        _hybrid_candidate(standards[0], "IS 1: 2000", "First", rank=1),
                        _hybrid_candidate(standards[1], "IS 2: 2000", "Second", rank=2),
                    ]
                )
                reranker = CrossEncoderRerankerService(
                    "fake-reranker",
                    model=FakeCrossEncoderModel({"Second": 5.0, "First": 1.0}),
                )
                service = RerankedSearchService(hybrid, reranker, rerank_k=12)

                result = await service.search(session, "second", limit=1)

            assert hybrid.calls == [("second", 12)]
            assert [candidate.standard_code for candidate in result.candidates] == ["IS 2: 2000"]
            assert result.timings_ms.semantic_ms == pytest.approx(1.0)
            assert result.timings_ms.bm25_ms == pytest.approx(2.0)
            assert result.timings_ms.rrf_ms == pytest.approx(3.0)
            assert result.timings_ms.hybrid_ms == pytest.approx(6.0)
            assert result.timings_ms.reranker_ms >= 0
            assert result.timings_ms.total_ms >= result.timings_ms.reranker_ms
        finally:
            await database.close()

    asyncio.run(run())


def test_reranker_lazy_loads_model_once(monkeypatch) -> None:
    created = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            created.append(model_name)

        def predict(
            self,
            pairs: list[list[str]],
            batch_size: int | None = None,
        ) -> list[list[float]]:
            return [[1.0] for _ in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    service = CrossEncoderRerankerService("fake-model")

    assert service._predict([["q", "a"]]) == [1.0]
    assert service._predict([["q", "b"]]) == [1.0]
    assert created == ["fake-model"]


def test_reranker_uses_configured_batch_size(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        ("IS 1: 2000", "First", "first"),
                        ("IS 2: 2000", "Second", "second"),
                    ],
                )
                model = FakeCrossEncoderModel()
                service = CrossEncoderRerankerService("fake-reranker", model=model, batch_size=16)

                await service.rerank(
                    session,
                    "batch me",
                    [
                        _hybrid_candidate(standards[0], "IS 1: 2000", "First", rank=1),
                        _hybrid_candidate(standards[1], "IS 2: 2000", "Second", rank=2),
                    ],
                    limit=2,
                )

            assert len(model.calls) == 1
            assert model.batch_sizes == [16]
        finally:
            await database.close()

    asyncio.run(run())


@pytest.mark.parametrize("rerank_k", [5, 10, 20])
def test_reranked_service_supports_configured_rerank_pool_sizes(tmp_path, rerank_k) -> None:
    class FakeHybridService:
        def __init__(self, candidates: list[HybridCandidate]) -> None:
            self.candidates = candidates
            self.calls: list[tuple[str, int]] = []

        async def search(self, session, query: str, limit: int) -> HybridSearchResult:
            self.calls.append((query, limit))
            return HybridSearchResult(
                query=query,
                candidates=self.candidates[:limit],
                timings_ms=HybridTimings(
                    semantic_ms=1.0,
                    bm25_ms=2.0,
                    rrf_ms=3.0,
                    total_ms=6.0,
                ),
            )

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        (f"IS {number}: 2000", f"Candidate {number}", str(number))
                        for number in range(1, 22)
                    ],
                )
                candidates = [
                    _hybrid_candidate(
                        standard,
                        standard.standard_id,
                        standard.title,
                        rank=rank,
                    )
                    for rank, standard in enumerate(standards, start=1)
                ]
                hybrid = FakeHybridService(candidates)
                reranker = CrossEncoderRerankerService(
                    "fake-reranker",
                    model=FakeCrossEncoderModel(),
                )
                service = RerankedSearchService(
                    hybrid,
                    reranker,
                    rerank_k=rerank_k,
                    return_k=5,
                )

                result = await service.search(session, "candidate", limit=5)

            assert hybrid.calls == [("candidate", rerank_k)]
            assert len(result.candidates) == 5
        finally:
            await database.close()

    asyncio.run(run())


def test_reranked_service_respects_return_k_and_pool_membership(tmp_path) -> None:
    class FakeHybridService:
        def __init__(self, candidates: list[HybridCandidate]) -> None:
            self.candidates = candidates

        async def search(self, session, query: str, limit: int) -> HybridSearchResult:
            return HybridSearchResult(
                query=query,
                candidates=self.candidates[:limit],
                timings_ms=HybridTimings(
                    semantic_ms=1.0,
                    bm25_ms=2.0,
                    rrf_ms=3.0,
                    total_ms=6.0,
                ),
            )

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                standards = await _seed_standards(
                    session,
                    [
                        (f"IS {number}: 2000", f"Candidate {number}", str(number))
                        for number in range(1, 8)
                    ],
                )
                hybrid_candidates = [
                    _hybrid_candidate(
                        standard,
                        standard.standard_id,
                        standard.title,
                        rank=rank,
                    )
                    for rank, standard in enumerate(standards[:5], start=1)
                ]
                service = RerankedSearchService(
                    FakeHybridService(hybrid_candidates),
                    CrossEncoderRerankerService(
                        "fake-reranker",
                        model=FakeCrossEncoderModel(
                            {f"Candidate {number}": float(number) for number in range(1, 8)}
                        ),
                    ),
                    rerank_k=5,
                    return_k=3,
                )

                result = await service.search(session, "candidate", limit=5)

            assert len(result.candidates) == 3
            assert {candidate.standard_id for candidate in result.candidates} <= {
                candidate.standard_id for candidate in hybrid_candidates
            }
            assert all(
                candidate.standard_id != str(standards[6].id) for candidate in result.candidates
            )
            assert len({candidate.standard_id for candidate in result.candidates}) == len(
                result.candidates
            )
        finally:
            await database.close()

    asyncio.run(run())


def test_reranked_endpoint_works_without_changing_existing_search_endpoints(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{database_path}",
        rerank_k=4,
        return_k=2,
    )
    app = create_app(settings)
    app.state.qdrant.client = FakeQdrantClient()
    app.state.embedding_service = _embedding_service()
    app.state.reranker_service = CrossEncoderRerankerService(
        settings.reranker_model,
        model=FakeCrossEncoderModel(),
    )

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
            json={"query": "clay tiles for a roof ridge", "limit": 3},
        )
        reranked_response = client.post(
            "/api/v1/search/reranked",
            json={"query": "clay tiles for a roof ridge", "limit": 3},
        )
        empty_response = client.post(
            "/api/v1/search/reranked",
            json={"query": "   ", "limit": 3},
        )
        bad_limit_response = client.post(
            "/api/v1/search/reranked",
            json={"query": "clay tile", "limit": 0},
        )

    assert semantic_response.status_code == 200
    assert hybrid_response.status_code == 200
    assert reranked_response.status_code == 200
    body = reranked_response.json()
    assert body["query"] == "clay tiles for a roof ridge"
    assert len(body["candidates"]) <= 2
    assert body["candidates"][0]["rank"] == 1
    assert "reranker_score" in body["candidates"][0]
    assert "rrf_rank" in body["candidates"][0]
    assert "rrf_score" in body["candidates"][0]
    assert "semantic_rank" in body["candidates"][0]
    assert "bm25_rank" in body["candidates"][0]
    assert set(body["timings_ms"]) == {
        "semantic_ms",
        "bm25_ms",
        "rrf_ms",
        "hybrid_ms",
        "reranker_ms",
        "total_ms",
    }
    assert empty_response.status_code == 400
    assert empty_response.json()["detail"]["code"] == "INVALID_SEARCH_QUERY"
    assert bad_limit_response.status_code == 422
