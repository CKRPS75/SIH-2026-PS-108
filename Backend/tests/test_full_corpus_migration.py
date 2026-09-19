import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.base import Base
from app.db.models import Standard
from app.db.session import Database
from app.schemas.search import ProductAwareSearchRequest
from app.services.hybrid_search_service import HybridCandidate, HybridSearchResult, HybridTimings
from app.services.parsed_standards_corpus import (
    FULL_CORPUS_DATASET_NAME,
    audit_parsed_records,
    build_canonical_corpus,
    build_product_query,
    canonicalize_product,
    derive_product_metadata,
    load_canonical_dataset,
    product_compatibility,
    write_canonical_dataset,
)
from app.services.product_aware_search_service import ProductAwareSearchService
from app.services.reranker_training import (
    configure_last_n_layers_trainable,
    pairwise_ranknet_loss,
)
from scripts.ingest_parsed_standards import ingest_parsed_standards_dataset


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.requires_grad = True
        self._count = count

    def numel(self) -> int:
        return self._count


class FakeLayer:
    def __init__(self, parameter: FakeParameter) -> None:
        self._parameter = parameter

    def parameters(self):
        return [self._parameter]


class FakeTorchModel:
    def __init__(self) -> None:
        self.layer_params = [FakeParameter(10) for _ in range(6)]
        self.classifier_param = FakeParameter(5)
        self.roberta = SimpleNamespace(
            encoder=SimpleNamespace(
                layer=[FakeLayer(parameter) for parameter in self.layer_params]
            )
        )

    def parameters(self):
        return [*self.layer_params, self.classifier_param]

    def named_parameters(self):
        for index, parameter in enumerate(self.layer_params):
            yield f"roberta.encoder.layer.{index}.weight", parameter
        yield "classifier.weight", self.classifier_param


class FakeHybridService:
    def __init__(self, candidates: list[HybridCandidate]) -> None:
        self.candidates = candidates
        self.queries = []

    async def search(self, session, query: str, limit: int):
        self.queries.append((query, limit))
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


class FakeReranker:
    def __init__(self) -> None:
        self.seen_candidates = []

    async def rerank(self, session, query: str, candidates: list[HybridCandidate], *, limit: int):
        from app.services.reranker_service import RerankedCandidate

        self.seen_candidates.append([candidate.standard_code for candidate in candidates])
        return [
            RerankedCandidate(
                rank=rank,
                standard_id=candidate.standard_id,
                standard_code=candidate.standard_code,
                title=candidate.title,
                reranker_score=float(100 - rank),
                rrf_rank=candidate.rank,
                rrf_score=candidate.rrf_score,
                semantic_rank=candidate.semantic_rank,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
            )
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]


class SlowReranker:
    def __init__(self) -> None:
        self.seen_candidates = []

    async def rerank(self, session, query: str, candidates: list[HybridCandidate], *, limit: int):
        self.seen_candidates.append([candidate.standard_code for candidate in candidates])
        await asyncio.sleep(1)
        raise AssertionError("slow reranker should be timed out before returning")


class SkippedCpuReranker:
    last_inference_ms = None
    last_success = True
    last_fallback_reason = None
    device = "cpu"
    model_loaded = False

    def should_skip(self) -> bool:
        return True

    def skip_reason(self) -> str:
        return "cpu_reranker_disabled"

    def mark_skipped(self) -> None:
        self.last_inference_ms = 0.0
        self.last_success = False
        self.last_fallback_reason = "cpu_reranker_disabled"

    async def rerank(self, session, query: str, candidates: list[HybridCandidate], *, limit: int):
        raise AssertionError("skipped CPU reranker should not be called")


async def _create_database(tmp_path: Path) -> Database:
    database_path = tmp_path / "standards.db"
    database = Database(
        SimpleNamespace(database_url=f"sqlite+aiosqlite:///{database_path}")
    )
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return database


def test_corpus_audit_reports_duplicates_and_known_is_2116() -> None:
    records = [
        _source_record("IS 2116: 1980", "Door frames", "Scope text", "Full text"),
        _source_record("IS 2116: 1980", "Door frames", "Scope text", "Full text"),
        _source_record("", "", "", ""),
    ]

    audit = audit_parsed_records(records)

    assert audit.total_records == 3
    assert audit.duplicate_codes[canonicalize_standard_id("IS 2116: 1980")] == 2
    assert audit.known_duplicate_is_2116_records == 2
    assert audit.missing_is_code == 1
    assert audit.missing_title == 1


def test_build_canonical_corpus_deduplicates_and_preserves_provenance(tmp_path: Path) -> None:
    records = [
        _source_record("IS 1: 2000", "Sluice valve", "Requirements for sluice valves.", "x" * 200),
        _source_record("IS 1: 2000", "Sluice valve", "Requirements for sluice valves.", "x" * 200),
    ]

    canonical, warnings = build_canonical_corpus(records)
    path = tmp_path / "canonical.json"
    write_canonical_dataset(canonical, path)
    loaded = load_canonical_dataset(path)

    assert len(loaded) == 1
    assert loaded[0].standard_code_norm == canonicalize_standard_id("IS 1: 2000")
    assert loaded[0].source_record_count == 2
    assert len(loaded[0].source_provenance) == 2
    assert warnings == []


def test_product_metadata_and_query_representation_are_product_first() -> None:
    metadata = derive_product_metadata(
        "CAST IRON REFLUX VALVES",
        "Requirements for valves used in water works purposes.",
    )

    assert metadata.canonical_product == "valve"
    assert metadata.product_confidence == "high"
    assert build_product_query("valve", "used on a steel water pipeline") == (
        "PRODUCT: valve\nWORK DESCRIPTION: used on a steel water pipeline"
    )


def test_enriched_metadata_reconstructs_title_scope_and_evidence() -> None:
    records = [
        _source_record(
            "IS 10: 2000",
            "FABRICATED PVC FITTINGS FOR",
            "",
            (
                "1. Scope-Requirements for fabricated PVC straight reducers for potable "
                "water supplies.\n"
                "2. Requirements-The fittings shall conform.\n"
                "SUMMARY OF\n"
                "IS 10 : 2000 FABRICATED PVC FITTINGS FOR\n"
                "POTABLE WATER SUPPLIES\n"
                "PART 3 SPECIFIC REQUIREMENTS FOR STRAIGHT REDUCER\n"
                "(First Revision)"
            ),
        )
    ]

    canonical, _ = build_canonical_corpus(records)
    record = canonical[0]

    assert record.canonical_title_expanded == (
        "FABRICATED PVC FITTINGS FOR POTABLE WATER SUPPLIES PART 3 SPECIFIC "
        "REQUIREMENTS FOR STRAIGHT REDUCER"
    )
    assert record.scope_reconstructed.startswith("Requirements for fabricated PVC")
    assert record.canonical_product == "pipe_fitting"
    assert record.product_subtype == "reducer"
    assert record.material == "PVC"
    assert record.application == "potable water supply"
    assert record.function == "carry_potable_water"
    assert record.primary_subject == "reducer"
    assert record.metadata_evidence["canonical_product"]
    assert "SUBTYPE: reducer" in record.retrieval_text
    assert "SUBJECT: reducer" in record.retrieval_text


def test_non_product_standard_links_family_without_forcing_product() -> None:
    metadata = derive_product_metadata(
        "METHODS OF TEST FOR HYDRAULIC CEMENT",
        "Covers physical testing of hydraulic cement.",
        "The cement test method describes apparatus and procedure.",
    )

    assert metadata.standard_kind == "test_method"
    assert metadata.canonical_product is None
    assert metadata.primary_subject == "METHODS OF TEST FOR HYDRAULIC CEMENT"
    assert metadata.applies_to_product_families == ["cement"]


def test_generated_full_corpus_metadata_quality_report_is_complete() -> None:
    dataset_path = Path("data/canonical/parsed_standards_canonical.json")
    report_path = Path("data/metadata/metadata_enrichment_report.json")
    if not dataset_path.exists() or not report_path.exists():
        return

    records = json.loads(dataset_path.read_text(encoding="utf-8"))["records"]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(records) == 558
    assert sum(bool(record.get("primary_subject")) for record in records) == 558
    assert report["original_null_products"] == 338
    null_resolution = report["null_resolution"]
    assert sum(null_resolution.values()) == 338
    assert report["product_coverage"]["before"] == 220
    assert report["product_coverage"]["after"] >= report["product_coverage"]["before"]


def test_product_canonicalization_and_compatibility() -> None:
    product, confidence = canonicalize_product("non-return valve")

    assert (product, confidence) == ("valve", "high")
    assert product_compatibility("valve", "valve") == "compatible"
    assert product_compatibility("valve", "pipe") == "incompatible"
    assert product_compatibility(None, "pipe") == "unknown_query_product"


def test_product_aware_request_schema_separates_product_and_description() -> None:
    request = ProductAwareSearchRequest(
        product="valve",
        description="required on a steel water pipeline",
    )

    assert request.product == "valve"
    assert "steel water pipeline" in request.description


def test_product_gate_places_compatible_candidates_before_lexical_traps(tmp_path: Path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                valve = _standard("IS 1: 2000", "Valve", "valve")
                pipe = _standard("IS 2: 2000", "Steel pipe", "pipe")
                session.add_all([pipe, valve])
                await session.commit()
                hybrid = FakeHybridService(
                    [
                        _hybrid_candidate(pipe, rank=1),
                        _hybrid_candidate(valve, rank=2),
                    ]
                )
                reranker = FakeReranker()
                service = ProductAwareSearchService(hybrid, reranker, rerank_k=5, return_k=2)

                result = await service.search(
                    session,
                    product="valve",
                    description="for a steel water pipeline",
                    limit=2,
                )

            assert reranker.seen_candidates[0] == ["IS 1: 2000", "IS 2: 2000"]
            assert result.candidates[0].product_compatibility == "compatible"
            assert result.candidates[1].product_compatibility == "incompatible"
        finally:
            await database.close()

    asyncio.run(run())


def test_product_gate_falls_back_for_unknown_product(tmp_path: Path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                pipe = _standard("IS 2: 2000", "Steel pipe", "pipe")
                session.add(pipe)
                await session.commit()
                reranker = FakeReranker()
                service = ProductAwareSearchService(
                    FakeHybridService([_hybrid_candidate(pipe, rank=1)]),
                    reranker,
                    rerank_k=5,
                    return_k=1,
                )

                result = await service.search(
                    session,
                    product="widget",
                    description="for a steel water pipeline",
                    limit=1,
                )

            assert result.product_confidence == "low"
            assert reranker.seen_candidates[0] == ["IS 2: 2000"]
        finally:
            await database.close()

    asyncio.run(run())


def test_product_gate_uses_applies_to_family_metadata(tmp_path: Path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                insulation = _standard("IS 3: 2000", "Pipe insulation", "thermal_insulation")
                insulation.applies_to_product_families = ["pipe_water_drainage"]
                session.add(insulation)
                await session.commit()
                reranker = FakeReranker()
                service = ProductAwareSearchService(
                    FakeHybridService([_hybrid_candidate(insulation, rank=1)]),
                    reranker,
                    rerank_k=5,
                    return_k=1,
                )

                result = await service.search(
                    session,
                    product="pipe",
                    description="cellular glass pipe thermal insulation",
                    limit=1,
                )

            assert result.candidates[0].product_compatibility == "applies_to_family"
        finally:
            await database.close()

    asyncio.run(run())


def test_product_aware_search_falls_back_when_reranker_is_slow(tmp_path: Path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                valve = _standard("IS 1: 2000", "Valve", "valve")
                pipe = _standard("IS 2: 2000", "Steel pipe", "pipe")
                session.add_all([pipe, valve])
                await session.commit()
                service = ProductAwareSearchService(
                    FakeHybridService(
                        [
                            _hybrid_candidate(pipe, rank=1),
                            _hybrid_candidate(valve, rank=2),
                        ]
                    ),
                    SlowReranker(),
                    rerank_k=5,
                    return_k=2,
                    reranker_timeout_s=0.01,
                )

                result = await service.search(
                    session,
                    product="valve",
                    description="for a steel water pipeline",
                    limit=2,
                )

            assert [candidate.standard_code for candidate in result.candidates] == [
                "IS 1: 2000",
                "IS 2: 2000",
            ]
            assert result.candidates[0].product_compatibility == "compatible"
            assert result.candidates[0].reranker_score == 0.0
            assert result.candidates[0].reranker_score_source == "rrf_timeout_fallback"
            assert result.timings_ms.reranker_timeout is True
            assert result.timings_ms.reranker_ms < 500
        finally:
            await database.close()

    asyncio.run(run())


def test_product_aware_search_skips_cpu_reranker_without_cross_encoder_scores(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                valve = _standard("IS 1: 2000", "Valve", "valve")
                pipe = _standard("IS 2: 2000", "Steel pipe", "pipe")
                session.add_all([pipe, valve])
                await session.commit()
                service = ProductAwareSearchService(
                    FakeHybridService(
                        [
                            _hybrid_candidate(pipe, rank=1),
                            _hybrid_candidate(valve, rank=2),
                        ]
                    ),
                    SkippedCpuReranker(),
                    rerank_k=5,
                    return_k=2,
                )

                result = await service.search(
                    session,
                    product="valve",
                    description="for a steel water pipeline",
                    limit=2,
                )

            assert [candidate.standard_code for candidate in result.candidates] == [
                "IS 1: 2000",
                "IS 2: 2000",
            ]
            assert result.candidates[0].reranker_score == 0.0
            assert result.candidates[0].rrf_score > 0
            assert result.candidates[0].raw_cross_encoder_score is None
            assert result.candidates[0].reranker_score_source == "rrf_cpu_reranker_disabled"
            assert result.timings_ms.reranker_used is False
            assert result.timings_ms.reranker_success is False
            assert result.timings_ms.reranker_timeout is False
            assert result.timings_ms.reranker_device == "cpu"
            assert result.timings_ms.reranker_model_loaded is False
            assert result.timings_ms.reranker_fallback_reason == "cpu_reranker_disabled"
            assert result.timings_ms.reranker_inference_ms == 0.0
            assert result.timings_ms.reranker_ms < 50
        finally:
            await database.close()

    asyncio.run(run())


def test_full_corpus_ingestion_is_idempotent(tmp_path: Path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            canonical, _ = build_canonical_corpus(
                [_source_record("IS 1: 2000", "Sluice valve", "Valve scope", "x" * 200)]
            )
            path = tmp_path / "canonical.json"
            write_canonical_dataset(canonical, path)
            async with database.session_factory() as session:
                first = await ingest_parsed_standards_dataset(path, session)
                second = await ingest_parsed_standards_dataset(path, session)

            assert first.inserted == 1
            assert second.inserted == 0
            assert second.unchanged == 1
            assert second.postgres_count == 1
            assert second.postgres_unique_count == 1
        finally:
            await database.close()

    asyncio.run(run())


def test_pairwise_ranknet_loss_prefers_positive_over_negative() -> None:
    good = pairwise_ranknet_loss([3.0], [0.0])
    bad = pairwise_ranknet_loss([0.0], [3.0])

    assert good < bad


def test_last_four_layer_unfreezing_keeps_classifier_trainable() -> None:
    model = SimpleNamespace(model=FakeTorchModel())

    report = configure_last_n_layers_trainable(model, last_n=4)

    assert report["trainable_parameters"] == 45
    assert model.model.classifier_param.requires_grad
    assert [parameter.requires_grad for parameter in model.model.layer_params] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def _source_record(code: str, title: str, scope: str, full_text: str) -> dict[str, object]:
    return {
        "is_code": code,
        "is_code_norm": code,
        "title": title,
        "revision": code.split(":")[-1].strip() if ":" in code else None,
        "page_start": 1,
        "page_end": 2,
        "scope": scope,
        "full_text": full_text,
    }


def _standard(code: str, title: str, product: str) -> Standard:
    canonical_id = canonicalize_standard_id(code)
    return Standard(
        id=uuid4(),
        standard_id=code,
        canonical_id=canonical_id,
        standard_code_norm=canonical_id,
        base_code=extract_base_code(canonical_id),
        title=title,
        status="SOURCE_PARSED",
        source_dataset=FULL_CORPUS_DATASET_NAME,
        retrieval_text=f"STANDARD: {code}\nPRODUCT: {product}\nTITLE: {title}",
        canonical_product=product,
        standard_kind="product_standard",
        family="water_valve" if product == "valve" else "pipe_water_drainage",
        product_aliases=[product],
        source_provenance=[],
        search_profile={
            "canonical_product": product,
            "standard_kind": "product_standard",
        },
    )


def _hybrid_candidate(standard: Standard, *, rank: int) -> HybridCandidate:
    return HybridCandidate(
        rank=rank,
        standard_id=str(standard.id),
        standard_code=standard.standard_id,
        title=standard.title,
        rrf_score=1 / (60 + rank),
        semantic_rank=rank,
        semantic_score=1 - (rank / 10),
        bm25_rank=rank,
        bm25_score=10 - rank,
    )
