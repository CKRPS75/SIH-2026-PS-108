import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.api.v1.tenders import (
    _identified_product,
    _recommendation,
    _search_payload,
    _technical_detail,
)
from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.models import Standard
from app.docs import _docs_html
from app.main import create_app
from app.schemas.tenders import ExtractedTenderItem, TenderTextAnalysisRequest
from app.services.gemini_tender_extractor import (
    GeminiTenderExtractionPayload,
    GeminiTenderExtractor,
)
from app.services.standard_detail_enrichment import build_readable_technical_details

UPVC_TENDER = """Tender Item No. 07 — Supply and Installation of UPVC Drainage Pipes

Supply, delivery, installation, jointing and testing of unplasticized polyvinyl chloride
(UPVC) pipes for soil, waste and rainwater drainage applications in residential and
commercial buildings. The pipes shall be suitable for conveying domestic wastewater,
soil discharge and rainwater from internal and external building drainage systems.

The pipes shall be of 110 mm nominal diameter unless otherwise specified. The scope
shall include UPVC fittings, bends, tees, couplers, clamps and supports.
"""


def test_tender_text_schema_requires_text_and_limits_result_count() -> None:
    request = TenderTextAnalysisRequest(text="UPVC pipe for building drainage", limit=3)

    assert request.text.startswith("UPVC")
    assert request.limit == 3


def test_deterministic_single_item_extraction_is_structured_and_grounded() -> None:
    async def run() -> None:
        extractor = GeminiTenderExtractor(api_key=None, model_name="gemini-test")

        result = await extractor.extract(UPVC_TENDER)

        assert result.gemini_success is False
        assert result.fallback_reason == "missing_api_key_deterministic_fallback"
        assert len(result.items) == 1
        item = result.items[0]
        assert item.item_no == "07"
        assert item.normalized_product == "pipe"
        assert "UPVC" in item.material
        assert "soil drainage" in item.application
        assert "waste drainage" in item.application
        assert "rainwater drainage" in item.application
        assert "fittings" in item.associated_components
        assert "couplers" in item.associated_components
        assert item.dimensions["diameter"]["value"] == 110
        assert item.attribute_evidence

    asyncio.run(run())


def test_multi_item_extraction_does_not_leak_attributes_between_items() -> None:
    text = """Item 1: UPVC drainage pipe for soil waste and rainwater in buildings.
Item 2: 43 grade ordinary Portland cement for concrete work.
Item 3: Grade C hexagonal bolt for fastening."""

    async def run() -> None:
        result = await GeminiTenderExtractor(api_key=None, model_name="gemini-test").extract(text)

        assert len(result.items) == 3
        assert result.items[0].normalized_product == "pipe"
        assert "UPVC" in result.items[0].material
        assert result.items[1].normalized_product == "cement"
        assert "UPVC" not in result.items[1].material
        assert result.items[2].normalized_product == "bolt"
        assert "hexagonal bolt" in result.items[2].product

    asyncio.run(run())


def test_negative_constraints_are_preserved_for_flooring_not_roofing() -> None:
    async def run() -> None:
        result = await GeminiTenderExtractor(api_key=None, model_name="gemini-test").extract(
            "Supply clay tiles for internal flooring, explicitly not roofing."
        )

        item = result.items[0]
        assert "flooring" in item.application
        assert "roofing" not in item.application
        assert any("roofing" in value for value in item.excluded_attributes)

    asyncio.run(run())


def test_gemini_structured_response_is_sanitized_and_does_not_pass_is_codes() -> None:
    payload = GeminiTenderExtractionPayload(
        items=[
            ExtractedTenderItem(
                raw_text="Need pipe. Do not recommend IS 13592.",
                product="UPVC pipe IS 13592",
                normalized_product="pipe",
                normalized_description="UPVC pipe for drainage IS 13592",
                material=["UPVC"],
                confidence=0.9,
            )
        ]
    )
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        client=_FakeGeminiClient(parsed=payload),
    )

    async def run() -> None:
        result = await extractor.extract("Need UPVC pipe")

        assert result.gemini_success is True
        assert "IS 13592" not in result.items[0].product
        assert "IS 13592" not in result.items[0].normalized_description

    asyncio.run(run())


def test_gemini_normalized_product_is_canonicalized_before_search_payload() -> None:
    payload = GeminiTenderExtractionPayload(
        items=[
            ExtractedTenderItem(
                raw_text="Grade C hexagonal bolt for fastening",
                product="hexagonal bolt",
                normalized_product="Hexagonal Bolt",
                normalized_description="Grade C hexagonal bolt for fastening",
                confidence=0.9,
            )
        ]
    )
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        client=_FakeGeminiClient(parsed=payload),
    )

    async def run() -> None:
        result = await extractor.extract("Grade C hexagonal bolt for fastening")
        search_payload = _search_payload(result.items[0], 5)

        assert result.items[0].normalized_product == "bolt"
        assert result.items[0].normalized_description == "Grade C hexagonal bolt"
        assert search_payload.product == "bolt"

    asyncio.run(run())


def test_gemini_upvc_building_drainage_description_is_grounded_and_stable() -> None:
    payload = GeminiTenderExtractionPayload(
        items=[
            ExtractedTenderItem(
                raw_text=(
                    "UPVC pipes for soil, waste and rainwater drainage applications "
                    "in residential and commercial buildings."
                ),
                product="UPVC drainage pipes",
                normalized_product="UPVC Drainage Pipe",
                normalized_description=(
                    "UPVC drainage pipes for soil, waste and rainwater "
                    "gravity drainage applications"
                ),
                material=["UPVC"],
                confidence=0.9,
            )
        ]
    )
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        client=_FakeGeminiClient(parsed=payload),
    )

    async def run() -> None:
        result = await extractor.extract("UPVC pipes for soil, waste and rainwater")

        assert result.items[0].normalized_product == "pipe"
        assert result.items[0].normalized_description == (
            "UPVC pipe for soil and waste discharge inside and outside buildings "
            "including rainwater drainage"
        )

    asyncio.run(run())


def test_gemini_opc_43_description_keeps_grade_first() -> None:
    payload = GeminiTenderExtractionPayload(
        items=[
            ExtractedTenderItem(
                raw_text="Supply 43 grade ordinary Portland cement for concrete work.",
                product="ordinary Portland cement",
                normalized_product="cement",
                normalized_description="ordinary portland cement 43 grade for concrete work",
                confidence=0.9,
            )
        ]
    )
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        client=_FakeGeminiClient(parsed=payload),
    )

    async def run() -> None:
        result = await extractor.extract("43 grade ordinary Portland cement")

        assert result.items[0].normalized_description == "43 grade ordinary Portland cement"

    asyncio.run(run())


def test_gemini_malformed_response_uses_grounded_fallback() -> None:
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        client=_FakeGeminiClient(text="not-json"),
    )

    async def run() -> None:
        result = await extractor.extract("UPVC pipe for soil waste rainwater drainage")

        assert result.gemini_success is False
        assert result.items
        assert result.fallback_reason == "invalid_structured_response_deterministic_fallback"

    asyncio.run(run())


def test_gemini_timeout_uses_grounded_fallback() -> None:
    extractor = GeminiTenderExtractor(
        api_key="test",
        model_name="gemini-test",
        timeout_s=0.001,
        client=_SlowGeminiClient(),
    )

    async def run() -> None:
        result = await extractor.extract("UPVC pipe for soil waste rainwater drainage")

        assert result.gemini_success is False
        assert result.items
        assert result.fallback_reason == "timeout_deterministic_fallback"

    asyncio.run(run())


def test_search_payload_uses_normalized_description_without_contractual_noise() -> None:
    item = ExtractedTenderItem(
        raw_text="Supply and install UPVC pipe.",
        product="UPVC drainage pipe",
        normalized_product="pipe",
        normalized_description="UPVC pipe, soil drainage, waste drainage, rainwater drainage",
        confidence=0.8,
    )

    payload = _search_payload(item, 5)

    assert payload.product == "pipe"
    assert "Supply and install" not in payload.description
    assert payload.limit == 5


def test_standard_detail_enrichment_excludes_full_text_but_allows_excerpt() -> None:
    standard = _standard("IS 13592: 1992", "UPVC drainage pipe", "pipe")
    standard.full_text = "Full text " * 200
    detail = _technical_detail(standard)

    assert detail.standard_code == "IS 13592: 1992"
    assert detail.canonical_product == "pipe"
    assert detail.full_text_excerpt is not None
    assert len(detail.full_text_excerpt) <= 500


def test_identified_product_is_prominent_in_response_shape() -> None:
    item = ExtractedTenderItem(
        item_no="11",
        raw_text="Supply clay flooring tiles for internal flooring.",
        product="Clay Flooring Tile",
        normalized_product="tile",
        normalized_description="clay flooring tile for internal flooring",
        material=["clay"],
        application=["interior flooring"],
        installation_context=["building flooring"],
        confidence=0.9,
    )

    identified = _identified_product(item)

    assert identified.display_name == "Clay Flooring Tile"
    assert identified.canonical_product == "tile"
    assert identified.material == ["clay"]
    assert identified.application == ["interior flooring"]


def test_top_one_recommendation_keeps_rank_and_standard_prominent() -> None:
    standard = _standard("IS 1478: 1992", "CLAY FLOORING TILES", "tile")
    candidate = SimpleNamespace(
        rank=1,
        standard_code="IS 1478: 1992",
        standard_id=str(standard.id),
        title="CLAY FLOORING TILES",
        rrf_rank=1,
        rrf_score=0.05,
        semantic_rank=2,
        semantic_score=0.8,
        bm25_rank=1,
        bm25_score=14.0,
        product_compatibility="compatible",
        constraint_score=12.0,
        constraint_flags=["PRODUCT_EXACT_PRODUCT", "APPLICATION_EXACT_MATCH"],
        final_score=12.05,
        raw_cross_encoder_score=None,
        reranker_score=0.0,
        reranker_score_source="rrf_cpu_reranker_disabled",
    )

    recommendation = _recommendation(candidate, standard)

    assert recommendation.rank == 1
    assert recommendation.standard_code == "IS 1478: 1992"
    assert recommendation.ranking is not None
    assert recommendation.ranking.rrf_score == 0.05
    assert recommendation.ranking.constraint_flags == [
        "PRODUCT_EXACT_PRODUCT",
        "APPLICATION_EXACT_MATCH",
    ]


def test_readable_technical_details_extract_dimensions_with_evidence() -> None:
    standard = _standard("IS 1478: 1992", "CLAY FLOORING TILES", "tile")
    standard.full_text = (
        "Scope - Requirements for dimensions, quality and strength for clay flooring tiles. "
        "Dimensions i) 150 x 150 x 15 mm ii) 200 x 200 x 20 mm. "
        "Thickness - Average + 2 mm, individual + 1 mm. "
        "Water absorption 10 percent, Max."
    )

    details = build_readable_technical_details(standard)

    assert any(
        fact.label == "Nominal size" and "150 \u00d7 150 \u00d7 15 mm" in fact.value
        for fact in details.nominal_sizes
    )
    assert all(fact.type == "NOMINAL_SIZE" for fact in details.nominal_sizes)
    assert all(fact.evidence for fact in details.nominal_sizes)
    assert any("Water absorption" in fact.label for fact in details.performance_requirements)


def test_readable_technical_details_do_not_fabricate_missing_dimensions() -> None:
    standard = _standard("IS 9999: 2026", "GENERIC MATERIAL", "material")
    standard.full_text = "Scope - Requirements for a generic material without stated sizes."

    details = build_readable_technical_details(standard)

    assert details.dimensions == []
    assert details.temperature_ranges == []


def test_is_1478_context_aware_details_are_not_polluted_by_table_numbers() -> None:
    standard = _canonical_standard("IS 1478: 1992")

    details = build_readable_technical_details(standard)

    size_values = {fact.value for fact in details.nominal_sizes}
    dimensional_values = {fact.value for fact in details.dimensional_requirements}
    performance_values = {fact.value for fact in details.performance_requirements}

    assert details.material == "clay"
    assert details.material != "steel"
    assert "150 \u00d7 150 \u00d7 15 mm" in size_values
    assert "250 \u00d7 250 \u00d7 30 mm" in size_values
    assert {fact.value for fact in details.classification} == {"Class 1", "Class 2", "Class 3"}
    assert "maximum 3 mm" in dimensional_values
    assert "Average +5 mm; Individual +2 mm" in dimensional_values
    assert "Average +2 mm; Individual +1 mm" in dimensional_values
    assert (
        "maximum 2 percent along edges; maximum 1.5 percent along diagonals"
        in dimensional_values
    )
    assert details.temperature_ranges == []
    assert not any(fact.label == "Length" and fact.value == "5 mm" for fact in details.dimensions)
    assert not any(fact.label == "Width" and fact.value == "5 mm" for fact in details.dimensions)
    assert not any(
        fact.label == "Thickness" and fact.value == "2 mm" for fact in details.dimensions
    )
    assert not any("40 C" in fact.value for fact in details.temperature_ranges)
    assert any("Class 1 = 10 percent" in value for value in performance_values)
    assert any("Class 3 = 40 mm" in value for value in performance_values)


def test_standard_full_text_is_not_returned_excessively() -> None:
    standard = _standard(
        "IS 8154: 1993",
        "PREORMED CALCIUM SILICATE INSULATION",
        "thermal_insulation",
    )
    standard.full_text = "Requirements for temperatures upto 650 0C. " + (
        "Long source text. " * 200
    )

    details = build_readable_technical_details(standard)
    legacy_detail = _technical_detail(standard)

    assert details.source_excerpt is not None
    assert len(details.source_excerpt) <= 700
    assert legacy_detail.full_text_excerpt is not None
    assert len(legacy_detail.full_text_excerpt) <= 500
    assert any("650" in fact.value for fact in details.temperature_ranges)


def test_docs_and_openapi_expose_tender_analyzer() -> None:
    app = create_app()
    docs_html = _docs_html(app)
    openapi = app.openapi()

    assert "Tender Analyzer" in docs_html
    assert "UPVC Drainage Mock Tender" in docs_html
    assert "3-Item Construction Tender" in docs_html
    assert "min-height: 360px" in docs_html
    assert "View Original Tender" in docs_html
    assert "View Source Evidence" in docs_html
    assert "Technical Details From Standard" in docs_html
    assert "technicalDetailsSection" in docs_html
    assert "/api/v1/tenders/analyze-text" in openapi["paths"]


class _FakeGeminiClient:
    def __init__(self, parsed=None, text: str | None = None) -> None:
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self._parsed = parsed
        self._text = text

    def _generate_content(self, **kwargs):
        return SimpleNamespace(parsed=self._parsed, text=self._text)


class _SlowGeminiClient:
    def __init__(self) -> None:
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        time.sleep(0.05)
        return SimpleNamespace(parsed=GeminiTenderExtractionPayload(items=[]))


def _standard(code: str, title: str, product: str) -> Standard:
    canonical_id = canonicalize_standard_id(code)
    return Standard(
        id=uuid4(),
        standard_id=code,
        canonical_id=canonical_id,
        base_code=extract_base_code(canonical_id),
        title=title,
        status="ACTIVE",
        canonical_product=product,
        standard_kind="product_standard",
        material="UPVC",
        application="building drainage",
        function="wastewater conveyance",
        family="pipe_water_drainage",
        metadata_confidence="high",
        metadata_evidence={"canonical_product": ["title"]},
        source_provenance=[],
        search_profile={},
    )


def _canonical_standard(code: str) -> Standard:
    path = Path("data/canonical/parsed_standards_canonical.json")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in dataset["records"] if item["standard_code"] == code)
    canonical_id = canonicalize_standard_id(record["standard_code"])
    return Standard(
        id=uuid4(),
        standard_id=record["standard_code"],
        canonical_id=canonical_id,
        base_code=extract_base_code(canonical_id),
        title=record["title"],
        status="ACTIVE",
        scope=record.get("scope"),
        scope_text=record.get("scope"),
        scope_reconstructed=record.get("scope_reconstructed"),
        full_text=record.get("full_text"),
        retrieval_text=record.get("retrieval_text"),
        canonical_product=record.get("canonical_product"),
        product_subtype=record.get("product_subtype"),
        primary_subject=record.get("primary_subject"),
        material=record.get("material"),
        application=record.get("application"),
        function=record.get("function"),
        family=record.get("family"),
        metadata_confidence=record.get("metadata_confidence"),
        metadata_evidence=record.get("metadata_evidence") or {},
        source_provenance=record.get("source_provenance") or [],
        search_profile={},
    )
