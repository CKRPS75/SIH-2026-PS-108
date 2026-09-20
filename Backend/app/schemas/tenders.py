from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.search import ProductAwareSearchTimings


class TenderTextAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    limit: int = Field(default=5, ge=1, le=30)


class AttributeEvidence(BaseModel):
    field: str
    value: str
    evidence: str
    inferred: bool = False


class ExtractedTenderItem(BaseModel):
    item_no: str | None = None
    raw_text: str
    product: str
    normalized_product: str | None = None
    normalized_description: str
    material: list[str] = Field(default_factory=list)
    application: list[str] = Field(default_factory=list)
    function: list[str] = Field(default_factory=list)
    installation_context: list[str] = Field(default_factory=list)
    associated_components: list[str] = Field(default_factory=list)
    dimensions: dict[str, Any] = Field(default_factory=dict)
    technical_requirements: list[str] = Field(default_factory=list)
    excluded_attributes: list[str] = Field(default_factory=list)
    attribute_evidence: list[AttributeEvidence] = Field(default_factory=list)
    ambiguity: bool = False
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "material",
        "application",
        "function",
        "installation_context",
        "associated_components",
        "technical_requirements",
        "excluded_attributes",
        "missing_information",
        mode="before",
    )
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]


class TenderExtractionResult(BaseModel):
    items: list[ExtractedTenderItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gemini_used: bool = False
    gemini_success: bool = False
    gemini_latency_ms: float = 0.0
    fallback_reason: str | None = None


class TenderSearchPayload(BaseModel):
    product: str
    description: str
    limit: int


class IdentifiedTenderProduct(BaseModel):
    display_name: str
    canonical_product: str | None = None
    material: list[str] = Field(default_factory=list)
    application: list[str] = Field(default_factory=list)
    function: list[str] = Field(default_factory=list)
    installation_context: list[str] = Field(default_factory=list)
    important_technical_attributes: list[AttributeEvidence] = Field(default_factory=list)


class TechnicalFact(BaseModel):
    label: str
    value: str
    evidence: str | None = None
    type: str | None = None
    name: str | None = None
    qualifier: str | None = None
    unit: str | None = None


class StandardTechnicalDetails(BaseModel):
    overview: str | None = None
    scope: str | None = None
    product: str | None = None
    subtype: str | None = None
    material: str | None = None
    application: str | None = None
    function: str | None = None
    primary_subject: str | None = None
    family: str | None = None
    nominal_sizes: list[TechnicalFact] = Field(default_factory=list)
    classification: list[TechnicalFact] = Field(default_factory=list)
    dimensional_requirements: list[TechnicalFact] = Field(default_factory=list)
    dimensions: list[TechnicalFact] = Field(default_factory=list)
    performance_requirements: list[TechnicalFact] = Field(default_factory=list)
    workmanship_requirements: list[TechnicalFact] = Field(default_factory=list)
    test_requirements: list[TechnicalFact] = Field(default_factory=list)
    grades: list[TechnicalFact] = Field(default_factory=list)
    temperature_ranges: list[TechnicalFact] = Field(default_factory=list)
    other_requirements: list[TechnicalFact] = Field(default_factory=list)
    source_excerpt: str | None = None


class StandardSourceMetadata(BaseModel):
    source_dataset: str | None = None
    source_revision: str | None = None
    source_page_start: int | None = None
    source_page_end: int | None = None
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)
    metadata_confidence: str | None = None
    metadata_evidence: dict[str, Any] = Field(default_factory=dict)
    publication_year: int | None = None
    lifecycle_status: str | None = None
    lifecycle_status_note: str | None = None


class TenderRecommendationRanking(BaseModel):
    final_score: float | None = None
    rrf_rank: int
    rrf_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    product_compatibility: str
    constraint_score: float | None = None
    constraint_flags: list[str] = Field(default_factory=list)
    raw_cross_encoder_score: float | None = None
    reranker_score: float
    reranker_score_source: str | None = None


class TechnicalStandardDetail(BaseModel):
    standard_code: str
    standard_id: str
    title: str
    canonical_title_expanded: str | None = None
    scope: str | None = None
    scope_text: str | None = None
    canonical_product: str | None = None
    product_subtype: str | None = None
    primary_subject: str | None = None
    material: str | None = None
    application: str | None = None
    function: str | None = None
    family: str | None = None
    standard_kind: str | None = None
    metadata_confidence: str | None = None
    metadata_evidence: dict[str, Any] = Field(default_factory=dict)
    source_dataset: str | None = None
    source_revision: str | None = None
    source_page_start: int | None = None
    source_page_end: int | None = None
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)
    publication_year: int | None = None
    lifecycle_status: str | None = None
    lifecycle_status_note: str | None = None
    full_text_excerpt: str | None = None


class TenderRecommendation(BaseModel):
    rank: int
    standard_code: str
    standard_id: str
    title: str
    rrf_rank: int
    rrf_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    product_compatibility: str
    constraint_score: float | None = None
    constraint_flags: list[str] = Field(default_factory=list)
    final_score: float | None = None
    raw_cross_encoder_score: float | None = None
    reranker_score: float
    reranker_score_source: str | None = None
    match_summary: list[str] = Field(default_factory=list)
    technical_detail: TechnicalStandardDetail | None = None
    technical_details: StandardTechnicalDetails | None = None
    ranking: TenderRecommendationRanking | None = None
    source: StandardSourceMetadata | None = None


class TenderAnalyzedItem(BaseModel):
    item_no: str | None = None
    raw_text: str
    identified_product: IdentifiedTenderProduct
    structured_requirement: ExtractedTenderItem
    search_payload: TenderSearchPayload
    recommendations: list[TenderRecommendation] = Field(default_factory=list)
    search_timings_ms: ProductAwareSearchTimings | None = None
    warnings: list[str] = Field(default_factory=list)


class TenderAnalysisDiagnostics(BaseModel):
    status: str
    warnings: list[str] = Field(default_factory=list)
    extraction: TenderExtractionResult
    total_latency_ms: float
    persistence: str = "deferred_text_analysis_only"


class TenderTextAnalysisResponse(BaseModel):
    text_length: int
    original_tender_text: str
    item_count: int
    limit: int
    items: list[TenderAnalyzedItem] = Field(default_factory=list)
    diagnostics: TenderAnalysisDiagnostics
