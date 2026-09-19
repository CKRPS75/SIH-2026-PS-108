from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.standards import StandardSummary


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    language: str = "auto"
    include_allied: bool = True
    include_compliance: bool = True


class SearchResponse(BaseModel):
    raw_query: str
    normalized_query: str
    detected_language: str
    entities: dict[str, Any] = Field(default_factory=dict)
    primary_standards: list[StandardSummary] = Field(default_factory=list)
    allied_standards: list[StandardSummary] = Field(default_factory=list)
    compliance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    trace_id: UUID


class ComplianceResult(BaseModel):
    status: str
    scheme: str | None = None
    order_ref: str | None = None
    evidence: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    trace_id: str | None = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    limit: int = Field(default=5, ge=1, le=30)


class SemanticSearchCandidate(BaseModel):
    rank: int
    standard_id: str
    standard_code: str
    title: str
    score: float


class SemanticSearchResponse(BaseModel):
    query: str
    candidates: list[SemanticSearchCandidate] = Field(default_factory=list)


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    limit: int = Field(default=5, ge=1, le=30)


class HybridSearchCandidate(BaseModel):
    rank: int
    standard_id: str
    standard_code: str
    title: str
    rrf_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


class HybridSearchTimings(BaseModel):
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    total_ms: float


class HybridSearchResponse(BaseModel):
    query: str
    candidates: list[HybridSearchCandidate] = Field(default_factory=list)
    timings_ms: HybridSearchTimings


class RerankedSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    limit: int = Field(default=5, ge=1, le=30)


class RerankedSearchCandidate(BaseModel):
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


class RerankedSearchTimings(BaseModel):
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    hybrid_ms: float
    reranker_ms: float
    total_ms: float


class RerankedSearchResponse(BaseModel):
    query: str
    candidates: list[RerankedSearchCandidate] = Field(default_factory=list)
    timings_ms: RerankedSearchTimings


class ProductAwareSearchRequest(BaseModel):
    product: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=5000)
    limit: int = Field(default=5, ge=1, le=30)


class SemanticQueryInterpretation(BaseModel):
    normalized_product: str | None = None
    product_aliases: list[str] = Field(default_factory=list)
    true_product_aliases: list[str] = Field(default_factory=list)
    known_subtypes: list[str] = Field(default_factory=list)
    query_subtype: str | None = None
    subtype: str | None = None
    subtype_family: str | None = None
    cement_type: str | None = None
    pozzolana_source: str | None = None
    head_shape: str | None = None
    form: str | None = None
    material: list[str] = Field(default_factory=list)
    context_material: list[str] = Field(default_factory=list)
    excluded_material: list[str] = Field(default_factory=list)
    function: list[str] = Field(default_factory=list)
    application: list[str] = Field(default_factory=list)
    medium: list[str] = Field(default_factory=list)
    installation_context: list[str] = Field(default_factory=list)
    excluded_function: list[str] = Field(default_factory=list)
    excluded_application: list[str] = Field(default_factory=list)
    excluded_subtype: list[str] = Field(default_factory=list)
    excluded_medium: list[str] = Field(default_factory=list)
    excluded_installation_context: list[str] = Field(default_factory=list)
    grade: str | None = None
    temperature_c: float | None = None
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    pressure: str | None = None
    explicit_constraints: list[str] = Field(default_factory=list)
    context_only_terms: list[str] = Field(default_factory=list)
    attribute_evidence: list[str] = Field(default_factory=list)
    ambiguity: bool = False
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class QueryInterpreterDiagnostics(BaseModel):
    mode: str
    gemini_used: bool = False
    gemini_success: bool = False
    gemini_latency_ms: float = 0.0
    gemini_fallback_reason: str | None = None


class ProductAwareSearchCandidate(BaseModel):
    rank: int
    standard_id: str
    standard_code: str
    title: str
    reranker_score: float
    rrf_rank: int
    rrf_score: float
    product_compatibility: str
    canonical_product: str | None = None
    standard_kind: str | None = None
    family: str | None = None
    constraint_score: float | None = None
    constraint_flags: list[str] = Field(default_factory=list)
    raw_cross_encoder_score: float | None = None
    reranker_score_source: str | None = None
    final_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


class ProductAwareSearchTimings(BaseModel):
    gemini_ms: float | None = None
    embedding_ms: float | None = None
    semantic_retrieval_ms: float | None = None
    semantic_ms: float
    bm25_ms: float
    rrf_ms: float
    hybrid_ms: float
    product_gate_ms: float
    constraint_ms: float | None = None
    reranker_ms: float
    reranker_device: str | None = None
    reranker_model_loaded: bool | None = None
    reranker_used: bool | None = None
    reranker_timeout: bool | None = None
    reranker_success: bool | None = None
    reranker_fallback_reason: str | None = None
    reranker_inference_ms: float | None = None
    postprocess_ms: float | None = None
    total_ms: float


class ProductAwareSearchResponse(BaseModel):
    product: str
    description: str
    query: str
    canonical_product: str | None = None
    product_confidence: str
    query_interpretation: SemanticQueryInterpretation | None = None
    query_interpreter: QueryInterpreterDiagnostics | None = None
    ambiguity: bool = False
    missing_information: list[str] = Field(default_factory=list)
    candidates: list[ProductAwareSearchCandidate] = Field(default_factory=list)
    timings_ms: ProductAwareSearchTimings
