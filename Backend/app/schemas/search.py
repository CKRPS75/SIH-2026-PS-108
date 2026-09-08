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
