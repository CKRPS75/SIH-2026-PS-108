from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StandardSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    standard_id: str
    canonical_id: str
    title: str
    status: str
    publication_year: int | None = None


class StandardDetail(StandardSummary):
    base_code: str
    part: str | None = None
    section: str | None = None
    scope_text: str | None = None
    division: str | None = None
    amendments_count: int
    source_url: str | None = None
    last_verified_at: datetime | None = None
    source_dataset: str | None = None
    source_record_id: str | None = None
    source_revision: str | None = None
    source_page_start: int | None = None
    source_page_end: int | None = None
    lifecycle_status: str | None = None
    lifecycle_status_note: str | None = None
    retrieval_text: str | None = None
    normalized_retrieval_text: str | None = None
    full_text: str | None = None
    search_profile: dict = Field(default_factory=dict)


class StandardValidationRequest(BaseModel):
    standard_id: str = Field(min_length=3, max_length=255)


class StandardValidationResponse(BaseModel):
    status: str
    standard: StandardDetail | None = None
    replacement: StandardSummary | None = None
    warnings: list[str] = Field(default_factory=list)


class AlliedStandardSummary(BaseModel):
    standard_id: str
    title: str | None = None
    relation_type: str
    evidence: str | None = None


class AlliedStandardsResponse(BaseModel):
    standard_id: str
    allied_standards: list[AlliedStandardSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
