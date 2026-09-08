from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StandardSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    standard_id: str
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


class StandardValidationRequest(BaseModel):
    standard_id: str = Field(min_length=3, max_length=255)


class StandardValidationResponse(BaseModel):
    status: str
    standard: StandardDetail | None = None
    replacement: StandardSummary | None = None
    warnings: list[str] = Field(default_factory=list)
