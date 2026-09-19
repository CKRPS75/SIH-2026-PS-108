from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PilotSearchProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    product_family: list[str]
    applications: list[str]
    materials: list[str]
    functions: list[str]
    properties: list[str]
    environments: list[str]
    grades: list[str]
    form_type: list[str]
    manufacturing_method: list[str]
    operating_conditions: list[str]
    dimensions: list[str]
    exclusions: list[str]
    aliases: list[str]
    discriminating_attributes: list[str]


class PilotStandardRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_revision: str | None
    source_page_start: int | None = Field(ge=1)
    source_page_end: int | None = Field(ge=1)
    source_scope: str
    lifecycle_status: str | None
    lifecycle_status_note: str
    search_profile: PilotSearchProfile
    retrieval_text: str
    full_text: str
    family: str | None = Field(default=None, min_length=1)

    @field_validator("record_id", "standard_code", "title", "lifecycle_status_note", mode="before")
    @classmethod
    def reject_blank_required_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("field must not be blank")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> "PilotStandardRecord":
        if (
            self.source_page_start is not None
            and self.source_page_end is not None
            and self.source_page_end < self.source_page_start
        ):
            raise ValueError("source_page_end must be greater than or equal to source_page_start")
        return self


class PilotStandardsDataset(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    records: list[PilotStandardRecord]
    family_counts: dict[str, int] | None = None
