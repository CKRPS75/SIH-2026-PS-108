from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.models import Standard
from app.schemas.pilot_standard import PilotStandardRecord, PilotStandardsDataset

_WHITESPACE = re.compile(r"\s+")
_STANDARD_PREFIX = re.compile(r"^IS(?:\s*/\s*IEC)?\s+\d{2,6}\b")
_PART = re.compile(r"\(PART\s+([^)]+)\)")
_SECTION = re.compile(r"\bSEC\s+(\d+)\b")
_PUBLICATION_YEAR = re.compile(r":(\d{4})\s*$")


@dataclass(frozen=True)
class StandardPreview:
    standard_code: str
    product: str
    application: str
    material: str


@dataclass
class IngestionReport:
    dataset_name: str = ""
    expected_records: int = 0
    records_found: int = 0
    schema_valid_records: int = 0
    inserted_records: int = 0
    updated_records: int = 0
    unchanged_records: int = 0
    skipped_records: int = 0
    duplicate_standard_codes: list[str] = field(default_factory=list)
    duplicate_record_ids: list[str] = field(default_factory=list)
    missing_required_fields: list[str] = field(default_factory=list)
    empty_retrieval_texts: list[str] = field(default_factory=list)
    invalid_standard_codes: list[str] = field(default_factory=list)
    invalid_source_pages: list[str] = field(default_factory=list)
    search_profile_errors: list[str] = field(default_factory=list)
    database_verification_count: int = 0
    status: str = "FAIL"
    errors: list[str] = field(default_factory=list)
    previews: list[StandardPreview] = field(default_factory=list)

    @property
    def duplicate_standard_code_count(self) -> int:
        return len(self.duplicate_standard_codes)

    @property
    def duplicate_record_id_count(self) -> int:
        return len(self.duplicate_record_ids)

    @property
    def missing_required_field_count(self) -> int:
        return len(self.missing_required_fields)

    @property
    def empty_retrieval_text_count(self) -> int:
        return len(self.empty_retrieval_texts)

    @property
    def skipped_record_count(self) -> int:
        return self.skipped_records


def normalize_search_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value.strip().lower())


async def ingest_standards_dataset(path: str, session: AsyncSession) -> IngestionReport:
    report = IngestionReport()
    raw_data = _load_json(Path(path), report)
    if raw_data is None:
        return report

    report.dataset_name = str(raw_data.get("dataset_name") or "")
    report.expected_records = int(raw_data.get("record_count") or 0)
    report.records_found = len(raw_data.get("records") or [])

    dataset = _validate_schema(raw_data, report)
    if dataset is None:
        return report

    report.dataset_name = dataset.dataset_name
    report.expected_records = dataset.record_count
    report.records_found = len(dataset.records)
    report.schema_valid_records = len(dataset.records)
    _validate_dataset_rules(dataset, report)
    if report.errors:
        return report

    canonical_ids = []
    for record in dataset.records:
        canonical_id = canonicalize_standard_id(record.standard_code)
        canonical_ids.append(canonical_id)
        existing = await _get_existing_standard(session, canonical_id)
        values = _standard_values(dataset.dataset_name, record, canonical_id)
        if existing is None:
            session.add(Standard(**values))
            report.inserted_records += 1
            continue

        if _has_changes(existing, values):
            for key, value in values.items():
                setattr(existing, key, value)
            report.updated_records += 1
        else:
            report.unchanged_records += 1

    await session.commit()
    report.database_verification_count = await _verification_count(
        session, dataset.dataset_name, canonical_ids
    )
    report.previews = [_preview(record) for record in dataset.records]
    report.status = (
        "PASS" if report.database_verification_count == len(set(canonical_ids)) else "FAIL"
    )
    if report.status == "FAIL":
        report.errors.append("database verification count did not match ingested records")
    return report


def _load_json(path: Path, report: IngestionReport) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.errors.append(f"dataset file not found: {path}")
    except json.JSONDecodeError as exc:
        report.errors.append(f"dataset JSON is invalid: {exc}")
    return None


def _validate_schema(
    raw_data: dict[str, Any], report: IngestionReport
) -> PilotStandardsDataset | None:
    try:
        return PilotStandardsDataset.model_validate(raw_data)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            message = f"{location}: {error['msg']}"
            if error["type"] == "missing":
                if "search_profile" in error["loc"]:
                    report.search_profile_errors.append(message)
                else:
                    report.missing_required_fields.append(message)
            elif "search_profile" in error["loc"]:
                report.search_profile_errors.append(message)
            else:
                report.errors.append(message)
        report.errors.append("dataset schema validation failed")
        return None


def _validate_dataset_rules(dataset: PilotStandardsDataset, report: IngestionReport) -> None:
    if len(dataset.records) != dataset.record_count:
        report.errors.append(
            f"record_count mismatch: expected {dataset.record_count}, found {len(dataset.records)}"
        )

    record_ids = [record.record_id for record in dataset.records]
    report.duplicate_record_ids = sorted(_duplicates(record_ids))

    canonical_ids: list[str] = []
    for record in dataset.records:
        canonical_id = canonicalize_standard_id(record.standard_code)
        if not _STANDARD_PREFIX.match(canonical_id):
            report.invalid_standard_codes.append(record.standard_code)
        canonical_ids.append(canonical_id)
        if not record.retrieval_text.strip():
            report.empty_retrieval_texts.append(record.record_id)
        if (
            record.source_page_start is not None
            and record.source_page_end is not None
            and record.source_page_end < record.source_page_start
        ):
            report.invalid_source_pages.append(record.record_id)

    report.duplicate_standard_codes = sorted(_duplicates(canonical_ids))
    if dataset.family_counts is not None:
        actual_family_counts: dict[str, int] = {}
        for record in dataset.records:
            family = record.family or ""
            actual_family_counts[family] = actual_family_counts.get(family, 0) + 1
        if actual_family_counts != dataset.family_counts:
            report.errors.append(
                "family_counts mismatch: "
                f"expected {dataset.family_counts}, found {actual_family_counts}"
            )
    if report.duplicate_record_ids:
        report.errors.append("duplicate record_id values found")
    if report.duplicate_standard_codes:
        report.errors.append("duplicate canonical standard_code values found")
    if report.empty_retrieval_texts:
        report.errors.append("empty retrieval_text values found")
    if report.invalid_standard_codes:
        report.errors.append("standard_code normalization failed")
    if report.invalid_source_pages:
        report.errors.append("invalid source page values found")
    if report.search_profile_errors:
        report.errors.append("search_profile validation failed")
    report.skipped_records = len(dataset.records) if report.errors else 0


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicate_values: set[str] = set()
    for value in values:
        if value in seen:
            duplicate_values.add(value)
        seen.add(value)
    return duplicate_values


async def _get_existing_standard(session: AsyncSession, canonical_id: str) -> Standard | None:
    result = await session.execute(select(Standard).where(Standard.canonical_id == canonical_id))
    return result.scalar_one_or_none()


async def _verification_count(
    session: AsyncSession, dataset_name: str, canonical_ids: list[str]
) -> int:
    result = await session.execute(
        select(func.count(Standard.id)).where(
            Standard.source_dataset == dataset_name,
            Standard.canonical_id.in_(canonical_ids),
        )
    )
    return int(result.scalar_one())


def _standard_values(
    dataset_name: str, record: PilotStandardRecord, canonical_id: str
) -> dict[str, Any]:
    search_profile = record.search_profile.model_dump()
    return {
        "standard_id": record.standard_code,
        "canonical_id": canonical_id,
        "base_code": extract_base_code(canonical_id),
        "part": _extract_first(_PART, canonical_id),
        "section": _extract_first(_SECTION, canonical_id),
        "title": record.title,
        "scope_text": record.source_scope,
        "publication_year": _extract_publication_year(canonical_id),
        "status": record.lifecycle_status or "UNKNOWN",
        "amendments_count": 0,
        "keywords": _keywords(search_profile),
        "source_dataset": dataset_name,
        "source_record_id": record.record_id,
        "source_revision": record.source_revision,
        "source_page_start": record.source_page_start,
        "source_page_end": record.source_page_end,
        "lifecycle_status": record.lifecycle_status,
        "lifecycle_status_note": record.lifecycle_status_note,
        "retrieval_text": record.retrieval_text,
        "normalized_retrieval_text": normalize_search_text(record.retrieval_text),
        "full_text": record.full_text,
        "search_profile": search_profile,
    }


def _extract_first(pattern: re.Pattern[str], value: str) -> str | None:
    match = pattern.search(value)
    return match.group(1) if match else None


def _extract_publication_year(canonical_id: str) -> int | None:
    match = _PUBLICATION_YEAR.search(canonical_id)
    return int(match.group(1)) if match else None


def _keywords(search_profile: dict[str, list[str]]) -> list[str]:
    terms = {
        normalize_search_text(term)
        for values in search_profile.values()
        for term in values
        if term.strip()
    }
    return sorted(terms)


def _has_changes(existing: Standard, values: dict[str, Any]) -> bool:
    return any(getattr(existing, key) != value for key, value in values.items())


def _preview(record: PilotStandardRecord) -> StandardPreview:
    profile = record.search_profile
    return StandardPreview(
        standard_code=canonicalize_standard_id(record.standard_code).replace(":", ": "),
        product=", ".join(profile.product_family),
        application=", ".join(profile.applications),
        material=", ".join(profile.materials),
    )
