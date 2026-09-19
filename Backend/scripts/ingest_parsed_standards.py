from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.standards import extract_base_code  # noqa: E402
from app.db.models import Standard  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.parsed_standards_corpus import (  # noqa: E402
    DEFAULT_CANONICAL_DATASET,
    FULL_CORPUS_DATASET_NAME,
    CanonicalStandard,
    load_canonical_dataset,
)


@dataclass(frozen=True)
class ParsedIngestionReport:
    dataset_records: int
    inserted: int
    updated: int
    unchanged: int
    postgres_count: int
    postgres_unique_count: int


async def ingest_parsed_standards_dataset(path: Path, session) -> ParsedIngestionReport:
    records = load_canonical_dataset(path)
    inserted = 0
    updated = 0
    unchanged = 0
    for record in records:
        values = _standard_values(record)
        existing = await _existing_standard(session, record.standard_code_norm)
        if existing is None:
            session.add(Standard(**values))
            inserted += 1
        elif _has_changes(existing, values):
            for key, value in values.items():
                setattr(existing, key, value)
            updated += 1
        else:
            unchanged += 1
    await session.commit()
    postgres_count = await _count_full_corpus(session)
    postgres_unique_count = await _count_unique_full_corpus(session)
    return ParsedIngestionReport(
        dataset_records=len(records),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        postgres_count=postgres_count,
        postgres_unique_count=postgres_unique_count,
    )


async def main_async() -> None:
    args = _parse_args()
    database = Database(get_settings())
    try:
        async with database.session_factory() as session:
            report = await ingest_parsed_standards_dataset(args.dataset, session)
        print("Parsed standards ingestion")
        print(f"Dataset records: {report.dataset_records}")
        print(f"Inserted: {report.inserted}")
        print(f"Updated: {report.updated}")
        print(f"Unchanged: {report.unchanged}")
        print(f"PostgreSQL full-corpus count: {report.postgres_count}")
        print(f"PostgreSQL unique normalized codes: {report.postgres_unique_count}")
        if report.postgres_count != report.dataset_records:
            raise SystemExit("PostgreSQL count does not match canonical dataset count")
        if report.postgres_unique_count != report.dataset_records:
            raise SystemExit("PostgreSQL unique normalized count does not match dataset count")
    finally:
        await database.close()


def _standard_values(record: CanonicalStandard) -> dict[str, Any]:
    keywords = [
        value
        for value in [
            record.canonical_product,
            record.product_subtype,
            record.primary_subject,
            record.material,
            record.application,
            record.function,
            record.standard_kind,
            record.family,
            *(record.product_aliases or []),
            *(record.applies_to_product_families or []),
        ]
        if value
    ]
    search_profile = {
        "canonical_product": record.canonical_product,
        "product_subtype": record.product_subtype,
        "primary_subject": record.primary_subject,
        "product_aliases": record.product_aliases,
        "product_confidence": record.product_confidence,
        "product_derivation_source": record.product_derivation_source,
        "material": record.material,
        "application": record.application,
        "function": record.function,
        "applies_to_product_families": record.applies_to_product_families,
        "metadata_confidence": record.metadata_confidence,
        "metadata_evidence": record.metadata_evidence,
        "metadata_derivation_method": record.metadata_derivation_method,
        "review_required": record.review_required,
        "review_reason": record.review_reason,
        "standard_kind": record.standard_kind,
        "family": record.family,
        "source_provenance": record.source_provenance,
        "quality_flags": record.quality_flags,
    }
    return {
        "standard_id": record.standard_code,
        "canonical_id": record.standard_code_norm,
        "standard_code_norm": record.standard_code_norm,
        "base_code": extract_base_code(record.standard_code_norm),
        "part": None,
        "section": None,
        "title": record.title or record.standard_code,
        "scope_text": record.scope,
        "scope": record.scope,
        "division": None,
        "publication_year": _revision_year(record.revision),
        "status": "SOURCE_PARSED",
        "amendments_count": 0,
        "keywords": keywords,
        "source_dataset": FULL_CORPUS_DATASET_NAME,
        "source_record_id": record.standard_code_norm,
        "source_revision": record.revision,
        "source_page_start": record.page_start,
        "source_page_end": record.page_end,
        "page_start": record.page_start,
        "page_end": record.page_end,
        "lifecycle_status": None,
        "lifecycle_status_note": (
            "Historical parsed summary corpus; current lifecycle/QCO status not inferred."
        ),
        "retrieval_text": record.retrieval_text,
        "normalized_retrieval_text": " ".join(record.retrieval_text.casefold().split()),
        "full_text": record.full_text,
        "standard_kind": record.standard_kind,
        "canonical_product": record.canonical_product,
        "product_subtype": record.product_subtype,
        "primary_subject": record.primary_subject,
        "product_aliases": record.product_aliases,
        "material": record.material,
        "application": record.application,
        "function": record.function,
        "applies_to_product_families": record.applies_to_product_families,
        "product_confidence": record.product_confidence,
        "metadata_confidence": record.metadata_confidence,
        "metadata_evidence": record.metadata_evidence,
        "metadata_derivation_method": record.metadata_derivation_method,
        "review_required": record.review_required,
        "review_reason": record.review_reason,
        "raw_title": record.raw_title,
        "canonical_title_expanded": record.canonical_title_expanded,
        "title_evidence": record.title_evidence,
        "scope_reconstructed": record.scope_reconstructed,
        "scope_source": record.scope_source,
        "family": record.family,
        "source_provenance": record.source_provenance,
        "search_profile": search_profile,
    }


async def _existing_standard(session, canonical_id: str) -> Standard | None:
    result = await session.execute(select(Standard).where(Standard.canonical_id == canonical_id))
    return result.scalars().first()


async def _count_full_corpus(session) -> int:
    result = await session.execute(
        select(func.count(Standard.id)).where(Standard.source_dataset == FULL_CORPUS_DATASET_NAME)
    )
    return int(result.scalar_one())


async def _count_unique_full_corpus(session) -> int:
    result = await session.execute(
        select(func.count(func.distinct(Standard.canonical_id))).where(
            Standard.source_dataset == FULL_CORPUS_DATASET_NAME
        )
    )
    return int(result.scalar_one())


def _has_changes(existing: Standard, values: dict[str, Any]) -> bool:
    return any(getattr(existing, key) != value for key, value in values.items())


def _revision_year(revision: str | None) -> int | None:
    if not revision:
        return None
    for token in reversed(revision.replace(":", " ").split()):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest canonical parsed standards into DB.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CANONICAL_DATASET)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
