from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.db.session import Database
from app.services.standard_ingestion import IngestionReport, ingest_standards_dataset

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "standardwise_50_pilot_dataset.json"
)


async def main(dataset_path: Path) -> None:
    database = Database(get_settings())
    try:
        async with database.session_factory() as session:
            report = await ingest_standards_dataset(str(dataset_path), session)
        _print_report(report)
    finally:
        await database.close()


def _print_report(report: IngestionReport) -> None:
    print(f"Dataset: {report.dataset_name}")
    print()
    print(f"Expected records: {report.expected_records}")
    print(f"Records found: {report.records_found}")
    print(f"Schema-valid: {report.schema_valid_records}")
    print(f"Inserted: {report.inserted_records}")
    print(f"Updated: {report.updated_records}")
    print(f"Unchanged: {report.unchanged_records}")
    print(f"Skipped: {report.skipped_records}")
    print()
    print(f"Duplicate standard codes: {report.duplicate_standard_code_count}")
    print(f"Duplicate record IDs: {report.duplicate_record_id_count}")
    print(f"Missing required fields: {report.missing_required_field_count}")
    print(f"Empty retrieval texts: {report.empty_retrieval_text_count}")
    print()
    print(f"Database verification: {report.database_verification_count} / {report.records_found}")
    print()
    if report.previews:
        print("Preview:")
        for preview in report.previews:
            print()
            print(preview.standard_code)
            print(f"Product: {preview.product}")
            print(f"Application: {preview.application}")
            print(f"Material: {preview.material}")
        print()
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
        print()
    print(f"STATUS: {report.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a StandardWise pilot dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.dataset))
