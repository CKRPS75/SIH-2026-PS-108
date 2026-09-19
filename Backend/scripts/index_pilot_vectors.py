from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.standard_vector_index import (  # noqa: E402
    PilotIndexReport,
    StandardVectorIndex,
)

DEFAULT_DATASET_PATH = ROOT / "data" / "standardwise_50_pilot_dataset.json"


async def main(dataset_path: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = str(dataset["dataset_name"])
    expected_records = int(dataset["record_count"])
    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    embedding_service = BgeM3EmbeddingService(settings.embedding_model)
    index = StandardVectorIndex(
        qdrant.client,
        embedding_service,
        settings.qdrant_collection,
        dataset_name=dataset_name,
    )
    try:
        async with database.session_factory() as session:
            report = await index.index_pilot_standards(session, expected_records=expected_records)
        _print_report(report)
    finally:
        await database.close()
        await qdrant.close()


def _print_report(report: PilotIndexReport) -> None:
    print(f"Collection: {report.collection}")
    print()
    print(f"PostgreSQL pilot standards: {report.postgres_pilot_standards}")
    print(f"Valid retrieval texts: {report.valid_retrieval_texts}")
    print(f"Embeddings generated: {report.embeddings_generated}")
    print(f"Vectors inserted/updated: {report.vectors_inserted_or_updated}")
    print(
        "Qdrant pilot records verified: "
        f"{report.qdrant_pilot_records_verified} / {report.expected_records}"
    )
    print()
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
        print()
    print(f"STATUS: {report.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a StandardWise pilot dataset in Qdrant.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.dataset))
