from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.parsed_standards_corpus import FULL_CORPUS_DATASET_NAME  # noqa: E402
from app.services.standard_vector_index import (  # noqa: E402
    StandardVectorIndex,
    count_pilot_standards,
)


async def main_async() -> None:
    args = _parse_args()
    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    try:
        async with database.session_factory() as session:
            expected = await count_pilot_standards(session, FULL_CORPUS_DATASET_NAME)
            index = StandardVectorIndex(
                qdrant.client,
                BgeM3EmbeddingService(settings.embedding_model),
                args.collection,
                dataset_name=FULL_CORPUS_DATASET_NAME,
            )
            report = await index.index_standards(session, expected_records=expected)
        print("Full-corpus Qdrant indexing")
        print(f"Collection: {args.collection}")
        print(f"PostgreSQL standards: {report.postgres_pilot_standards}")
        print(f"Embeddings generated: {report.embeddings_generated}")
        print(f"Vectors upserted: {report.vectors_inserted_or_updated}")
        print(f"Qdrant count: {report.qdrant_pilot_records_verified}")
        print(f"Status: {report.status}")
        if report.errors:
            for error in report.errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
    finally:
        await database.close()
        await qdrant.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index parsed full corpus into Qdrant.")
    parser.add_argument("--collection", default="standardwise_standards_full")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
