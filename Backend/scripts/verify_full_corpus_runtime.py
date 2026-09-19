from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.models import Standard  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.bm25_service import Bm25LexicalSearchService  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.parsed_standards_corpus import (  # noqa: E402
    DEFAULT_CANONICAL_DATASET,
    FULL_CORPUS_DATASET_NAME,
)


@dataclass(frozen=True)
class RuntimeVerification:
    canonical_records: int
    postgres_records: int
    postgres_unique_normalized_codes: int
    postgres_duplicate_normalized_codes: int
    qdrant_collection: str
    qdrant_vectors: int
    qdrant_duplicate_point_ids: int
    stale_pilot_collection_vectors: int | None
    bm25_documents: int
    missing_retrieval_text: int
    retrieval_text_uses_full_text: int
    counts_consistent: bool
    errors: list[str]


async def main_async() -> None:
    args = _parse_args()
    report = await verify_runtime(args.canonical, args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print("Full-corpus runtime verification")
    print(f"Canonical records: {report.canonical_records}")
    print(f"PostgreSQL records: {report.postgres_records}")
    print(f"PostgreSQL unique normalized codes: {report.postgres_unique_normalized_codes}")
    print(f"PostgreSQL duplicate normalized codes: {report.postgres_duplicate_normalized_codes}")
    print(f"Qdrant collection: {report.qdrant_collection}")
    print(f"Qdrant vectors: {report.qdrant_vectors}")
    print(f"Qdrant duplicate point IDs: {report.qdrant_duplicate_point_ids}")
    print(f"BM25 documents: {report.bm25_documents}")
    print(f"Missing retrieval_text: {report.missing_retrieval_text}")
    print(f"Retrieval text appears to embed full_text: {report.retrieval_text_uses_full_text}")
    print(f"Counts consistent: {'YES' if report.counts_consistent else 'NO'}")
    print(f"Wrote: {args.output}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.errors and args.strict:
        raise SystemExit(1)


async def verify_runtime(canonical_path: Path, collection: str) -> RuntimeVerification:
    canonical = _load_canonical_records(canonical_path)
    canonical_count = len(canonical)
    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    errors: list[str] = []
    try:
        async with database.session_factory() as session:
            postgres_count = await _postgres_count(session)
            postgres_unique = await _postgres_unique_count(session)
            postgres_duplicates = await _postgres_duplicate_count(session)
            missing_retrieval_text = await _missing_retrieval_text_count(session)
            retrieval_text_full_text = await _retrieval_text_uses_full_text_count(session)
            bm25_documents = len(
                await Bm25LexicalSearchService(
                    dataset_name=FULL_CORPUS_DATASET_NAME
                )._fetch_standards(session)
            )
        qdrant_vectors = await _qdrant_count(qdrant.client, collection)
        stale_pilot_vectors = await _qdrant_count_or_none(qdrant.client, settings.qdrant_collection)
        duplicate_point_ids = 0
    finally:
        await database.close()
        await qdrant.close()

    if canonical_count != postgres_count:
        errors.append("canonical count does not match PostgreSQL full-corpus count")
    if canonical_count != postgres_unique:
        errors.append("canonical count does not match PostgreSQL unique normalized codes")
    if postgres_duplicates:
        errors.append("PostgreSQL has duplicate normalized IS codes")
    if canonical_count != qdrant_vectors:
        errors.append("canonical count does not match Qdrant vector count")
    if canonical_count != bm25_documents:
        errors.append("canonical count does not match BM25 document count")
    if missing_retrieval_text:
        errors.append("one or more full-corpus standards are missing retrieval_text")
    if retrieval_text_full_text:
        errors.append("one or more retrieval_text values appear to include the entire full_text")
    if collection == settings.qdrant_collection and qdrant_vectors == 50:
        errors.append("runtime collection appears to be the old 50-standard pilot index")

    return RuntimeVerification(
        canonical_records=canonical_count,
        postgres_records=postgres_count,
        postgres_unique_normalized_codes=postgres_unique,
        postgres_duplicate_normalized_codes=postgres_duplicates,
        qdrant_collection=collection,
        qdrant_vectors=qdrant_vectors,
        qdrant_duplicate_point_ids=duplicate_point_ids,
        stale_pilot_collection_vectors=stale_pilot_vectors,
        bm25_documents=bm25_documents,
        missing_retrieval_text=missing_retrieval_text,
        retrieval_text_uses_full_text=retrieval_text_full_text,
        counts_consistent=not errors,
        errors=errors,
    )


async def _postgres_count(session) -> int:
    result = await session.execute(
        select(func.count(Standard.id)).where(Standard.source_dataset == FULL_CORPUS_DATASET_NAME)
    )
    return int(result.scalar_one())


async def _postgres_unique_count(session) -> int:
    result = await session.execute(
        select(func.count(func.distinct(Standard.canonical_id))).where(
            Standard.source_dataset == FULL_CORPUS_DATASET_NAME
        )
    )
    return int(result.scalar_one())


async def _postgres_duplicate_count(session) -> int:
    subquery = (
        select(Standard.canonical_id)
        .where(Standard.source_dataset == FULL_CORPUS_DATASET_NAME)
        .group_by(Standard.canonical_id)
        .having(func.count(Standard.id) > 1)
        .subquery()
    )
    result = await session.execute(select(func.count()).select_from(subquery))
    return int(result.scalar_one())


async def _missing_retrieval_text_count(session) -> int:
    result = await session.execute(
        select(func.count(Standard.id))
        .where(Standard.source_dataset == FULL_CORPUS_DATASET_NAME)
        .where((Standard.retrieval_text.is_(None)) | (func.length(Standard.retrieval_text) == 0))
    )
    return int(result.scalar_one())


async def _retrieval_text_uses_full_text_count(session) -> int:
    result = await session.execute(
        select(Standard.retrieval_text, Standard.full_text).where(
            Standard.source_dataset == FULL_CORPUS_DATASET_NAME
        )
    )
    count = 0
    for retrieval_text, full_text in result.all():
        if full_text and retrieval_text and len(full_text) > 800:
            ratio = len(retrieval_text) / len(full_text)
            if ratio > 0.8:
                count += 1
    return count


async def _qdrant_count(client: Any, collection: str) -> int:
    result = await client.count(collection_name=collection, count_filter=None, exact=True)
    return int(getattr(result, "count", result))


async def _qdrant_count_or_none(client: Any, collection: str) -> int | None:
    try:
        return await _qdrant_count(client, collection)
    except Exception:  # noqa: BLE001 - collection may not exist
        return None


def _load_canonical_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", data)
    if not isinstance(records, list):
        raise ValueError("canonical file must contain a records list")
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify full-corpus runtime counts.")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DATASET)
    parser.add_argument("--collection", default="standardwise_standards_full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/results/full_corpus_runtime_verification.json"),
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
