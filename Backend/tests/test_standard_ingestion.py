import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.core.config import Settings
from app.core.standards import canonicalize_standard_id
from app.db.base import Base
from app.db.models import Standard
from app.db.session import Database
from app.main import create_app
from app.services.standard_ingestion import ingest_standards_dataset

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "standardwise_tile_pilot_dataset.json"
EXPANDED_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "standardwise_50_pilot_dataset.json"
)
EXPECTED_FAMILY_COUNTS = {
    "tile": 9,
    "pipe_water_drainage": 10,
    "thermal_insulation": 10,
    "cement": 8,
    "fastener": 7,
    "water_valve": 6,
}
EXPECTED_NEW_CODES = {
    "IS 12709: 1994",
    "IS 12818: 1992",
    "IS 13592: 1992",
    "IS 14333: 1996",
    "IS 14402: 1996",
    "IS 14735: 1999",
    "IS 1239 (Part 1): 2004",
    "IS 3589: 2001",
    "IS 4270: 2001",
    "IS 651: 1992",
    "IS 3677: 1985",
    "IS 4671: 1984",
    "IS 6598: 1972",
    "IS 7509: 1993",
    "IS 8154: 1993",
    "IS 8183: 1993",
    "IS 9428: 1993",
    "IS 9742: 1993",
    "IS 9842: 1994",
    "IS 12436: 1988",
    "IS 269: 1989",
    "IS 8112: 1989",
    "IS 455: 1989",
    "IS 1489 (Part 1): 1991",
    "IS 1489 (Part 2): 1991",
    "IS 3466: 1988",
    "IS 8041: 1990",
    "IS 8042: 1989",
    "IS 1363 (Part 1): 2002",
    "IS 1363 (Part 2): 2002",
    "IS 1363 (Part 3): 2002",
    "IS 1364 (Part 1): 2002",
    "IS 1365: 1978",
    "IS 2016: 1967",
    "IS 3757: 1985",
    "IS 5312 (Part 1): 2004",
    "IS 5312 (Part 2): 1986",
    "IS 9338: 1984",
    "IS 9739: 1981",
    "IS 14845: 2000",
    "IS 14846: 2000",
}


def _dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _expanded_dataset() -> dict[str, Any]:
    return json.loads(EXPANDED_DATASET_PATH.read_text(encoding="utf-8"))


def _write_dataset(tmp_path: Path, dataset: dict[str, Any]) -> Path:
    path = tmp_path / "pilot_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


async def _create_database(tmp_path: Path) -> Database:
    database_path = tmp_path / "standards.db"
    database = Database(Settings(database_url=f"sqlite+aiosqlite:///{database_path}"))
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return database


def test_valid_9_record_dataset_loads_successfully(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(DATASET_PATH), session)

            assert report.status == "PASS"
            assert report.expected_records == 9
            assert report.records_found == 9
            assert report.schema_valid_records == 9
            assert report.inserted_records == 9
            assert report.skipped_records == 0
            assert report.database_verification_count == 9
            assert len(report.previews) == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_expanded_dataset_has_expected_records_families_and_codes() -> None:
    dataset = _expanded_dataset()
    records = dataset["records"]
    canonical_codes = [canonicalize_standard_id(record["standard_code"]) for record in records]
    actual_family_counts: dict[str, int] = {}
    for record in records:
        family = record["family"]
        actual_family_counts[family] = actual_family_counts.get(family, 0) + 1

    tile_codes = {
        canonicalize_standard_id(record["standard_code"]) for record in _dataset()["records"]
    }
    expanded_tile_codes = {
        canonicalize_standard_id(record["standard_code"])
        for record in records
        if record["family"] == "tile"
    }

    assert dataset["record_count"] == 50
    assert len(records) == 50
    assert len(canonical_codes) == len(set(canonical_codes)) == 50
    assert actual_family_counts == dataset["family_counts"] == EXPECTED_FAMILY_COUNTS
    assert all(record["retrieval_text"].strip() for record in records)
    assert expanded_tile_codes == tile_codes
    assert {canonicalize_standard_id(code) for code in EXPECTED_NEW_CODES} == set(
        canonical_codes
    ) - tile_codes


def test_expanded_ingestion_reuses_tile_rows_and_is_idempotent(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                await ingest_standards_dataset(str(DATASET_PATH), session)
                initial_rows = list((await session.execute(select(Standard))).scalars().all())
                initial_ids = {row.canonical_id: row.id for row in initial_rows}

                first_report = await ingest_standards_dataset(str(EXPANDED_DATASET_PATH), session)
                rows = list((await session.execute(select(Standard))).scalars().all())
                second_report = await ingest_standards_dataset(str(EXPANDED_DATASET_PATH), session)

            assert first_report.status == "PASS"
            assert first_report.inserted_records == 41
            assert first_report.updated_records == 9
            assert len(rows) == 50
            assert len({row.canonical_id for row in rows}) == 50
            assert all(row.source_dataset == "standardwise_50_pilot_dataset" for row in rows)
            assert all(isinstance(row.id, UUID) for row in rows)
            assert {
                row.canonical_id: row.id for row in rows if row.canonical_id in initial_ids
            } == initial_ids
            assert second_report.status == "PASS"
            assert second_report.inserted_records == 0
            assert second_report.unchanged_records == 50
        finally:
            await database.close()

    asyncio.run(run())


def test_record_count_mismatch_fails(tmp_path) -> None:
    dataset = _dataset()
    dataset["record_count"] = 10
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert report.skipped_records == 9
            assert "record_count mismatch" in report.errors[0]
        finally:
            await database.close()

    asyncio.run(run())


def test_duplicate_record_id_fails(tmp_path) -> None:
    dataset = _dataset()
    dataset["records"][1]["record_id"] = dataset["records"][0]["record_id"]
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert report.duplicate_record_id_count == 1
            assert report.skipped_records == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_duplicate_normalized_standard_code_fails(tmp_path) -> None:
    dataset = _dataset()
    dataset["records"][1]["standard_code"] = dataset["records"][0]["standard_code"].lower()
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert report.duplicate_standard_code_count == 1
            assert report.skipped_records == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_missing_retrieval_text_fails(tmp_path) -> None:
    dataset = _dataset()
    del dataset["records"][0]["retrieval_text"]
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert report.missing_required_field_count == 1
        finally:
            await database.close()

    asyncio.run(run())


def test_empty_retrieval_text_fails(tmp_path) -> None:
    dataset = _dataset()
    dataset["records"][0]["retrieval_text"] = "  "
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert report.empty_retrieval_text_count == 1
            assert report.skipped_records == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_missing_required_search_profile_structure_fails(tmp_path) -> None:
    dataset = _dataset()
    del dataset["records"][0]["search_profile"]["applications"]
    path = _write_dataset(tmp_path, dataset)

    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                report = await ingest_standards_dataset(str(path), session)

            assert report.status == "FAIL"
            assert len(report.search_profile_errors) == 1
        finally:
            await database.close()

    asyncio.run(run())


def test_ingestion_is_idempotent(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                first_report = await ingest_standards_dataset(str(DATASET_PATH), session)
                second_report = await ingest_standards_dataset(str(DATASET_PATH), session)

                result = await session.execute(select(func.count(Standard.id)))
                count = result.scalar_one()

            assert first_report.inserted_records == 9
            assert second_report.inserted_records == 0
            assert second_report.unchanged_records == 9
            assert count == 9
        finally:
            await database.close()

    asyncio.run(run())


def test_all_9_records_are_present_in_postgresql_after_ingestion(tmp_path) -> None:
    async def run() -> None:
        database = await _create_database(tmp_path)
        try:
            async with database.session_factory() as session:
                await ingest_standards_dataset(str(DATASET_PATH), session)
                result = await session.execute(
                    select(Standard).where(
                        Standard.source_dataset == "standardwise_tile_pilot_dataset"
                    )
                )
                standards = list(result.scalars().all())

            assert len(standards) == 9
            assert {standard.canonical_id for standard in standards} >= {
                "IS 3367:1993",
                "IS 3951 (PART 1):1975",
                "IS 3951 (PART 2):1975",
                "IS 2690 (PART 1):1993",
                "IS 2690 (PART 2):1992",
            }
        finally:
            await database.close()

    asyncio.run(run())


def test_validation_endpoint_resolves_ingested_standard(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)

    async def seed_database() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with app.state.database.session_factory() as session:
            await ingest_standards_dataset(str(DATASET_PATH), session)

    asyncio.run(seed_database())
    with TestClient(app) as client:
        response = client.post("/api/v1/standards/validate", json={"standard_id": "IS 3367:1993"})

    assert response.status_code == 200
    standard = response.json()["standard"]
    assert standard["canonical_id"] == "IS 3367:1993"
    assert standard["title"] == "BURNT CLAY TILES FOR USE IN LINING"
    assert standard["search_profile"]["applications"] == [
        "irrigation canal lining",
        "drainage channel lining",
    ]


def test_invalid_standard_still_returns_standard_not_found(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)

    async def seed_database() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with app.state.database.session_factory() as session:
            await ingest_standards_dataset(str(DATASET_PATH), session)

    asyncio.run(seed_database())
    with TestClient(app) as client:
        response = client.post("/api/v1/standards/validate", json={"standard_id": "IS 99999"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "STANDARD_NOT_FOUND"
