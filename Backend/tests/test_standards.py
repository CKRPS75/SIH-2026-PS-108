import asyncio

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.base import Base
from app.db.models import Standard
from app.main import create_app
from app.repositories.standards import StandardRepository


def test_standard_identifier_normalization_preserves_part_and_year() -> None:
    assert canonicalize_standard_id(" is  16910 (Part 1) : 2020 ") == "IS 16910 (PART 1):2020"
    assert extract_base_code("IS 16910 (PART 1):2020") == "IS 16910"
    assert canonicalize_standard_id("IS/ IEC 60529 : 2018") == "IS/ IEC 60529:2018"


def test_validation_endpoint_uses_curated_metadata(tmp_path) -> None:
    database_path = tmp_path / "standards.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database_path}")
    app = create_app(settings)

    async def seed_database() -> None:
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with app.state.database.session_factory() as session:
            await StandardRepository(session).upsert(
                Standard(
                    standard_id="IS 16910 (Part 1): 2020",
                    canonical_id="placeholder",
                    base_code="placeholder",
                    title="Video surveillance systems - Part 1: General requirements",
                    status="ACTIVE",
                )
            )

    asyncio.run(seed_database())
    with TestClient(app) as client:
        found = client.post(
            "/api/v1/standards/validate", json={"standard_id": "is 16910 (part 1):2020"}
        )
        missing = client.post("/api/v1/standards/validate", json={"standard_id": "IS 99999"})

    assert found.status_code == 200
    assert found.json()["standard"]["standard_id"] == "IS 16910 (Part 1): 2020"
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "STANDARD_NOT_FOUND"
