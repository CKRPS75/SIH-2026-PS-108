from typing import Any

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local envs
    AsyncDriver = Any
    AsyncGraphDatabase = None

try:
    from qdrant_client import AsyncQdrantClient
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local envs
    AsyncQdrantClient = None

from app.core.config import Settings


class QdrantClientService:
    def __init__(self, settings: Settings) -> None:
        if AsyncQdrantClient is None:
            self.client = None
            return
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def is_ready(self) -> bool:
        if self.client is None:
            return False
        await self.client.get_collections()
        return True

    async def close(self) -> None:
        if self.client is None:
            return
        await self.client.close()


class Neo4jClientService:
    def __init__(self, settings: Settings) -> None:
        if AsyncGraphDatabase is None:
            self.driver = None
            return
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def is_ready(self) -> bool:
        if self.driver is None:
            return False
        await self.driver.verify_connectivity()
        return True

    async def close(self) -> None:
        if self.driver is None:
            return
        await self.driver.close()
