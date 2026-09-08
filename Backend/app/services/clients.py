from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


class QdrantClientService:
    def __init__(self, settings: Settings) -> None:
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def is_ready(self) -> bool:
        await self.client.get_collections()
        return True

    async def close(self) -> None:
        await self.client.close()


class Neo4jClientService:
    def __init__(self, settings: Settings) -> None:
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def is_ready(self) -> bool:
        await self.driver.verify_connectivity()
        return True

    async def close(self) -> None:
        await self.driver.close()
