from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


class Database:
    """Owns the async SQLAlchemy engine and request-session factory."""

    def __init__(self, settings: Settings) -> None:
        self.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def is_ready(self) -> bool:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True

    async def close(self) -> None:
        await self.engine.dispose()
