from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.standards import canonicalize_standard_id, extract_base_code
from app.db.models import Standard


class StandardRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, standard: Standard) -> Standard:
        standard.canonical_id = canonicalize_standard_id(standard.standard_id)
        standard.base_code = extract_base_code(standard.canonical_id)
        existing = await self.get_by_canonical_id(standard.canonical_id)
        if existing is None:
            self._session.add(standard)
            await self._session.commit()
            await self._session.refresh(standard)
            return standard

        for column in Standard.__table__.columns:
            if column.name not in {"id", "canonical_id"}:
                setattr(existing, column.name, getattr(standard, column.name))
        await self._session.commit()
        await self._session.refresh(existing)
        return existing

    async def get_by_canonical_id(self, standard_id: str) -> Standard | None:
        canonical_id = canonicalize_standard_id(standard_id)
        result = await self._session.execute(
            select(Standard).where(Standard.canonical_id == canonical_id)
        )
        return result.scalar_one_or_none()

    async def lookup_by_base_code(self, base_code: str) -> list[Standard]:
        result = await self._session.execute(
            select(Standard).where(Standard.base_code == extract_base_code(base_code))
        )
        return list(result.scalars().all())
