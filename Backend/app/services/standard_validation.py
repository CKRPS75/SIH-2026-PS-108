from app.db.models import Standard
from app.repositories.standards import StandardRepository
from app.schemas.standards import StandardSummary, StandardValidationResponse


class StandardValidationService:
    def __init__(self, repository: StandardRepository) -> None:
        self._repository = repository

    async def validate(self, standard_id: str) -> StandardValidationResponse:
        standard = await self._repository.get_by_canonical_id(standard_id)
        if standard is None:
            return StandardValidationResponse(
                status="STANDARD_NOT_FOUND",
                warnings=["The supplied identifier is absent from the curated standards corpus."],
            )

        replacement = await self._replacement(standard)
        warnings: list[str] = []
        if standard.status in {"SUPERSEDED", "WITHDRAWN"}:
            warnings.append(f"This standard is {standard.status.lower()} in curated metadata.")
        return StandardValidationResponse(
            status=standard.status,
            standard=standard,
            replacement=replacement,
            warnings=warnings,
        )

    async def _replacement(self, standard: Standard) -> StandardSummary | None:
        if standard.replaced_by_id is None:
            return None
        candidates = await self._repository.lookup_by_base_code(standard.base_code)
        replacement = next(
            (item for item in candidates if item.id == standard.replaced_by_id), None
        )
        return StandardSummary.model_validate(replacement) if replacement else None
