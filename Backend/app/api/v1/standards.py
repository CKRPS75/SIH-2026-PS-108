from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.repositories.standards import StandardRepository
from app.schemas.standards import (
    AlliedStandardsResponse,
    AlliedStandardSummary,
    StandardValidationRequest,
    StandardValidationResponse,
)
from app.services.allied_standards import AlliedStandardNotFound, AlliedStandardsGraphService
from app.services.standard_validation import StandardValidationService

router = APIRouter(prefix="/api/v1/standards", tags=["standards"])


@router.post("/validate", response_model=StandardValidationResponse)
async def validate_standard(
    request: StandardValidationRequest,
    session: AsyncSession = Depends(get_session),
) -> StandardValidationResponse:
    result = await StandardValidationService(StandardRepository(session)).validate(
        request.standard_id
    )
    if result.status == "STANDARD_NOT_FOUND":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "STANDARD_NOT_FOUND", "message": result.warnings[0]},
        )
    return result


@router.get("/{standard_id}/allied", response_model=AlliedStandardsResponse)
async def allied_standards(
    standard_id: str,
    request: Request,
) -> AlliedStandardsResponse:
    service = AlliedStandardsGraphService(_neo4j_driver(request))
    try:
        relations, warning = await service.get_allied_or_empty(standard_id)
    except AlliedStandardNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "STANDARD_NOT_FOUND", "message": str(exc)},
        ) from exc
    warnings = [warning] if warning else []
    return AlliedStandardsResponse(
        standard_id=standard_id,
        allied_standards=[
            AlliedStandardSummary(
                standard_id=relation.target_standard_id,
                title=relation.target_title,
                relation_type=relation.relation_type,
                evidence=relation.evidence,
            )
            for relation in relations
        ],
        warnings=warnings,
    )


def _neo4j_driver(request: Request) -> Any:
    neo4j = request.app.state.neo4j
    return getattr(neo4j, "driver", neo4j)
