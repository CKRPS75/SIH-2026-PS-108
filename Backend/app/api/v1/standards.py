from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.repositories.standards import StandardRepository
from app.schemas.standards import StandardValidationRequest, StandardValidationResponse
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
