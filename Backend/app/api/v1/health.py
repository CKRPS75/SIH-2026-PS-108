from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "version": request.app.state.settings.app_version,
    }


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    checks = await request.app.state.dependencies.check_all()
    embedding_ready = bool(getattr(request.app.state, "embedding_ready", True))
    ready_to_serve = checks["postgres"] == "ready" and embedding_ready
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready_to_serve else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if ready_to_serve else "not_ready", "dependencies": checks},
    )
