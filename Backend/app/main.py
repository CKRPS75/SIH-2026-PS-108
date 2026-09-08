from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.v1.health import router as health_router
from app.api.v1.standards import router as standards_router
from app.core.config import Settings, get_settings
from app.db.session import Database
from app.services.clients import Neo4jClientService, QdrantClientService
from app.services.dependencies import DependencyHealthService


async def unavailable() -> bool:
    return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await app.state.database.close()
        await app.state.qdrant.close()
        await app.state.neo4j.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name, version=resolved_settings.app_version, lifespan=lifespan
    )
    app.state.settings = resolved_settings
    app.state.database = Database(resolved_settings)
    app.state.qdrant = QdrantClientService(resolved_settings)
    app.state.neo4j = Neo4jClientService(resolved_settings)
    app.state.dependencies = DependencyHealthService(
        app.state.database.is_ready,
        app.state.qdrant.is_ready,
        app.state.neo4j.is_ready,
    )

    @app.middleware("http")
    async def attach_trace_id(request: Request, call_next):
        request.state.trace_id = request.headers.get("X-Trace-ID", str(uuid4()))
        response = await call_next(request)
        response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    app.include_router(health_router)
    app.include_router(standards_router)
    return app


app = create_app()
