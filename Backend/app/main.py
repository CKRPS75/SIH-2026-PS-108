from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router
from app.api.v1.standards import router as standards_router
from app.core.config import Settings, get_settings
from app.db.session import Database
from app.docs import register_docs_routes
from app.services.clients import Neo4jClientService, QdrantClientService
from app.services.dependencies import DependencyHealthService
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.gemini_query_interpreter import GeminiQueryInterpreter
from app.services.reranker_service import CrossEncoderRerankerService


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
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
        docs_url=None,
    )
    app.state.settings = resolved_settings
    app.state.database = Database(resolved_settings)
    app.state.qdrant = QdrantClientService(resolved_settings)
    app.state.neo4j = Neo4jClientService(resolved_settings)
    app.state.embedding_service = BgeM3EmbeddingService(resolved_settings.embedding_model)
    app.state.reranker_service = CrossEncoderRerankerService(
        resolved_settings.reranker_model,
        batch_size=resolved_settings.rerank_batch_size,
        reranker_mode=resolved_settings.reranker_mode,
    )
    app.state.query_interpreter = GeminiQueryInterpreter(
        api_key=resolved_settings.gemini_api_key,
        model_name=resolved_settings.gemini_model,
        timeout_s=resolved_settings.query_interpreter_timeout_s,
        cache_ttl_s=resolved_settings.query_interpreter_cache_ttl_s,
        cache_max_size=resolved_settings.query_interpreter_cache_max_size,
    )
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
    app.include_router(search_router)
    register_docs_routes(app)
    return app


app = create_app()
