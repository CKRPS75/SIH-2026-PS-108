from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.services.embedding_service import BgeM3EmbeddingService
from app.services.gemini_query_interpreter import GeminiQueryInterpreter
from app.services.gemini_tender_extractor import GeminiTenderExtractor
from app.services.reranker_service import CrossEncoderRerankerService


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in request.app.state.database.session():
        yield session


def get_qdrant_client(request: Request) -> Any:
    return request.app.state.qdrant.client


def get_embedding_service(request: Request) -> BgeM3EmbeddingService:
    return request.app.state.embedding_service


def get_reranker_service(request: Request) -> CrossEncoderRerankerService:
    return request.app.state.reranker_service


def get_query_interpreter(request: Request) -> GeminiQueryInterpreter:
    return request.app.state.query_interpreter


def get_tender_extractor(request: Request) -> GeminiTenderExtractor:
    return request.app.state.tender_extractor


def get_settings_from_app(request: Request) -> Settings:
    return request.app.state.settings
