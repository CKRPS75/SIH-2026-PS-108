from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RERANKER_MODEL = BACKEND_ROOT / "models" / "bge-reranker-large"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "StandardWise API"
    app_env: str = "development"
    app_version: str = "0.1.0"
    database_url: str = "postgresql+asyncpg://standardwise:password@localhost:5432/standardwise"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "standards_v1"
    qdrant_full_collection: str = "standardwise_standards_full"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change_me"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = str(LOCAL_RERANKER_MODEL)
    translation_provider: str = "indictrans2"
    llm_provider: str = "ollama"
    llm_model: str = "llama3:8b"
    dense_k: int = Field(default=30, ge=1)
    bm25_k: int = Field(default=30, ge=1)
    hybrid_semantic_k: int = Field(default=20, ge=1)
    hybrid_bm25_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_k: int = Field(default=10, ge=1)
    rerank_batch_size: int = Field(default=8, ge=1)
    reranker_mode: Literal["auto", "enabled", "disabled"] = "auto"
    product_aware_reranker_timeout_s: float = Field(default=8.0, gt=0)
    query_interpreter_mode: str = "disabled"
    query_interpreter_timeout_s: float = Field(default=5.0, gt=0)
    query_interpreter_cache_ttl_s: float = Field(default=300.0, gt=0)
    query_interpreter_cache_max_size: int = Field(default=128, ge=1)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    return_k: int = Field(default=5, ge=1)
    upload_max_mb: int = Field(default=25, ge=1)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
