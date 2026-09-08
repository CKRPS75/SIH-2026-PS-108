from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "StandardWise API"
    app_env: str = "development"
    app_version: str = "0.1.0"
    database_url: str = "postgresql+asyncpg://standardwise:password@localhost:5432/standardwise"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "standards_v1"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change_me"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-large"
    translation_provider: str = "indictrans2"
    llm_provider: str = "ollama"
    llm_model: str = "llama3:8b"
    dense_k: int = Field(default=30, ge=1)
    bm25_k: int = Field(default=30, ge=1)
    rerank_k: int = Field(default=20, ge=1)
    return_k: int = Field(default=5, ge=1)
    upload_max_mb: int = Field(default=25, ge=1)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
