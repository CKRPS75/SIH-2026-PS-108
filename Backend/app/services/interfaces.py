from abc import ABC, abstractmethod
from typing import Any


class TranslationService(ABC):
    @abstractmethod
    async def normalize(self, text: str, language: str = "auto") -> str: ...


class EmbeddingService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class VectorSearchService(ABC):
    @abstractmethod
    async def search(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]: ...


class LexicalSearchService(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int) -> list[dict[str, Any]]: ...


class RerankerService(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...


class GraphService(ABC):
    @abstractmethod
    async def dependencies(self, standard_id: str, hops: int = 1) -> list[dict[str, Any]]: ...


class ComplianceService(ABC):
    @abstractmethod
    async def check(
        self, standard_id: str, product_category: str | None = None
    ) -> dict[str, Any]: ...


class DocumentParser(ABC):
    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> list[dict[str, Any]]: ...
