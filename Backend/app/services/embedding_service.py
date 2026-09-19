from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar


class BgeM3EmbeddingService:
    """Lazy, reusable BGE-M3 embedding service."""

    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model = model

    async def embed(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._validate_texts(texts)
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self._encode([text])[0]

    def embedding_dimension(self) -> int:
        model = self._load_model()
        dimension = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        if dimension is not None:
            return int(dimension)
        return len(self.embed_query("standardwise embedding dimension probe"))

    def warmup(self) -> int:
        """Load BGE-M3 once and run a tiny embedding probe."""
        return len(self.embed_query("standardwise warmup"))

    def _load_model(self) -> Any:
        if self._model is None:
            cached_model = self._model_cache.get(self.model_name)
            if cached_model is not None:
                self._model = cached_model
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "sentence-transformers is required for BAAI/bge-m3 embeddings"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self._model_cache[self.model_name] = self._model
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load_model()
        encoded = model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return [_as_float_list(vector) for vector in encoded]

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"text at index {index} must not be empty")


def _as_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
