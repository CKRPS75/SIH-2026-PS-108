from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Standard
from app.services.standard_vector_index import PILOT_DATASET_NAME

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Bm25Candidate:
    standard_id: str
    standard_code: str
    title: str
    bm25_rank: int
    bm25_score: float


class Bm25LexicalSearchService:
    def __init__(
        self,
        *,
        dataset_name: str = PILOT_DATASET_NAME,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._dataset_name = dataset_name
        self._k1 = k1
        self._b = b

    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 20,
    ) -> list[Bm25Candidate]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must not be empty")

        standards = await self._fetch_standards(session)
        if not standards:
            return []

        documents = [tokenize(standard.retrieval_text or "") for standard in standards]
        bm25 = _Bm25Okapi(documents, k1=self._k1, b=self._b)
        scores = bm25.score(query_tokens)
        ranked = [
            (standard, score)
            for standard, score in zip(standards, scores, strict=True)
            if score > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].canonical_id))

        return [
            Bm25Candidate(
                standard_id=str(standard.id),
                standard_code=standard.standard_id,
                title=standard.title,
                bm25_rank=rank,
                bm25_score=score,
            )
            for rank, (standard, score) in enumerate(ranked[:limit], start=1)
        ]

    async def _fetch_standards(self, session: AsyncSession) -> list[Standard]:
        result = await session.execute(
            select(Standard)
            .where(Standard.source_dataset == self._dataset_name)
            .where(Standard.retrieval_text.is_not(None))
            .order_by(Standard.canonical_id)
        )
        return list(result.scalars().all())


class _Bm25Okapi:
    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float,
        b: float,
    ) -> None:
        self._documents = documents
        self._k1 = k1
        self._b = b
        self._doc_lengths = [len(document) for document in documents]
        self._average_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._term_frequencies = [Counter(document) for document in documents]
        self._idf = self._compute_idf(documents)

    def score(self, query_tokens: list[str]) -> list[float]:
        return [
            self._score_document(term_frequency, doc_length, query_tokens)
            for term_frequency, doc_length in zip(
                self._term_frequencies,
                self._doc_lengths,
                strict=True,
            )
        ]

    def _score_document(
        self,
        term_frequency: Counter[str],
        doc_length: int,
        query_tokens: list[str],
    ) -> float:
        if not self._average_doc_length:
            return 0.0

        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if frequency == 0:
                continue
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * doc_length / self._average_doc_length
            )
            score += self._idf[token] * frequency * (self._k1 + 1) / denominator
        return score

    @staticmethod
    def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
        document_count = len(documents)
        document_frequencies: Counter[str] = Counter()
        for document in documents:
            document_frequencies.update(set(document))
        return {
            token: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(" ".join(text.lower().split()))
