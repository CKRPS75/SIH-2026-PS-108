from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.standards import canonicalize_standard_id


class AlliedStandardNotFound(ValueError):
    pass


@dataclass(frozen=True)
class AlliedStandardRelation:
    source_standard_id: str
    target_standard_id: str
    relation_type: str
    target_title: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class AlliedLoadReport:
    input_relations: int
    nodes_merged: int
    edges_merged: int


class AlliedStandardsGraphService:
    def __init__(self, driver: Any) -> None:
        self._driver = driver

    async def get_allied(self, standard_id: str) -> list[AlliedStandardRelation]:
        canonical_id = canonicalize_standard_id(standard_id)
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Standard {standard_id: $standard_id})
                    OPTIONAL MATCH (source)-[edge:ALLIED_WITH]-(target:Standard)
                    RETURN
                      source.standard_id AS source_standard_id,
                      target.standard_id AS target_standard_id,
                      target.title AS target_title,
                      edge.relation_type AS relation_type,
                      edge.evidence AS evidence
                    ORDER BY relation_type, target_standard_id
                    """,
                    standard_id=canonical_id,
                )
                records = await result.data()
        except Exception as exc:  # noqa: BLE001 - callers decide whether to degrade
            raise RuntimeError("Neo4j allied standards lookup failed") from exc

        if not records:
            raise AlliedStandardNotFound(f"{canonical_id} was not found in the standards graph")
        return [
            AlliedStandardRelation(
                source_standard_id=record["source_standard_id"],
                target_standard_id=record["target_standard_id"],
                target_title=record.get("target_title"),
                relation_type=record["relation_type"],
                evidence=record.get("evidence"),
            )
            for record in records
            if record.get("target_standard_id")
        ]

    async def get_allied_or_empty(
        self,
        standard_id: str,
    ) -> tuple[list[AlliedStandardRelation], str | None]:
        try:
            return await self.get_allied(standard_id), None
        except AlliedStandardNotFound:
            raise
        except Exception as exc:  # noqa: BLE001 - degradation boundary
            return [], str(exc)

    async def load_relations(
        self,
        relations: list[AlliedStandardRelation],
    ) -> AlliedLoadReport:
        payload = [
            {
                "source_standard_id": canonicalize_standard_id(relation.source_standard_id),
                "target_standard_id": canonicalize_standard_id(relation.target_standard_id),
                "target_title": relation.target_title,
                "relation_type": _normalize_relation_type(relation.relation_type),
                "evidence": relation.evidence,
            }
            for relation in relations
        ]
        if not payload:
            return AlliedLoadReport(input_relations=0, nodes_merged=0, edges_merged=0)
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    UNWIND $relations AS relation
                    MERGE (source:Standard {standard_id: relation.source_standard_id})
                    MERGE (target:Standard {standard_id: relation.target_standard_id})
                    SET target.title = coalesce(relation.target_title, target.title)
                    MERGE (source)-[edge:ALLIED_WITH {
                      relation_type: relation.relation_type
                    }]->(target)
                    SET edge.evidence = relation.evidence
                    RETURN
                      count(DISTINCT source) + count(DISTINCT target) AS nodes_merged,
                      count(edge) AS edges_merged
                    """,
                    relations=payload,
                )
                summary = (await result.data())[0]
        except Exception as exc:  # noqa: BLE001 - normalize graph load failures
            raise RuntimeError("Neo4j allied standards load failed") from exc

        return AlliedLoadReport(
            input_relations=len(payload),
            nodes_merged=int(summary["nodes_merged"]),
            edges_merged=int(summary["edges_merged"]),
        )


def _normalize_relation_type(value: str) -> str:
    normalized = "_".join(value.upper().split())
    if not normalized:
        raise ValueError("relation_type must not be empty")
    return normalized
