from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Standard
from app.services.gemini_query_interpreter import SemanticQueryIntent, build_enriched_query
from app.services.hybrid_search_service import HybridCandidate, HybridSearchService
from app.services.parsed_standards_corpus import (
    FAMILY_BY_PRODUCT,
    canonicalize_product,
    product_compatibility,
)
from app.services.reranker_service import (
    CrossEncoderRerankerService,
    RerankedCandidate,
    RerankedTimings,
)


@dataclass(frozen=True)
class ProductAwareCandidate:
    rank: int
    standard_id: str
    standard_code: str
    title: str
    reranker_score: float
    rrf_rank: int
    rrf_score: float
    product_compatibility: str
    canonical_product: str | None
    standard_kind: str | None
    family: str | None
    function: str | None = None
    constraint_score: float | None = None
    constraint_flags: list[str] | None = None
    raw_cross_encoder_score: float | None = None
    reranker_score_source: str | None = None
    final_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


@dataclass(frozen=True)
class ProductAwareTimings(RerankedTimings):
    product_gate_ms: float
    gemini_ms: float | None = None
    embedding_ms: float | None = None
    semantic_retrieval_ms: float | None = None
    constraint_ms: float | None = None
    postprocess_ms: float | None = None
    reranker_device: str | None = None
    reranker_model_loaded: bool | None = None
    reranker_used: bool | None = None
    reranker_timeout: bool | None = None
    reranker_success: bool | None = None
    reranker_fallback_reason: str | None = None
    reranker_inference_ms: float | None = None


@dataclass(frozen=True)
class ProductAwareSearchResult:
    product: str
    description: str
    query: str
    canonical_product: str | None
    product_confidence: str
    query_interpretation: SemanticQueryIntent | None
    ambiguity: bool
    missing_information: list[str]
    candidates: list[ProductAwareCandidate]
    timings_ms: ProductAwareTimings


@dataclass(frozen=True)
class ConstraintAssessment:
    score: float
    flags: list[str]


STATE_SCORES = {
    "EXACT_MATCH": 4.0,
    "STRONG_COMPATIBLE": 3.0,
    "PARTIAL_MATCH": 1.5,
    "UNKNOWN": 0.0,
    "CONTRADICTION": -8.0,
}

MATERIAL_GROUPS = {
    "metal": {
        "metal",
        "steel",
        "cast iron",
        "malleable cast iron",
        "brass",
        "copper alloy",
        "aluminium",
    },
    "plastic": {
        "HDPE",
        "LDPE",
        "PVC",
        "UPVC",
        "CPVC",
        "polypropylene",
        "polyethylene",
        "plastic",
    },
    "composite": {"GRP"},
    "fibrous_insulation": {"fibrous", "mineral wool", "rock wool", "slag wool", "glass wool"},
    "cementitious": {"cement", "finishing cement", "concrete", "ferrocement"},
    "ceramic": {"clay", "sandstone"},
    "calcium_silicate": {"calcium silicate"},
    "polyurethane": {"polyurethane", "PUR"},
}

MATERIAL_GROUP_BY_VALUE = {
    material: group for group, materials in MATERIAL_GROUPS.items() for material in materials
}


class ProductAwareSearchService:
    def __init__(
        self,
        hybrid_service: HybridSearchService,
        reranker_service: CrossEncoderRerankerService,
        *,
        rerank_k: int = 10,
        return_k: int = 5,
        reranker_timeout_s: float = 8.0,
    ) -> None:
        self._hybrid_service = hybrid_service
        self._reranker_service = reranker_service
        self._rerank_k = rerank_k
        self._return_k = return_k
        self._reranker_timeout_s = reranker_timeout_s

    async def search(
        self,
        session: AsyncSession,
        *,
        product: str,
        description: str,
        limit: int | None = None,
        query_intent: SemanticQueryIntent | None = None,
        gemini_ms: float | None = None,
    ) -> ProductAwareSearchResult:
        limit = self._return_k if limit is None else min(limit, self._return_k)
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not product.strip():
            raise ValueError("product must not be empty")
        if not description.strip():
            raise ValueError("description must not be empty")

        query = build_enriched_query(product, description, query_intent)
        canonical_product, confidence = _query_product(product, query_intent)
        started_at = perf_counter()
        hybrid_result = await self._hybrid_service.search(
            session,
            query,
            limit=max(self._rerank_k, limit),
        )

        gate_started_at = perf_counter()
        standards_by_id = await _standards_by_id(
            session,
            {candidate.standard_id for candidate in hybrid_result.candidates},
        )
        gated_candidates, compatibility_by_id = _apply_product_gate(
            hybrid_result.candidates,
            standards_by_id,
            canonical_product=canonical_product if confidence == "high" else None,
        )
        product_gate_ms = (perf_counter() - gate_started_at) * 1000

        constraint_started_at = perf_counter()
        constraint_by_id = _score_constraints(
            gated_candidates,
            standards_by_id,
            query_intent=query_intent,
            canonical_product=canonical_product if confidence == "high" else None,
        )
        constrained_candidates = (
            _order_by_constraints(gated_candidates, constraint_by_id)
            if query_intent is not None
            else gated_candidates
        )
        constraint_ms = (perf_counter() - constraint_started_at) * 1000

        reranker_started_at = perf_counter()
        reranker_timeout = False
        reranker_used = False
        skip_reason = _reranker_skip_reason(self._reranker_service)
        if skip_reason:
            _mark_reranker_skipped(self._reranker_service, skip_reason)
            reranked = _fallback_reranked_candidates(
                constrained_candidates,
                limit=limit,
                score_source=_fallback_score_source(skip_reason),
            )
        else:
            reranker_used = True
            try:
                reranked = await asyncio.wait_for(
                    self._reranker_service.rerank(
                        session,
                        query,
                        constrained_candidates,
                        limit=limit,
                    ),
                    timeout=self._reranker_timeout_s,
                )
            except TimeoutError:
                reranker_timeout = True
                self._reranker_service.last_success = False
                self._reranker_service.last_fallback_reason = "timeout"
                reranked = _fallback_reranked_candidates(
                    constrained_candidates,
                    limit=limit,
                    score_source="rrf_timeout_fallback",
                )
        reranker_ms = (perf_counter() - reranker_started_at) * 1000
        postprocess_started_at = perf_counter()
        candidates = _prioritize_function_matches(
            _decorate_candidates(
                reranked,
                standards_by_id,
                compatibility_by_id,
                constraint_by_id,
            ),
            description,
        )
        if query_intent is not None:
            candidates = _final_constraint_order(candidates)
        postprocess_ms = (perf_counter() - postprocess_started_at) * 1000
        timings = hybrid_result.timings_ms
        return ProductAwareSearchResult(
            product=product,
            description=description,
            query=query,
            canonical_product=canonical_product,
            product_confidence=confidence,
            query_interpretation=query_intent,
            ambiguity=bool(query_intent.ambiguity) if query_intent else False,
            missing_information=query_intent.missing_information if query_intent else [],
            candidates=candidates,
            timings_ms=ProductAwareTimings(
                gemini_ms=gemini_ms,
                embedding_ms=timings.embedding_ms,
                semantic_retrieval_ms=timings.semantic_retrieval_ms,
                semantic_ms=timings.semantic_ms,
                bm25_ms=timings.bm25_ms,
                rrf_ms=timings.rrf_ms,
                hybrid_ms=timings.total_ms,
                product_gate_ms=product_gate_ms,
                constraint_ms=constraint_ms,
                reranker_ms=reranker_ms,
                postprocess_ms=postprocess_ms,
                reranker_device=getattr(self._reranker_service, "device", None),
                reranker_model_loaded=getattr(self._reranker_service, "model_loaded", None),
                reranker_used=reranker_used,
                reranker_timeout=reranker_timeout,
                reranker_success=getattr(self._reranker_service, "last_success", None),
                reranker_fallback_reason=getattr(
                    self._reranker_service,
                    "last_fallback_reason",
                    None,
                ),
                reranker_inference_ms=getattr(self._reranker_service, "last_inference_ms", None),
                total_ms=(perf_counter() - started_at) * 1000,
            ),
        )


def _apply_product_gate(
    candidates: list[HybridCandidate],
    standards_by_id: dict[str, Standard],
    *,
    canonical_product: str | None,
) -> tuple[list[HybridCandidate], dict[str, str]]:
    compatibility_by_id = {}
    for candidate in candidates:
        standard = standards_by_id.get(candidate.standard_id)
        candidate_product = _effective_value(standard, "canonical_product") if standard else None
        compatibility_by_id[candidate.standard_id] = _metadata_aware_compatibility(
            canonical_product=canonical_product,
            candidate_product=candidate_product,
            standard=standard,
        )

    if canonical_product is None:
        return candidates, compatibility_by_id

    compatible = [
        candidate
        for candidate in candidates
        if compatibility_by_id[candidate.standard_id]
        in {"compatible", "adjacent", "same_family", "applies_to_family"}
    ]
    unknown = [
        candidate
        for candidate in candidates
        if compatibility_by_id[candidate.standard_id] == "unknown_candidate_product"
    ]
    incompatible = [
        candidate
        for candidate in candidates
        if compatibility_by_id[candidate.standard_id] == "incompatible"
    ]
    if not compatible:
        return candidates, compatibility_by_id
    return [*compatible, *unknown, *incompatible], compatibility_by_id


def _query_product(
    product: str,
    query_intent: SemanticQueryIntent | None,
) -> tuple[str | None, str]:
    if query_intent and query_intent.normalized_product:
        return query_intent.normalized_product, "high"
    return canonicalize_product(product)


def _effective_value(standard: Standard | None, field: str) -> str | None:
    if standard is None:
        return None
    text = _standard_text(standard)
    code = getattr(standard, "standard_id", "")

    if field == "canonical_product":
        if "glass fibre reinforced" in text and "pipe" in text and "insulation" not in text:
            return "pipe"
        if "reflux" in text and "valve" in text:
            return "valve"
        return standard.canonical_product
    if field == "family":
        if _effective_value(standard, "canonical_product") == "pipe":
            return "pipe_water_drainage"
        if _effective_value(standard, "canonical_product") == "valve":
            return "water_valve"
        return standard.family
    if field == "product_subtype":
        if "pressure reducing valve" in text:
            return "pressure_reducing_valve"
        if "reflux" in text or "check valve" in text or "non return" in text:
            return "check_valve"
        if "part 1 fly ash based" in text or code == "IS 1489 (Part 1): 1991":
            return "portland_pozzolana_fly_ash"
        if "part 2 calcined clay based" in text or code == "IS 1489 (Part 2): 1991":
            return "portland_pozzolana_calcined_clay"
        return standard.product_subtype
    if field == "subtype_family":
        return (
            "check_valve"
            if _effective_value(standard, "product_subtype") == "check_valve"
            else None
        )
    if field == "cement_type":
        subtype = _effective_value(standard, "product_subtype") or ""
        if "portland_pozzolana" in subtype:
            return "ppc"
        if "ordinary_portland" in subtype:
            return "opc"
        return None
    if field == "pozzolana_source":
        subtype = _effective_value(standard, "product_subtype")
        if subtype == "portland_pozzolana_fly_ash":
            return "fly_ash"
        if subtype == "portland_pozzolana_calcined_clay":
            return "calcined_clay"
        return None
    if field == "material":
        if "glass fibre reinforced" in text or "glass reinforced plastic" in text:
            return "GRP"
        if "high density polyethylene" in text or "hdpe" in text:
            return "HDPE"
        if "unplasticized polyvinyl chloride" in text or "unplasticised polyvinyl chloride" in text:
            return "UPVC"
        if "calcium silicate" in text:
            return "calcium silicate"
        if "preformed fibrous" in text or "fibrous pipe insulation" in text:
            return "fibrous"
        if "clay flooring" in text:
            return "clay"
        return standard.material
    if field == "application":
        if "potable water" in text or "drinking water" in text:
            return "potable water supply"
        if "industrial waste" in text or "other than potable" in text or "non potable" in text:
            return "industrial waste"
        if "sewerage" in text or "sewage" in text:
            return "sewerage"
        if "flooring" in text and "roofing" in text:
            return "flooring roofing"
        if "flooring" in text:
            return "flooring"
        if "roofing" in text:
            return "roofing"
        if "thermal insulation" in text or "insulation" in text:
            return "thermal insulation"
        return standard.application
    if field == "function":
        if _effective_value(standard, "product_subtype") == "pressure_reducing_valve":
            return "reduce_pressure"
        if _effective_value(standard, "product_subtype") == "check_valve":
            return "prevent_reverse_flow"
        if _effective_value(standard, "application") == "potable water supply":
            return "carry_potable_water"
        if _effective_value(standard, "application") in {"sewerage", "industrial waste"}:
            return "carry_sewage"
        if _effective_value(standard, "application") == "flooring":
            return "flooring"
        if _effective_value(standard, "application") == "roofing":
            return "roofing"
        return standard.function
    if field == "primary_subject":
        subtype = _effective_value(standard, "product_subtype")
        if subtype:
            return subtype.replace("_", " ")
        return standard.primary_subject
    if field == "form":
        return "preformed" if "preformed" in text or "preormed" in text else None
    return None


def _standard_text(standard: Standard) -> str:
    return _norm(
        " ".join(
            value
            for value in [
                getattr(standard, "standard_id", ""),
                standard.title,
                standard.scope_text,
                standard.retrieval_text,
                standard.primary_subject,
            ]
            if value
        )
    )


def _metadata_aware_compatibility(
    *,
    canonical_product: str | None,
    candidate_product: str | None,
    standard: Standard | None,
) -> str:
    compatibility = product_compatibility(canonical_product, candidate_product)
    if compatibility != "incompatible" or canonical_product is None or standard is None:
        return compatibility
    query_family = FAMILY_BY_PRODUCT.get(canonical_product)
    applies_to = set(standard.applies_to_product_families or [])
    if query_family and query_family in applies_to:
        return "applies_to_family"
    if query_family and _effective_value(standard, "family") == query_family:
        return "same_family"
    product_terms = [canonical_product, canonical_product.replace("_", " ")]
    source_text = " ".join(
        value
        for value in [
            _effective_value(standard, "primary_subject"),
            _effective_value(standard, "product_subtype"),
            _effective_value(standard, "material"),
            _effective_value(standard, "application"),
            _effective_value(standard, "function"),
            standard.title,
        ]
        if value
    )
    if any(_contains_phrase(source_text, term) for term in product_terms):
        return "compatible"
    return compatibility


def _score_constraints(
    candidates: list[HybridCandidate],
    standards_by_id: dict[str, Standard],
    *,
    query_intent: SemanticQueryIntent | None,
    canonical_product: str | None,
) -> dict[str, ConstraintAssessment]:
    if query_intent is None:
        return {
            candidate.standard_id: ConstraintAssessment(score=0.0, flags=[])
            for candidate in candidates
        }
    return {
        candidate.standard_id: _score_candidate_constraints(
            standards_by_id.get(candidate.standard_id),
            query_intent=query_intent,
            canonical_product=canonical_product,
        )
        for candidate in candidates
    }


def _score_candidate_constraints(
    standard: Standard | None,
    *,
    query_intent: SemanticQueryIntent,
    canonical_product: str | None,
) -> ConstraintAssessment:
    if standard is None:
        return ConstraintAssessment(score=0.0, flags=["UNKNOWN_CANDIDATE"])

    score = 0.0
    flags: list[str] = []
    product_state = _product_state(canonical_product, standard)
    score += {
        "EXACT_PRODUCT": 8.0,
        "APPLIES_TO_PRODUCT": 5.0,
        "ADJACENT_PRODUCT": 2.0,
        "SAME_FAMILY": 0.75,
        "UNKNOWN": 0.0,
        "CONTRADICTION": -10.0,
    }[product_state]
    flags.append(f"PRODUCT_{product_state}")

    score += _attribute_score(
        query_values=query_intent.material,
        candidate_values=[
            _effective_value(standard, "material"),
            _effective_value(standard, "primary_subject"),
            _effective_value(standard, "product_subtype"),
            standard.title,
            standard.scope_text,
        ],
        field="MATERIAL",
        state_for=_material_state,
        flags=flags,
    )
    score += _attribute_score(
        query_values=query_intent.function,
        candidate_values=[
            _effective_value(standard, "function"),
            _effective_value(standard, "product_subtype"),
            standard.title,
        ],
        field="FUNCTION",
        state_for=_function_state,
        flags=flags,
    )
    score += _attribute_score(
        query_values=query_intent.application,
        candidate_values=[
            _effective_value(standard, "application"),
            standard.scope_text,
            standard.title,
        ],
        field="APPLICATION",
        state_for=_text_state,
        flags=flags,
    )
    if query_intent.cement_type:
        score += _attribute_score(
            query_values=[query_intent.cement_type],
            candidate_values=[
                _effective_value(standard, "cement_type"),
                _effective_value(standard, "product_subtype"),
                standard.title,
            ],
            field="CEMENT_TYPE",
            state_for=_text_state,
            flags=flags,
        )
    if query_intent.pozzolana_source:
        score += _attribute_score(
            query_values=[query_intent.pozzolana_source],
            candidate_values=[
                _effective_value(standard, "pozzolana_source"),
                _effective_value(standard, "product_subtype"),
                standard.title,
                standard.scope_text,
            ],
            field="POZZOLANA_SOURCE",
            state_for=_pozzolana_state,
            flags=flags,
        )
    if query_intent.subtype_family:
        score += _attribute_score(
            query_values=[query_intent.subtype_family],
            candidate_values=[
                _effective_value(standard, "subtype_family"),
                _effective_value(standard, "product_subtype"),
                standard.title,
            ],
            field="SUBTYPE_FAMILY",
            state_for=_subtype_state,
            flags=flags,
        )
    if query_intent.form:
        score += _attribute_score(
            query_values=[query_intent.form],
            candidate_values=[
                _effective_value(standard, "form"),
                _effective_value(standard, "primary_subject"),
                standard.title,
                standard.scope_text,
            ],
            field="FORM",
            state_for=_text_state,
            flags=flags,
        )
    if query_intent.subtype:
        score += _attribute_score(
            query_values=[query_intent.subtype],
            candidate_values=[
                _effective_value(standard, "product_subtype"),
                _effective_value(standard, "primary_subject"),
                standard.title,
            ],
            field="SUBTYPE",
            state_for=_subtype_state,
            flags=flags,
        )
    score += _excluded_score(query_intent, standard, flags)
    if query_intent.grade:
        score += _grade_score(query_intent.grade, standard, flags)
    if query_intent.temperature_c is not None:
        score += _temperature_score(query_intent.temperature_c, standard, flags)
    return ConstraintAssessment(score=score, flags=flags)


def _attribute_score(
    *,
    query_values: list[str],
    candidate_values: list[str | None],
    field: str,
    state_for: Callable[[str, str], str],
    flags: list[str],
) -> float:
    query = [value for value in query_values if value]
    candidate_text = " ".join(value for value in candidate_values if value)
    if not query:
        return 0.0
    if not candidate_text.strip():
        flags.append(f"{field}_UNKNOWN")
        return 0.0
    states = [state_for(value, candidate_text) for value in query]
    state = _best_state(states)
    flags.append(f"{field}_{state}")
    return STATE_SCORES[state]


def _excluded_score(
    query_intent: SemanticQueryIntent,
    standard: Standard,
    flags: list[str],
) -> float:
    candidate_text = " ".join(
        value
        for value in [
            _effective_value(standard, "material"),
            _effective_value(standard, "function"),
            _effective_value(standard, "application"),
            _effective_value(standard, "product_subtype"),
            standard.title,
            standard.scope_text,
        ]
        if value
    )
    score = 0.0
    checks = [
        ("EXCLUDED_MATERIAL", query_intent.excluded_material, _material_state),
        ("EXCLUDED_FUNCTION", query_intent.excluded_function, _function_state),
        ("EXCLUDED_APPLICATION", query_intent.excluded_application, _text_state),
        ("EXCLUDED_SUBTYPE", query_intent.excluded_subtype, _subtype_state),
        ("EXCLUDED_MEDIUM", query_intent.excluded_medium, _text_state),
        ("EXCLUDED_INSTALLATION", query_intent.excluded_installation_context, _text_state),
    ]
    for field, values, state_for in checks:
        if not values:
            continue
        states = [state_for(value, candidate_text) for value in values]
        state = _best_state(states)
        if state in {"EXACT_MATCH", "STRONG_COMPATIBLE", "PARTIAL_MATCH"}:
            flags.append(f"{field}_CONTRADICTION")
            score += STATE_SCORES["CONTRADICTION"]
        else:
            flags.append(f"{field}_CLEAR")
    return score


def _grade_score(grade: str, standard: Standard, flags: list[str]) -> float:
    candidate_text = _norm(
        " ".join([_effective_value(standard, "product_subtype") or "", standard.title or ""])
    )
    grade_key = grade.lower()
    if f"{grade_key} grade" in candidate_text or f"grade {grade_key}" in candidate_text:
        flags.append("GRADE_MATCH")
        return STATE_SCORES["EXACT_MATCH"]
    known_grades = {"33", "43", "53"}
    wrong_known_grade = any(f"{other} grade" in candidate_text for other in known_grades - {grade})
    wrong_letter_grade = (
        grade.isalpha()
        and "grade " in candidate_text
        and f"grade {grade.lower()}" not in candidate_text
    )
    if wrong_known_grade or wrong_letter_grade:
        flags.append("GRADE_CONTRADICTION")
        return STATE_SCORES["CONTRADICTION"]
    flags.append("GRADE_UNKNOWN")
    return 0.0


def _temperature_score(temperature_c: float, standard: Standard, flags: list[str]) -> float:
    candidate_text = " ".join(
        [standard.title or "", standard.scope_text or "", standard.retrieval_text or ""]
    )
    ranges = _temperature_ranges(candidate_text)
    if not ranges:
        flags.append("TEMPERATURE_UNKNOWN")
        return 0.0
    if any(low <= temperature_c <= high for low, high in ranges):
        flags.append("TEMPERATURE_MATCH")
        return STATE_SCORES["EXACT_MATCH"]
    flags.append("TEMPERATURE_CONTRADICTION")
    return STATE_SCORES["CONTRADICTION"]


def _temperature_ranges(text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    normalized = _norm(text)
    for match in re.finditer(
        r"(?:up to|upto|not exceeding|maximum|max)\s+(\d+(?:\.\d+)?)\s*(?:degree[s]?)?\s*c",
        normalized,
    ):
        ranges.append((-273.15, float(match.group(1))))
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:degree[s]?)?\s*c",
        normalized,
    ):
        low = float(match.group(1))
        high = float(match.group(2))
        ranges.append((min(low, high), max(low, high)))
    return ranges


def _order_by_constraints(
    candidates: list[HybridCandidate],
    constraint_by_id: dict[str, ConstraintAssessment],
) -> list[HybridCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            min(constraint_by_id[candidate.standard_id].score, 0),
            constraint_by_id[candidate.standard_id].score,
            -candidate.rank,
        ),
        reverse=True,
    )


def _final_constraint_order(candidates: list[ProductAwareCandidate]) -> list[ProductAwareCandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            (candidate.constraint_score or 0) < 0,
            -(candidate.final_score or 0),
            candidate.rrf_rank,
        ),
    )
    return [replace(candidate, rank=rank) for rank, candidate in enumerate(ordered, start=1)]


def _product_state(canonical_product: str | None, standard: Standard) -> str:
    if canonical_product is None:
        return "UNKNOWN"
    candidate_product = _effective_value(standard, "canonical_product")
    if candidate_product is None:
        return "UNKNOWN"
    if canonical_product == candidate_product:
        return "EXACT_PRODUCT"
    query_family = FAMILY_BY_PRODUCT.get(canonical_product)
    if query_family and query_family in set(standard.applies_to_product_families or []):
        return "APPLIES_TO_PRODUCT"
    compatibility = product_compatibility(canonical_product, candidate_product)
    if compatibility == "adjacent":
        return "ADJACENT_PRODUCT"
    if compatibility == "same_family":
        return "SAME_FAMILY"
    return "CONTRADICTION" if compatibility == "incompatible" else "UNKNOWN"


def _best_state(states: list[str]) -> str:
    order = ["EXACT_MATCH", "STRONG_COMPATIBLE", "PARTIAL_MATCH", "UNKNOWN", "CONTRADICTION"]
    for state in order:
        if state in states:
            return state
    return "UNKNOWN"


def _material_state(query_value: str, candidate_text: str) -> str:
    query = _material_family(query_value)
    candidate = _material_family(candidate_text)
    if not query or not candidate:
        return "UNKNOWN"
    if query == candidate:
        return "EXACT_MATCH"
    if {query, candidate} in [{"clay", "sandstone"}]:
        return "CONTRADICTION"
    if {query, candidate} == {"plastic", "GRP"}:
        return "STRONG_COMPATIBLE"
    query_group = MATERIAL_GROUP_BY_VALUE.get(query)
    candidate_group = MATERIAL_GROUP_BY_VALUE.get(candidate)
    if query_group == candidate_group and query_group in {"metal", "plastic", "fibrous_insulation"}:
        return "STRONG_COMPATIBLE"
    if query_group and candidate_group and query_group != candidate_group:
        return "CONTRADICTION"
    if _text_state(query_value, candidate_text) != "UNKNOWN":
        return _text_state(query_value, candidate_text)
    return "UNKNOWN"


def _material_family(value: str) -> str:
    text = _norm(value)
    if "calcium silicate" in text:
        return "calcium silicate"
    if "mineral wool" in text:
        return "mineral wool"
    if "rock wool" in text:
        return "rock wool"
    if "slag wool" in text:
        return "slag wool"
    if "glass wool" in text:
        return "glass wool"
    if (
        "grp" in text
        or "gfrp" in text
        or "glass fibre reinforced" in text
        or "glass fiber reinforced" in text
        or "glass reinforced plastic" in text
    ):
        return "GRP"
    if "fibrous" in text or "fibre" in text or "fiber" in text:
        return "fibrous"
    if "upvc" in text or "unplasticized polyvinyl chloride" in text:
        return "UPVC"
    if "cpvc" in text or "chlorinated polyvinyl chloride" in text:
        return "CPVC"
    if "polyvinyl chloride" in text or "pvc" in text:
        return "PVC"
    if "hdpe" in text or "high density polyethylene" in text:
        return "HDPE"
    if "ldpe" in text or "low density polyethylene" in text:
        return "LDPE"
    if "polyethylene" in text:
        return "polyethylene"
    if "polypropylene" in text:
        return "polypropylene"
    if "plastic" in text:
        return "plastic"
    if "polyurethane" in text or " pur " in f" {text} ":
        return "polyurethane"
    if "upvc" in text:
        return "UPVC"
    if "steel" in text or "mild steel" in text:
        return "steel"
    if "metal" in text:
        return "metal"
    if "malleable cast iron" in text:
        return "malleable cast iron"
    if "cast iron" in text:
        return "cast iron"
    if "brass" in text:
        return "brass"
    if "copper alloy" in text:
        return "copper alloy"
    if "finishing cement" in text:
        return "finishing cement"
    if "cement" in text:
        return "cement"
    if "concrete" in text:
        return "concrete"
    if "ferrocement" in text:
        return "ferrocement"
    if "sandstone" in text:
        return "sandstone"
    if "clay" in text:
        return "clay"
    return ""


def _function_state(query_value: str, candidate_text: str) -> str:
    return _subtype_state(query_value, candidate_text)


def _subtype_state(query_value: str, candidate_text: str) -> str:
    query = _norm(query_value)
    candidate = _norm(candidate_text)
    if query in candidate or candidate in query:
        return "EXACT_MATCH"
    check_terms = {
        "check valve",
        "reflux",
        "non return",
        "non-return",
        "swing check",
        "prevent reverse flow",
    }
    if query in {"prevent reverse flow", "prevent_reverse_flow", "check valve", "check_valve"}:
        if any(term in candidate for term in check_terms):
            return "STRONG_COMPATIBLE"
        if "pressure reducing" in candidate:
            return "CONTRADICTION"
    if "pressure reducing" in query and any(term in candidate for term in check_terms):
        return "CONTRADICTION"
    return _text_state(query_value, candidate_text)


def _pozzolana_state(query_value: str, candidate_text: str) -> str:
    query = _norm(query_value)
    candidate = _norm(candidate_text)
    if query == "fly ash":
        if "fly ash" in candidate and "calcined clay" not in candidate:
            return "EXACT_MATCH"
        if "calcined clay" in candidate:
            return "CONTRADICTION"
    if query == "calcined clay":
        if "calcined clay" in candidate:
            return "EXACT_MATCH"
        if "fly ash" in candidate:
            return "CONTRADICTION"
    return _text_state(query_value, candidate_text)


def _text_state(query_value: str, candidate_text: str) -> str:
    query = _norm(query_value)
    candidate = _norm(candidate_text)
    if query in {"potable", "potable water", "potable water supply"} and (
        "other than potable" in candidate or "non potable" in candidate
    ):
        return "CONTRADICTION"
    if query in candidate or candidate in query:
        return "EXACT_MATCH"
    if query == "sewerage":
        return (
            "STRONG_COMPATIBLE"
            if "sewage" in candidate or "sewerage" in candidate or "drainage" in candidate
            else "UNKNOWN"
        )
    if query == "industrial waste":
        return (
            "STRONG_COMPATIBLE"
            if "industrial waste" in candidate
            or "effluent" in candidate
            or "other than potable" in candidate
            else "UNKNOWN"
        )
    if query == "potable water supply":
        return (
            "STRONG_COMPATIBLE"
            if "potable water" in candidate or "drinking water" in candidate
            else "UNKNOWN"
        )
    if query == "water pipeline":
        return (
            "PARTIAL_MATCH"
            if "water" in candidate or "pipeline" in candidate or "water supply" in candidate
            else "UNKNOWN"
        )
    return "UNKNOWN"


def _norm(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.casefold().replace("_", " ").split())
    normalized_phrase = " ".join(phrase.casefold().replace("_", " ").split())
    return f" {normalized_phrase} " in f" {normalized_text} "


def _decorate_candidates(
    candidates: list[RerankedCandidate],
    standards_by_id: dict[str, Standard],
    compatibility_by_id: dict[str, str],
    constraint_by_id: dict[str, ConstraintAssessment] | None = None,
) -> list[ProductAwareCandidate]:
    decorated = []
    for rank, candidate in enumerate(candidates, start=1):
        standard = standards_by_id.get(candidate.standard_id)
        constraint = (constraint_by_id or {}).get(
            candidate.standard_id,
            ConstraintAssessment(score=0.0, flags=[]),
        )
        decorated.append(
            ProductAwareCandidate(
                rank=rank,
                standard_id=candidate.standard_id,
                standard_code=candidate.standard_code,
                title=candidate.title,
                reranker_score=candidate.reranker_score,
                rrf_rank=candidate.rrf_rank,
                rrf_score=candidate.rrf_score,
                product_compatibility=compatibility_by_id.get(
                    candidate.standard_id,
                    "unknown_candidate_product",
                ),
                canonical_product=_effective_value(standard, "canonical_product"),
                standard_kind=standard.standard_kind if standard else None,
                family=_effective_value(standard, "family"),
                function=_effective_value(standard, "function"),
                constraint_score=constraint.score,
                constraint_flags=constraint.flags,
                raw_cross_encoder_score=candidate.raw_cross_encoder_score,
                reranker_score_source=candidate.score_source,
                final_score=_final_score(candidate, constraint),
                semantic_rank=candidate.semantic_rank,
                semantic_score=candidate.semantic_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
            )
        )
    return decorated


def _final_score(candidate: RerankedCandidate, constraint: ConstraintAssessment) -> float:
    relevance = (
        candidate.reranker_score
        if candidate.score_source == "cross_encoder"
        else candidate.rrf_score
    )
    return relevance + constraint.score


def _fallback_reranked_candidates(
    candidates: list[HybridCandidate],
    *,
    limit: int,
    score_source: str,
) -> list[RerankedCandidate]:
    return [
        RerankedCandidate(
            rank=rank,
            standard_id=candidate.standard_id,
            standard_code=candidate.standard_code,
            title=candidate.title,
            reranker_score=0.0,
            rrf_rank=candidate.rank,
            rrf_score=candidate.rrf_score,
            semantic_rank=candidate.semantic_rank,
            semantic_score=candidate.semantic_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            raw_cross_encoder_score=None,
            score_source=score_source,
        )
        for rank, candidate in enumerate(candidates[:limit], start=1)
    ]


def _reranker_skip_reason(reranker_service: CrossEncoderRerankerService) -> str | None:
    should_skip = getattr(reranker_service, "should_skip", None)
    if not callable(should_skip) or not should_skip():
        return None
    skip_reason = getattr(reranker_service, "skip_reason", None)
    if callable(skip_reason):
        return skip_reason() or "reranker_skipped"
    return "reranker_skipped"


def _mark_reranker_skipped(
    reranker_service: CrossEncoderRerankerService,
    fallback_reason: str,
) -> None:
    mark_skipped = getattr(reranker_service, "mark_skipped", None)
    if callable(mark_skipped):
        mark_skipped()
        return
    reranker_service.last_success = False
    reranker_service.last_fallback_reason = fallback_reason
    reranker_service.last_inference_ms = 0.0


def _fallback_score_source(fallback_reason: str) -> str:
    if fallback_reason == "cpu_reranker_disabled":
        return "rrf_cpu_reranker_disabled"
    if fallback_reason == "reranker_disabled":
        return "rrf_reranker_disabled"
    return "rrf_reranker_skipped"


async def _standards_by_id(
    session: AsyncSession,
    standard_ids: set[str],
) -> dict[str, Standard]:
    if not standard_ids:
        return {}
    result = await session.execute(
        select(Standard).where(Standard.id.in_([UUID(standard_id) for standard_id in standard_ids]))
    )
    return {str(standard.id): standard for standard in result.scalars().all()}


def _prioritize_function_matches(
    candidates: list[ProductAwareCandidate],
    description: str,
) -> list[ProductAwareCandidate]:
    target_function = _description_function(description)
    if target_function is None:
        return candidates
    matching = [candidate for candidate in candidates if candidate.function == target_function]
    if not matching:
        return candidates
    non_matching = [candidate for candidate in candidates if candidate.function != target_function]
    return [
        replace(candidate, rank=rank)
        for rank, candidate in enumerate([*matching, *non_matching], start=1)
    ]


def _description_function(description: str) -> str | None:
    text = " ".join(description.casefold().split())
    if "reverse flow" in text or "non return" in text or "non-return" in text:
        return "prevent_reverse_flow"
    if "reduce pressure" in text or "pressure reducing" in text:
        return "reduce_pressure"
    if "release air" in text or "air relief" in text:
        return "release_air"
    return None
