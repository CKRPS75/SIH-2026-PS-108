from types import SimpleNamespace

from app.services.gemini_query_interpreter import SemanticQueryIntent
from app.services.product_aware_search_service import (
    _material_state,
    _score_candidate_constraints,
    _subtype_state,
)


def test_material_taxonomy_states() -> None:
    assert _material_state("clay", "sandstone tile") == "CONTRADICTION"
    assert _material_state("plastic", "steel pipe") == "CONTRADICTION"
    assert _material_state("plastic", "concrete pipe") == "CONTRADICTION"
    assert _material_state("plastic", "cast iron pipe") == "CONTRADICTION"
    assert _material_state("plastic", "HDPE pipe") == "STRONG_COMPATIBLE"
    assert _material_state("plastic", "UPVC pipe") == "STRONG_COMPATIBLE"
    assert _material_state("plastic", "GRP pipe") == "STRONG_COMPATIBLE"
    assert _material_state("metal", "cast iron fitting") == "STRONG_COMPATIBLE"
    assert _material_state("fibrous", "rock wool insulation") == "STRONG_COMPATIBLE"
    assert _material_state("fibrous", "calcium silicate insulation") == "CONTRADICTION"
    assert _material_state("fibrous", "PUR insulation") == "CONTRADICTION"
    assert _material_state("fibrous", "finishing cement") == "CONTRADICTION"


def test_subtype_taxonomy_states() -> None:
    assert _subtype_state("check_valve", "reflux valve for water") == "STRONG_COMPATIBLE"
    assert _subtype_state("check_valve", "pressure reducing valve") == "CONTRADICTION"


def test_grade_unknown_and_match_are_distinct() -> None:
    matching = _score_candidate_constraints(
        _standard(title="Hexagon head bolts Grade C", product="bolt"),
        query_intent=SemanticQueryIntent(normalized_product="bolt", grade="C"),
        canonical_product="bolt",
    )
    unknown = _score_candidate_constraints(
        _standard(title="Hexagon head bolts", product="bolt"),
        query_intent=SemanticQueryIntent(normalized_product="bolt", grade="C"),
        canonical_product="bolt",
    )

    assert "GRADE_MATCH" in matching.flags
    assert "GRADE_UNKNOWN" in unknown.flags


def test_temperature_range_match_and_contradiction() -> None:
    matching = _score_candidate_constraints(
        _standard(title="Insulation up to 650 C", product="thermal_insulation"),
        query_intent=SemanticQueryIntent(
            normalized_product="thermal_insulation",
            temperature_c=600,
        ),
        canonical_product="thermal_insulation",
    )
    contradiction = _score_candidate_constraints(
        _standard(title="Insulation up to 450 C", product="thermal_insulation"),
        query_intent=SemanticQueryIntent(
            normalized_product="thermal_insulation",
            temperature_c=600,
        ),
        canonical_product="thermal_insulation",
    )

    assert "TEMPERATURE_MATCH" in matching.flags
    assert "TEMPERATURE_CONTRADICTION" in contradiction.flags


def test_explicit_bolt_dominates_same_family_fasteners() -> None:
    bolt = _score_candidate_constraints(
        _standard(title="Hexagon head bolt", product="bolt", family="fastener"),
        query_intent=SemanticQueryIntent(normalized_product="bolt"),
        canonical_product="bolt",
    )
    nut = _score_candidate_constraints(
        _standard(title="Hexagon nut", product="nut", family="fastener"),
        query_intent=SemanticQueryIntent(normalized_product="bolt"),
        canonical_product="bolt",
    )

    assert "PRODUCT_EXACT_PRODUCT" in bolt.flags
    assert bolt.score > nut.score


def _standard(
    *,
    title: str,
    product: str,
    family: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        title=title,
        canonical_product=product,
        family=family,
        applies_to_product_families=[],
        material=None,
        function=None,
        application=None,
        product_subtype=None,
        primary_subject=title,
        scope_text=title,
        retrieval_text=title,
        product_aliases=[],
    )
