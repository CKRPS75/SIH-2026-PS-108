import json
from pathlib import Path

from scripts import generate_reranker_training_data as generator


def test_eligibility_assigns_one_valid_status_per_full_corpus_record() -> None:
    path = Path("data/canonical/parsed_standards_canonical.json")
    if not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    rows = [generator._eligibility_row(record) for record in records]

    assert len(rows) == 558
    assert {row["eligibility"] for row in rows} <= generator.VALID_ELIGIBILITY
    assert all(row["reason"] for row in rows)


def test_review_required_record_does_not_become_safe() -> None:
    record = _record(review_required=True, review_reason="needs human review")

    row = generator._eligibility_row(record)

    assert row["eligibility"] == "REVIEW_REQUIRED"


def test_generator_derives_queries_without_concepts_map() -> None:
    record = _record()
    delattr(generator, "CONCEPTS") if hasattr(generator, "CONCEPTS") else None

    queries = generator._query_candidates(record)

    assert queries
    assert all("IS 1" not in query["query"] for query in queries)
    assert any("grounded valve" in query["query"] for query in queries)


def test_query_grounding_and_leakage_filtering() -> None:
    record = _record()
    queries, leakage = generator._build_positive_queries(
        [record],
        frozen_queries=["grounded valve standard"],
    )

    assert leakage.exact
    assert queries
    assert all(query["grounding_evidence"] for query in queries)
    assert all(query["eligibility"] == "SAFE_FOR_TRAINING" for query in queries)


def test_split_is_deterministic_and_disjoint() -> None:
    queries = [
        {"query_id": f"q{i:03d}", "standard_code": "IS 1: 2000", "query": f"query {i}"}
        for i in range(40)
    ]

    first_train, first_validation = generator._split_queries(queries)
    second_train, second_validation = generator._split_queries(queries)

    assert [item["query_id"] for item in first_train] == [
        item["query_id"] for item in second_train
    ]
    assert [item["query_id"] for item in first_validation] == [
        item["query_id"] for item in second_validation
    ]
    assert {item["query_id"] for item in first_train}.isdisjoint(
        {item["query_id"] for item in first_validation}
    )


def test_hard_negatives_exclude_positive_and_have_valid_categories() -> None:
    positive = _record(code="IS 1: 2000", product="valve", family="water_valve")
    records = [
        positive,
        _record(code="IS 2: 2000", product="valve", family="water_valve", subtype="sluice valve"),
        _record(code="IS 3: 2000", product="pipe", family="pipe_water_drainage"),
        _record(code="IS 4: 2000", product="valve", family="water_valve", material="cast iron"),
        _record(code="IS 5: 2000", product="cement", family="cement"),
    ]

    negatives = generator._hard_negatives(positive, records)

    assert all(item["standard_code"] != positive["standard_code"] for item in negatives)
    assert {item["category"] for item in negatives} <= generator.VALID_NEGATIVE_CATEGORIES


def test_ambiguous_queries_are_not_supervised_positives() -> None:
    records = [
        _record(code="IS 1: 2000", product="cement", family="cement"),
        _record(code="IS 2: 2000", product="cement", family="cement", subtype="slag cement"),
    ]
    ambiguous = generator._ambiguous_queries(records)
    queries, _ = generator._build_positive_queries(records, frozen_queries=[])
    ambiguous_keys = {generator._query_key(record["query"]) for record in ambiguous}

    assert ambiguous
    assert all(generator._query_key(query["query"]) not in ambiguous_keys for query in queries)


def _record(
    *,
    code: str = "IS 1: 2000",
    product: str = "valve",
    family: str = "water_valve",
    subtype: str = "grounded valve",
    material: str = "cast iron",
    review_required: bool = False,
    review_reason: str | None = None,
) -> dict[str, object]:
    return {
        "standard_code": code,
        "standard_code_norm": code.replace(" ", ""),
        "title": f"{subtype} for water supply",
        "canonical_title_expanded": None,
        "canonical_product": product,
        "product_subtype": subtype,
        "primary_subject": subtype,
        "function": "prevent_reverse_flow",
        "material": material,
        "application": "water supply",
        "family": family,
        "standard_kind": "product_standard",
        "scope": f"Covers {material} {subtype} for water supply.",
        "scope_reconstructed": None,
        "product_aliases": [product],
        "metadata_confidence": "high",
        "product_confidence": "high",
        "metadata_evidence": {
            "canonical_product": ["title"],
            "primary_subject": ["title"],
            "product_subtype": ["title"],
            "function": ["scope"],
        },
        "title_evidence": [],
        "review_required": review_required,
        "review_reason": review_reason,
        "quality_flags": [],
    }
