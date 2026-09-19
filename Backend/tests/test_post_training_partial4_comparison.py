from scripts.evaluate_post_training_partial4_comparison import (
    COMPATIBLE,
    INCOMPATIBLE_DATASET,
    EvalCase,
    _classify_error,
    _metrics,
    classify_cases_for_dataset,
)


def test_classify_cases_marks_missing_standard_incompatible() -> None:
    cases = [
        EvalCase("q1", "known", "IS 1: 2000"),
        EvalCase("q2", "missing", "IS 14845: 2000"),
    ]

    status = classify_cases_for_dataset("old_39_regression", cases, {"IS 1:2000"})

    assert status["status"] == INCOMPATIBLE_DATASET
    assert status["missing_standard_codes"] == ["IS 14845: 2000"]
    assert status["missing_cases"][0]["query_id"] == "q2"


def test_classify_cases_accepts_unscored_ambiguous_suite() -> None:
    status = classify_cases_for_dataset(
        "ambiguous_product_diagnostics",
        [EvalCase("ambiguous", "", product="valve", description="for water supply")],
        set(),
    )

    assert status["status"] == COMPATIBLE
    assert status["missing_standard_codes"] == []


def test_error_classification_separates_retrieval_miss_from_reranker_error() -> None:
    corpus = {
        "IS 1:2000": {"standard_code": "IS 1: 2000", "function": "carry_water"},
        "IS 2:2000": {"standard_code": "IS 2: 2000", "function": "drain_water"},
    }

    assert _classify_error("IS 1: 2000", None, None, "IS 2: 2000", corpus) == "RETRIEVAL_MISS"
    assert (
        _classify_error("IS 1: 2000", 3, None, "IS 2: 2000", corpus)
        == "FUNCTION_CONFUSION"
    )


def test_error_classification_detects_part_variant_confusion() -> None:
    assert (
        _classify_error(
            "IS 1489 (Part 1): 1991",
            2,
            2,
            "IS 1489 (Part 2): 1991",
            {},
        )
        == "PART_OR_VARIANT_CONFUSION"
    )


def test_metrics_report_counts_and_rates() -> None:
    metrics = _metrics(
        [
            {"expected_rank": 1},
            {"expected_rank": 3},
            {"expected_rank": None},
        ]
    )

    assert metrics["hit_at_1"] == 1
    assert metrics["hit_at_3"] == 2
    assert metrics["hit_at_5"] == 2
    assert metrics["hit_at_1_rate"] == 1 / 3
    assert metrics["mrr"] == (1 + 1 / 3) / 3
