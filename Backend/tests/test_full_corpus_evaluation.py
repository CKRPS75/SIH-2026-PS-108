from scripts.evaluate_full_corpus_baselines import EvalCase, _classify_error, _metrics


def test_candidate_recall_metrics_include_top_10_and_top_20() -> None:
    rows = [
        {"expected_rank": 1},
        {"expected_rank": 7},
        {"expected_rank": 15},
        {"expected_rank": None},
    ]

    metrics = _metrics(rows)

    assert metrics["hit_at_1"] == 1
    assert metrics["hit_at_5"] == 1
    assert metrics["hit_at_10"] == 2
    assert metrics["candidate_recall_at_20"] == 0.75


def test_error_classification_distinguishes_retrieval_miss_and_part_confusion() -> None:
    retrieval_miss = _classify_error(
        EvalCase(
            query_id="q1",
            query="x",
            expected_standard="IS 1: 2000",
        ),
        {"expected_rank": None},
        {"expected_rank": None, "top1_standard": "IS 2: 2000"},
    )
    part_confusion = _classify_error(
        EvalCase(
            query_id="q2",
            query="x",
            expected_standard="IS 1489 (Part 1): 1991",
        ),
        {"expected_rank": 1},
        {"expected_rank": 2, "top1_standard": "IS 1489 (Part 2): 1991"},
    )

    assert retrieval_miss == "RETRIEVAL_MISS"
    assert part_confusion == "PART_OR_VARIANT_CONFUSION"
