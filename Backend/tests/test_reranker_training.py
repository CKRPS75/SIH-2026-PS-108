import json
from pathlib import Path

import pytest

from app.services.reranker_training import (
    CheckpointMetrics,
    ScoredRerankerPair,
    binary_cross_entropy_with_logits,
    frozen_query_leakage,
    group_pairs_by_query,
    load_pair_examples,
    pair_accuracy_from_logits,
    ranking_metrics,
    score_pair_groups,
    select_best_checkpoint,
    standardwise_reranker_best_path,
)


class FakeScoringModel:
    def __init__(self, scores_by_standard: dict[str, float]) -> None:
        self.scores_by_standard = scores_by_standard
        self.calls = []

    def predict(self, pairs, batch_size: int, activation_fn=None):
        self.calls.append((pairs, batch_size, activation_fn))
        return [
            self.scores_by_standard.get(_standard_code_from_document(document), 0.0)
            for _, document in pairs
        ]


def test_load_pair_examples_parses_labels_and_pair_types(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_record("q1", "query", "IS 1: 2000", 1, "positive"),
            _pair_record("q1", "query", "IS 2: 2000", 0, "hard_negative"),
        ],
    )

    pairs = load_pair_examples(path)

    assert [pair.label for pair in pairs] == [1, 0]
    assert [pair.pair_type for pair in pairs] == ["positive", "hard_negative"]
    assert pairs[0].query == "query"


def test_load_pair_examples_rejects_bad_label_pair_type_combo(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [_pair_record("q1", "query", "IS 1: 2000", 1, "hard_negative")])

    with pytest.raises(ValueError, match="positive label"):
        load_pair_examples(path)


def test_group_pairs_by_query_requires_one_positive_per_query(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_record("q1", "query", "IS 1: 2000", 1, "positive"),
            _pair_record("q1", "query", "IS 2: 2000", 0, "hard_negative"),
            _pair_record("q1", "query", "IS 3: 2000", 0, "hard_negative"),
            _pair_record("q1", "query", "IS 4: 2000", 0, "hard_negative"),
            _pair_record("q1", "query", "IS 5: 2000", 0, "hard_negative"),
        ],
    )

    grouped = group_pairs_by_query(load_pair_examples(path))

    assert list(grouped) == ["q1"]
    assert sum(pair.label == 1 for pair in grouped["q1"]) == 1
    assert sum(pair.label == 0 for pair in grouped["q1"]) == 4


def test_validation_ranking_metrics_rank_positive_within_query_groups(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_record("q1", "first", "IS 1: 2000", 1, "positive"),
            _pair_record("q1", "first", "IS 2: 2000", 0, "hard_negative"),
            _pair_record("q2", "second", "IS 3: 2000", 1, "positive"),
            _pair_record("q2", "second", "IS 4: 2000", 0, "hard_negative"),
            _pair_record("q2", "second", "IS 5: 2000", 0, "hard_negative"),
        ],
    )
    groups = group_pairs_by_query(load_pair_examples(path))
    scored_groups = {
        "q1": [
            ScoredRerankerPair(pair=groups["q1"][0], score=0.9),
            ScoredRerankerPair(pair=groups["q1"][1], score=0.1),
        ],
        "q2": [
            ScoredRerankerPair(pair=groups["q2"][0], score=0.2),
            ScoredRerankerPair(pair=groups["q2"][1], score=0.8),
            ScoredRerankerPair(pair=groups["q2"][2], score=0.6),
        ],
    }

    metrics = ranking_metrics(scored_groups)

    assert metrics["query_count"] == 2
    assert metrics["hit_at_1"] == 1
    assert metrics["hit_at_3"] == 2
    assert metrics["hit_at_5"] == 2
    assert metrics["mrr"] == pytest.approx((1 + (1 / 3)) / 2)


def test_score_pair_groups_keeps_query_groups_and_uses_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_record("q1", "query", "IS 1: 2000", 1, "positive"),
            _pair_record("q1", "query", "IS 2: 2000", 0, "hard_negative"),
        ],
    )
    groups = group_pairs_by_query(load_pair_examples(path))
    model = FakeScoringModel({"IS 1: 2000": 10.0, "IS 2: 2000": -2.0})

    scored_groups = score_pair_groups(model, groups, batch_size=8)

    assert [scored.score for scored in scored_groups["q1"]] == [10.0, -2.0]
    assert model.calls[0][1] == 8


def test_pair_loss_and_accuracy_use_binary_logits() -> None:
    labels = [1, 0, 1, 0]
    logits = [3.0, -3.0, -1.0, 1.0]

    assert binary_cross_entropy_with_logits(labels, logits) > 0
    assert pair_accuracy_from_logits(labels, logits) == pytest.approx(0.5)


def test_select_best_checkpoint_prioritizes_validation_mrr_then_hit_at_1(tmp_path: Path) -> None:
    epoch_1 = CheckpointMetrics(
        epoch=1,
        checkpoint_path=tmp_path / "epoch-1",
        train_loss=0.2,
        validation_loss=0.4,
        validation_hit_at_1=0.80,
        validation_hit_at_3=1.0,
        validation_hit_at_5=1.0,
        validation_mrr=0.90,
    )
    epoch_2 = CheckpointMetrics(
        epoch=2,
        checkpoint_path=tmp_path / "epoch-2",
        train_loss=0.1,
        validation_loss=0.5,
        validation_hit_at_1=0.85,
        validation_hit_at_3=1.0,
        validation_hit_at_5=1.0,
        validation_mrr=0.90,
    )

    assert select_best_checkpoint([epoch_1, epoch_2]) == epoch_2


def test_standardwise_best_path_is_fine_tuned_checkpoint_path(tmp_path: Path) -> None:
    assert standardwise_reranker_best_path(tmp_path) == tmp_path / "best"


def test_frozen_query_leakage_catches_training_overlap(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [_pair_record("q1", "Need cement for finishing work.", "IS 1: 2000")])
    pairs = load_pair_examples(path)

    leakage = frozen_query_leakage(pairs, ["I need cement for finishing work."])

    assert leakage["exact"] == []
    assert leakage["near"]


def _pair_record(
    query_id: str,
    query: str,
    standard_code: str,
    label: int = 1,
    pair_type: str = "positive",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": query,
        "document": f"{standard_code}\nTitle\nfamily: test\nsource_scope: test scope",
        "label": label,
        "standard_code": standard_code,
        "pair_type": pair_type,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _standard_code_from_document(document: str) -> str:
    return document.splitlines()[0]
