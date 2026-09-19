from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_ROOT


@dataclass(frozen=True)
class RerankerPair:
    query_id: str
    query: str
    document: str
    label: int
    standard_code: str
    pair_type: str


@dataclass(frozen=True)
class ScoredRerankerPair:
    pair: RerankerPair
    score: float


@dataclass(frozen=True)
class CheckpointMetrics:
    epoch: int
    checkpoint_path: Path
    train_loss: float
    validation_loss: float
    validation_hit_at_1: float
    validation_hit_at_3: float
    validation_hit_at_5: float
    validation_mrr: float


def load_pair_examples(path: Path) -> list[RerankerPair]:
    pairs: list[RerankerPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = _parse_label(record.get("label"), line_number)
            pair_type = str(record.get("pair_type", "")).strip()
            if label == 1 and pair_type != "positive":
                raise ValueError(f"{path}:{line_number} positive label must use pair_type=positive")
            if label == 0 and pair_type != "hard_negative":
                raise ValueError(
                    f"{path}:{line_number} negative label must use pair_type=hard_negative"
                )
            pairs.append(
                RerankerPair(
                    query_id=_required_text(record, "query_id", path, line_number),
                    query=_required_text(record, "query", path, line_number),
                    document=_required_text(record, "document", path, line_number),
                    label=label,
                    standard_code=_required_text(record, "standard_code", path, line_number),
                    pair_type=pair_type,
                )
            )
    return pairs


def group_pairs_by_query(pairs: Sequence[RerankerPair]) -> dict[str, list[RerankerPair]]:
    grouped: dict[str, list[RerankerPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.query_id].append(pair)

    for query_id, query_pairs in grouped.items():
        positives = [pair for pair in query_pairs if pair.label == 1]
        if len(positives) != 1:
            raise ValueError(f"{query_id} must contain exactly one positive pair")
        queries = {pair.query for pair in query_pairs}
        if len(queries) != 1:
            raise ValueError(f"{query_id} contains inconsistent query text")
    return dict(grouped)


def binary_cross_entropy_with_logits(labels: Sequence[int], logits: Sequence[float]) -> float:
    if len(labels) != len(logits):
        raise ValueError("labels and logits must have the same length")
    if not labels:
        raise ValueError("at least one label is required")
    total = 0.0
    for label, logit in zip(labels, logits, strict=True):
        if label not in {0, 1}:
            raise ValueError("labels must be 0 or 1")
        total += max(logit, 0.0) - (logit * label) + math.log1p(math.exp(-abs(logit)))
    return total / len(labels)


def pair_accuracy_from_logits(labels: Sequence[int], logits: Sequence[float]) -> float:
    if len(labels) != len(logits):
        raise ValueError("labels and logits must have the same length")
    if not labels:
        raise ValueError("at least one label is required")
    correct = sum(
        (logit >= 0.0) == bool(label) for label, logit in zip(labels, logits, strict=True)
    )
    return correct / len(labels)


def score_pairs(
    model: Any,
    pairs: Sequence[RerankerPair],
    *,
    batch_size: int,
    activation_fn: Any | None = None,
) -> list[float]:
    inputs = [[pair.query, pair.document] for pair in pairs]
    kwargs: dict[str, Any] = {"batch_size": batch_size}
    if activation_fn is not None:
        kwargs["activation_fct"] = activation_fn
    raw_scores = model.predict(inputs, **kwargs)
    return as_float_scores(raw_scores)


def as_float_scores(raw_scores: Any) -> list[float]:
    if hasattr(raw_scores, "detach"):
        raw_scores = raw_scores.detach().cpu()
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, int | float):
        return [float(raw_scores)]
    return [_as_float(score) for score in raw_scores]


def score_pair_groups(
    model: Any,
    grouped_pairs: dict[str, list[RerankerPair]],
    *,
    batch_size: int,
    activation_fn: Any | None = None,
) -> dict[str, list[ScoredRerankerPair]]:
    ordered_pairs = [pair for pairs in grouped_pairs.values() for pair in pairs]
    scores = score_pairs(
        model,
        ordered_pairs,
        batch_size=batch_size,
        activation_fn=activation_fn,
    )
    scored_by_query: dict[str, list[ScoredRerankerPair]] = defaultdict(list)
    for pair, score in zip(ordered_pairs, scores, strict=True):
        scored_by_query[pair.query_id].append(ScoredRerankerPair(pair=pair, score=score))
    return dict(scored_by_query)


def ranking_metrics(scored_groups: dict[str, list[ScoredRerankerPair]]) -> dict[str, float]:
    if not scored_groups:
        raise ValueError("at least one query group is required")
    positive_ranks = []
    for query_id, scored_pairs in scored_groups.items():
        positives = [scored_pair for scored_pair in scored_pairs if scored_pair.pair.label == 1]
        if len(positives) != 1:
            raise ValueError(f"{query_id} must contain exactly one positive pair")
        ranked = sorted(
            scored_pairs,
            key=lambda scored_pair: (-scored_pair.score, scored_pair.pair.standard_code),
        )
        positive_pair = positives[0].pair
        rank = next(
            index
            for index, scored_pair in enumerate(ranked, start=1)
            if scored_pair.pair == positive_pair
        )
        positive_ranks.append(rank)

    query_count = len(positive_ranks)
    return {
        "query_count": query_count,
        "hit_at_1": sum(rank <= 1 for rank in positive_ranks),
        "hit_at_3": sum(rank <= 3 for rank in positive_ranks),
        "hit_at_5": sum(rank <= 5 for rank in positive_ranks),
        "mrr": sum(1 / rank for rank in positive_ranks) / query_count,
        "hit_at_1_rate": sum(rank <= 1 for rank in positive_ranks) / query_count,
        "hit_at_3_rate": sum(rank <= 3 for rank in positive_ranks) / query_count,
        "hit_at_5_rate": sum(rank <= 5 for rank in positive_ranks) / query_count,
    }


def select_best_checkpoint(results: Sequence[CheckpointMetrics]) -> CheckpointMetrics:
    if not results:
        raise ValueError("at least one checkpoint result is required")
    return max(
        results,
        key=lambda result: (
            result.validation_mrr,
            result.validation_hit_at_1,
            -result.validation_loss,
            -result.epoch,
        ),
    )


def standardwise_reranker_best_path(output_root: Path | None = None) -> Path:
    root = output_root or BACKEND_ROOT / "models" / "standardwise-reranker"
    return root / "best"


def frozen_query_leakage(
    training_pairs: Sequence[RerankerPair],
    frozen_queries: Iterable[str],
    *,
    near_threshold: float = 0.92,
) -> dict[str, list[str]]:
    training_queries = {pair.query for pair in training_pairs}
    exact_frozen = {_query_key(query): query for query in frozen_queries}
    exact = []
    near = []
    for query in sorted(training_queries):
        key = _query_key(query)
        if key in exact_frozen:
            exact.append(f"{query} == {exact_frozen[key]}")
            continue
        for frozen_query in frozen_queries:
            ratio = SequenceMatcher(None, key, _query_key(frozen_query)).ratio()
            if ratio >= near_threshold:
                near.append(f"{query} ~= {frozen_query}")
                break
    return {"exact": exact, "near": near}


def checkpoint_metrics_to_dict(metrics: CheckpointMetrics) -> dict[str, str | int | float]:
    return {
        "epoch": metrics.epoch,
        "checkpoint_path": str(metrics.checkpoint_path),
        "train_loss": metrics.train_loss,
        "validation_loss": metrics.validation_loss,
        "validation_hit_at_1": metrics.validation_hit_at_1,
        "validation_hit_at_3": metrics.validation_hit_at_3,
        "validation_hit_at_5": metrics.validation_hit_at_5,
        "validation_mrr": metrics.validation_mrr,
    }


def pairwise_ranknet_loss(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    if len(positive_scores) != len(negative_scores):
        raise ValueError("positive_scores and negative_scores must have the same length")
    if not positive_scores:
        raise ValueError("at least one score pair is required")
    total = 0.0
    for positive, negative in zip(positive_scores, negative_scores, strict=True):
        margin = positive - negative
        total += math.log1p(math.exp(-margin))
    return total / len(positive_scores)


def configure_last_n_layers_trainable(cross_encoder: Any, *, last_n: int) -> dict[str, int | float]:
    if last_n < 1:
        raise ValueError("last_n must be at least 1")
    model = getattr(cross_encoder, "model", None)
    if model is None:
        raise RuntimeError("CrossEncoder does not expose its underlying torch model")
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_name_markers = ("classifier", "score", "classification_head")
    encoder_layers = _find_encoder_layers(model)
    for name, parameter in model.named_parameters():
        if any(marker in name for marker in trainable_name_markers):
            parameter.requires_grad = True
    for layer in encoder_layers[-last_n:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": trainable / total if total else 0.0,
    }


def _find_encoder_layers(model: Any) -> list[Any]:
    candidates = [
        "base_model.encoder.layer",
        "roberta.encoder.layer",
        "deberta.encoder.layer",
        "bert.encoder.layer",
        "xlm_roberta.encoder.layer",
    ]
    for path in candidates:
        value = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        try:
            layers = list(value)
        except TypeError:
            continue
        if layers:
            return layers
    raise RuntimeError("Could not locate transformer encoder layers")


def _parse_label(value: Any, line_number: int) -> int:
    if value in {0, 1}:
        return int(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    raise ValueError(f"line {line_number} label must be 0 or 1")


def _required_text(record: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number} missing non-empty {key}")
    return value


def _as_float(score: Any) -> float:
    if isinstance(score, Sequence) and not isinstance(score, str):
        if not score:
            raise ValueError("score sequence must not be empty")
        score = score[0]
    return float(score)


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())
