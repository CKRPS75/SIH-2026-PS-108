from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.standards import canonicalize_standard_id, extract_base_code  # noqa: E402
from scripts.evaluate_post_training_partial4_comparison import (  # noqa: E402
    COMPATIBLE,
    FileBm25Index,
    ModelSpec,
    _build_suites,
    _candidate_pool,
    _candidate_text,
    _classify_error,
    _description_function,
    _load_cached_candidate_pools,
    _load_corpus,
    _product_compatibility,
    _rerank_rows,
    classify_cases_for_dataset,
)

DEFAULT_CANONICAL_PATH = Path("data/canonical/parsed_standards_canonical.json")
DEFAULT_FULL_EVAL_PATH = Path("data/evaluation/full_corpus_eval.jsonl")
DEFAULT_PRODUCT_HOLDOUT_PATH = Path("data/evaluation/product_aware_holdout.jsonl")
DEFAULT_AMBIGUOUS_EVAL_PATH = Path("data/evaluation/full_corpus_ambiguous_eval.jsonl")
DEFAULT_BASELINE_PATH = Path("data/evaluation/results/full_corpus_eval_baselines.json")
DEFAULT_REGRESSION_HYBRID_PATH = Path("data/evaluation/results/full_corpus_hybrid_regression.json")
DEFAULT_SCORE_CSV = Path("data/evaluation/results/reranker_score_delta_audit.csv")
DEFAULT_SCORE_JSON = Path("data/evaluation/results/reranker_score_delta_audit.json")
DEFAULT_RANK_JSON = Path("data/evaluation/results/reranker_rank_change_audit.json")
DEFAULT_REPORT = Path("data/evaluation/results/evaluation_integrity_audit.md")

HEAD_MARKERS = ("classifier", "score", "classification_head")
KNOWN_ERROR_TARGETS = {
    "part_variant_8008": ("full_200_frozen", "fc_eval_0069"),
    "function_12709_14402": ("full_200_frozen", "fc_eval_0063"),
    "material_6760_1365": ("full_200_frozen", "fc_eval_0152"),
    "product_aware_1626": ("product_aware_holdout", "pa_holdout_0042"),
    "product_aware_1703": ("product_aware_holdout", "pa_holdout_0044"),
}


@dataclass(frozen=True)
class ScorePair:
    query_id: str
    query: str
    standard_code: str
    pair_type: str


def main() -> None:
    args = _parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _validate_device(args.device)

    corpus = _load_corpus(args.canonical)
    corpus_by_code = {
        canonicalize_standard_id(record["standard_code"]): record for record in corpus["records"]
    }
    bm25_index = FileBm25Index(corpus["records"])
    suites = _build_suites(args)
    cached_pools = _load_cached_candidate_pools(args)
    suite_statuses = {
        name: classify_cases_for_dataset(name, cases, set(corpus_by_code))
        for name, cases in suites.items()
    }
    model_specs = _model_specs(args)

    print("Hashing checkpoints...")
    checkpoint_audit = {spec.key: _checkpoint_hash(spec) for spec in model_specs}

    print("Comparing parameter deltas...")
    parameter_audit = _parameter_audit(model_specs)

    print("Auditing independent model loads...")
    model_load_audit = _model_load_audit(model_specs, args)

    print("Building fixed candidate pools...")
    evaluation_cases = _evaluation_cases(
        suites,
        suite_statuses,
        bm25_index,
        cached_pools,
        args.rerank_k,
    )

    print("Scoring fixed pools with fresh model instances...")
    ranking_audit, pair_scores_by_key, score_pair_rows, current_errors, retrieval_misses = (
        _score_and_rank_audit(
            model_specs,
            args,
            evaluation_cases,
            corpus_by_code,
            bm25_index,
            cached_pools,
        )
    )

    score_delta_audit = _score_delta_audit(score_pair_rows, pair_scores_by_key)
    rank_change_audit = _rank_change_audit(ranking_audit)
    conclusion = _conclusion(
        parameter_audit, score_delta_audit, rank_change_audit, model_load_audit
    )

    payload = {
        "checkpoint_hashes": checkpoint_audit,
        "parameter_delta_audit": parameter_audit,
        "model_load_audit": model_load_audit,
        "score_delta_summary": score_delta_audit,
        "rank_change_audit": rank_change_audit,
        "benchmark_compatibility": suite_statuses,
        "retrieval_misses": retrieval_misses,
        "current_error_audit": current_errors,
        "integrity_conclusion": conclusion,
        "metrics_trustworthy": conclusion != "EVALUATION_IMPLEMENTATION_BUG",
    }

    args.score_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_score_csv(args.score_csv, score_pair_rows, pair_scores_by_key)
    args.score_json.write_text(json.dumps(score_delta_audit, indent=2), encoding="utf-8")
    args.rank_json.write_text(
        json.dumps(
            {
                "rank_change_audit": rank_change_audit,
                "retrieval_misses": retrieval_misses,
                "current_error_audit": current_errors,
                "benchmark_compatibility": suite_statuses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.report.write_text(_human_report(payload), encoding="utf-8")
    print(f"Wrote {args.score_csv}")
    print(f"Wrote {args.score_json}")
    print(f"Wrote {args.rank_json}")
    print(f"Wrote {args.report}")


def _validate_device(device: str) -> None:
    if device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")


def _model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        ModelSpec("pretrained_bge_large", "pretrained BAAI/bge-reranker-large", args.pretrained),
        ModelSpec("old_head_only", "old head-only checkpoint", args.old_head),
        ModelSpec("partial_last4", "new partial-last-4 checkpoint", args.partial4),
    ]


def _checkpoint_hash(spec: ModelSpec) -> dict[str, Any]:
    weight_file = _weight_file(spec.path)
    digest = hashlib.sha256()
    with weight_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "model_label": spec.label,
        "model_path": str(spec.path),
        "resolved_model_path": str(spec.path.resolve()),
        "weight_filename": weight_file.name,
        "weight_file": str(weight_file),
        "file_size_bytes": weight_file.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _weight_file(path: Path) -> Path:
    candidates = [path / "model.safetensors", path / "pytorch_model.bin"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported weight file found in {path}")


def _parameter_audit(model_specs: list[ModelSpec]) -> dict[str, Any]:
    comparisons = [
        (model_specs[0], model_specs[1]),
        (model_specs[0], model_specs[2]),
        (model_specs[1], model_specs[2]),
    ]
    return {
        f"{left.key}_vs_{right.key}": _compare_weight_files(
            _weight_file(left.path), _weight_file(right.path)
        )
        for left, right in comparisons
    }


def _compare_weight_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import safe_open

    layer_indices = _layer_indices(left_path)
    early_layer = min(layer_indices) if layer_indices else None
    final_layer = max(layer_indices) if layer_indices else None
    fourth_from_last = final_layer - 3 if final_layer is not None else None
    groups = {
        "classification_or_scoring_head": TensorAccumulator(),
        "final_transformer_layer": TensorAccumulator(),
        "fourth_from_last_transformer_layer": TensorAccumulator(),
        "early_frozen_transformer_layer": TensorAccumulator(),
    }
    all_tensors = TensorAccumulator()
    tensors_compared = 0
    tensors_with_nonzero_differences = 0

    with (
        safe_open(left_path, framework="pt", device="cpu") as left_file,
        safe_open(
            right_path,
            framework="pt",
            device="cpu",
        ) as right_file,
    ):
        left_keys = set(left_file.keys())
        right_keys = set(right_file.keys())
        common_keys = sorted(left_keys & right_keys)
        for key in common_keys:
            left = left_file.get_tensor(key)
            right = right_file.get_tensor(key)
            diff = torch.abs(left.float() - right.float())
            max_delta = float(diff.max().item()) if diff.numel() else 0.0
            sum_delta = float(diff.sum().item())
            count = diff.numel()
            nonzero = bool(torch.any(diff != 0).item()) if count else False
            tensors_compared += 1
            tensors_with_nonzero_differences += int(nonzero)
            all_tensors.add(count, sum_delta, max_delta, nonzero)
            group_name = _parameter_group(key, early_layer, final_layer, fourth_from_last)
            if group_name:
                groups[group_name].add(count, sum_delta, max_delta, nonzero)
            del left, right, diff

    return {
        "left_weight_file": str(left_path),
        "right_weight_file": str(right_path),
        "total_tensors_compared": tensors_compared,
        "tensors_with_nonzero_differences": tensors_with_nonzero_differences,
        "maximum_absolute_parameter_delta": all_tensors.max_abs_delta,
        "mean_absolute_parameter_delta": all_tensors.mean_abs_delta,
        "representative_parameter_groups": {
            name: accumulator.to_dict() for name, accumulator in groups.items()
        },
    }


class TensorAccumulator:
    def __init__(self) -> None:
        self.tensor_count = 0
        self.nonzero_tensor_count = 0
        self.element_count = 0
        self.sum_abs_delta = 0.0
        self.max_abs_delta = 0.0

    @property
    def mean_abs_delta(self) -> float:
        return self.sum_abs_delta / self.element_count if self.element_count else 0.0

    def add(
        self, element_count: int, sum_abs_delta: float, max_abs_delta: float, nonzero: bool
    ) -> None:
        self.tensor_count += 1
        self.nonzero_tensor_count += int(nonzero)
        self.element_count += element_count
        self.sum_abs_delta += sum_abs_delta
        self.max_abs_delta = max(self.max_abs_delta, max_abs_delta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_count": self.tensor_count,
            "nonzero_tensor_count": self.nonzero_tensor_count,
            "maximum_absolute_parameter_delta": self.max_abs_delta,
            "mean_absolute_parameter_delta": self.mean_abs_delta,
        }


def _layer_indices(weight_path: Path) -> list[int]:
    from safetensors.torch import safe_open

    indices = set()
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = _layer_match(key)
            if match is not None:
                indices.add(match)
    return sorted(indices)


def _parameter_group(
    key: str,
    early_layer: int | None,
    final_layer: int | None,
    fourth_from_last: int | None,
) -> str | None:
    if any(marker in key for marker in HEAD_MARKERS):
        return "classification_or_scoring_head"
    layer = _layer_match(key)
    if layer is None:
        return None
    if layer == final_layer:
        return "final_transformer_layer"
    if layer == fourth_from_last:
        return "fourth_from_last_transformer_layer"
    if layer == early_layer:
        return "early_frozen_transformer_layer"
    return None


def _layer_match(key: str) -> int | None:
    match = re.search(r"(?:encoder\.layer|layers)\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def _model_load_audit(
    model_specs: list[ModelSpec], args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows = []
    for spec in model_specs:
        model = _load_cross_encoder(spec.path, args)
        try:
            rows.append(
                {
                    "logical_model_label": spec.key,
                    "requested_checkpoint_path": str(spec.path),
                    "resolved_checkpoint_path": str(spec.path.resolve()),
                    "weight_file_actually_loaded": str(_weight_file(spec.path)),
                    "cross_encoder_object_id": id(model),
                    "underlying_model_object_id": id(getattr(model, "model", None)),
                    "tokenizer_class": type(getattr(model, "tokenizer", None)).__name__,
                    "model_class": type(getattr(model, "model", None)).__name__,
                }
            )
        finally:
            _release_model(model)
    object_ids = [row["underlying_model_object_id"] for row in rows]
    return {
        "fresh_model_instance_per_label": len(object_ids) == len(set(object_ids)),
        "score_cache_present": False,
        "score_cache_key_includes_model_identity": None,
        "loads": rows,
    }


def _load_cross_encoder(path: Path, args: argparse.Namespace) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(path), device=args.device, local_files_only=args.local_files_only)


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        return


def _evaluation_cases(
    suites: dict[str, list[Any]],
    suite_statuses: dict[str, Any],
    bm25_index: FileBm25Index,
    cached_pools: dict[str, dict[str, list[dict[str, Any]]]],
    rerank_k: int,
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for suite_name, cases in suites.items():
        if suite_statuses[suite_name]["status"] != COMPATIBLE:
            continue
        rows = []
        for case in cases:
            query = _case_query(suite_name, case)
            pool, source = _candidate_pool(
                suite_name, case, query, rerank_k, bm25_index, cached_pools
            )
            rows.append(
                {"case": case, "query": query, "candidate_pool": pool, "retrieval_source": source}
            )
        result[suite_name] = rows
    return result


def _case_query(suite_name: str, case: Any) -> str:
    if suite_name in {"product_aware_holdout", "ambiguous_product_diagnostics"}:
        from app.services.parsed_standards_corpus import build_product_query

        return build_product_query(case.product or "", case.description or "")
    return case.query


def _score_and_rank_audit(
    model_specs: list[ModelSpec],
    args: argparse.Namespace,
    evaluation_cases: dict[str, list[dict[str, Any]]],
    corpus_by_code: dict[str, dict[str, Any]],
    bm25_index: FileBm25Index,
    cached_pools: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], dict[str, float]],
    list[ScorePair],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    score_pairs = _select_score_pairs(evaluation_cases, corpus_by_code)
    pair_scores_by_key: dict[tuple[str, str], dict[str, float]] = {}
    ranking_by_model: dict[str, dict[str, dict[str, Any]]] = {spec.key: {} for spec in model_specs}
    raw_pool_scores: dict[str, dict[str, dict[str, float]]] = {spec.key: {} for spec in model_specs}

    for spec in model_specs:
        print(f"Raw-score/rank audit for {spec.key}...")
        model = _load_cross_encoder(spec.path, args)
        try:
            pair_scores = _score_pairs(model, score_pairs, corpus_by_code, args.rerank_batch_size)
            for pair_key, score in pair_scores.items():
                pair_scores_by_key.setdefault(pair_key, {})[spec.key] = score
            for suite_name, rows in evaluation_cases.items():
                for row in rows:
                    case = row["case"]
                    query = row["query"]
                    pool = row["candidate_pool"]
                    compatibility_by_code = None
                    prioritize_function = None
                    if suite_name in {"product_aware_holdout", "ambiguous_product_diagnostics"}:
                        from app.services.parsed_standards_corpus import canonicalize_product

                        canonical_product, confidence = canonicalize_product(case.product or "")
                        compatibility_by_code = {
                            candidate["standard_code"]: _product_compatibility(
                                canonical_product if confidence == "high" else None,
                                corpus_by_code.get(
                                    canonicalize_standard_id(candidate["standard_code"]), {}
                                ),
                            )
                            for candidate in pool
                        }
                        prioritize_function = _description_function(case.description or "")
                    reranked = _rerank_rows(
                        model,
                        query,
                        pool,
                        corpus_by_code,
                        return_k=args.rerank_k,
                        batch_size=args.rerank_batch_size,
                        compatibility_by_code=compatibility_by_code,
                        prioritize_function=prioritize_function,
                    )
                    case_key = f"{suite_name}:{case.query_id}"
                    ranking_by_model[spec.key][case_key] = {
                        "suite": suite_name,
                        "query_id": case.query_id,
                        "query": query,
                        "expected_standard": case.expected_standard,
                        "retrieval_source": row["retrieval_source"],
                        "candidate_pool": pool,
                        "ranking": [candidate["standard_code"] for candidate in reranked],
                        "scores": {
                            candidate["standard_code"]: candidate["reranker_score"]
                            for candidate in reranked
                        },
                        "expected_pool_rank": _expected_rank(pool, case.expected_standard),
                        "expected_reranked_rank": _rank_of(reranked, case.expected_standard),
                        "top1_standard": reranked[0]["standard_code"] if reranked else None,
                    }
                    raw_pool_scores[spec.key][case_key] = {
                        candidate["standard_code"]: candidate["reranker_score"]
                        for candidate in reranked
                    }
        finally:
            _release_model(model)

    current_errors = _current_error_audit(ranking_by_model, raw_pool_scores, corpus_by_code)
    retrieval_misses = _retrieval_misses(ranking_by_model["partial_last4"])
    return ranking_by_model, pair_scores_by_key, score_pairs, current_errors, retrieval_misses


def _select_score_pairs(
    evaluation_cases: dict[str, list[dict[str, Any]]],
    corpus_by_code: dict[str, dict[str, Any]],
) -> list[ScorePair]:
    selected: dict[tuple[str, str], ScorePair] = {}

    def add(query_id: str, query: str, standard_code: str, pair_type: str) -> None:
        if canonicalize_standard_id(standard_code) not in corpus_by_code:
            return
        selected.setdefault(
            (query_id, canonicalize_standard_id(standard_code)),
            ScorePair(query_id, query, standard_code, pair_type),
        )

    for suite_name, rows in evaluation_cases.items():
        for row in rows:
            case = row["case"]
            if not case.expected_standard:
                continue
            query = row["query"]
            pool = row["candidate_pool"]
            expected_code = canonicalize_standard_id(case.expected_standard)
            for candidate in pool:
                if canonicalize_standard_id(candidate["standard_code"]) == expected_code:
                    add(case.query_id, query, candidate["standard_code"], "easy_positive")
                    break
            for candidate in pool[:3]:
                if canonicalize_standard_id(candidate["standard_code"]) == expected_code:
                    continue
                pair_type = _pair_type(
                    case.expected_standard, candidate["standard_code"], corpus_by_code
                )
                add(case.query_id, query, candidate["standard_code"], pair_type)

    for _label, (suite_name, query_id) in KNOWN_ERROR_TARGETS.items():
        row = next(
            (
                row
                for row in evaluation_cases.get(suite_name, [])
                if row["case"].query_id == query_id
            ),
            None,
        )
        if row is None:
            continue
        case = row["case"]
        query = row["query"]
        if case.expected_standard:
            add(query_id, query, case.expected_standard, "known_error_expected")
        for candidate in row["candidate_pool"][:5]:
            add(
                query_id,
                query,
                candidate["standard_code"],
                _pair_type(case.expected_standard, candidate["standard_code"], corpus_by_code),
            )

    return list(selected.values())[: max(30, min(len(selected), 80))]


def _pair_type(expected: str, candidate: str, corpus_by_code: dict[str, dict[str, Any]]) -> str:
    expected_code = canonicalize_standard_id(expected)
    candidate_code = canonicalize_standard_id(candidate)
    if expected_code == candidate_code:
        return "easy_positive"
    if (
        extract_base_code(expected_code) == extract_base_code(candidate_code)
        or "PART" in expected_code
        or "PART" in candidate_code
    ):
        return "part_variant_confusion"
    expected_record = corpus_by_code.get(expected_code, {})
    candidate_record = corpus_by_code.get(candidate_code, {})
    if _different_nonempty(expected_record, candidate_record, "function"):
        return "function_confusion"
    if _different_nonempty(expected_record, candidate_record, "product_subtype"):
        return "subtype_confusion"
    if _different_nonempty(expected_record, candidate_record, "material") or _different_nonempty(
        expected_record, candidate_record, "application"
    ):
        return "material_application_confusion"
    if (
        expected_record.get("family")
        and candidate_record.get("family")
        and expected_record.get("family") != candidate_record.get("family")
    ):
        return "cross_product_lexical_trap"
    return "same_product_hard_negative"


def _score_pairs(
    model: Any,
    score_pairs: list[ScorePair],
    corpus_by_code: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[tuple[str, str], float]:
    import torch

    model_pairs = [
        [pair.query, _candidate_text(corpus_by_code[canonicalize_standard_id(pair.standard_code)])]
        for pair in score_pairs
    ]
    raw_scores = model.predict(
        model_pairs,
        batch_size=batch_size,
        activation_fct=torch.nn.Identity(),
    )
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    return {
        (pair.query_id, canonicalize_standard_id(pair.standard_code)): _as_float_score(score)
        for pair, score in zip(score_pairs, raw_scores, strict=True)
    }


def _as_float_score(score: Any) -> float:
    if isinstance(score, list | tuple):
        return float(score[0]) if score else 0.0
    return float(score)


def _score_delta_audit(
    score_pairs: list[ScorePair],
    pair_scores_by_key: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    comparisons = {
        "head_only_minus_pretrained": ("old_head_only", "pretrained_bge_large"),
        "partial_last4_minus_pretrained": ("partial_last4", "pretrained_bge_large"),
        "partial_last4_minus_head_only": ("partial_last4", "old_head_only"),
    }
    summary = {}
    for name, (left, right) in comparisons.items():
        differences = []
        identical = 0
        for pair in score_pairs:
            scores = pair_scores_by_key[
                (pair.query_id, canonicalize_standard_id(pair.standard_code))
            ]
            diff = scores[left] - scores[right]
            differences.append(abs(diff))
            identical += int(scores[left] == scores[right])
        summary[name] = _difference_summary(differences, identical)
    summary["pair_count"] = len(score_pairs)
    return summary


def _difference_summary(values: list[float], identical_count: int) -> dict[str, Any]:
    return {
        "mean_absolute_score_difference": mean(values) if values else 0.0,
        "median_absolute_score_difference": median(values) if values else 0.0,
        "maximum_absolute_score_difference": max(values) if values else 0.0,
        "exactly_identical_score_count": identical_count,
    }


def _rank_change_audit(ranking_by_model: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    pretrained = ranking_by_model["pretrained_bge_large"]
    head = ranking_by_model["old_head_only"]
    partial = ranking_by_model["partial_last4"]
    compatible_scored_keys = [key for key, row in partial.items() if row.get("expected_standard")]

    def topn(row: dict[str, Any], n: int) -> list[str]:
        return row["ranking"][:n]

    partial_any_k20_changed = [
        key for key in compatible_scored_keys if topn(partial[key], 20) != topn(pretrained[key], 20)
    ]
    partial_top5_changed = [
        key for key in compatible_scored_keys if topn(partial[key], 5) != topn(pretrained[key], 5)
    ]
    return {
        "compatible_scored_query_count": len(compatible_scored_keys),
        "head_only_top1_changed_vs_pretrained": sum(
            head[key]["top1_standard"] != pretrained[key]["top1_standard"]
            for key in compatible_scored_keys
        ),
        "partial_last4_top1_changed_vs_pretrained": sum(
            partial[key]["top1_standard"] != pretrained[key]["top1_standard"]
            for key in compatible_scored_keys
        ),
        "partial_last4_top3_ordering_changed_vs_pretrained": sum(
            topn(partial[key], 3) != topn(pretrained[key], 3) for key in compatible_scored_keys
        ),
        "partial_last4_top5_ordering_changed_vs_pretrained": len(partial_top5_changed),
        "partial_last4_any_k20_ordering_changed_vs_pretrained": len(partial_any_k20_changed),
        "partial_last4_spearman_mean": mean(
            _spearman(pretrained[key]["ranking"], partial[key]["ranking"])
            for key in compatible_scored_keys
        )
        if compatible_scored_keys
        else 0.0,
        "head_only_spearman_mean": mean(
            _spearman(pretrained[key]["ranking"], head[key]["ranking"])
            for key in compatible_scored_keys
        )
        if compatible_scored_keys
        else 0.0,
        "ranking_status": "RANKING_CHANGED"
        if partial_any_k20_changed
        else "SCORES_CHANGED_RANKING_UNCHANGED",
        "partial_last4_changed_query_keys": partial_any_k20_changed[:50],
    }


def _spearman(left: list[str], right: list[str]) -> float:
    common = [code for code in left if code in set(right)]
    if len(common) < 2:
        return 1.0 if left == right else 0.0
    left_rank = {code: index for index, code in enumerate(left, start=1)}
    right_rank = {code: index for index, code in enumerate(right, start=1)}
    n = len(common)
    squared = sum((left_rank[code] - right_rank[code]) ** 2 for code in common)
    return 1 - (6 * squared) / (n * (n * n - 1))


def _current_error_audit(
    ranking_by_model: dict[str, dict[str, dict[str, Any]]],
    raw_pool_scores: dict[str, dict[str, dict[str, float]]],
    corpus_by_code: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for label, (suite_name, query_id) in KNOWN_ERROR_TARGETS.items():
        case_key = f"{suite_name}:{query_id}"
        partial = ranking_by_model["partial_last4"].get(case_key)
        if not partial:
            continue
        expected = partial["expected_standard"]
        competitor_codes = [partial["top1_standard"]]
        if expected:
            competitor_codes.insert(0, expected)
        score_rows = {}
        for code in dict.fromkeys(code for code in competitor_codes if code):
            score_rows[code] = {
                model_key: raw_pool_scores[model_key].get(case_key, {}).get(code)
                for model_key in ranking_by_model
            }
        rows.append(
            {
                "label": label,
                "suite": suite_name,
                "query_id": query_id,
                "query": partial["query"],
                "expected_standard": expected,
                "candidate_rank_before_reranking": partial["expected_pool_rank"],
                "pretrained_rank": ranking_by_model["pretrained_bge_large"][case_key][
                    "expected_reranked_rank"
                ],
                "head_only_rank": ranking_by_model["old_head_only"][case_key][
                    "expected_reranked_rank"
                ],
                "partial_last4_rank": partial["expected_reranked_rank"],
                "top1_partial_last4": partial["top1_standard"],
                "scores": score_rows,
                "classification": _current_error_classification(partial, corpus_by_code),
            }
        )
    return rows


def _current_error_classification(
    row: dict[str, Any],
    corpus_by_code: dict[str, dict[str, Any]],
) -> str:
    classification = _classify_error(
        row["expected_standard"],
        row["expected_pool_rank"],
        row["expected_reranked_rank"],
        row["top1_standard"],
        corpus_by_code,
    )
    if classification == "PART_OR_VARIANT_CONFUSION":
        return "PART_VARIANT_CONFUSION"
    if classification == "RERANKER_ERROR":
        return "OTHER_RERANKER_ERROR"
    return classification


def _retrieval_misses(partial_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    misses = []
    for row in partial_rows.values():
        if not row.get("expected_standard") or row["expected_pool_rank"] is not None:
            continue
        misses.append(
            {
                "suite": row["suite"],
                "query_id": row["query_id"],
                "query": row["query"],
                "expected_standard": row["expected_standard"],
                "semantic_rank": None,
                "bm25_rank": None,
                "hybrid_rank": None,
                "candidate_cutoff_position": 20,
                "retrieval_source": row["retrieval_source"],
            }
        )
    return misses


def _expected_rank(candidates: list[dict[str, Any]], expected: str) -> int | None:
    if not expected:
        return None
    expected_code = canonicalize_standard_id(expected)
    for candidate in candidates:
        if canonicalize_standard_id(candidate["standard_code"]) == expected_code:
            return int(candidate["rank"])
    return None


def _rank_of(candidates: list[dict[str, Any]], expected: str) -> int | None:
    return _expected_rank(candidates, expected)


def _different_nonempty(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    return bool(left.get(key) and right.get(key) and left.get(key) != right.get(key))


def _write_score_csv(
    path: Path,
    score_pairs: list[ScorePair],
    pair_scores_by_key: dict[tuple[str, str], dict[str, float]],
) -> None:
    fieldnames = [
        "query_id",
        "query",
        "standard_code",
        "pair_type",
        "pretrained_score",
        "head_only_score",
        "partial_last4_score",
        "head_minus_pretrained",
        "partial_minus_pretrained",
        "partial_minus_head",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in score_pairs:
            scores = pair_scores_by_key[
                (pair.query_id, canonicalize_standard_id(pair.standard_code))
            ]
            writer.writerow(
                {
                    "query_id": pair.query_id,
                    "query": pair.query,
                    "standard_code": pair.standard_code,
                    "pair_type": pair.pair_type,
                    "pretrained_score": scores["pretrained_bge_large"],
                    "head_only_score": scores["old_head_only"],
                    "partial_last4_score": scores["partial_last4"],
                    "head_minus_pretrained": scores["old_head_only"]
                    - scores["pretrained_bge_large"],
                    "partial_minus_pretrained": scores["partial_last4"]
                    - scores["pretrained_bge_large"],
                    "partial_minus_head": scores["partial_last4"] - scores["old_head_only"],
                }
            )


def _conclusion(
    parameter_audit: dict[str, Any],
    score_delta_audit: dict[str, Any],
    rank_change_audit: dict[str, Any],
    model_load_audit: dict[str, Any],
) -> str:
    if not model_load_audit["fresh_model_instance_per_label"]:
        return "EVALUATION_IMPLEMENTATION_BUG"
    changed = any(
        report["tensors_with_nonzero_differences"] > 0 for report in parameter_audit.values()
    )
    score_changed = any(
        value.get("maximum_absolute_score_difference", 0.0) > 0.0
        for key, value in score_delta_audit.items()
        if key != "pair_count"
    )
    if not changed or not score_changed:
        return "C. MODEL_NOT_MEANINGFULLY_CHANGED".split(". ", 1)[1]
    if rank_change_audit["partial_last4_any_k20_ordering_changed_vs_pretrained"] > 0:
        return "MODEL_CHANGED_AND_RANKINGS_CHANGED"
    return "MODEL_CHANGED_BUT_RANKINGS_UNCHANGED"


def _human_report(payload: dict[str, Any]) -> str:
    lines = [
        "# StandardWise Reranker Evaluation Integrity Audit",
        "",
        f"Final conclusion: {payload['integrity_conclusion']}",
        f"Frozen metrics trustworthy: {payload['metrics_trustworthy']}",
        "",
        "## Checkpoint Hashes",
    ]
    for key, row in payload["checkpoint_hashes"].items():
        lines.append(
            f"- {key}: {row['weight_filename']} "
            f"size={row['file_size_bytes']} "
            f"sha256={row['sha256']}"
        )
    lines.extend(["", "## Parameter Delta Summary"])
    for key, row in payload["parameter_delta_audit"].items():
        nonzero = row["tensors_with_nonzero_differences"]
        max_delta = row["maximum_absolute_parameter_delta"]
        mean_delta = row["mean_absolute_parameter_delta"]
        lines.append(
            f"- {key}: tensors={row['total_tensors_compared']} "
            f"nonzero={nonzero} max={max_delta:.8g} mean={mean_delta:.8g}"
        )
        for group, group_row in row["representative_parameter_groups"].items():
            group_nonzero = group_row["nonzero_tensor_count"]
            group_max = group_row["maximum_absolute_parameter_delta"]
            group_mean = group_row["mean_absolute_parameter_delta"]
            lines.append(
                f"  - {group}: tensors={group_row['tensor_count']} "
                f"nonzero={group_nonzero} max={group_max:.8g} "
                f"mean={group_mean:.8g}"
            )
    lines.extend(["", "## Model Loading Verification"])
    fresh_loads = payload["model_load_audit"]["fresh_model_instance_per_label"]
    lines.append(f"Fresh model instance per label: {fresh_loads}")
    lines.append(f"Score cache present: {payload['model_load_audit']['score_cache_present']}")
    for row in payload["model_load_audit"]["loads"]:
        object_id = row["underlying_model_object_id"]
        lines.append(
            f"- {row['logical_model_label']}: requested={row['requested_checkpoint_path']} "
            f"resolved={row['resolved_checkpoint_path']} "
            f"weight={row['weight_file_actually_loaded']} "
            f"tokenizer={row['tokenizer_class']} "
            f"model={row['model_class']} object={object_id}"
        )
    lines.extend(["", "## Raw Score Deltas"])
    lines.append(f"Audited pair count: {payload['score_delta_summary']['pair_count']}")
    for key, row in payload["score_delta_summary"].items():
        if key == "pair_count":
            continue
        lines.append(
            f"- {key}: mean_abs={row['mean_absolute_score_difference']:.8g} "
            f"median_abs={row['median_absolute_score_difference']:.8g} "
            f"max_abs={row['maximum_absolute_score_difference']:.8g} "
            f"identical={row['exactly_identical_score_count']}"
        )
    rank = payload["rank_change_audit"]
    lines.extend(["", "## Rank Changes"])
    for key, value in rank.items():
        if key != "partial_last4_changed_query_keys":
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Benchmark Compatibility"])
    for name, status in payload["benchmark_compatibility"].items():
        line = f"- {name}: {status['status']} ({status['query_count']} queries)"
        if status.get("missing_standard_codes"):
            line += f" missing={', '.join(status['missing_standard_codes'])}"
        lines.append(line)
    lines.extend(["", f"## Retrieval Misses ({len(payload['retrieval_misses'])})"])
    for row in payload["retrieval_misses"]:
        lines.append(
            f"- {row['suite']}:{row['query_id']} expected={row['expected_standard']} "
            f"cutoff={row['candidate_cutoff_position']} source={row['retrieval_source']}"
        )
    lines.extend(["", "## Current Error Audit"])
    for row in payload["current_error_audit"]:
        pool_rank = row["candidate_rank_before_reranking"]
        lines.append(
            f"- {row['label']}: expected={row['expected_standard']} "
            f"top1={row['top1_partial_last4']} "
            f"pool_rank={pool_rank} pretrained_rank={row['pretrained_rank']} "
            f"head_rank={row['head_only_rank']} partial_rank={row['partial_last4_rank']} "
            f"class={row['classification']}"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit StandardWise reranker evaluation integrity."
    )
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--full-eval", type=Path, default=DEFAULT_FULL_EVAL_PATH)
    parser.add_argument("--product-holdout", type=Path, default=DEFAULT_PRODUCT_HOLDOUT_PATH)
    parser.add_argument("--ambiguous-eval", type=Path, default=DEFAULT_AMBIGUOUS_EVAL_PATH)
    parser.add_argument("--full-baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--regression-hybrid", type=Path, default=DEFAULT_REGRESSION_HYBRID_PATH)
    parser.add_argument("--pretrained", type=Path, default=Path("models/bge-reranker-large"))
    parser.add_argument("--old-head", type=Path, default=Path("models/standardwise-reranker/best"))
    parser.add_argument(
        "--partial4",
        type=Path,
        default=Path("models/full-corpus-reranker-partial4/best"),
    )
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--score-json", type=Path, default=DEFAULT_SCORE_JSON)
    parser.add_argument("--rank-json", type=Path, default=DEFAULT_RANK_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
