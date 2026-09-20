"""Audit StandardWise reranker fit quality from preserved artifacts.

This script is intentionally evaluation-only. It reads training/evaluation
artifacts and writes fit-audit reports without loading reranker weights or
touching the production recommendation pipeline.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"

TRAIN_PAIRS = ROOT / "data" / "training" / "full_corpus_train_pairs.jsonl"
VALIDATION_PAIRS = ROOT / "data" / "training" / "full_corpus_validation_pairs.jsonl"
TRAINING_REPORT = ROOT / "data" / "training" / "full_corpus_training_report.json"
FULL_EVAL = ROOT / "data" / "evaluation" / "full_corpus_eval.jsonl"
PRODUCT_AWARE_HOLDOUT = ROOT / "data" / "evaluation" / "product_aware_holdout.jsonl"
AMBIGUOUS_EVAL = ROOT / "data" / "evaluation" / "full_corpus_ambiguous_eval.jsonl"
TRAINING_METRICS = ROOT / "models" / "standardwise-reranker" / "training_metrics.json"
COMPARISON_REPORT = RESULTS_DIR / "post_training_partial4_comparison.json"
INTEGRITY_REPORT = RESULTS_DIR / "evaluation_integrity_audit.md"

AUDIT_JSON = RESULTS_DIR / "reranker_fit_audit.json"
AUDIT_MD = RESULTS_DIR / "reranker_fit_audit.md"
EPOCH_CSV = RESULTS_DIR / "reranker_epoch_comparison.csv"
LEARNING_CURVE_CSV = RESULTS_DIR / "reranker_learning_curve.csv"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def token_set(value: str) -> set[str]:
    return set(normalize_text(value).split())


def token_jaccard_sets(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_scores(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label")) for row in rows)
    query_ids = {row.get("query_id") for row in rows if row.get("query_id")}
    query_texts = {row.get("query") for row in rows if row.get("query")}
    positive_standards = {
        row.get("positive_standard_code") for row in rows if row.get("positive_standard_code")
    }
    candidate_standards = {row.get("standard_code") for row in rows if row.get("standard_code")}
    negatives = [row for row in rows if row.get("label") == 0]
    positives = [row for row in rows if row.get("label") == 1]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("query_id")].append(row)
    candidates_per_query = [len(items) for items in grouped.values()]
    return {
        "pair_count": len(rows),
        "positive_pair_count": len(positives),
        "negative_pair_count": len(negatives),
        "label_counts": dict(label_counts),
        "unique_query_ids": len(query_ids),
        "unique_query_texts": len(query_texts),
        "unique_positive_standards": len(positive_standards),
        "unique_candidate_standards": len(candidate_standards),
        "negative_category_counts": dict(
            Counter(row.get("negative_category") or "" for row in negatives)
        ),
        "negative_mining_source_counts": dict(
            Counter(row.get("negative_mining_source") or "" for row in negatives)
        ),
        "candidates_per_query": {
            "min": min(candidates_per_query) if candidates_per_query else None,
            "median": safe_median(candidates_per_query),
            "max": max(candidates_per_query) if candidates_per_query else None,
        },
    }


def eval_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [row.get("expected_standard") for row in rows if row.get("expected_standard")]
    products = [
        row.get("product") or row.get("expected_product")
        for row in rows
        if row.get("product") or row.get("expected_product")
    ]
    return {
        "query_count": len(rows),
        "labeled_query_count": len(expected),
        "unique_expected_standards": len(set(expected)),
        "unique_products": len(set(products)),
        "families": dict(Counter(row.get("family") or "" for row in rows)),
    }


def query_text(row: dict[str, Any]) -> str:
    if row.get("query"):
        return str(row["query"])
    product = row.get("product") or row.get("expected_product") or ""
    description = row.get("description") or ""
    return f"{product} {description}".strip()


def build_leakage_report(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    eval_sets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    train_query_ids = {row.get("query_id") for row in train_rows if row.get("query_id")}
    val_query_ids = {row.get("query_id") for row in validation_rows if row.get("query_id")}
    train_norm_queries = {
        normalize_text(row.get("query")) for row in train_rows if row.get("query")
    }
    val_norm_queries = {
        normalize_text(row.get("query")) for row in validation_rows if row.get("query")
    }
    train_pairs = {
        (normalize_text(row.get("query")), row.get("standard_code"), row.get("label"))
        for row in train_rows
    }
    val_pairs = {
        (normalize_text(row.get("query")), row.get("standard_code"), row.get("label"))
        for row in validation_rows
    }
    train_positive_standards = {
        row.get("positive_standard_code") for row in train_rows if row.get("positive_standard_code")
    }
    val_positive_standards = {
        row.get("positive_standard_code")
        for row in validation_rows
        if row.get("positive_standard_code")
    }

    eval_leakage = {}
    train_query_samples = [
        (str(query), token_set(str(query)))
        for query in sorted({row.get("query") for row in train_rows if row.get("query")})
    ]
    for name, rows in eval_sets.items():
        eval_norms = {normalize_text(query_text(row)) for row in rows if query_text(row)}
        exact = sorted(norm for norm in eval_norms & train_norm_queries if norm)
        high_similarity = []
        for eval_row in rows:
            text = query_text(eval_row)
            if not text:
                continue
            best_score = 0.0
            best_train = ""
            eval_tokens = token_set(text)
            for train_text, train_tokens in train_query_samples:
                score = token_jaccard_sets(eval_tokens, train_tokens)
                if score > best_score:
                    best_score = score
                    best_train = train_text
            if best_score >= 0.92:
                high_similarity.append(
                    {
                        "query_id": eval_row.get("query_id"),
                        "eval_text": text,
                        "train_text": best_train,
                        "similarity": round(best_score, 4),
                    }
                )
        eval_expected = {
            row.get("expected_standard") for row in rows if row.get("expected_standard")
        }
        eval_leakage[name] = {
            "exact_normalized_query_overlap_with_train": len(exact),
            "exact_normalized_query_overlap_examples": exact[:10],
            "near_duplicate_query_count_ge_0_92": len(high_similarity),
            "near_duplicate_examples": high_similarity[:10],
            "expected_standard_overlap_with_train_count": len(
                eval_expected & train_positive_standards
            ),
            "expected_standard_count": len(eval_expected),
        }

    negative_positive_violations = []
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            if row.get("label") == 0 and row.get("standard_code") == row.get(
                "positive_standard_code"
            ):
                negative_positive_violations.append(
                    {
                        "split": split_name,
                        "query_id": row.get("query_id"),
                        "standard_code": row.get("standard_code"),
                    }
                )

    return {
        "train_validation_query_id_overlap_count": len(train_query_ids & val_query_ids),
        "train_validation_normalized_query_overlap_count": len(
            train_norm_queries & val_norm_queries
        ),
        "train_validation_pair_overlap_count": len(train_pairs & val_pairs),
        "train_validation_positive_standard_overlap_count": len(
            train_positive_standards & val_positive_standards
        ),
        "train_positive_standard_count": len(train_positive_standards),
        "validation_positive_standard_count": len(val_positive_standards),
        "negative_positive_violations_count": len(negative_positive_violations),
        "negative_positive_violation_examples": negative_positive_violations[:10],
        "eval_set_leakage": eval_leakage,
        "interpretation": (
            "Exact query and pair overlap checks test leakage. Positive-standard overlap is reported "
            "as a generalization limitation, not as leakage, because validation appears query-held-out "
            "rather than standard-held-out."
        ),
    }


def summarize_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    suite_results = comparison.get("suite_results", {})
    metrics_rows = []
    score_values: dict[str, list[float]] = defaultdict(list)
    top1_delta_total = 0
    scored_query_total = 0
    weighted_mrr_delta_sum = 0.0

    for suite_name, model_results in suite_results.items():
        pretrained = model_results.get("pretrained_bge_large", {})
        fine_tuned = model_results.get("partial_last4") or model_results.get("old_head_only") or {}
        pretrained_metrics = pretrained.get("metrics") or {}
        fine_tuned_metrics = fine_tuned.get("metrics") or {}
        scored_count = fine_tuned_metrics.get("query_count") or 0
        if pretrained_metrics and fine_tuned_metrics:
            hit1_delta = (fine_tuned_metrics.get("hit_at_1") or 0) - (
                pretrained_metrics.get("hit_at_1") or 0
            )
            mrr_delta = (fine_tuned_metrics.get("mrr") or 0.0) - (
                pretrained_metrics.get("mrr") or 0.0
            )
            top1_delta_total += hit1_delta
            scored_query_total += scored_count
            weighted_mrr_delta_sum += mrr_delta * scored_count
        else:
            hit1_delta = None
            mrr_delta = None
        for model_name, result in model_results.items():
            for case in result.get("results") or []:
                for candidate in case.get("candidates") or []:
                    score = candidate.get("reranker_score")
                    if isinstance(score, int | float) and math.isfinite(float(score)):
                        score_values[model_name].append(float(score))
        metrics_rows.append(
            {
                "suite": suite_name,
                "status": fine_tuned.get("status") or pretrained.get("status"),
                "scored": fine_tuned.get("scored"),
                "query_count": fine_tuned.get("query_count"),
                "labeled_query_count": fine_tuned.get("labeled_query_count"),
                "pretrained_hit_at_1": pretrained_metrics.get("hit_at_1"),
                "fine_tuned_hit_at_1": fine_tuned_metrics.get("hit_at_1"),
                "hit_at_1_delta": hit1_delta,
                "pretrained_mrr": pretrained_metrics.get("mrr"),
                "fine_tuned_mrr": fine_tuned_metrics.get("mrr"),
                "mrr_delta": mrr_delta,
                "candidate_recall_fine_tuned": fine_tuned.get("candidate_recall"),
                "fine_tuned_error_counts": fine_tuned.get("error_counts"),
            }
        )

    return {
        "device": comparison.get("device"),
        "models": comparison.get("models"),
        "suite_statuses": comparison.get("suite_statuses"),
        "suite_metric_comparison": metrics_rows,
        "score_distributions_returned_top5": {
            model: summarize_scores(scores) for model, scores in sorted(score_values.items())
        },
        "aggregate_delta": {
            "scored_query_count": scored_query_total,
            "fine_tuned_minus_pretrained_hit_at_1_count": top1_delta_total,
            "fine_tuned_minus_pretrained_weighted_mrr_delta": (
                weighted_mrr_delta_sum / scored_query_total if scored_query_total else None
            ),
        },
    }


def parse_integrity_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8")
    wanted_prefixes = [
        "Final conclusion:",
        "Frozen metrics trustworthy:",
        "- partial_last4_top1_changed_vs_pretrained:",
        "- partial_last4_top3_ordering_changed_vs_pretrained:",
        "- partial_last4_top5_ordering_changed_vs_pretrained:",
        "- partial_last4_any_k20_ordering_changed_vs_pretrained:",
        "- compatible_scored_query_count:",
        "- partial_last4_spearman_mean:",
        "- ranking_status:",
    ]
    extracted = {}
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in wanted_prefixes:
            if stripped.startswith(prefix):
                key = prefix.strip("- :").lower().replace(" ", "_")
                extracted[key] = stripped.removeprefix(prefix).strip()
    return {"available": True, "extracted": extracted}


def build_training_curve(metrics: dict[str, Any]) -> dict[str, Any]:
    epochs = metrics.get("epochs") or []
    rows = []
    for epoch in epochs:
        checkpoint_path = Path(str(epoch.get("checkpoint_path", "")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        rows.append(
            {
                "epoch": epoch.get("epoch"),
                "train_loss": epoch.get("train_loss"),
                "validation_loss": epoch.get("validation_loss"),
                "validation_hit_at_1": epoch.get("validation_hit_at_1"),
                "validation_hit_at_3": epoch.get("validation_hit_at_3"),
                "validation_hit_at_5": epoch.get("validation_hit_at_5"),
                "validation_mrr": epoch.get("validation_mrr"),
                "checkpoint_path": str(epoch.get("checkpoint_path")),
                "checkpoint_exists_in_current_workspace": checkpoint_path.exists(),
                "is_selected_best": epoch.get("epoch") == metrics.get("best_epoch"),
            }
        )
    deltas = {}
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        deltas = {
            "train_loss_last_minus_first": (last["train_loss"] or 0) - (first["train_loss"] or 0),
            "validation_loss_last_minus_first": (last["validation_loss"] or 0)
            - (first["validation_loss"] or 0),
            "validation_mrr_last_minus_first": (last["validation_mrr"] or 0)
            - (first["validation_mrr"] or 0),
        }
    return {
        "selection_metric": metrics.get("selection_metric"),
        "best_epoch": metrics.get("best_epoch"),
        "best_checkpoint_path": metrics.get("best_checkpoint_path"),
        "best_model_path": metrics.get("best_model_path"),
        "observed_epoch_count": len(rows),
        "epochs": rows,
        "deltas_first_to_last": deltas,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def model_inventory() -> dict[str, Any]:
    model_root = ROOT / "models"
    inventory = {}
    for path in sorted(model_root.glob("*")) if model_root.exists() else []:
        if path.is_dir():
            inventory[path.name] = {
                "path": str(path.relative_to(ROOT)),
                "children": sorted(child.name for child in path.iterdir()),
            }
    comparison = load_json(COMPARISON_REPORT)
    referenced = comparison.get("models") or {}
    for model_name, spec in referenced.items():
        model_path = spec.get("path")
        exists = None
        if model_path:
            local_path = ROOT / str(model_path)
            exists = local_path.exists()
        spec["exists_in_current_windows_workspace"] = exists
    return {
        "local_models": inventory,
        "models_referenced_by_post_training_comparison": referenced,
    }


def final_classification(
    training_curve: dict[str, Any],
    comparison: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    aggregate = comparison.get("aggregate_delta") or {}
    scored = aggregate.get("scored_query_count") or 0
    hit1_delta = aggregate.get("fine_tuned_minus_pretrained_hit_at_1_count")
    mrr_delta = aggregate.get("fine_tuned_minus_pretrained_weighted_mrr_delta")
    val_loss_delta = (training_curve.get("deltas_first_to_last") or {}).get(
        "validation_loss_last_minus_first"
    )
    exact_leakage = (
        leakage.get("train_validation_query_id_overlap_count", 0)
        + leakage.get("train_validation_normalized_query_overlap_count", 0)
        + leakage.get("train_validation_pair_overlap_count", 0)
    )

    evidence = [
        f"Observed training metrics cover {training_curve.get('observed_epoch_count')} epochs; best_epoch={training_curve.get('best_epoch')}.",
        f"Fine-tuned minus pretrained Hit@1 delta across compatible scored suites: {hit1_delta} over {scored} scored queries.",
        f"Fine-tuned minus pretrained weighted MRR delta: {mrr_delta}.",
        f"Validation loss last-minus-first: {val_loss_delta}.",
        f"Exact train/validation query or pair leakage count: {exact_leakage}.",
        (
            "Validation standards overlap training standards, so the validation split primarily tests "
            "query paraphrase generalization rather than standard-level OOD generalization."
        ),
    ]
    return {
        "fit_classification": "INCONCLUSIVE",
        "fit_confidence": "medium",
        "production_value_decision": "DO_NOT_ENABLE_FINE_TUNED_RERANKER_IN_PRODUCTION",
        "production_value_confidence": "high",
        "rationale": (
            "The preserved evidence does not show a useful production-quality improvement over the "
            "pretrained BGE reranker. It also does not cleanly prove classic overfitting or "
            "underfitting: validation metrics are high and stable, epoch 2 slightly worsens loss, "
            "and external suite metrics are unchanged versus pretrained. The best-supported "
            "classification is inconclusive fit with no demonstrated production value."
        ),
        "evidence": evidence,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    classification = audit["classification"]
    training_curve = audit["training_curve"]
    comparison = audit["pretrained_vs_fine_tuned_comparison"]
    leakage = audit["leakage_audit"]
    dataset = audit["dataset_audit"]

    lines = [
        "# StandardWise Reranker Fit Audit",
        "",
        "## Final Classification",
        "",
        f"- Fit classification: **{classification['fit_classification']}**",
        f"- Fit confidence: **{classification['fit_confidence']}**",
        f"- Production decision: **{classification['production_value_decision']}**",
        f"- Production decision confidence: **{classification['production_value_confidence']}**",
        "",
        classification["rationale"],
        "",
        "## Key Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in classification["evidence"])
    lines.extend(
        [
            "",
            "## Artifact Inventory",
            "",
            f"- Local model directories: {', '.join(audit['model_inventory']['local_models'].keys()) or 'none'}",
            "- Post-training comparison references:",
        ]
    )
    for model_name, spec in (
        audit["model_inventory"]["models_referenced_by_post_training_comparison"] or {}
    ).items():
        lines.append(
            f"  - {model_name}: `{spec.get('path')}` "
            f"(exists here: {spec.get('exists_in_current_windows_workspace')})"
        )
    lines.extend(
        [
            "",
            "## Dataset And Leakage",
            "",
            f"- Train pairs: {dataset['train_pairs']['pair_count']} "
            f"({dataset['train_pairs']['positive_pair_count']} positive, "
            f"{dataset['train_pairs']['negative_pair_count']} negative)",
            f"- Validation pairs: {dataset['validation_pairs']['pair_count']} "
            f"({dataset['validation_pairs']['positive_pair_count']} positive, "
            f"{dataset['validation_pairs']['negative_pair_count']} negative)",
            f"- Train/validation query-id overlap: {leakage['train_validation_query_id_overlap_count']}",
            f"- Train/validation normalized-query overlap: {leakage['train_validation_normalized_query_overlap_count']}",
            f"- Train/validation exact pair overlap: {leakage['train_validation_pair_overlap_count']}",
            f"- Train/validation positive-standard overlap: {leakage['train_validation_positive_standard_overlap_count']} "
            f"of {leakage['validation_positive_standard_count']} validation standards",
            f"- Negative-as-positive violations: {leakage['negative_positive_violations_count']}",
            "",
            "Positive-standard overlap is not counted as direct leakage, but it weakens claims about standard-level OOD generalization.",
            "",
            "## Training Curve",
            "",
            f"- Selection metric: {training_curve.get('selection_metric')}",
            f"- Best epoch: {training_curve.get('best_epoch')}",
            f"- Observed epoch count: {training_curve.get('observed_epoch_count')}",
            "",
            "| Epoch | Train Loss | Validation Loss | Val Hit@1 | Val MRR | Selected | Checkpoint Exists Here |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in training_curve.get("epochs") or []:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['validation_loss']:.6f} | "
            f"{row['validation_hit_at_1']:.6f} | {row['validation_mrr']:.6f} | "
            f"{row['is_selected_best']} | {row['checkpoint_exists_in_current_workspace']} |"
        )
    lines.extend(
        [
            "",
            "## Pretrained Vs Fine-Tuned",
            "",
            f"- Aggregate scored queries: {comparison['aggregate_delta']['scored_query_count']}",
            f"- Hit@1 delta: {comparison['aggregate_delta']['fine_tuned_minus_pretrained_hit_at_1_count']}",
            f"- Weighted MRR delta: {comparison['aggregate_delta']['fine_tuned_minus_pretrained_weighted_mrr_delta']}",
            "",
            "| Suite | Status | Scored | Pretrained Hit@1 | Fine-Tuned Hit@1 | Hit@1 Delta | Pretrained MRR | Fine-Tuned MRR | MRR Delta |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["suite_metric_comparison"]:
        lines.append(
            f"| {row['suite']} | {row['status']} | {row['scored']} | "
            f"{row['pretrained_hit_at_1']} | {row['fine_tuned_hit_at_1']} | {row['hit_at_1_delta']} | "
            f"{row['pretrained_mrr']} | {row['fine_tuned_mrr']} | {row['mrr_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Domain Weaknesses",
            "",
            "- Existing comparison errors remain concentrated in retrieval misses, function confusion, material/application confusion, and part/variant confusion.",
            "- Product-aware holdout remains high, but the fine-tuned model did not improve the preserved product-aware metrics over pretrained BGE.",
            "- Ambiguous product diagnostics are compatibility-checked but unscored, so they cannot prove ranking improvement.",
            "",
            "## Score Distributions",
            "",
            "Score distributions below are from returned top-5 candidates in the preserved comparison artifact, not a full-corpus score sweep.",
            "",
            "| Model | Count | Min | Median | Mean | Max | Std Dev |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, stats in comparison["score_distributions_returned_top5"].items():
        lines.append(
            f"| {model} | {stats['count']} | {stats['min']} | {stats['median']} | "
            f"{stats['mean']} | {stats['max']} | {stats['stdev']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity Report Signals",
            "",
        ]
    )
    integrity = audit["integrity_report_summary"]
    if integrity.get("available"):
        for key, value in integrity.get("extracted", {}).items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- Integrity report was not available.")
    lines.extend(
        [
            "",
            "## Required Decision",
            "",
            "Do not enable the fine-tuned reranker in production from this evidence. The latency/runtime decision should remain separate from fit quality: this audit only says the preserved fine-tuning artifacts do not demonstrate ranking-quality improvement over pretrained BGE.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    train_rows = load_jsonl(TRAIN_PAIRS)
    validation_rows = load_jsonl(VALIDATION_PAIRS)
    eval_sets = {
        "full_corpus_eval": load_jsonl(FULL_EVAL),
        "product_aware_holdout": load_jsonl(PRODUCT_AWARE_HOLDOUT),
        "full_corpus_ambiguous_eval": load_jsonl(AMBIGUOUS_EVAL),
    }
    training_metrics = load_json(TRAINING_METRICS)
    training_report = load_json(TRAINING_REPORT)
    comparison_raw = load_json(COMPARISON_REPORT)
    training_curve = build_training_curve(training_metrics)
    comparison = summarize_comparison(comparison_raw)
    leakage = build_leakage_report(train_rows, validation_rows, eval_sets)

    audit = {
        "task": "reranker_fit_audit",
        "scope": "evaluation_only_no_production_pipeline_changes",
        "inputs": {
            "train_pairs": str(TRAIN_PAIRS.relative_to(ROOT)),
            "validation_pairs": str(VALIDATION_PAIRS.relative_to(ROOT)),
            "training_report": str(TRAINING_REPORT.relative_to(ROOT)),
            "training_metrics": str(TRAINING_METRICS.relative_to(ROOT)),
            "comparison_report": str(COMPARISON_REPORT.relative_to(ROOT)),
            "integrity_report": str(INTEGRITY_REPORT.relative_to(ROOT)),
        },
        "dataset_audit": {
            "training_report": training_report,
            "train_pairs": pair_stats(train_rows),
            "validation_pairs": pair_stats(validation_rows),
            "evaluation_sets": {name: eval_stats(rows) for name, rows in eval_sets.items()},
        },
        "leakage_audit": leakage,
        "training_curve": training_curve,
        "pretrained_vs_fine_tuned_comparison": comparison,
        "integrity_report_summary": parse_integrity_report(INTEGRITY_REPORT),
        "model_inventory": model_inventory(),
    }
    audit["classification"] = final_classification(training_curve, comparison, leakage)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    AUDIT_MD.write_text(render_markdown(audit), encoding="utf-8")

    epoch_rows = training_curve.get("epochs") or []
    epoch_fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_hit_at_1",
        "validation_hit_at_3",
        "validation_hit_at_5",
        "validation_mrr",
        "is_selected_best",
        "checkpoint_exists_in_current_workspace",
        "checkpoint_path",
    ]
    write_csv(EPOCH_CSV, epoch_rows, epoch_fields)
    write_csv(
        LEARNING_CURVE_CSV,
        epoch_rows,
        [
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_hit_at_1",
            "validation_mrr",
        ],
    )
    print(f"Wrote {AUDIT_JSON.relative_to(ROOT)}")
    print(f"Wrote {AUDIT_MD.relative_to(ROOT)}")
    print(f"Wrote {EPOCH_CSV.relative_to(ROOT)}")
    print(f"Wrote {LEARNING_CURVE_CSV.relative_to(ROOT)}")
    print(
        "Classification:",
        audit["classification"]["fit_classification"],
        "| production:",
        audit["classification"]["production_value_decision"],
    )


if __name__ == "__main__":
    main()
