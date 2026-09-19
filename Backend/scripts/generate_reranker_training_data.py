from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_reranked_retrieval_50 import (  # noqa: E402
    AMBIGUOUS_DIAGNOSTIC_QUERIES,
    HOLDOUT_CASES,
    UNSEEN_HOLDOUT_CASES,
)
from scripts.evaluate_semantic_retrieval_50 import PRIMARY_BENCHMARK  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "data" / "canonical" / "parsed_standards_canonical.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "training"
RANDOM_SEED = 5108
TRAIN_RATIO = 0.85
VALID_ELIGIBILITY = {"SAFE_FOR_TRAINING", "REVIEW_REQUIRED", "SEARCH_ONLY"}
VALID_NEGATIVE_CATEGORIES = {
    "same_product_or_family",
    "close_variant",
    "cross_product_lexical_trap",
}
NON_PRODUCT_KINDS = {"test_method", "code_of_practice", "terminology", "installation"}
UNSAFE_FLAGS = {
    "missing_title",
    "missing_primary_subject",
    "metadata_review_required",
    "duplicate_source_records",
}


@dataclass(frozen=True)
class LeakageReport:
    exact: list[str]
    near: list[str]


def main(dataset_path: Path, output_dir: Path) -> None:
    records = _load_records(dataset_path)
    records_by_code = {record["standard_code"]: record for record in records}
    frozen_queries = _frozen_queries()

    eligibility = [_eligibility_row(record) for record in records]
    eligibility_by_code = {row["standard_code"]: row for row in eligibility}
    safe_records = [
        record
        for record in records
        if eligibility_by_code[record["standard_code"]]["eligibility"] == "SAFE_FOR_TRAINING"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    ambiguous_queries = _ambiguous_queries(records)
    query_records, rejected_leakage = _build_positive_queries(safe_records, frozen_queries)
    train_queries, validation_queries = _split_queries(query_records)
    train_pairs = _build_pairs(train_queries, records, records_by_code)
    validation_pairs = _build_pairs(validation_queries, records, records_by_code)
    validation_counts = _validation_counts(
        records_by_code,
        eligibility_by_code,
        query_records,
        train_pairs,
        validation_pairs,
        ambiguous_queries,
        frozen_queries,
    )
    report = _training_report(
        records,
        eligibility,
        query_records,
        train_queries,
        validation_queries,
        train_pairs,
        validation_pairs,
        ambiguous_queries,
        validation_counts,
        rejected_leakage,
    )

    _write_eligibility_csv(output_dir / "full_corpus_training_eligibility.csv", eligibility)
    _write_jsonl(output_dir / "full_corpus_queries.jsonl", query_records)
    _write_jsonl(output_dir / "full_corpus_train_pairs.jsonl", train_pairs)
    _write_jsonl(output_dir / "full_corpus_validation_pairs.jsonl", validation_pairs)
    _write_jsonl(output_dir / "full_corpus_ambiguous_queries.jsonl", ambiguous_queries)
    _write_review_csv(
        output_dir / "full_corpus_training_review.csv",
        query_records,
        [*train_pairs, *validation_pairs],
        records_by_code,
    )
    (output_dir / "full_corpus_training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Generated full-corpus reranker training data:")
    print(f"  Corpus records: {len(records)}")
    print(f"  SAFE_FOR_TRAINING: {report['safe_for_training']}")
    print(f"  REVIEW_REQUIRED: {report['review_required']}")
    print(f"  SEARCH_ONLY: {report['search_only']}")
    print(f"  Positive queries: {len(query_records)}")
    print(f"  Training queries: {len(train_queries)}")
    print(f"  Validation queries: {len(validation_queries)}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(validation_pairs)}")
    print(f"  Ambiguous queries: {len(ambiguous_queries)}")
    print(f"  Exact frozen leakage: {validation_counts['exact_frozen_leakage']}")
    print(f"  Near frozen leakage: {validation_counts['near_frozen_leakage']}")
    print(f"  Output directory: {output_dir}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("canonical dataset must contain a records array")
    if len(records) != 558:
        raise ValueError(f"expected 558 canonical records, found {len(records)}")
    return records


def _eligibility_row(record: dict[str, Any]) -> dict[str, Any]:
    flags = set(record.get("quality_flags") or [])
    review_required = bool(record.get("review_required"))
    if review_required:
        eligibility = "REVIEW_REQUIRED"
        reason = record.get("review_reason") or "metadata review required"
    elif flags & UNSAFE_FLAGS:
        eligibility = "REVIEW_REQUIRED"
        reason = f"unsafe quality flags: {', '.join(sorted(flags & UNSAFE_FLAGS))}"
    elif record.get("standard_kind") in NON_PRODUCT_KINDS:
        eligibility = "SEARCH_ONLY"
        reason = "non-product standard kept searchable but not used as procurement positive"
    elif not record.get("primary_subject"):
        eligibility = "REVIEW_REQUIRED"
        reason = "missing primary subject"
    elif not record.get("canonical_product") and not record.get("product_subtype"):
        eligibility = "SEARCH_ONLY"
        reason = "no source-grounded product or subtype for supervised procurement label"
    elif record.get("metadata_confidence") == "low" or record.get("product_confidence") == "low":
        eligibility = "REVIEW_REQUIRED"
        reason = "low confidence metadata"
    elif not _has_grounding(record):
        eligibility = "REVIEW_REQUIRED"
        reason = "missing source grounding evidence"
    else:
        eligibility = "SAFE_FOR_TRAINING"
        reason = "grounded product/subject metadata supports supervised positive"
    return {
        "standard_code": record["standard_code"],
        "eligibility": eligibility,
        "reason": reason,
        "standard_kind": record.get("standard_kind") or "",
        "canonical_product": record.get("canonical_product") or "",
        "primary_subject": record.get("primary_subject") or "",
        "product_subtype": record.get("product_subtype") or "",
        "function": record.get("function") or "",
        "material": record.get("material") or "",
        "application": record.get("application") or "",
        "metadata_confidence": record.get("metadata_confidence") or "",
        "product_confidence": record.get("product_confidence") or "",
        "review_required": review_required,
        "review_reason": record.get("review_reason") or "",
        "quality_flags": ";".join(sorted(flags)),
    }


def _has_grounding(record: dict[str, Any]) -> bool:
    evidence = record.get("metadata_evidence") or {}
    grounded_fields = {"canonical_product", "primary_subject", "product_subtype", "function"}
    return any(evidence.get(field) for field in grounded_fields)


def _build_positive_queries(
    records: list[dict[str, Any]],
    frozen_queries: list[str],
) -> tuple[list[dict[str, Any]], LeakageReport]:
    query_records = []
    seen: set[str] = set()
    rejected_exact = []
    rejected_near = []
    for record in sorted(records, key=lambda item: item["standard_code_norm"]):
        candidates = _query_candidates(record)
        accepted_index = 0
        for candidate in candidates:
            query = candidate["query"]
            key = _query_key(query)
            if not key or key in seen:
                continue
            leakage = _leakage_for_query(query, frozen_queries)
            if leakage.exact:
                rejected_exact.extend(leakage.exact)
                continue
            if leakage.near:
                rejected_near.extend(leakage.near)
                continue
            accepted_index += 1
            seen.add(key)
            query_records.append(
                {
                    "query_id": f"fcq_{_slug(record['standard_code'])}_{accepted_index:02d}",
                    "standard_code": record["standard_code"],
                    "query": query,
                    "product": candidate["product"],
                    "description": candidate["description"],
                    "style": candidate["style"],
                    "specificity": candidate["specificity"],
                    "source_fields": candidate["source_fields"],
                    "grounding_evidence": _grounding_evidence(record, candidate["source_fields"]),
                    "eligibility": "SAFE_FOR_TRAINING",
                    "review_status": "generated_from_metadata",
                }
            )
    return query_records, LeakageReport(sorted(set(rejected_exact)), sorted(set(rejected_near)))


def _query_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    product = _product_phrase(record)
    subject = _clean_phrase(record.get("primary_subject"))
    subtype = _clean_phrase(record.get("product_subtype"))
    function = _function_phrase(record.get("function"))
    material = _clean_phrase(record.get("material"))
    application = _clean_phrase(record.get("application"))
    title = _clean_phrase(record.get("canonical_title_expanded") or record.get("title"))
    aliases = [_clean_phrase(alias) for alias in record.get("product_aliases") or []]
    aliases = [alias for alias in aliases if alias and alias != product][:3]
    values: list[dict[str, Any]] = []

    def add(
        style: str,
        specificity: str,
        query: str,
        description: str,
        source_fields: list[str],
        *,
        product_value: str = product,
    ) -> None:
        query = _normalize_space(query)
        description = _normalize_space(description)
        if not query or _contains_standard_code(query):
            return
        values.append(
            {
                "query": query,
                "product": product_value,
                "description": description,
                "style": style,
                "specificity": specificity,
                "source_fields": source_fields,
            }
        )

    if subtype:
        add("product_only", "specific", f"{subtype} standard", subtype, ["product_subtype"])
    if subject:
        add("product_only", "specific", f"{subject} standard", subject, ["primary_subject"])
    if function:
        add(
            "product_function",
            "specific",
            f"{product} required to {function}",
            f"required to {function}",
            ["canonical_product", "function"],
        )
    if material:
        add(
            "product_material",
            "specific",
            f"{material} {product}",
            f"{material} construction or composition",
            ["canonical_product", "material"],
        )
    if application:
        add(
            "product_application",
            "specific",
            f"{product} for {application}",
            f"for {application}",
            ["canonical_product", "application"],
        )
    if subtype and function:
        add(
            "subtype_function",
            "specific",
            f"{subtype} to {function}",
            f"{subtype} required to {function}",
            ["product_subtype", "function"],
        )
    if material and application:
        add(
            "material_application",
            "specific",
            f"{material} {product} for {application}",
            f"{material} product for {application}",
            ["canonical_product", "material", "application"],
        )
    if subtype and application:
        add(
            "subtype_application",
            "specific",
            f"{subtype} for {application}",
            f"{subtype} used for {application}",
            ["product_subtype", "application"],
        )
    if product and (application or function or material):
        descriptor = "; ".join(part for part in [application, function, material] if part)
        add(
            "tender_sentence",
            "specific",
            f"Supply {product} meeting Indian Standard requirements for {descriptor}",
            descriptor,
            _present_fields(record, ["canonical_product", "application", "function", "material"]),
        )
    if title:
        add(
            "natural_language",
            "specific",
            f"Need the Indian Standard covering {title}",
            title,
            ["title"],
        )
    if application:
        add(
            "boq_short",
            "general_but_sufficient",
            f"{product}, {application}",
            application,
            ["canonical_product", "application"],
        )
    for alias in aliases:
        add(
            "alias",
            "general_but_sufficient",
            f"{alias} for {application or material or subject}",
            application or material or subject,
            [
                "product_aliases",
                "application" if application else "material" if material else "primary_subject",
            ],
            product_value=alias,
        )
    return values[:12]


def _product_phrase(record: dict[str, Any]) -> str:
    for field in ["canonical_product", "product_subtype", "primary_subject"]:
        value = _clean_phrase(record.get(field))
        if value:
            return value
    return "standard"


def _function_phrase(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("_", " ")


def _present_fields(record: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if record.get(field)]


def _grounding_evidence(record: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    evidence = record.get("metadata_evidence") or {}
    result = {}
    for field in fields:
        value = record.get(field)
        if field == "title":
            value = record.get("canonical_title_expanded") or record.get("title")
        if field == "product_aliases":
            value = record.get("product_aliases")
        result[field] = {
            "value": _evidence_value(value),
            "source": evidence.get(field) or record.get("title_evidence") or ["canonical_metadata"],
        }
    return result


def _evidence_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_space(value)[:220]
    if isinstance(value, list):
        return [_normalize_space(item)[:120] for item in value[:5]]
    return value


def _build_pairs(
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    records_by_code: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for query in queries:
        positive_code = query["standard_code"]
        positive_record = records_by_code[positive_code]
        pairs.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "positive_standard_code": positive_code,
                "positive_document": _document_text(positive_record),
                "negative_standard_code": "",
                "negative_document": "",
                "negative_category": "",
                "negative_mining_source": "",
                "document": _document_text(positive_record),
                "label": 1,
                "standard_code": positive_code,
                "pair_type": "positive",
            }
        )
        for negative in _hard_negatives(positive_record, records):
            negative_record = records_by_code[negative["standard_code"]]
            pairs.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "positive_standard_code": positive_code,
                    "positive_document": _document_text(positive_record),
                    "negative_standard_code": negative["standard_code"],
                    "negative_document": _document_text(negative_record),
                    "negative_category": negative["category"],
                    "negative_mining_source": negative["source"],
                    "document": _document_text(negative_record),
                    "label": 0,
                    "standard_code": negative["standard_code"],
                    "pair_type": "hard_negative",
                }
            )
    return pairs


def _hard_negatives(
    positive: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, str]]:
    scored = []
    for record in records:
        if record["standard_code"] == positive["standard_code"]:
            continue
        category = _negative_category(positive, record)
        score = _negative_score(positive, record, category)
        scored.append((score, record["standard_code"], category))
    scored.sort(key=lambda item: (-item[0], item[1]))
    desired = [
        ("same_product_or_family", 2),
        ("close_variant", 1),
        ("cross_product_lexical_trap", 1),
    ]
    selected: list[dict[str, str]] = []
    selected_codes: set[str] = set()
    for category, count in desired:
        for _, code, candidate_category in scored:
            if len([item for item in selected if item["category"] == category]) >= count:
                break
            if candidate_category == category and code not in selected_codes:
                selected.append(
                    {
                        "standard_code": code,
                        "category": category,
                        "source": "metadata_similarity",
                    }
                )
                selected_codes.add(code)
    for _, code, category in scored:
        if len(selected) >= limit:
            break
        if code not in selected_codes:
            selected.append(
                {
                    "standard_code": code,
                    "category": category,
                    "source": "metadata_similarity",
                }
            )
            selected_codes.add(code)
    return selected[:limit]


def _negative_category(positive: dict[str, Any], candidate: dict[str, Any]) -> str:
    if positive.get("canonical_product") and (
        positive.get("canonical_product") == candidate.get("canonical_product")
    ):
        return "same_product_or_family"
    if positive.get("family") and positive.get("family") == candidate.get("family"):
        return "same_product_or_family"
    variant_fields = ["product_subtype", "material", "application", "function", "primary_subject"]
    if any(
        _token_overlap(positive.get(field), candidate.get(field)) >= 0.35
        for field in variant_fields
    ):
        return "close_variant"
    return "cross_product_lexical_trap"


def _negative_score(positive: dict[str, Any], candidate: dict[str, Any], category: str) -> float:
    base = _token_overlap(_source_text(positive), _source_text(candidate))
    category_bonus = {
        "same_product_or_family": 0.30,
        "close_variant": 0.18,
        "cross_product_lexical_trap": 0.05,
    }[category]
    return base + category_bonus


def _source_text(record: dict[str, Any]) -> str:
    return " ".join(
        _normalize_space(record.get(field) or "")
        for field in [
            "canonical_product",
            "product_subtype",
            "primary_subject",
            "function",
            "material",
            "application",
            "family",
            "standard_kind",
            "title",
            "scope",
            "scope_reconstructed",
        ]
    )


def _document_text(record: dict[str, Any]) -> str:
    lines = [f"STANDARD: {record['standard_code']}"]
    for label, field in [
        ("PRODUCT", "canonical_product"),
        ("SUBTYPE", "product_subtype"),
        ("SUBJECT", "primary_subject"),
        ("FUNCTION", "function"),
        ("MATERIAL", "material"),
        ("APPLICATION", "application"),
        ("TYPE", "standard_kind"),
    ]:
        if record.get(field):
            lines.append(f"{label}: {record[field]}")
    title = record.get("canonical_title_expanded") or record.get("title")
    if title:
        lines.append(f"TITLE: {_normalize_space(title)}")
    scope = record.get("scope") or record.get("scope_reconstructed")
    if scope:
        lines.append(f"SCOPE: {_normalize_space(scope)}")
    return "\n".join(lines)


def _split_queries(
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(queries)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    validation_count = round(len(shuffled) * (1 - TRAIN_RATIO))
    validation_ids = {record["query_id"] for record in shuffled[:validation_count]}
    train = sorted(
        [record for record in queries if record["query_id"] not in validation_ids],
        key=lambda item: item["query_id"],
    )
    validation = sorted(
        [record for record in queries if record["query_id"] in validation_ids],
        key=lambda item: item["query_id"],
    )
    return train, validation


def _ambiguous_queries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("review_required"):
            continue
        key = (
            record.get("canonical_product")
            or record.get("product_subtype")
            or record.get("family")
        )
        if key:
            groups[str(key)].append(record)
    ambiguous = []
    for key, grouped in sorted(groups.items()):
        if len(grouped) < 2:
            continue
        plausible = sorted(record["standard_code"] for record in grouped[:8])
        if len(plausible) < 2:
            continue
        ambiguous.append(
            {
                "query": f"Need {key.replace('_', ' ')} standard",
                "reason": (
                    "metadata supports multiple plausible standards without "
                    "subtype/material/application discriminator"
                ),
                "plausible_standards": plausible,
                "missing_discriminator": "specific part, grade, material, function, or application",
            }
        )
    return ambiguous[:50]


def _validation_counts(
    records_by_code: dict[str, dict[str, Any]],
    eligibility_by_code: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
    ambiguous_queries: list[dict[str, Any]],
    frozen_queries: list[str],
) -> dict[str, int]:
    all_pairs = train_pairs + validation_pairs
    query_by_id = {query["query_id"]: query for query in queries}
    train_ids = {pair["query_id"] for pair in train_pairs}
    validation_ids = {pair["query_id"] for pair in validation_pairs}
    leakage = _leakage_for_queries([query["query"] for query in queries], frozen_queries)
    ambiguous_keys = {_query_key(record["query"]) for record in ambiguous_queries}
    unknown = sum(1 for pair in all_pairs if pair["standard_code"] not in records_by_code)
    positive_as_negative = sum(
        1
        for pair in all_pairs
        if pair["label"] == 0
        and query_by_id.get(pair["query_id"], {}).get("standard_code") == pair["standard_code"]
    )
    non_safe_positive = sum(
        1
        for query in queries
        if eligibility_by_code[query["standard_code"]]["eligibility"] != "SAFE_FOR_TRAINING"
    )
    ambiguous_supervised = sum(
        1 for query in queries if _query_key(query["query"]) in ambiguous_keys
    )
    return {
        "exact_frozen_leakage": len(leakage.exact),
        "near_frozen_leakage": len(leakage.near),
        "train_validation_overlap": len(train_ids & validation_ids),
        "unknown_standard_references": unknown,
        "positive_as_negative_violations": positive_as_negative,
        "non_safe_positive_violations": non_safe_positive,
        "ambiguous_supervised_violations": ambiguous_supervised,
    }


def _training_report(
    records: list[dict[str, Any]],
    eligibility: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    train_queries: list[dict[str, Any]],
    validation_queries: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
    ambiguous_queries: list[dict[str, Any]],
    validation_counts: dict[str, int],
    rejected_leakage: LeakageReport,
) -> dict[str, Any]:
    eligibility_counts = Counter(row["eligibility"] for row in eligibility)
    query_counts = Counter(query["standard_code"] for query in queries)
    all_pairs = train_pairs + validation_pairs
    negative_pairs = [pair for pair in all_pairs if pair["label"] == 0]
    return {
        "corpus_records": len(records),
        "safe_for_training": eligibility_counts["SAFE_FOR_TRAINING"],
        "review_required": eligibility_counts["REVIEW_REQUIRED"],
        "search_only": eligibility_counts["SEARCH_ONLY"],
        "positive_queries": len(queries),
        "train_queries": len(train_queries),
        "validation_queries": len(validation_queries),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "standards_with_queries": len(query_counts),
        "standards_without_queries": len(records) - len(query_counts),
        "query_count_distribution_per_standard": _distribution(query_counts),
        "negative_category_counts": dict(
            Counter(pair["negative_category"] for pair in negative_pairs)
        ),
        "negative_mining_source_counts": dict(
            Counter(pair["negative_mining_source"] for pair in negative_pairs)
        ),
        "ambiguous_queries": len(ambiguous_queries),
        **validation_counts,
        "eligibility_reason_counts": dict(Counter(row["reason"] for row in eligibility)),
        "rejected_exact_frozen_leakage_candidates": len(rejected_leakage.exact),
        "rejected_near_frozen_leakage_candidates": len(rejected_leakage.near),
    }


def _distribution(counts: Counter[str]) -> dict[str, Any]:
    values = list(counts.values())
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _frozen_queries() -> list[str]:
    queries = [case.query for case in PRIMARY_BENCHMARK]
    queries.extend(case.query for case in HOLDOUT_CASES + UNSEEN_HOLDOUT_CASES)
    queries.extend(case.query for case in AMBIGUOUS_DIAGNOSTIC_QUERIES)
    evaluation_dir = ROOT / "data" / "evaluation"
    for path in evaluation_dir.glob("*.jsonl"):
        for record in _read_jsonl(path):
            if record.get("query"):
                queries.append(str(record["query"]))
            if record.get("product") and record.get("description"):
                queries.append(f"{record['product']} {record['description']}")
    return sorted(set(query for query in queries if query.strip()))


def _leakage_for_queries(queries: list[str], frozen_queries: list[str]) -> LeakageReport:
    exact = []
    near = []
    for query in queries:
        leakage = _leakage_for_query(query, frozen_queries)
        exact.extend(leakage.exact)
        near.extend(leakage.near)
    return LeakageReport(sorted(set(exact)), sorted(set(near)))


def _leakage_for_query(query: str, frozen_queries: list[str]) -> LeakageReport:
    exact = []
    near = []
    key = _query_key(query)
    for frozen in frozen_queries:
        frozen_key = _query_key(frozen)
        if key == frozen_key:
            exact.append(f"{query} == {frozen}")
        elif _near_duplicate(query, frozen):
            near.append(f"{query} ~= {frozen}")
    return LeakageReport(exact, near)


def _near_duplicate(left: str, right: str) -> bool:
    left_key = _query_key(left)
    right_key = _query_key(right)
    if not left_key or not right_key:
        return False
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    token_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence_similarity = SequenceMatcher(None, left_key, right_key).ratio()
    return token_similarity >= 0.90 or sequence_similarity >= 0.94


def _write_eligibility_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "standard_code",
        "eligibility",
        "reason",
        "standard_kind",
        "canonical_product",
        "primary_subject",
        "product_subtype",
        "function",
        "material",
        "application",
        "metadata_confidence",
        "product_confidence",
        "review_required",
        "review_reason",
        "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_review_csv(
    path: Path,
    queries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    records_by_code: dict[str, dict[str, Any]],
) -> None:
    negatives_by_query: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        if pair["label"] == 0:
            negatives_by_query[pair["query_id"]].append(pair["standard_code"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "positive_standard",
                "positive_title",
                "product",
                "description",
                "style",
                "specificity",
                "source_fields",
                "hard_negative_codes",
                "review_status",
                "review_notes",
            ],
        )
        writer.writeheader()
        for query in queries:
            standard = records_by_code[query["standard_code"]]
            writer.writerow(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "positive_standard": query["standard_code"],
                    "positive_title": _normalize_space(
                        standard.get("canonical_title_expanded") or standard.get("title") or ""
                    ),
                    "product": query["product"],
                    "description": query["description"],
                    "style": query["style"],
                    "specificity": query["specificity"],
                    "source_fields": ";".join(query["source_fields"]),
                    "hard_negative_codes": ";".join(negatives_by_query[query["query_id"]]),
                    "review_status": query["review_status"],
                    "review_notes": "",
                }
            )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _contains_standard_code(query: str) -> bool:
    return bool(re.search(r"\bIS\s*\d{1,5}\b", query, flags=re.IGNORECASE))


def _token_overlap(left: Any, right: Any) -> float:
    left_tokens = set(_tokens(str(left or "")))
    right_tokens = set(_tokens(str(right or "")))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold().replace("_", " "))


def _query_key(query: str) -> str:
    return " ".join(_tokens(query))


def _normalize_space(text: Any) -> str:
    return " ".join(str(text).split())


def _clean_phrase(value: Any) -> str:
    text = _normalize_space(value or "")
    return text.replace("_", " ")


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", code.casefold()).strip("_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata-grounded full-corpus reranker training data."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.dataset, args.output_dir)
