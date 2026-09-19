from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
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
DEFAULT_TRAINING_DIR = ROOT / "data" / "training"
VALID_ELIGIBILITY = {"SAFE_FOR_TRAINING", "REVIEW_REQUIRED", "SEARCH_ONLY"}
VALID_NEGATIVE_CATEGORIES = {
    "same_product_or_family",
    "close_variant",
    "cross_product_lexical_trap",
}
VALID_MINING_SOURCES = {"metadata_similarity"}


def main(dataset_path: Path, training_dir: Path) -> int:
    records = _load_records(dataset_path)
    corpus_by_code = {record["standard_code"]: record for record in records}
    eligibility = _read_eligibility(training_dir / "full_corpus_training_eligibility.csv")
    queries = _read_jsonl(training_dir / "full_corpus_queries.jsonl")
    train_pairs = _read_jsonl(training_dir / "full_corpus_train_pairs.jsonl")
    validation_pairs = _read_jsonl(training_dir / "full_corpus_validation_pairs.jsonl")
    ambiguous_queries = _read_jsonl(training_dir / "full_corpus_ambiguous_queries.jsonl")
    report = json.loads(
        (training_dir / "full_corpus_training_report.json").read_text(encoding="utf-8")
    )
    all_pairs = train_pairs + validation_pairs

    errors: dict[str, list[str]] = defaultdict(list)
    warnings: dict[str, list[str]] = defaultdict(list)
    _check_eligibility(records, eligibility, errors)
    _check_query_records(queries, corpus_by_code, eligibility, errors)
    _check_pairs(queries, all_pairs, train_pairs, validation_pairs, corpus_by_code, errors)
    _check_frozen_leakage(queries, errors)
    _check_ambiguous_queries(queries, ambiguous_queries, corpus_by_code, errors, warnings)
    _check_report(report, errors)

    query_counts = Counter(query["standard_code"] for query in queries)
    negative_pairs = [pair for pair in all_pairs if pair.get("label") == 0]
    train_query_ids = {pair["query_id"] for pair in train_pairs}
    validation_query_ids = {pair["query_id"] for pair in validation_pairs}

    print("Full-corpus reranker training data validation report")
    print(f"Corpus standards: {len(corpus_by_code)}")
    print(f"Eligibility rows: {len(eligibility)}")
    print(f"Positive query records: {len(queries)}")
    print(f"Training query count: {len(train_query_ids)}")
    print(f"Validation query count: {len(validation_query_ids)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    print(f"Ambiguous queries: {len(ambiguous_queries)}")
    print(f"Standards with queries: {len(query_counts)}")
    if query_counts:
        print(
            "Queries per standard min/median/max: "
            f"{min(query_counts.values())}/{median(query_counts.values()):.0f}/"
            f"{max(query_counts.values())}"
        )
    _print_count_section(
        "Eligibility counts",
        Counter(row["eligibility"] for row in eligibility.values()),
    )
    _print_count_section(
        "Negative category counts",
        Counter(pair.get("negative_category", "") for pair in negative_pairs),
    )
    _print_count_section(
        "Negative mining source counts",
        Counter(pair.get("negative_mining_source", "") for pair in negative_pairs),
    )

    print()
    print("Validation issue summary:")
    for name in sorted(errors):
        print(f"  {name}: {len(errors[name])}")
    for name, items in sorted(warnings.items()):
        print(f"  warning {name}: {len(items)}")

    if any(errors.values()):
        print()
        print("Critical validation errors:")
        for name, items in sorted(errors.items()):
            if not items:
                continue
            print(f"{name}:")
            for item in items[:25]:
                print(f"  - {item}")
            if len(items) > 25:
                print(f"  ... {len(items) - 25} more")
        return 1

    print()
    print("PASS: full-corpus training data is valid and has no critical leakage.")
    return 0


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("canonical dataset must contain records")
    return records


def _read_eligibility(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["standard_code"]: row for row in csv.DictReader(handle)}


def _check_eligibility(
    records: list[dict[str, Any]],
    eligibility: dict[str, dict[str, str]],
    errors: dict[str, list[str]],
) -> None:
    corpus_codes = {record["standard_code"] for record in records}
    missing = sorted(corpus_codes - set(eligibility))
    extra = sorted(set(eligibility) - corpus_codes)
    errors["missing_eligibility_rows"].extend(missing)
    errors["unknown_eligibility_rows"].extend(extra)
    for code, row in eligibility.items():
        if row.get("eligibility") not in VALID_ELIGIBILITY:
            errors["invalid_eligibility"].append(f"{code}: {row.get('eligibility')}")
        if not row.get("reason"):
            errors["missing_eligibility_reason"].append(code)


def _check_query_records(
    queries: list[dict[str, Any]],
    corpus_by_code: dict[str, dict[str, Any]],
    eligibility: dict[str, dict[str, str]],
    errors: dict[str, list[str]],
) -> None:
    query_ids = [query.get("query_id", "") for query in queries]
    errors["duplicate_query_ids"].extend(_duplicates(query_ids))
    query_keys = [_query_key(query.get("query", "")) for query in queries]
    errors["duplicate_queries"].extend(_duplicates(query_keys))
    query_standard_keys = [
        f"{_query_key(query.get('query', ''))}|{query.get('standard_code', '')}"
        for query in queries
    ]
    errors["duplicate_query_standard_pairs"].extend(_duplicates(query_standard_keys))
    for query in queries:
        query_id = str(query.get("query_id", "<missing>"))
        code = str(query.get("standard_code", ""))
        if code not in corpus_by_code:
            errors["unknown_standard_code"].append(f"{query_id}: {code}")
            continue
        if eligibility.get(code, {}).get("eligibility") != "SAFE_FOR_TRAINING":
            errors["non_safe_positive"].append(
                f"{query_id}: {code} is {eligibility.get(code, {}).get('eligibility')}"
            )
        if query.get("eligibility") != "SAFE_FOR_TRAINING":
            errors["query_not_marked_safe"].append(query_id)
        if not query.get("source_fields"):
            errors["missing_source_grounding"].append(query_id)
        if not query.get("grounding_evidence"):
            errors["missing_source_grounding"].append(query_id)
        if _contains_standard_code(str(query.get("query", ""))):
            errors["query_contains_standard_code"].append(query_id)
        if not str(query.get("product", "")).strip():
            errors["missing_query_product"].append(query_id)
        if not str(query.get("description", "")).strip():
            errors["missing_query_description"].append(query_id)


def _check_pairs(
    queries: list[dict[str, Any]],
    all_pairs: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
    corpus_by_code: dict[str, dict[str, Any]],
    errors: dict[str, list[str]],
) -> None:
    query_by_id = {query["query_id"]: query for query in queries}
    train_ids = {pair.get("query_id") for pair in train_pairs}
    validation_ids = {pair.get("query_id") for pair in validation_pairs}
    errors["train_validation_overlap"].extend(
        sorted(str(item) for item in train_ids & validation_ids)
    )
    pair_keys = [
        (
            pair.get("query_id"),
            pair.get("standard_code"),
            pair.get("label"),
            pair.get("pair_type"),
        )
        for pair in all_pairs
    ]
    errors["duplicate_pair_corruption"].extend(str(item) for item in _duplicates(pair_keys))
    pairs_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in all_pairs:
        query_id = str(pair.get("query_id", ""))
        pairs_by_query[query_id].append(pair)
        code = str(pair.get("standard_code", ""))
        if code not in corpus_by_code:
            errors["unknown_standard_code"].append(f"{query_id}: {code}")
        if pair.get("query") != query_by_id.get(query_id, {}).get("query"):
            errors["pair_query_mismatch"].append(query_id)
        if pair.get("label") == 1:
            if pair.get("pair_type") != "positive":
                errors["invalid_pair_type"].append(query_id)
            if code != query_by_id.get(query_id, {}).get("standard_code"):
                errors["positive_pair_wrong_standard"].append(query_id)
        elif pair.get("label") == 0:
            if pair.get("pair_type") != "hard_negative":
                errors["invalid_pair_type"].append(query_id)
            positive_code = query_by_id.get(query_id, {}).get("standard_code")
            if code == positive_code:
                errors["positive_as_negative"].append(query_id)
            if pair.get("negative_category") not in VALID_NEGATIVE_CATEGORIES:
                errors["invalid_negative_category"].append(query_id)
            if pair.get("negative_mining_source") not in VALID_MINING_SOURCES:
                errors["invalid_negative_mining_source"].append(query_id)
            if not pair.get("positive_document") or not pair.get("negative_document"):
                errors["missing_pair_documents"].append(query_id)
        else:
            errors["invalid_pair_label"].append(f"{query_id}: {pair.get('label')}")
    for query_id, pairs in pairs_by_query.items():
        positives = [pair for pair in pairs if pair.get("label") == 1]
        negatives = [pair for pair in pairs if pair.get("label") == 0]
        if len(positives) != 1:
            errors["invalid_positive_count"].append(query_id)
        if len(negatives) < 1:
            errors["missing_negatives"].append(query_id)


def _check_frozen_leakage(
    queries: list[dict[str, Any]],
    errors: dict[str, list[str]],
) -> None:
    frozen_queries = _frozen_queries()
    for query in queries:
        for frozen in frozen_queries:
            if _query_key(query["query"]) == _query_key(frozen):
                errors["exact_frozen_leakage"].append(f"{query['query']} == {frozen}")
            elif _near_duplicate(query["query"], frozen):
                errors["near_frozen_leakage"].append(f"{query['query']} ~= {frozen}")


def _check_ambiguous_queries(
    queries: list[dict[str, Any]],
    ambiguous_queries: list[dict[str, Any]],
    corpus_by_code: dict[str, dict[str, Any]],
    errors: dict[str, list[str]],
    warnings: dict[str, list[str]],
) -> None:
    query_keys = {_query_key(query["query"]) for query in queries}
    for index, ambiguous in enumerate(ambiguous_queries, start=1):
        key = _query_key(ambiguous.get("query", ""))
        if key in query_keys:
            errors["ambiguous_query_used_as_positive"].append(ambiguous.get("query", str(index)))
        plausible = ambiguous.get("plausible_standards") or []
        if len(plausible) < 2:
            warnings["ambiguous_queries_with_fewer_than_two_plausible_standards"].append(
                ambiguous.get("query", str(index))
            )
        if not ambiguous.get("missing_discriminator"):
            errors["ambiguous_missing_discriminator"].append(ambiguous.get("query", str(index)))
        for code in plausible:
            if code not in corpus_by_code:
                errors["unknown_standard_code"].append(f"ambiguous:{index}: {code}")


def _check_report(report: dict[str, Any], errors: dict[str, list[str]]) -> None:
    for field in [
        "corpus_records",
        "safe_for_training",
        "review_required",
        "search_only",
        "positive_queries",
        "train_queries",
        "validation_queries",
        "train_pairs",
        "validation_pairs",
        "exact_frozen_leakage",
        "near_frozen_leakage",
    ]:
        if field not in report:
            errors["missing_report_fields"].append(field)
    if report.get("exact_frozen_leakage") != 0:
        errors["exact_frozen_leakage"].append("report exact_frozen_leakage is non-zero")
    if report.get("near_frozen_leakage") != 0:
        errors["near_frozen_leakage"].append("report near_frozen_leakage is non-zero")


def _frozen_queries() -> list[str]:
    queries = [case.query for case in PRIMARY_BENCHMARK]
    queries.extend(case.query for case in HOLDOUT_CASES + UNSEEN_HOLDOUT_CASES)
    queries.extend(case.query for case in AMBIGUOUS_DIAGNOSTIC_QUERIES)
    for path in (ROOT / "data" / "evaluation").glob("*.jsonl"):
        for record in _read_jsonl(path):
            if record.get("query"):
                queries.append(str(record["query"]))
            if record.get("product") and record.get("description"):
                queries.append(f"{record['product']} {record['description']}")
    return sorted(set(query for query in queries if query.strip()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def _duplicates(values: list[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _near_duplicate(left: str, right: str) -> bool:
    left_key = _query_key(left)
    right_key = _query_key(right)
    if not left_key or not right_key or left_key == right_key:
        return False
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    token_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence_similarity = SequenceMatcher(None, left_key, right_key).ratio()
    return token_similarity >= 0.90 or sequence_similarity >= 0.94


def _query_key(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", query.casefold()))


def _contains_standard_code(query: str) -> bool:
    return bool(re.search(r"\bIS\s*\d{1,5}\b", query, flags=re.IGNORECASE))


def _print_count_section(title: str, counts: Counter[str]) -> None:
    print()
    print(f"{title}:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate full-corpus reranker training data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    args = parser.parse_args()
    raise SystemExit(main(args.dataset, args.training_dir))
