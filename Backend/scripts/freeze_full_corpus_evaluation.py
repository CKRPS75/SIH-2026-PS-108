from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.parsed_standards_corpus import (  # noqa: E402
    DEFAULT_CANONICAL_DATASET,
    CanonicalStandard,
    load_canonical_dataset,
)

DEFAULT_EVAL = Path("data/evaluation/full_corpus_eval.jsonl")
DEFAULT_AMBIGUOUS = Path("data/evaluation/full_corpus_ambiguous_eval.jsonl")
DEFAULT_PRODUCT_HOLDOUT = Path("data/evaluation/product_aware_holdout.jsonl")


def main() -> None:
    args = _parse_args()
    records = load_canonical_dataset(args.canonical)
    evaluation = build_full_corpus_eval(records, target=args.target)
    ambiguous = build_ambiguous_eval()
    product_holdout = build_product_aware_holdout(records, target=args.product_target)
    _write_jsonl(args.eval_output, evaluation)
    _write_jsonl(args.ambiguous_output, ambiguous)
    _write_jsonl(args.product_holdout_output, product_holdout)
    print("Frozen full-corpus evaluation sets")
    print(f"Full-corpus eval cases: {len(evaluation)} -> {args.eval_output}")
    print(f"Ambiguous eval cases: {len(ambiguous)} -> {args.ambiguous_output}")
    print(f"Product-aware holdout cases: {len(product_holdout)} -> {args.product_holdout_output}")


def build_full_corpus_eval(
    records: list[CanonicalStandard],
    *,
    target: int = 200,
) -> list[dict[str, object]]:
    usable = [
        record
        for record in records
        if record.title
        and record.scope
        and record.standard_code
        and "missing_scope" not in record.quality_flags
    ]
    buckets: dict[tuple[str, str, str], list[CanonicalStandard]] = {}
    for record in usable:
        key = (
            record.family or "unknown_family",
            record.canonical_product or "unknown_product",
            record.standard_kind,
        )
        buckets.setdefault(key, []).append(record)
    selected: list[CanonicalStandard] = []
    while len(selected) < target:
        added = False
        for key in sorted(buckets):
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
    cases = []
    for index, record in enumerate(selected, start=1):
        product = record.canonical_product or _title_phrase(record)
        cases.append(
            {
                "query_id": f"fc_eval_{index:04d}",
                "query": f"Need {product} standard for {_scope_phrase(record.scope or '')}",
                "expected_standard": record.standard_code,
                "expected_product": record.canonical_product,
                "family": record.family,
                "standard_kind": record.standard_kind,
                "source_fields": ["title", "scope"],
            }
        )
    return cases


def build_product_aware_holdout(
    records: list[CanonicalStandard],
    *,
    target: int = 75,
) -> list[dict[str, object]]:
    usable = [
        record
        for record in records
        if record.canonical_product
        and record.product_confidence in {"high", "medium"}
        and record.scope
        and record.title
    ]
    selected = usable[:target]
    cases = [
        {
            "query_id": f"pa_holdout_{index:04d}",
            "product": record.canonical_product,
            "description": _product_description(record),
            "expected_standard": record.standard_code,
            "expected_product": record.canonical_product,
            "family": record.family,
            "standard_kind": record.standard_kind,
            "source_fields": ["title", "scope"],
        }
        for index, record in enumerate(selected, start=1)
    ]
    cases.extend(
        [
            {
                "query_id": "pa_holdout_valve_steel_main",
                "product": "valve",
                "description": "installed in a steel water distribution main",
                "expected_standard": "",
                "expected_product": "valve",
                "family": "water_valve",
                "standard_kind": "product_standard",
                "source_fields": ["diagnostic"],
            },
            {
                "query_id": "pa_holdout_valve_reverse_flow",
                "product": "valve",
                "description": "installed in a steel water line to prevent reverse flow",
                "expected_standard": "",
                "expected_product": "valve",
                "family": "water_valve",
                "standard_kind": "product_standard",
                "source_fields": ["diagnostic"],
            },
        ]
    )
    return cases


def build_ambiguous_eval() -> list[dict[str, object]]:
    return [
        {
            "query_id": "fc_ambiguous_valve_water_supply",
            "product": "valve",
            "description": "for water supply",
            "reason": "valve function is underspecified",
        },
        {
            "query_id": "fc_ambiguous_cement_repair",
            "product": "cement",
            "description": "for repair",
            "reason": "repair condition does not identify cement subtype",
        },
        {
            "query_id": "fc_ambiguous_pipe_water_supply",
            "product": "pipe",
            "description": "for water supply",
            "reason": "pipe material and service pressure are underspecified",
        },
    ]


def _product_description(record: CanonicalStandard) -> str:
    scope = _scope_phrase(record.scope or "")
    title = " ".join((record.title or "").lower().split())
    return f"{scope} ({title})" if title else scope


def _scope_phrase(scope: str) -> str:
    text = " ".join(scope.split())
    text = text.rstrip(".")
    return text[:220].rstrip()


def _title_phrase(record: CanonicalStandard) -> str:
    return " ".join((record.title or record.standard_code).lower().split()[:5])


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze full-corpus evaluation sets.")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DATASET)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--ambiguous-output", type=Path, default=DEFAULT_AMBIGUOUS)
    parser.add_argument("--product-holdout-output", type=Path, default=DEFAULT_PRODUCT_HOLDOUT)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--product-target", type=int, default=75)
    return parser.parse_args()


if __name__ == "__main__":
    main()
