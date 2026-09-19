from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.parsed_standards_corpus import (  # noqa: E402
    DEFAULT_CANONICAL_DATASET,
    DEFAULT_METADATA_DIR,
    DEFAULT_PARSED_SOURCE,
    DEFAULT_PRODUCT_REVIEW_CSV,
    build_canonical_corpus,
    load_parsed_records,
    write_canonical_dataset,
    write_metadata_enrichment_reports,
    write_product_review,
)


def main() -> None:
    args = _parse_args()
    old_records = []
    if args.output.exists():
        from app.services.parsed_standards_corpus import load_canonical_dataset  # noqa: PLC0415

        old_records = load_canonical_dataset(args.output)
    records = load_parsed_records(args.source)
    canonical_records, warnings = build_canonical_corpus(records)
    write_canonical_dataset(canonical_records, args.output)
    write_product_review(canonical_records, args.product_review)
    metadata_report = write_metadata_enrichment_reports(
        old_records or canonical_records,
        canonical_records,
        args.metadata_dir,
    )
    print("Canonical parsed standards corpus")
    print(f"Source records: {len(records)}")
    print(f"Canonical standards: {len(canonical_records)}")
    print(f"Primary subject coverage: {metadata_report['coverage']['primary_subject']}")
    print(f"Product coverage: {metadata_report['product_coverage']['after']}")
    print(f"Duplicate/conflict warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}")
    if len(warnings) > 20:
        print(f"... {len(warnings) - 20} more warnings")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.product_review}")
    print(f"Wrote metadata reports under: {args.metadata_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic canonical full corpus from parsed standards."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_PARSED_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_CANONICAL_DATASET)
    parser.add_argument("--product-review", type=Path, default=DEFAULT_PRODUCT_REVIEW_CSV)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()
