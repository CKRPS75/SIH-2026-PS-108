from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.parsed_standards_corpus import (  # noqa: E402
    DEFAULT_AUDIT_CSV,
    DEFAULT_AUDIT_JSON,
    DEFAULT_PARSED_SOURCE,
    audit_parsed_records,
    load_parsed_records,
    write_audit_reports,
)


def main() -> None:
    args = _parse_args()
    records = load_parsed_records(args.source)
    audit = audit_parsed_records(records)
    write_audit_reports(audit, args.json_output, args.csv_output)
    print("Parsed standards audit")
    print(f"Source: {args.source}")
    print(f"Total records: {audit.total_records}")
    print(f"Unique normalized standard codes: {audit.unique_normalized_standard_codes}")
    print(f"Duplicate codes: {len(audit.duplicate_codes)}")
    print(f"Known IS 2116:1980 duplicate records: {audit.known_duplicate_is_2116_records}")
    print(f"Missing is_code: {audit.missing_is_code}")
    print(f"Missing is_code_norm: {audit.missing_is_code_norm}")
    print(f"Missing title: {audit.missing_title}")
    print(f"Missing scope: {audit.missing_scope}")
    print(f"Missing full_text: {audit.missing_full_text}")
    print(f"Empty full_text: {audit.empty_full_text}")
    print(f"Suspiciously short records: {audit.suspiciously_short_records}")
    print(f"Suspiciously long records: {audit.suspiciously_long_records}")
    print(f"Possible merged-record artifacts: {audit.possible_merged_record_artifacts}")
    print(f"Duplicate title/code combinations: {len(audit.duplicate_title_code_combinations)}")
    print(f"Page-range anomalies: {audit.page_range_anomalies}")
    print(f"Wrote: {args.json_output}")
    print(f"Wrote: {args.csv_output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parsed_standards.json.")
    parser.add_argument("--source", type=Path, default=DEFAULT_PARSED_SOURCE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_AUDIT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    main()
