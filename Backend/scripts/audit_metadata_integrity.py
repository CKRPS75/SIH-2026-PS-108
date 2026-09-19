from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.parsed_standards_corpus import (
    DEFAULT_CANONICAL_DATASET,
    FAMILY_BY_PRODUCT,
    PRODUCT_ALIASES,
    canonicalize_product,
)

JSON_OUTPUT = Path("data/evaluation/results/metadata_integrity_audit.json")
MD_OUTPUT = Path("data/evaluation/results/metadata_integrity_audit.md")


def main() -> None:
    records = json.loads(DEFAULT_CANONICAL_DATASET.read_text(encoding="utf-8"))["records"]
    findings = [_audit_record(record) for record in records]
    flagged = [finding for finding in findings if finding["issues"]]
    report = {
        "source": str(DEFAULT_CANONICAL_DATASET),
        "total_records": len(records),
        "flagged_records": len(flagged),
        "repaired_records": 0,
        "issue_counts": dict(Counter(issue for finding in flagged for issue in finding["issues"])),
        "findings": flagged,
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_OUTPUT.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


def _audit_record(record: dict[str, Any]) -> dict[str, Any]:
    product = record.get("canonical_product")
    title_text = " ".join(
        str(record.get(field) or "")
        for field in [
            "title",
            "canonical_title_expanded",
            "primary_subject",
            "product_subtype",
            "retrieval_text",
        ]
    )
    title_product, confidence = canonicalize_product(title_text)
    issues = []
    evidence = {
        "title_product": title_product,
        "title_product_confidence": confidence,
        "canonical_product": product,
        "family": record.get("family"),
        "applies_to_product_families": record.get("applies_to_product_families") or [],
    }
    if product and title_product and product != title_product:
        product_family = FAMILY_BY_PRODUCT.get(product)
        title_family = FAMILY_BY_PRODUCT.get(title_product)
        if product_family != title_family:
            issues.append("PRODUCT_TITLE_CONFLICT")
        else:
            issues.append("PRODUCT_TITLE_SAME_FAMILY_MISMATCH")
    family = record.get("family")
    if (
        product
        and family
        and FAMILY_BY_PRODUCT.get(product)
        and FAMILY_BY_PRODUCT[product] != family
    ):
        issues.append("PRODUCT_FAMILY_CONFLICT")
    if product and not _has_product_evidence(product, title_text):
        issues.append("WEAK_PRODUCT_TEXT_EVIDENCE")
    return {
        "standard_code": record.get("standard_code"),
        "title": record.get("title"),
        "issues": issues,
        "evidence": evidence,
        "recommended_action": "review" if issues else "none",
    }


def _has_product_evidence(product: str, text: str) -> bool:
    normalized = f" {' '.join(text.casefold().replace('_', ' ').split())} "
    return any(
        f" {' '.join(alias.casefold().split())} " in normalized
        for alias in PRODUCT_ALIASES.get(product, set())
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Metadata Integrity Audit",
        "",
        f"Source: `{report['source']}`",
        f"Total records: {report['total_records']}",
        f"Flagged records: {report['flagged_records']}",
        f"Automatically repaired records: {report['repaired_records']}",
        "",
        "## Issue Counts",
        "",
    ]
    for issue, count in sorted(report["issue_counts"].items()):
        lines.append(f"- `{issue}`: {count}")
    lines.extend(["", "## Flagged Records", ""])
    for finding in report["findings"][:100]:
        lines.append(
            f"- `{finding['standard_code']}`: {', '.join(finding['issues'])} -- "
            f"{finding.get('title') or ''}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
