from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings  # noqa: E402
from app.db.models import Standard  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.gemini_query_interpreter import SemanticQueryIntent  # noqa: E402
from app.services.product_aware_search_service import (  # noqa: E402
    _effective_value,
    _score_candidate_constraints,
    _text_state,
)

OUTPUT_DIR = Path("data/evaluation/results")

CHECK_CODES = [
    "IS 5312 (Part 1): 2004",
    "IS 5312 (Part 2): 1986",
    "IS 9338: 1984",
    "IS 778: 1984",
    "IS 13114: 1991",
]
PRESSURE_CODES = ["IS 9739: 1981"]
AUDIT_CODES = [
    "IS 5312 (Part 1): 2004",
    "IS 5312 (Part 2): 1986",
    "IS 9338: 1984",
    "IS 9739: 1981",
    "IS 14846: 2000",
]

GROUPS: list[dict[str, Any]] = [
    {
        "id": "reverse_flow",
        "label": "VALVE GROUP A - reverse flow",
        "expected_family": CHECK_CODES,
        "expected_intent": {
            "normalized_product": "valve",
            "subtype_family": "check_valve",
            "function": "prevent_reverse_flow",
            "medium": "water",
        },
        "queries": [
            ("A1", "valve", "valve that prevents water from flowing backwards"),
            ("A2", "valve", "non-return valve for water line"),
            ("A3", "valve", "check valve for water pipeline"),
            ("A4", "valve", "reflux valve for water line"),
            ("A5", "valve", "water should flow only one way and must not come back"),
        ],
    },
    {
        "id": "pressure_reducing",
        "label": "VALVE GROUP B - pressure reducing",
        "expected_family": PRESSURE_CODES,
        "expected_intent": {
            "normalized_product": "valve",
            "subtype": "pressure_reducing_valve",
            "function": "reduce_pressure",
            "medium": "water",
        },
        "queries": [
            ("B1", "valve", "valve to reduce downstream pressure in water supply"),
            ("B2", "valve", "pressure reducing valve for water pipeline"),
            ("B3", "valve", "water pressure regulator valve for downstream pressure control"),
        ],
    },
    {
        "id": "generic_ambiguity",
        "label": "VALVE GROUP C - generic ambiguity",
        "expected_family": [],
        "expected_behavior": "ambiguous_valve_function",
        "expected_intent": {
            "normalized_product": "valve",
            "medium": "water",
        },
        "queries": [
            ("C1", "valve", "valve for water pipeline"),
        ],
    },
]


async def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    async with _TraceContext(args) as trace_context:
        for group in GROUPS:
            for query_id, product, description in group["queries"]:
                rows.append(
                    await _run_case(
                        args,
                        trace_context,
                        group=group,
                        query_id=query_id,
                        product=product,
                        description=description,
                    )
                )

        metadata_audit = await trace_context.metadata_audit(AUDIT_CODES, rows)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "phase": args.phase,
        "base_url": args.base_url,
        "query_count": len(rows),
        "groups": GROUPS,
        "summary": _summary(rows),
        "group_metrics": _group_metrics(rows),
        "metadata_audit": metadata_audit,
        "metadata_changes": _metadata_changes(metadata_audit),
        "constraint_changes": [],
        "results": rows,
    }

    output_json = OUTPUT_DIR / f"valve_diagnostic_{args.phase}.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.phase == "after" and args.before_json:
        before = json.loads(Path(args.before_json).read_text(encoding="utf-8"))
        (OUTPUT_DIR / "valve_diagnostic_before_after.md").write_text(
            _before_after_markdown(before, payload),
            encoding="utf-8",
        )
    else:
        (OUTPUT_DIR / f"valve_diagnostic_{args.phase}.md").write_text(
            _markdown(payload),
            encoding="utf-8",
        )
    print(f"Wrote {output_json}")


class _TraceContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.settings = Settings()
        self.database = Database(self.settings)
        self.args = args

    async def __aenter__(self) -> _TraceContext:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        await self.database.close()

    async def trace(
        self,
        live_payload: dict[str, Any],
        product: str,
        description: str,
    ) -> dict[str, Any]:
        _ = (product, description)
        intent_payload = live_payload.get("query_interpretation") or {}
        intent = SemanticQueryIntent.model_validate(intent_payload) if intent_payload else None
        query = live_payload.get("query") or ""
        debug_trace = live_payload.get("debug_trace") or {}
        semantic = debug_trace.get("semantic_top20") or []
        bm25 = debug_trace.get("bm25_top20") or []
        rrf = debug_trace.get("rrf_pool") or []
        final = debug_trace.get("final_top10") or []
        codes = {
            item.get("standard_code")
            for item in [*semantic, *bm25, *rrf, *final]
            if item.get("standard_code")
        }
        async with self.database.session_factory() as session:
            standards_by_code = await _standards_by_code(session, codes)
        return {
            "retrieval_query_text": query,
            "semantic_top20": [
                _retrieval_candidate(candidate, standards_by_code, "semantic")
                for candidate in semantic
            ],
            "bm25_top20": [
                _retrieval_candidate(candidate, standards_by_code, "bm25") for candidate in bm25
            ],
            "rrf_pool": [_hybrid_candidate(candidate, standards_by_code) for candidate in rrf],
            "final_top10_trace": [
                _final_candidate(candidate, standards_by_code, intent) for candidate in final
            ],
            "retrieval_presence": _retrieval_presence(semantic, bm25, rrf, final),
        }

    async def metadata_audit(
        self,
        audit_codes: list[str],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        seen_codes = set(audit_codes)
        for row in rows:
            for candidate in row.get("trace", {}).get("final_top10_trace", []):
                seen_codes.add(candidate["standard_code"])
        async with self.database.session_factory() as session:
            standards = await _standards_by_code(session, seen_codes)
        return {
            code: _metadata_record(standard)
            for code, standard in sorted(standards.items())
            if code in seen_codes
        }


async def _run_case(
    args: argparse.Namespace,
    trace_context: _TraceContext,
    *,
    group: dict[str, Any],
    query_id: str,
    product: str,
    description: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now().isoformat()
    live_payload: dict[str, Any] | None = None
    error = None
    try:
        live_payload = _call_live_api(args.base_url, product, description, args.timeout_s)
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact records failures
        error = f"{exc.__class__.__name__}: {exc}"

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    row = {
        "ok": error is None,
        "query_id": query_id,
        "group_id": group["id"],
        "product": product,
        "raw_query": description,
        "started_at": started_at,
        "latency_ms": latency_ms,
        "expected_behavior": group.get("expected_behavior"),
        "expected_family_codes": group.get("expected_family", []),
        "expected_intent": group.get("expected_intent", {}),
        "error": error,
    }
    if live_payload is None:
        row["failure_category"] = ["LIVE_REQUEST_FAILURE"]
        row["passed"] = False
        return row

    row.update(
        {
            "gemini_raw_interpretation": live_payload.get("query_interpretation"),
            "raw_gemini_note": (
                "The live API exposes the post-normalization interpretation. "
                "The pre-normalization Gemini object is not returned by the API."
            ),
            "normalized_intent": live_payload.get("query_interpretation"),
            "grounded_evidence": (live_payload.get("query_interpretation") or {}).get(
                "attribute_evidence",
                [],
            ),
            "negative_constraints": _negative_constraints(
                live_payload.get("query_interpretation") or {}
            ),
            "retrieval_query_text": live_payload.get("query"),
            "query_interpreter": live_payload.get("query_interpreter"),
            "ambiguity": live_payload.get("ambiguity"),
            "missing_information": live_payload.get("missing_information") or [],
            "live_top5": [
                _response_candidate(candidate)
                for candidate in live_payload.get("candidates", [])
            ],
            "timings_ms": live_payload.get("timings_ms"),
        }
    )
    try:
        row["trace"] = await trace_context.trace(live_payload, product, description)
    except Exception as exc:  # noqa: BLE001 - trace should not hide live behavior
        row["trace_error"] = f"{exc.__class__.__name__}: {exc}"
        row["trace"] = {}

    row["failure_category"] = _failure_categories(group, row)
    row["passed"] = not row["failure_category"]
    return row


def _call_live_api(
    base_url: str,
    product: str,
    description: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps({"product": product, "description": description, "limit": 10}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/search/product-aware",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Query-Interpreter-Mode": "gemini",
            "X-Debug-Trace": "true",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _response_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": candidate.get("rank"),
        "standard_code": candidate.get("standard_code"),
        "title": candidate.get("title"),
        "canonical_product": candidate.get("canonical_product"),
        "family": candidate.get("family"),
        "function": candidate.get("function"),
        "semantic_rank": candidate.get("semantic_rank"),
        "semantic_score": candidate.get("semantic_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "rrf_rank": candidate.get("rrf_rank"),
        "rrf_score": candidate.get("rrf_score"),
        "constraint_flags": candidate.get("constraint_flags") or [],
        "constraint_score": candidate.get("constraint_score"),
        "final_score": candidate.get("final_score"),
    }


def _retrieval_candidate(
    candidate: dict[str, Any],
    standards_by_code: dict[str, Standard],
    source: str,
) -> dict[str, Any]:
    code = candidate.get("standard_code")
    standard = standards_by_code.get(code)
    return {
        "rank": candidate.get("rank"),
        "standard_code": code,
        "title": candidate.get("title"),
        "semantic_score": candidate.get("semantic_score") if source == "semantic" else None,
        "bm25_score": candidate.get("bm25_score") if source == "bm25" else None,
        "metadata": _metadata_record(standard),
    }


def _hybrid_candidate(
    candidate: dict[str, Any],
    standards_by_code: dict[str, Standard],
) -> dict[str, Any]:
    standard = standards_by_code.get(candidate.get("standard_code"))
    return {
        "standard_code": candidate.get("standard_code"),
        "title": candidate.get("title"),
        "semantic_rank": candidate.get("semantic_rank"),
        "semantic_score": candidate.get("semantic_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "rrf_rank": candidate.get("rank"),
        "rrf_score": candidate.get("rrf_score"),
        "metadata": _metadata_record(standard),
    }


def _final_candidate(
    candidate: dict[str, Any],
    standards_by_code: dict[str, Standard],
    intent: SemanticQueryIntent | None,
) -> dict[str, Any]:
    code = candidate.get("standard_code")
    standard = standards_by_code.get(code)
    constraint = (
        _score_candidate_constraints(
            standard,
            query_intent=intent,
            canonical_product=intent.normalized_product,
        )
        if standard is not None and intent is not None
        else None
    )
    flags = constraint.flags if constraint else (candidate.get("constraint_flags") or [])
    medium_state = _medium_state(intent, standard)
    return {
        "standard_code": code,
        "title": candidate.get("title"),
        "canonical_product": candidate.get("canonical_product"),
        "product_subtype": _effective_value(standard, "product_subtype"),
        "primary_subject": _effective_value(standard, "primary_subject"),
        "product_aliases": standard.product_aliases if standard else [],
        "material": _effective_value(standard, "material"),
        "application": _effective_value(standard, "application"),
        "function": _effective_value(standard, "function"),
        "applies_to_product_families": standard.applies_to_product_families if standard else [],
        "family": _effective_value(standard, "family"),
        "metadata_confidence": standard.metadata_confidence if standard else None,
        "metadata_evidence": standard.metadata_evidence if standard else {},
        "semantic_rank": candidate.get("semantic_rank"),
        "semantic_score": candidate.get("semantic_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "rrf_rank": candidate.get("rrf_rank"),
        "rrf_score": candidate.get("rrf_score"),
        "product_state": _flag_state(flags, "PRODUCT"),
        "subtype_state": _flag_state(flags, "SUBTYPE"),
        "subtype_family_state": _flag_state(flags, "SUBTYPE_FAMILY"),
        "function_state": _flag_state(flags, "FUNCTION"),
        "application_state": _flag_state(flags, "APPLICATION"),
        "medium_state": medium_state,
        "constraint_flags": flags,
        "constraint_score": candidate.get("constraint_score"),
        "final_score": candidate.get("final_score"),
        "final_rank": candidate.get("rank"),
    }


def _metadata_record(standard: Standard | None) -> dict[str, Any] | None:
    if standard is None:
        return None
    fields = [
        "canonical_product",
        "product_subtype",
        "primary_subject",
        "material",
        "application",
        "function",
        "family",
    ]
    effective = {field: _effective_value(standard, field) for field in fields}
    stored = {
        "canonical_product": standard.canonical_product,
        "product_subtype": standard.product_subtype,
        "primary_subject": standard.primary_subject,
        "product_aliases": standard.product_aliases,
        "material": standard.material,
        "application": standard.application,
        "function": standard.function,
        "applies_to_product_families": standard.applies_to_product_families,
        "family": standard.family,
        "metadata_confidence": standard.metadata_confidence,
        "metadata_evidence": standard.metadata_evidence,
        "retrieval_text": standard.retrieval_text,
    }
    return {
        "standard_code": standard.standard_id,
        "title": standard.title,
        **stored,
        "effective": effective,
        "runtime_corrections": {
            field: {"stored": stored.get(field), "effective": value}
            for field, value in effective.items()
            if stored.get(field) != value
        },
    }


def _metadata_changes(metadata_audit: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for code, record in metadata_audit.items():
        corrections = (record or {}).get("runtime_corrections") or {}
        if corrections:
            changes.append({"standard_code": code, "runtime_corrections": corrections})
    return changes


async def _standards_by_code(session: Any, codes: set[str]) -> dict[str, Standard]:
    if not codes:
        return {}
    result = await session.execute(select(Standard).where(Standard.standard_id.in_(sorted(codes))))
    return {standard.standard_id: standard for standard in result.scalars().all()}


def _retrieval_presence(
    semantic: list[Any],
    bm25: list[Any],
    rrf: list[Any],
    final: list[Any],
) -> dict[str, Any]:
    return {
        code: {
            "semantic_top20_rank": _rank_by_code(semantic, code),
            "bm25_top20_rank": _rank_by_code(bm25, code),
            "rrf_pool_rank": _rank_by_code(rrf, code),
            "final_top10_rank": _rank_by_code(final, code),
        }
        for code in CHECK_CODES
    }


def _rank_by_code(candidates: list[Any], code: str) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        candidate_code = candidate.get("standard_code") if isinstance(candidate, dict) else None
        if candidate_code == code:
            return candidate.get("rank") or candidate.get("rrf_rank") or index
    return None


def _failure_categories(group: dict[str, Any], row: dict[str, Any]) -> list[str]:
    failures = []
    diagnostics = row.get("query_interpreter") or {}
    if diagnostics.get("gemini_success") is False:
        failures.append("INTERPRETATION_FAILURE")
    intent = row.get("normalized_intent") or {}
    if not _intent_matches(group, intent):
        failures.append("NORMALIZATION_FAILURE")
    if group.get("id") == "generic_ambiguity":
        if not row.get("ambiguity") or not _has_valve_missing_info(row):
            failures.append("AMBIGUITY_FAILURE")
        if intent.get("subtype") or intent.get("subtype_family") or intent.get("function"):
            failures.append("NORMALIZATION_FAILURE")
        return sorted(set(failures))

    final_top = row.get("trace", {}).get("final_top10_trace", [])
    top1 = final_top[0]["standard_code"] if final_top else None
    if group["id"] == "pressure_reducing" and top1 != "IS 9739: 1981":
        failures.append("FUSION_FAILURE")
    if group["id"] == "reverse_flow":
        presence = row.get("trace", {}).get("retrieval_presence", {})
        is_5312_present = any(
            presence.get(code, {}).get("semantic_top20_rank")
            or presence.get(code, {}).get("bm25_top20_rank")
            or presence.get(code, {}).get("rrf_pool_rank")
            for code in ["IS 5312 (Part 1): 2004", "IS 5312 (Part 2): 1986"]
        )
        if not is_5312_present:
            failures.append("RETRIEVAL_FAILURE")
        if not _top_family(final_top, CHECK_CODES):
            failures.append("FUSION_FAILURE")
        if final_top and _top_contradicts_reverse_flow(final_top[0]):
            failures.append("CONSTRAINT_FAILURE")
    return sorted(set(failures))


def _intent_matches(group: dict[str, Any], intent: dict[str, Any]) -> bool:
    expected = group.get("expected_intent") or {}
    for field, value in expected.items():
        if field in {"normalized_product", "subtype", "subtype_family"}:
            if intent.get(field) != value:
                return False
            continue
        values = intent.get(field) or []
        if isinstance(values, str):
            values = [values]
        if value not in values:
            return False
    return True


def _top_family(final_top: list[dict[str, Any]], codes: list[str]) -> bool:
    return bool(final_top) and final_top[0]["standard_code"] in set(codes)


def _top_contradicts_reverse_flow(candidate: dict[str, Any]) -> bool:
    return candidate.get("function") in {"reduce_pressure", "release_air", "isolate_flow"}


def _has_valve_missing_info(row: dict[str, Any]) -> bool:
    missing = " ".join(row.get("missing_information") or []).casefold()
    return "valve" in missing and ("function" in missing or "type" in missing)


def _medium_state(intent: SemanticQueryIntent | None, standard: Standard | None) -> str | None:
    if intent is None or standard is None or not intent.medium:
        return None
    candidate_text = " ".join(
        value
        for value in [
            _effective_value(standard, "application"),
            _effective_value(standard, "function"),
            standard.title,
            standard.scope_text,
            standard.retrieval_text,
        ]
        if value
    )
    states = [_text_state(value, candidate_text) for value in intent.medium]
    return _best_state(states)


def _best_state(states: list[str]) -> str | None:
    for state in ["EXACT_MATCH", "STRONG_COMPATIBLE", "PARTIAL_MATCH", "UNKNOWN", "CONTRADICTION"]:
        if state in states:
            return state
    return None


def _flag_state(flags: list[str], prefix: str) -> str | None:
    token = f"{prefix}_"
    for flag in flags:
        if flag.startswith(token):
            return flag.removeprefix(token)
    return None


def _negative_constraints(intent: dict[str, Any]) -> dict[str, list[str]]:
    return {
        key: intent.get(key) or []
        for key in [
            "excluded_material",
            "excluded_application",
            "excluded_function",
            "excluded_subtype",
            "excluded_medium",
            "excluded_installation_context",
        ]
        if intent.get(key)
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("ok")]
    latencies = [row["latency_ms"] for row in ok]
    return {
        "completed_query_count": len(ok),
        "failed_query_count": len(rows) - len(ok),
        "gemini_failure_count": sum(
            (row.get("query_interpreter") or {}).get("gemini_success") is False for row in ok
        ),
        "passed_count": sum(not row.get("failure_category") for row in ok),
        "top1_consistency": _top_consistency(ok, rank=1),
        "top3_consistency": _top_consistency(ok, rank=3),
        "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 95), 3) if latencies else None,
    }


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    metrics = {}
    for group_id, items in grouped.items():
        top1 = [
            (item.get("trace", {}).get("final_top10_trace") or [{}])[0].get("standard_code")
            for item in items
            if item.get("trace", {}).get("final_top10_trace")
        ]
        top3 = [
            tuple(
                candidate["standard_code"]
                for candidate in item.get("trace", {}).get("final_top10_trace", [])[:3]
            )
            for item in items
            if item.get("trace", {}).get("final_top10_trace")
        ]
        intents = [_intent_key(item.get("normalized_intent") or {}) for item in items]
        metrics[group_id] = {
            "query_count": len(items),
            "passed_count": sum(not item.get("failure_category") for item in items),
            "top1_values": top1,
            "top1_consistency_count": _mode_count(top1),
            "top3_consistency_count": _mode_count(top3),
            "equivalent_normalized_intent_count": _mode_count(intents),
            "failure_categories": sorted(
                {
                    category
                    for item in items
                    for category in (item.get("failure_category") or [])
                }
            ),
        }
    return metrics


def _top_consistency(rows: list[dict[str, Any]], *, rank: int) -> dict[str, Any]:
    groups = _group_metrics(rows)
    values = [metrics[f"top{rank}_consistency_count"] for metrics in groups.values()]
    totals = [metrics["query_count"] for metrics in groups.values()]
    return {"consistent": sum(values), "total": sum(totals)}


def _intent_key(intent: dict[str, Any]) -> tuple[Any, ...]:
    return (
        intent.get("normalized_product"),
        intent.get("subtype"),
        intent.get("subtype_family"),
        tuple(intent.get("function") or []),
        tuple(intent.get("medium") or []),
        intent.get("ambiguity"),
    )


def _mode_count(values: list[Any]) -> int:
    if not values:
        return 0
    counts = Counter(values)
    return max(counts.values())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _before_after_markdown(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_by_id = {row["query_id"]: row for row in before["results"]}
    lines = [
        "# StandardWise Valve Diagnostic Before/After",
        "",
        "## Summary",
        f"- Before: {_summary_line(before['summary'])}",
        f"- After: {_summary_line(after['summary'])}",
        "",
        "## Per Query",
        "",
        (
            "| Query | Description | Before Top-5 | After Top-5 | Before failure | "
            "After failure | Latency after |"
        ),
        "|---|---|---|---|---|---|---:|",
    ]
    for row in after["results"]:
        before_row = before_by_id.get(row["query_id"], {})
        lines.append(
            "| "
            f"{row['query_id']} | {row['raw_query']} | "
            f"{_top_codes(before_row)} | {_top_codes(row)} | "
            f"{before_row.get('failure_category')} | {row.get('failure_category')} | "
            f"{row.get('latency_ms')} |"
        )
    lines.extend(["", "## Metadata Changes"])
    changes = after.get("metadata_changes") or []
    if not changes:
        lines.append("- None.")
    for change in changes:
        lines.append(f"- {change['standard_code']}: {change['runtime_corrections']}")
    lines.extend(["", "## Group Metrics"])
    for group_id, metrics in after.get("group_metrics", {}).items():
        lines.append(f"- {group_id}: {metrics}")
    return "\n".join(lines) + "\n"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# StandardWise Valve Diagnostic",
        "",
        f"- Phase: {payload['phase']}",
        f"- Summary: {_summary_line(payload['summary'])}",
    ]
    return "\n".join(lines) + "\n"


def _summary_line(summary: dict[str, Any]) -> str:
    return (
        f"completed={summary['completed_query_count']} "
        f"failed={summary['failed_query_count']} "
        f"gemini_failures={summary['gemini_failure_count']} "
        f"passed={summary['passed_count']} "
        f"latency mean/median/p95="
        f"{summary['mean_latency_ms']}/{summary['median_latency_ms']}/"
        f"{summary['p95_latency_ms']} ms"
    )


def _top_codes(row: dict[str, Any]) -> str:
    final_top = row.get("trace", {}).get("final_top10_trace") or row.get("live_top5") or []
    return ", ".join(candidate.get("standard_code", "") for candidate in final_top[:5])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live valve-only StandardWise diagnostic.")
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--before-json",
        default=str(OUTPUT_DIR / "valve_diagnostic_before.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
