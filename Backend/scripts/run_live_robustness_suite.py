from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("data/evaluation/results")

GROUPS: list[dict[str, Any]] = [
    {
        "id": "grp_potable_water",
        "product": "pipe",
        "expected_codes": ["IS 12709: 1994"],
        "expected_intent": {"material": "GRP", "application": "potable water supply"},
        "queries": [
            "GRP pipe for potable drinking water",
            "GFRP pipe for drinking water supply",
            "glass fibre reinforced plastic pipe for potable water",
            "glass reinforced plastic pipe carrying drinking water",
        ],
    },
    {
        "id": "grp_industrial_waste",
        "product": "pipe",
        "expected_codes": ["IS 14402: 1996"],
        "expected_intent": {"material": "GRP", "application": "sewerage"},
        "queries": [
            "GRP pipe for industrial waste and non potable water",
            "GFRP pipe for industrial effluent",
            "glass fibre reinforced pipe for non-potable industrial waste",
        ],
    },
    {
        "id": "ppc_fly_ash",
        "product": "cement",
        "expected_codes": ["IS 1489 (Part 1): 1991"],
        "expected_intent": {"cement_type": "ppc", "pozzolana_source": "fly_ash"},
        "queries": [
            "Portland pozzolana cement made using fly ash",
            "PPC with fly ash",
            "fly ash based Portland pozzolana cement",
            "cement where the pozzolana source is fly ash",
        ],
    },
    {
        "id": "ppc_calcined_clay",
        "product": "cement",
        "expected_codes": ["IS 1489 (Part 2): 1991"],
        "expected_intent": {"cement_type": "ppc", "pozzolana_source": "calcined_clay"},
        "queries": [
            "Portland pozzolana cement made using calcined clay",
            "PPC with calcined clay",
            "calcined clay based PPC",
        ],
    },
    {
        "id": "reverse_flow_valve",
        "product": "valve",
        "expected_codes": ["IS 5312 (Part 1): 2004", "IS 5312 (Part 2): 1986", "IS 9338: 1984"],
        "expected_intent": {"subtype_family": "check_valve", "function": "prevent_reverse_flow"},
        "queries": [
            "valve that prevents water from flowing backwards",
            "non-return valve for water line",
            "check valve for water pipeline",
            "reflux valve for water line",
            "water should only flow one way through the valve",
        ],
    },
    {
        "id": "pressure_reducing_valve",
        "product": "valve",
        "expected_codes": ["IS 9739: 1981"],
        "expected_intent": {"subtype": "pressure_reducing_valve", "function": "reduce_pressure"},
        "queries": [
            "pressure reducing valve for water supply",
            "valve to reduce downstream water pressure",
            "water pressure regulator valve",
        ],
    },
    {
        "id": "hdpe_sewage",
        "product": "pipe",
        "expected_codes": ["IS 14333: 1996"],
        "expected_intent": {"material": "HDPE", "application": "sewerage"},
        "queries": [
            "HDPE pipe for municipal sewage system",
            "high density polyethylene sewage pipe",
            "municipal sewer pipe made from HDPE",
        ],
    },
    {
        "id": "opc_ambiguity",
        "product": "cement",
        "expected_codes": ["IS 269: 1989", "IS 8112: 1989", "IS 12269: 1987"],
        "expected_behavior": "ambiguity_true_grade_missing",
        "expected_intent": {"cement_type": "opc"},
        "queries": ["ordinary Portland cement"],
    },
    {
        "id": "interior_flooring_negation",
        "product": "tile",
        "expected_codes": ["IS 1478: 1992"],
        "expected_intent": {"application": "flooring", "excluded_application": "roofing"},
        "queries": [
            "tile for interior flooring, not roofing",
            "floor tile for indoor use and not roof application",
        ],
    },
]


def main() -> None:
    args = _parse_args()
    results = []
    for group in GROUPS:
        for query in group["queries"]:
            results.append(_run_query(args.base_url, group, query, timeout_s=args.timeout_s))
    payload = {
        "generated_at": datetime.now().isoformat(),
        "phase": args.phase,
        "base_url": args.base_url,
        "query_count": len(results),
        "groups": GROUPS,
        "summary": _summary(results),
        "group_metrics": _group_metrics(results),
        "results": results,
    }
    output_json = Path(args.output_json or OUTPUT_DIR / f"live_robustness_{args.phase}.json")
    output_md = Path(
        args.output_md
        or (
            OUTPUT_DIR / "live_robustness_before_after.md"
            if args.phase == "after"
            else OUTPUT_DIR / f"live_robustness_{args.phase}.md"
        )
    )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.phase == "after" and args.before_json:
        before = json.loads(Path(args.before_json).read_text(encoding="utf-8"))
        output_md.write_text(_before_after_markdown(before, payload), encoding="utf-8")
    else:
        output_md.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


def _run_query(
    base_url: str,
    group: dict[str, Any],
    description: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps({"product": group["product"], "description": description, "limit": 5}).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/search/product-aware",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Query-Interpreter-Mode": "gemini",
        },
        method="POST",
    )
    started = time.perf_counter()
    started_at = datetime.now().isoformat()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - started) * 1000
        candidates = payload.get("candidates") or []
        codes = [candidate.get("standard_code") for candidate in candidates]
        expected_codes = group["expected_codes"]
        expected_rank = _rank_of_any(codes, expected_codes)
        return {
            "ok": True,
            "group_id": group["id"],
            "product": group["product"],
            "raw_query": description,
            "started_at": started_at,
            "latency_ms": round(elapsed_ms, 3),
            "expected_standard_codes": expected_codes,
            "expected_behavior": group.get("expected_behavior"),
            "expected_intent": group.get("expected_intent", {}),
            "top1": codes[0] if codes else None,
            "top3": codes[:3],
            "expected_rank": expected_rank,
            "gemini_interpretation": payload.get("query_interpretation"),
            "normalized_intent": payload.get("query_interpretation"),
            "grounded_evidence": (payload.get("query_interpretation") or {}).get(
                "attribute_evidence", []
            ),
            "negative_constraints": _negative_constraints(
                payload.get("query_interpretation") or {}
            ),
            "ambiguity": payload.get("ambiguity"),
            "missing_information": payload.get("missing_information") or [],
            "query_interpreter": payload.get("query_interpreter"),
            "timings_ms": payload.get("timings_ms"),
            "candidates": [_candidate_trace(candidate) for candidate in candidates],
            "failure_category": _failure_category(group, payload, expected_rank),
            "passed": _passed(group, payload, expected_rank),
        }
    except TimeoutError as exc:
        return _error(group, description, started_at, started, f"timeout: {exc}")
    except urllib.error.URLError as exc:
        return _error(group, description, started_at, started, f"url_error: {exc.reason}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script records failures
        return _error(group, description, started_at, started, f"{exc.__class__.__name__}: {exc}")


def _candidate_trace(candidate: dict[str, Any]) -> dict[str, Any]:
    flags = candidate.get("constraint_flags") or []
    return {
        "standard_code": candidate.get("standard_code"),
        "title": candidate.get("title"),
        "semantic_rank": candidate.get("semantic_rank"),
        "semantic_score": candidate.get("semantic_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "rrf_rank": candidate.get("rrf_rank"),
        "rrf_score": candidate.get("rrf_score"),
        "product_state": _flag_state(flags, "PRODUCT"),
        "material_state": _flag_state(flags, "MATERIAL"),
        "subtype_state": _flag_state(flags, "SUBTYPE"),
        "function_state": _flag_state(flags, "FUNCTION"),
        "application_state": _flag_state(flags, "APPLICATION"),
        "grade_state": _flag_state(flags, "GRADE"),
        "temperature_state": _flag_state(flags, "TEMPERATURE"),
        "negative_constraint_states": [flag for flag in flags if flag.startswith("EXCLUDED_")],
        "constraint_flags": flags,
        "constraint_score": candidate.get("constraint_score"),
        "final_score": candidate.get("final_score"),
        "final_rank": candidate.get("rank"),
    }


def _failure_category(
    group: dict[str, Any],
    payload: dict[str, Any],
    expected_rank: int | None,
) -> str | None:
    if expected_rank == 1 and _passed(group, payload, expected_rank):
        return None
    diagnostics = payload.get("query_interpreter") or {}
    if diagnostics.get("gemini_success") is False:
        return "INTERPRETATION_FAILURE"
    intent = payload.get("query_interpretation") or {}
    expected_intent = group.get("expected_intent") or {}
    if not _intent_matches(intent, expected_intent):
        return "NORMALIZATION_FAILURE"
    if expected_rank is None:
        return "RETRIEVAL_FAILURE"
    top = (payload.get("candidates") or [{}])[0]
    flags = top.get("constraint_flags") or []
    if any(flag.endswith("_CONTRADICTION") for flag in flags):
        return "CONSTRAINT_FAILURE"
    return "FUSION_FAILURE"


def _passed(group: dict[str, Any], payload: dict[str, Any], expected_rank: int | None) -> bool:
    if group.get("expected_behavior") == "ambiguity_true_grade_missing":
        missing = " ".join(payload.get("missing_information") or []).casefold()
        return bool(payload.get("ambiguity")) and "grade" in missing and expected_rank is not None
    return expected_rank == 1


def _intent_matches(intent: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key in {"subtype", "subtype_family", "cement_type", "pozzolana_source"}:
            if intent.get(key) != expected_value:
                return False
            continue
        values = intent.get(key) or []
        if isinstance(values, str):
            values = [values]
        if expected_value not in values:
            return False
    return True


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


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [result for result in results if result.get("ok")]
    latencies = [float(result["latency_ms"]) for result in ok]
    labeled = [result for result in ok if result.get("expected_standard_codes")]
    expected_ranks = [result.get("expected_rank") for result in labeled]
    return {
        "completed_query_count": len(ok),
        "failed_query_count": len(results) - len(ok),
        "gemini_failure_count": sum(
            (result.get("query_interpreter") or {}).get("gemini_success") is False for result in ok
        ),
        "top1_accuracy": _hit_rate(expected_ranks, 1),
        "top3_accuracy": _hit_rate(expected_ranks, 3),
        "passed_count": sum(result.get("passed") is True for result in ok),
        "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 95), 3) if latencies else None,
    }


def _group_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["group_id"]].append(result)
    metrics = {}
    for group_id, rows in grouped.items():
        ok_rows = [row for row in rows if row.get("ok")]
        keys = [_intent_key(row.get("normalized_intent") or {}) for row in ok_rows]
        top1 = [row.get("top1") for row in ok_rows]
        top3 = [tuple(row.get("top3") or []) for row in ok_rows]
        metrics[group_id] = {
            "paraphrase_count": len(rows),
            "completed_count": len(ok_rows),
            "equivalent_normalized_intent_count": _mode_count(keys),
            "expected_top1_count": sum(row.get("expected_rank") == 1 for row in ok_rows),
            "top3_consistency_count": _mode_count(top3),
            "top1_values": top1,
        }
    return metrics


def _intent_key(intent: dict[str, Any]) -> tuple[Any, ...]:
    return (
        intent.get("normalized_product"),
        tuple(intent.get("material") or []),
        tuple(intent.get("application") or []),
        tuple(intent.get("function") or []),
        intent.get("subtype"),
        intent.get("subtype_family"),
        intent.get("cement_type"),
        intent.get("pozzolana_source"),
        intent.get("grade"),
    )


def _before_after_markdown(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_by_query = {row["raw_query"]: row for row in before["results"]}
    lines = [
        "# StandardWise Live Robustness Before/After",
        "",
        "## Summary",
        f"- Before: {_summary_line(before['summary'])}",
        f"- After: {_summary_line(after['summary'])}",
        "",
        "## Per Query",
        "",
        (
            "| Group | Query | Top-1 before | Top-1 after | Expected rank before | "
            "Expected rank after | Category after | Outcome | Latency after |"
        ),
        "|---|---|---|---|---:|---:|---|---|---:|",
    ]
    for row in after["results"]:
        before_row = before_by_query.get(row["raw_query"], {})
        outcome = _outcome(before_row, row)
        lines.append(
            "| "
            f"{row['group_id']} | {row['raw_query']} | "
            f"{before_row.get('top1')} | {row.get('top1')} | "
            f"{before_row.get('expected_rank')} | {row.get('expected_rank')} | "
            f"{row.get('failure_category')} | {outcome} | {row.get('latency_ms')} |"
        )
    lines.extend(["", "## Paraphrase Consistency"])
    for group_id, metrics in after["group_metrics"].items():
        lines.append(
            "- "
            f"{group_id}: {metrics['equivalent_normalized_intent_count']}/"
            f"{metrics['paraphrase_count']} equivalent intents, "
            f"{metrics['expected_top1_count']}/{metrics['paraphrase_count']} expected Top-1, "
            f"{metrics['top3_consistency_count']}/{metrics['paraphrase_count']} Top-3 consistency"
        )
    return "\n".join(lines) + "\n"


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# StandardWise Live Robustness", "", f"- Phase: {payload['phase']}"]
    lines.append(f"- Summary: {_summary_line(payload['summary'])}")
    return "\n".join(lines) + "\n"


def _summary_line(summary: dict[str, Any]) -> str:
    return (
        f"completed {summary['completed_query_count']}, "
        f"gemini_failures {summary['gemini_failure_count']}, "
        f"top1 {summary['top1_accuracy']}, top3 {summary['top3_accuracy']}, "
        f"latency mean/median/p95 {summary['mean_latency_ms']}/"
        f"{summary['median_latency_ms']}/{summary['p95_latency_ms']} ms"
    )


def _outcome(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_rank = before.get("expected_rank")
    after_rank = after.get("expected_rank")
    if before_rank is None and after_rank is not None:
        return "improved"
    if before_rank is not None and after_rank is None:
        return "regressed"
    if before_rank and after_rank:
        if after_rank < before_rank:
            return "improved"
        if after_rank > before_rank:
            return "regressed"
    if before.get("top1") != after.get("top1"):
        return "changed"
    return "unchanged"


def _hit_rate(ranks: list[int | None], k: int) -> dict[str, Any]:
    total = len(ranks)
    hits = sum(rank is not None and rank <= k for rank in ranks)
    return {"hits": hits, "total": total, "rate": round(hits / total, 4) if total else None}


def _rank_of_any(codes: list[str], expected: list[str]) -> int | None:
    expected_set = {_norm_code(code) for code in expected}
    for rank, code in enumerate(codes, start=1):
        if _norm_code(code) in expected_set:
            return rank
    return None


def _norm_code(code: str) -> str:
    return "".join(str(code).upper().split())


def _flag_state(flags: list[str], prefix: str) -> str | None:
    token = f"{prefix}_"
    for flag in flags:
        if flag.startswith(token):
            return flag.removeprefix(token)
    return None


def _mode_count(values: list[Any]) -> int:
    if not values:
        return 0
    return max(values.count(value) for value in set(values))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _error(
    group: dict[str, Any],
    description: str,
    started_at: str,
    started: float,
    error: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "group_id": group["id"],
        "product": group["product"],
        "raw_query": description,
        "started_at": started_at,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "expected_standard_codes": group["expected_codes"],
        "error": error,
        "failure_category": "INTERPRETATION_FAILURE",
        "passed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live StandardWise robustness suite.")
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=35.0)
    parser.add_argument("--before-json", default=str(OUTPUT_DIR / "live_robustness_before.json"))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
