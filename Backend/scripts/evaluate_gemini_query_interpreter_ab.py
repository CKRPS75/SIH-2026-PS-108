from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_INPUT = Path("data/evaluation/messy_query_eval.jsonl")
DEFAULT_JSON = Path("data/evaluation/results/gemini_query_interpreter_ab_test.json")
DEFAULT_MD = Path("data/evaluation/results/gemini_query_interpreter_ab_test.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current StandardWise product-aware search against "
            "Gemini query interpretation."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    records = _load_jsonl(args.input)
    disabled = _evaluate_mode(records, api_base_url=args.api_base_url, mode="disabled")
    gemini = _evaluate_mode(records, api_base_url=args.api_base_url, mode="gemini")
    report = {
        "input": str(args.input),
        "api_base_url": args.api_base_url,
        "record_count": len(records),
        "current": disabled,
        "gemini": gemini,
        "delta": {
            key: gemini.get(key, 0) - disabled.get(key, 0)
            for key in ["top1", "top3", "top5", "mrr", "cross_product_errors"]
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md_output.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")


def _evaluate_mode(
    records: list[dict[str, Any]],
    *,
    api_base_url: str,
    mode: str,
) -> dict[str, Any]:
    reciprocal_ranks = []
    top1 = top3 = top5 = 0
    ambiguity_correct = 0
    ambiguity_total = 0
    contradiction_errors = 0
    cross_product_errors = 0
    fallbacks = 0
    latencies = []
    errors = []
    for record in records:
        started_at = time.perf_counter()
        try:
            response = _post_product_aware(api_base_url, record, mode)
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append({"record": record, "error": str(exc)})
            continue
        latencies.append((time.perf_counter() - started_at) * 1000)
        expected_codes = set(record.get("expected_standard_codes") or [])
        returned_codes = [
            candidate["standard_code"] for candidate in response.get("candidates", [])
        ]
        rank = _first_rank(returned_codes, expected_codes)
        if rank == 1:
            top1 += 1
        if rank is not None and rank <= 3:
            top3 += 1
        if rank is not None and rank <= 5:
            top5 += 1
        reciprocal_ranks.append(0.0 if rank is None else 1 / rank)
        if "expected_ambiguity" in record:
            ambiguity_total += 1
            if bool(response.get("ambiguity")) == bool(record["expected_ambiguity"]):
                ambiguity_correct += 1
        if any(
            "CONTRADICTION" in flag
            for candidate in response.get("candidates", [])
            for flag in candidate.get("constraint_flags", [])
        ):
            contradiction_errors += 1
        if response.get("query_interpreter", {}).get("gemini_fallback_reason"):
            fallbacks += 1
        expected_product = record.get("expected_product")
        if expected_product and response.get("canonical_product") not in {expected_product, None}:
            cross_product_errors += 1
    count = max(len(records), 1)
    return {
        "top1": top1 / count,
        "top3": top3 / count,
        "top5": top5 / count,
        "mrr": sum(reciprocal_ranks) / count,
        "ambiguity_correct": ambiguity_correct,
        "ambiguity_total": ambiguity_total,
        "constraint_contradiction_errors": contradiction_errors,
        "cross_product_errors": cross_product_errors,
        "average_latency_ms": statistics.mean(latencies) if latencies else None,
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 95),
        "gemini_failure_fallback_count": fallbacks,
        "request_errors": errors,
    }


def _post_product_aware(api_base_url: str, record: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = {
        "product": record["product"],
        "description": record["description"],
        "limit": int(record.get("limit", 5)),
    }
    request = Request(
        f"{api_base_url.rstrip('/')}/api/v1/search/product-aware",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Query-Interpreter-Mode": mode,
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - local/dev harness URL
        return json.loads(response.read().decode("utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _first_rank(returned_codes: list[str], expected_codes: set[str]) -> int | None:
    if not expected_codes:
        return None
    for index, code in enumerate(returned_codes, start=1):
        if code in expected_codes:
            return index
    return None


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gemini Query Interpreter A/B Test",
        "",
        f"Input: `{report['input']}`",
        f"Records: {report['record_count']}",
        "",
        "| Metric | Current | Gemini | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["top1", "top3", "top5", "mrr", "cross_product_errors"]:
        current = report["current"].get(key)
        gemini = report["gemini"].get(key)
        delta = report["delta"].get(key)
        lines.append(f"| {key} | {_fmt(current)} | {_fmt(gemini)} | {_fmt(delta)} |")
    lines.extend(
        [
            "",
            f"Current average latency ms: {_fmt(report['current'].get('average_latency_ms'))}",
            f"Gemini average latency ms: {_fmt(report['gemini'].get('average_latency_ms'))}",
            f"Gemini fallback count: {report['gemini'].get('gemini_failure_fallback_count')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
