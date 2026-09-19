from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BEFORE = Path("data/evaluation/results/gemini_constraint_diagnostic_before.json")
AFTER = Path("data/evaluation/results/gemini_constraint_diagnostic_after.json")
OUTPUT = Path("data/evaluation/results/gemini_constraint_diagnostic_before_after.md")


def main() -> None:
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    after = json.loads(AFTER.read_text(encoding="utf-8"))
    lines = ["# Gemini Constraint Diagnostic Before/After", ""]
    for before_item, after_item in zip(before, after, strict=True):
        query = before_item["query"]["description"]
        lines.extend([f"## {query}", ""])
        lines.append(_summarize("Before", before_item))
        lines.append(_summarize("After", after_item))
        lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


def _summarize(label: str, item: dict[str, Any]) -> str:
    if not item.get("ok"):
        return f"- {label}: ERROR {item.get('error')} ({item.get('elapsed_ms')} ms)"
    response = item["response"]
    timings = response.get("timings_ms") or {}
    interpreter = response.get("query_interpreter") or {}
    top3 = []
    for candidate in (response.get("candidates") or [])[:3]:
        flags = ",".join(candidate.get("constraint_flags") or [])
        top3.append(
            f"{candidate.get('standard_code')} flags=[{flags}] "
            f"ce={candidate.get('raw_cross_encoder_score')} "
            f"final={candidate.get('final_score')}"
        )
    return (
        f"- {label}: top3={top3}; ambiguity={response.get('ambiguity')} "
        f"missing={response.get('missing_information')}; "
        f"gemini_success={interpreter.get('gemini_success')} "
        f"gemini_fallback={interpreter.get('gemini_fallback_reason')} "
        f"reranker_success={timings.get('reranker_success')} "
        f"reranker_timeout={timings.get('reranker_timeout')} "
        f"reranker_device={timings.get('reranker_device')} "
        f"latency_ms={item.get('elapsed_ms')}"
    )


if __name__ == "__main__":
    main()
