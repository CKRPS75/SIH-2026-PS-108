from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"
JSON_OUT = RESULTS_DIR / "tender_text_evaluation.json"
MD_OUT = RESULTS_DIR / "tender_text_evaluation.md"
ENDPOINT = "http://127.0.0.1:8000/api/v1/tenders/analyze-text"

CASES = [
    {
        "name": "UPVC drainage tender",
        "expected_items": 1,
        "expected_top1": "IS 13592",
        "text": (
            "Tender Item No. 07 - Supply and Installation of UPVC Drainage Pipes. "
            "Supply, delivery, installation, jointing and testing of unplasticized "
            "polyvinyl chloride (UPVC) pipes for soil, waste and rainwater drainage "
            "applications in residential and commercial buildings. The pipes shall "
            "be suitable for conveying domestic wastewater, soil discharge and "
            "rainwater from internal and external building drainage systems. The "
            "pipes shall be of 110 mm nominal diameter unless otherwise specified "
            "in the Bill of Quantities. The material shall be rigid UPVC, resistant "
            "to normal domestic wastewater, moisture and corrosion. Pipes shall have "
            "smooth internal surfaces and shall be suitable for gravity drainage "
            "applications. The scope of work shall include cutting, laying, jointing, "
            "fixing and connecting the pipes using compatible UPVC fittings, bends, "
            "tees, couplers and other accessories required for completion of the "
            "drainage system. All joints shall be watertight and shall be made using "
            "suitable manufacturer-recommended jointing methods. The contractor "
            "shall provide all necessary clamps, supports, fittings and accessories "
            "required for proper installation. The completed piping system shall be "
            "checked for leakage and satisfactory flow before acceptance. The "
            "supplied material shall be new, free from manufacturing defects and "
            "suitable for use in building soil, waste and rainwater disposal systems."
        ),
    },
    {
        "name": "OPC 43 tender",
        "expected_items": 1,
        "expected_top1": "IS 8112",
        "text": "Supply 43 grade ordinary Portland cement for concrete work.",
    },
    {
        "name": "PPC fly ash tender",
        "expected_items": 1,
        "expected_top1": "IS 1489 (Part 1)",
        "text": "Supply Portland pozzolana cement manufactured using fly ash.",
    },
    {
        "name": "HDPE sewage pipe tender",
        "expected_items": 1,
        "expected_top1": None,
        "text": "Supply HDPE pipe for municipal sewage conveyance.",
    },
    {
        "name": "Grade C bolt tender",
        "expected_items": 1,
        "expected_top1": "IS 1363 (Part 1)",
        "text": "Supply Grade C hexagonal bolts for fastening.",
    },
    {
        "name": "Pressure-reducing valve tender",
        "expected_items": 1,
        "expected_top1": None,
        "text": "Supply pressure reducing valve for water pipeline downstream pressure control.",
    },
    {
        "name": "Roofing flooring negative case",
        "expected_items": 1,
        "expected_top1": None,
        "text": "Supply clay tiles for internal flooring, explicitly not roofing.",
    },
    {
        "name": "Multi-item tender",
        "expected_items": 3,
        "expected_top1": None,
        "text": (
            "Item 1: UPVC drainage pipe for soil waste and rainwater in buildings.\n"
            "Item 2: 43 grade ordinary Portland cement for concrete work.\n"
            "Item 3: Grade C hexagonal bolt for fastening."
        ),
    },
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_run_case(case) for case in CASES]
    latencies = [row["total_latency_ms"] for row in rows if row["total_latency_ms"] is not None]
    summary = {
        "endpoint": ENDPOINT,
        "case_count": len(rows),
        "completed": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] != "ok" for row in rows),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }
    payload = {"summary": summary, "cases": rows}
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    MD_OUT.write_text(_render_md(payload), encoding="utf-8")
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {MD_OUT.relative_to(ROOT)}")


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        data = json.dumps({"text": case["text"], "limit": 5}).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - started) * 1000
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "name": case["name"],
            "status": "request_failed",
            "failure": str(exc),
            "expected_items": case["expected_items"],
            "expected_top1": case["expected_top1"],
            "total_latency_ms": None,
        }

    items = body.get("items", [])
    top1 = [
        (item.get("recommendations") or [{}])[0].get("standard_code")
        for item in items
        if item.get("recommendations")
    ]
    top3 = [
        [rec.get("standard_code") for rec in (item.get("recommendations") or [])[:3]]
        for item in items
    ]
    expected = case.get("expected_top1")
    expected_top1_hit = (
        None if expected is None else any(str(code or "").startswith(expected) for code in top1)
    )
    return {
        "name": case["name"],
        "status": "ok",
        "extraction_success": body.get("item_count") == case["expected_items"],
        "item_count": body.get("item_count"),
        "expected_items": case["expected_items"],
        "expected_top1": expected,
        "expected_top1_hit": expected_top1_hit,
        "top1": top1,
        "top3": top3,
        "gemini_latency_ms": (body.get("diagnostics", {}).get("extraction") or {}).get(
            "gemini_latency_ms"
        ),
        "total_latency_ms": latency_ms,
        "diagnostics_status": body.get("diagnostics", {}).get("status"),
        "warnings": body.get("diagnostics", {}).get("warnings", []),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Tender Text Evaluation",
        "",
        f"- Endpoint: `{payload['summary']['endpoint']}`",
        f"- Cases: {payload['summary']['case_count']}",
        f"- Completed: {payload['summary']['completed']}",
        f"- Failed: {payload['summary']['failed']}",
        f"- Mean latency ms: {payload['summary']['mean_latency_ms']}",
        f"- Median latency ms: {payload['summary']['median_latency_ms']}",
        "",
        "| Case | Items | Expected Top-1 | Hit | Top-1 | Top-3 | Latency ms |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for row in payload["cases"]:
        lines.append(
            "| {name} | {items} | {expected} | {hit} | {top1} | {top3} | {latency} |".format(
                name=row["name"],
                items=row.get("item_count"),
                expected=row.get("expected_top1"),
                hit=row.get("expected_top1_hit"),
                top1=", ".join(row.get("top1", [])),
                top3=json.dumps(row.get("top3", [])),
                latency=row.get("total_latency_ms"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
