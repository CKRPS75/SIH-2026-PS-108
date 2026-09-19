from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

DEFAULT_BASELINE = Path("data/evaluation/results/generalized_ranking_repair_after.json")
DEFAULT_CANONICAL = Path("data/canonical/parsed_standards_canonical.json")
DEFAULT_OUTPUT_JSON = Path("data/evaluation/results/gpu_reranker_ab_test.json")
DEFAULT_OUTPUT_MD = Path("data/evaluation/results/gpu_reranker_ab_test.md")


def main() -> None:
    args = _parse_args()
    if args.local_files_only:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    corpus = _corpus_by_code(args.canonical)
    environment = _environment(args.device)
    model_specs = _available_model_specs(args)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "task": "gpu_reranker_ab_test",
        "baseline_artifact": str(args.baseline),
        "canonical_artifact": str(args.canonical),
        "candidate_pool_source": (
            "generalized_ranking_repair_after.json final candidates; "
            "candidate pool size is limited to the preserved top-5 per query"
        ),
        "fusion": {
            "production": (
                "final_score = raw_cross_encoder_score + constraint_score; "
                "non-negative constraint scores sort before negative totals"
            ),
            "pure_cross_encoder": "sort by raw_cross_encoder_score desc, then baseline rank",
            "constraint_safe": (
                "same production score, but candidates with explicit *_CONTRADICTION "
                "flags sort after non-contradictions"
            ),
        },
        "environment": environment,
        "models": {},
        "baseline": _baseline_summary(baseline),
        "evaluations": {},
        "recommendation": {},
    }

    for spec in model_specs:
        print(f"Loading {spec['key']} from {spec['path']}", flush=True)
        model_result = _evaluate_model(spec, args, baseline, corpus)
        report["models"][spec["key"]] = {
            "label": spec["label"],
            "path": str(spec["path"]),
            "sha256": spec["sha256"],
            "load_time_ms": model_result.pop("load_time_ms"),
        }
        report["evaluations"][spec["key"]] = model_result
        _release_cuda()

    report["recommendation"] = _recommendation(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


def _evaluate_model(
    spec: dict[str, Any],
    args: argparse.Namespace,
    baseline: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder

    load_started = perf_counter()
    model = CrossEncoder(
        str(spec["path"]), device=args.device, local_files_only=args.local_files_only
    )
    load_time_ms = (perf_counter() - load_started) * 1000
    _warm_model(model, baseline, corpus, args.batch_size)

    rows = []
    inference_times = []
    total_times = []
    for index, item in enumerate(baseline["results"], start=1):
        started = perf_counter()
        response = item["response"]
        query_text = response["query"]
        candidates = response["candidates"]
        pairs = [[query_text, _candidate_text(corpus, candidate)] for candidate in candidates]
        inference_started = perf_counter()
        raw_scores = model.predict(pairs, batch_size=args.batch_size)
        inference_ms = (perf_counter() - inference_started) * 1000
        scores = _score_list(raw_scores)
        inference_times.append(inference_ms)

        pure = _pure_cross_encoder_rows(candidates, scores)
        production = _production_rows(candidates, scores, contradiction_safe=False)
        safe = _production_rows(candidates, scores, contradiction_safe=True)
        total_ms = (perf_counter() - started) * 1000
        total_times.append(total_ms)

        desired = _desired_codes(response)
        baseline_codes = [candidate["standard_code"] for candidate in candidates]
        row = {
            "index": index,
            "query": item["query"],
            "query_text": query_text,
            "candidate_count": len(candidates),
            "desired_standard_codes": desired,
            "baseline_top1": baseline_codes[0] if baseline_codes else None,
            "baseline_top3": baseline_codes[:3],
            "baseline_top5": baseline_codes[:5],
            "baseline_desired_rank": _rank_of_any(baseline_codes, desired),
            "gpu_top1": production[0]["standard_code"] if production else None,
            "gpu_top3": [row["standard_code"] for row in production[:3]],
            "gpu_top5": [row["standard_code"] for row in production[:5]],
            "gpu_desired_rank": _rank_of_any([row["standard_code"] for row in production], desired),
            "baseline_preserved_rank": _rank_of_any(
                [row["standard_code"] for row in production],
                [baseline_codes[0]] if baseline_codes else [],
            ),
            "outcome": _outcome(candidates, production, desired),
            "reranker_inference_ms": inference_ms,
            "total_eval_ms": total_ms,
            "production_candidates": production,
            "pure_cross_encoder_candidates": pure,
            "constraint_safe_candidates": safe,
            "regression_flags": _regression_flags(candidates, production, safe),
        }
        rows.append(row)

    return {
        "load_time_ms": round(load_time_ms, 3),
        "device": args.device,
        "batch_size": args.batch_size,
        "query_count": len(rows),
        "candidate_pool_size": {
            "min": min(row["candidate_count"] for row in rows),
            "max": max(row["candidate_count"] for row in rows),
            "mean": statistics.fmean(row["candidate_count"] for row in rows),
        },
        "latency": {
            "warm_reranker_inference_ms": _latency_summary(inference_times),
            "total_eval_ms": _latency_summary(total_times),
            "load_time_ms": round(load_time_ms, 3),
        },
        "metrics": {
            "explicit_desired_labels": _metrics(rows, label_source="explicit"),
            "baseline_preservation": _baseline_preservation_metrics(rows),
        },
        "per_query": rows,
        "improvements": [row for row in rows if row["outcome"] == "improved"],
        "regressions": [
            row for row in rows if row["outcome"] == "regressed" or row["regression_flags"]
        ],
    }


def _warm_model(
    model: Any, baseline: dict[str, Any], corpus: dict[str, dict[str, Any]], batch_size: int
) -> None:
    first = baseline["results"][0]
    query_text = first["response"]["query"]
    candidate = first["response"]["candidates"][0]
    model.predict([[query_text, _candidate_text(corpus, candidate)]], batch_size=batch_size)


def _pure_cross_encoder_rows(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[dict[str, Any]]:
    rows = [
        _candidate_row(candidate, score)
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    rows.sort(key=lambda row: (-row["raw_cross_encoder_score"], row["pre_rerank_rank"]))
    return [
        {**row, "post_rerank_rank": rank, "final_rank": rank}
        for rank, row in enumerate(rows, start=1)
    ]


def _production_rows(
    candidates: list[dict[str, Any]],
    scores: list[float],
    *,
    contradiction_safe: bool,
) -> list[dict[str, Any]]:
    rows = []
    for candidate, score in zip(candidates, scores, strict=True):
        row = _candidate_row(candidate, score)
        row["final_score"] = score + float(candidate.get("constraint_score") or 0.0)
        row["has_explicit_contradiction"] = any(
            str(flag).endswith("_CONTRADICTION") for flag in candidate.get("constraint_flags") or []
        )
        rows.append(row)
    rows.sort(key=lambda row: (-row["raw_cross_encoder_score"], row["pre_rerank_rank"]))
    for rank, row in enumerate(rows, start=1):
        row["post_rerank_rank"] = rank
    if contradiction_safe:
        rows.sort(
            key=lambda row: (
                row["has_explicit_contradiction"],
                float(row.get("constraint_score") or 0.0) < 0,
                -float(row["final_score"]),
                row["rrf_rank"],
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                float(row.get("constraint_score") or 0.0) < 0,
                -float(row["final_score"]),
                row["rrf_rank"],
            )
        )
    return [{**row, "final_rank": rank} for rank, row in enumerate(rows, start=1)]


def _candidate_row(candidate: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "standard_code": candidate["standard_code"],
        "title": candidate.get("title"),
        "pre_rerank_rank": candidate["rank"],
        "rrf_score": candidate.get("rrf_score"),
        "constraint_score": candidate.get("constraint_score"),
        "baseline_final_score": candidate.get("final_score"),
        "raw_cross_encoder_score": float(score),
        "reranker_score": float(score),
        "rrf_rank": candidate.get("rrf_rank"),
        "semantic_rank": candidate.get("semantic_rank"),
        "semantic_score": candidate.get("semantic_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "constraint_flags": candidate.get("constraint_flags") or [],
        "product_compatibility": candidate.get("product_compatibility"),
    }


def _candidate_text(corpus: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> str:
    record = corpus.get(_norm_code(candidate["standard_code"]))
    if not record:
        trace = candidate.get("trace") or {}
        return "\n".join(
            str(value)
            for value in [
                f"STANDARD: {candidate.get('standard_code')}",
                f"PRODUCT: {candidate.get('canonical_product')}",
                f"TYPE: {candidate.get('standard_kind')}",
                f"TITLE: {candidate.get('title')}",
                f"Product family: {candidate.get('family')}",
                f"Applications: {trace.get('application')}",
                f"Materials: {trace.get('material')}",
            ]
            if value and not str(value).endswith(": None")
        )
    lines = [
        f"STANDARD: {record.get('standard_code')}",
        f"PRODUCT: {record.get('canonical_product')}",
        f"TYPE: {record.get('standard_kind')}",
        f"TITLE: {record.get('title')}",
        f"Product family: {record.get('family')}",
        f"Applications: {record.get('application')}",
        f"Materials: {record.get('material')}",
        f"SCOPE: {record.get('scope')}",
    ]
    return "\n".join(line for line in lines if line and not line.endswith(": None"))


def _baseline_summary(baseline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in baseline["results"]:
        response = item["response"]
        candidates = response["candidates"]
        desired = _desired_codes(response)
        codes = [candidate["standard_code"] for candidate in candidates]
        rows.append(
            {
                "query": item["query"],
                "candidate_count": len(candidates),
                "top1": codes[0] if codes else None,
                "top3": codes[:3],
                "top5": codes[:5],
                "desired_standard_codes": desired,
                "desired_rank": _rank_of_any(codes, desired),
                "latency_ms": item.get("elapsed_ms"),
            }
        )
    return {
        "summary": baseline.get("summary"),
        "query_count": len(rows),
        "candidate_pool_size": {
            "min": min(row["candidate_count"] for row in rows),
            "max": max(row["candidate_count"] for row in rows),
            "mean": statistics.fmean(row["candidate_count"] for row in rows),
        },
        "metrics": {
            "explicit_desired_labels": _baseline_metrics(rows),
            "baseline_preservation": {
                "top1": len(rows),
                "top3": len(rows),
                "top5": len(rows),
                "mrr": 1.0,
                "query_count": len(rows),
            },
        },
        "per_query": rows,
    }


def _baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["desired_standard_codes"]]
    ranks = [row["desired_rank"] for row in labeled]
    return _rank_metrics(ranks)


def _metrics(rows: list[dict[str, Any]], *, label_source: str) -> dict[str, Any]:
    if label_source != "explicit":
        raise ValueError("unsupported label source")
    labeled = [row for row in rows if row["desired_standard_codes"]]
    ranks = [row["gpu_desired_rank"] for row in labeled]
    return _rank_metrics(ranks)


def _baseline_preservation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["baseline_preserved_rank"] for row in rows]
    return _rank_metrics(ranks)


def _rank_metrics(ranks: list[int | None]) -> dict[str, Any]:
    count = len(ranks)
    return {
        "query_count": count,
        "top1": sum(rank == 1 for rank in ranks),
        "top3": sum(rank is not None and rank <= 3 for rank in ranks),
        "top5": sum(rank is not None and rank <= 5 for rank in ranks),
        "mrr": round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / count, 6)
        if count
        else None,
    }


def _outcome(
    baseline_candidates: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    desired: list[str],
) -> str:
    baseline_codes = [candidate["standard_code"] for candidate in baseline_candidates]
    gpu_codes = [row["standard_code"] for row in gpu_rows]
    if desired:
        before = _rank_of_any(baseline_codes, desired) or math.inf
        after = _rank_of_any(gpu_codes, desired) or math.inf
        if after < before:
            return "improved"
        if after > before:
            return "regressed"
        return "unchanged"
    if baseline_codes and gpu_codes and baseline_codes[0] != gpu_codes[0]:
        return "changed_unlabeled"
    return "unchanged"


def _regression_flags(
    baseline_candidates: list[dict[str, Any]],
    production_rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
) -> list[str]:
    flags = []
    baseline_top1 = baseline_candidates[0]["standard_code"] if baseline_candidates else None
    production_top1 = production_rows[0]["standard_code"] if production_rows else None
    safe_top1 = safe_rows[0]["standard_code"] if safe_rows else None
    if production_top1 != baseline_top1:
        flags.append("top1_changed_vs_deterministic_baseline")
    top = production_rows[0] if production_rows else None
    if top and top.get("has_explicit_contradiction"):
        flags.append("production_top1_has_explicit_contradiction")
    if production_top1 != safe_top1:
        flags.append("constraint_safe_order_differs_from_production")
    return flags


def _desired_codes(response: dict[str, Any]) -> list[str]:
    return [str(code) for code in response.get("desired_standard_codes") or []]


def _rank_of_any(codes: list[str], desired: list[str]) -> int | None:
    desired_set = {_norm_code(code) for code in desired}
    if not desired_set:
        return None
    for rank, code in enumerate(codes, start=1):
        if _norm_code(code) in desired_set:
            return rank
    return None


def _score_list(raw_scores: Any) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, int | float):
        return [float(raw_scores)]
    return [float(score[0] if isinstance(score, list | tuple) else score) for score in raw_scores]


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _environment(device: str) -> dict[str, Any]:
    import platform

    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _available_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        {
            "key": "pretrained_bge_large",
            "label": "pretrained BGE reranker large",
            "path": args.pretrained,
        },
        {
            "key": "partial_last4",
            "label": "partial-last4 checkpoint",
            "path": args.partial4,
        },
    ]
    available = []
    for spec in specs:
        if spec["path"].exists():
            spec["sha256"] = _sha256_model(spec["path"])
            available.append(spec)
        else:
            print(f"Skipping unavailable checkpoint: {spec['path']}", flush=True)
    return available


def _sha256_model(path: Path) -> str | None:
    model_file = path / "model.safetensors"
    if not model_file.exists():
        return None
    digest = hashlib.sha256()
    with model_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_by_code(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return {_norm_code(record["standard_code"]): record for record in records}


def _norm_code(code: str) -> str:
    return "".join(str(code).upper().split())


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _recommendation(report: dict[str, Any]) -> dict[str, Any]:
    baseline_explicit = report["baseline"]["metrics"]["explicit_desired_labels"]
    baseline_top1 = baseline_explicit["top1"]
    baseline_mrr = baseline_explicit["mrr"] or 0.0
    best = None
    reasons = []
    for key, result in report["evaluations"].items():
        explicit = result["metrics"]["explicit_desired_labels"]
        preservation = result["metrics"]["baseline_preservation"]
        regressions = len(result["regressions"])
        if explicit["top1"] > baseline_top1 and regressions == 0:
            best = key
        if explicit["top1"] < baseline_top1 or (explicit["mrr"] or 0.0) < baseline_mrr:
            reasons.append(f"{key} regressed explicit desired-label quality")
        if preservation["top1"] < preservation["query_count"]:
            changed_count = preservation["query_count"] - preservation["top1"]
            reasons.append(
                f"{key} changed deterministic Top-1 on {changed_count} diagnostic queries"
            )
        if regressions:
            reasons.append(f"{key} has {regressions} regression-flagged queries")
    if best:
        return {
            "decision": "KEEP GPU RERANKER",
            "checkpoint": best,
            "reason": (
                "GPU reranker improved explicit desired-label Top-1 without regression flags."
            ),
        }
    return {
        "decision": "USE DETERMINISTIC PIPELINE WITHOUT RERANKER",
        "checkpoint": None,
        "reason": "; ".join(reasons)
        or (
            "GPU reranker did not provide meaningful quality improvement over "
            "the repaired deterministic baseline."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# StandardWise GPU Reranker A/B Test",
        "",
        "## Environment",
        f"- Python: `{report['environment']['python']}`",
        f"- Torch: `{report['environment']['torch']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        f"- GPU: `{report['environment']['cuda_device']}`",
        f"- Candidate pool source: {report['candidate_pool_source']}",
        "",
        "## Baseline",
    ]
    baseline_summary = report["baseline"]["summary"] or {}
    explicit = report["baseline"]["metrics"]["explicit_desired_labels"]
    preservation = report["baseline"]["metrics"]["baseline_preservation"]
    lines.extend(
        [
            f"- Completed: {baseline_summary.get('completed_query_count')}/15",
            f"- Timeouts: {baseline_summary.get('timeout_count')}",
            (
                f"- Latency: mean {baseline_summary.get('mean_latency_ms')} ms, "
                f"median {baseline_summary.get('median_latency_ms')} ms, "
                f"p95 {baseline_summary.get('p95_latency_ms')} ms"
            ),
            _metric_line("Explicit-label", explicit),
            _metric_line("Baseline-preservation", preservation),
            "",
            "## GPU Results",
        ]
    )
    for key, result in report["evaluations"].items():
        model = report["models"][key]
        explicit = result["metrics"]["explicit_desired_labels"]
        preservation = result["metrics"]["baseline_preservation"]
        latency = result["latency"]["warm_reranker_inference_ms"]
        lines.extend(
            [
                f"### {key}",
                f"- Checkpoint: `{model['path']}`",
                f"- SHA256: `{model['sha256']}`",
                f"- Load time: {model['load_time_ms']} ms",
                f"- Device / batch size: `{result['device']}` / {result['batch_size']}",
                (
                    f"- Candidate pool size: {result['candidate_pool_size']['min']}-"
                    f"{result['candidate_pool_size']['max']} candidates"
                ),
                _metric_line("Explicit-label", explicit),
                _metric_line("Baseline-preservation", preservation),
                (
                    f"- Warm reranker inference: mean {latency['mean']} ms, "
                    f"median {latency['median']} ms, p95 {latency['p95']} ms"
                ),
                f"- Regression-flagged queries: {len(result['regressions'])}",
                "",
            ]
        )
    lines.extend(["## Per-Query Comparison", ""])
    header = (
        "| # | Query | Baseline Top 1 | GPU Top 1 | "
        "Desired rank before -> after | Outcome | Regression flags |"
    )
    lines.extend([header, "|---:|---|---|---|---|---|---|"])
    first_key = next(iter(report["evaluations"]), None)
    if first_key:
        for row in report["evaluations"][first_key]["per_query"]:
            lines.append(
                "| "
                f"{row['index']} | "
                f"{row['query']['description']} | "
                f"{row['baseline_top1']} | "
                f"{row['gpu_top1']} | "
                f"{row['baseline_desired_rank']} -> {row['gpu_desired_rank']} | "
                f"{row['outcome']} | "
                f"{', '.join(row['regression_flags']) or '-'} |"
            )
    lines.extend(["", "## Recommendation"])
    rec = report["recommendation"]
    lines.extend([f"- Decision: **{rec['decision']}**", f"- Reason: {rec['reason']}", ""])
    return "\n".join(lines)


def _metric_line(label: str, metrics: dict[str, Any]) -> str:
    values = f"{metrics['top1']}/{metrics['top3']}/{metrics['top5']}/{metrics['mrr']}"
    return f"- {label} Top-1/Top-3/Top-5/MRR: {values}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU reranker A/B test for repaired diagnostics.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--pretrained", type=Path, default=Path("models/bge-reranker-large"))
    parser.add_argument(
        "--partial4",
        type=Path,
        default=Path("models/full-corpus-reranker-partial4/best"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


if __name__ == "__main__":
    main()
