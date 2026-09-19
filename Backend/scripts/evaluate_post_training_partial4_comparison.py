from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.standards import canonicalize_standard_id, extract_base_code  # noqa: E402
from app.services.bm25_service import _Bm25Okapi, tokenize  # noqa: E402
from app.services.parsed_standards_corpus import (  # noqa: E402
    FAMILY_BY_PRODUCT,
    build_product_query,
    canonicalize_product,
    product_compatibility,
)
from scripts.evaluate_reranked_retrieval_50 import (  # noqa: E402
    HOLDOUT_CASES,
    UNSEEN_HOLDOUT_CASES,
)
from scripts.evaluate_semantic_retrieval_50 import PRIMARY_BENCHMARK  # noqa: E402

COMPATIBLE = "COMPATIBLE"
INCOMPATIBLE_DATASET = "INCOMPATIBLE_DATASET"
ERROR = "ERROR"

DEFAULT_CANONICAL_PATH = Path("data/canonical/parsed_standards_canonical.json")
DEFAULT_FULL_EVAL_PATH = Path("data/evaluation/full_corpus_eval.jsonl")
DEFAULT_PRODUCT_HOLDOUT_PATH = Path("data/evaluation/product_aware_holdout.jsonl")
DEFAULT_AMBIGUOUS_EVAL_PATH = Path("data/evaluation/full_corpus_ambiguous_eval.jsonl")
DEFAULT_RESULT_PATH = Path("data/evaluation/results/post_training_partial4_comparison.json")
DEFAULT_REPORT_PATH = Path("data/evaluation/results/post_training_partial4_comparison.md")
DEFAULT_BASELINE_PATH = Path("data/evaluation/results/full_corpus_eval_baselines.json")
DEFAULT_REGRESSION_HYBRID_PATH = Path("data/evaluation/results/full_corpus_hybrid_regression.json")


@dataclass(frozen=True)
class EvalCase:
    query_id: str
    query: str
    expected_standard: str = ""
    product: str | None = None
    description: str | None = None
    family: str | None = None
    standard_kind: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: Path


def classify_cases_for_dataset(
    suite_name: str,
    cases: list[EvalCase],
    corpus_codes: set[str],
) -> dict[str, Any]:
    missing_cases = []
    for case in cases:
        if not case.expected_standard:
            continue
        expected_code = canonicalize_standard_id(case.expected_standard)
        if expected_code not in corpus_codes:
            missing_cases.append(
                {
                    "query_id": case.query_id,
                    "query": case.query,
                    "expected_standard": case.expected_standard,
                    "canonical_expected_standard": expected_code,
                }
            )
    if missing_cases:
        return {
            "name": suite_name,
            "status": INCOMPATIBLE_DATASET,
            "query_count": len(cases),
            "missing_standard_codes": sorted(
                {case["expected_standard"] for case in missing_cases}
            ),
            "missing_cases": missing_cases,
        }
    return {
        "name": suite_name,
        "status": COMPATIBLE,
        "query_count": len(cases),
        "missing_standard_codes": [],
        "missing_cases": [],
    }


async def main_async() -> None:
    args = _parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    corpus = _load_corpus(args.canonical)
    corpus_by_code = {
        canonicalize_standard_id(record["standard_code"]): record for record in corpus["records"]
    }
    bm25_index = FileBm25Index(corpus["records"])
    cached_pools = _load_cached_candidate_pools(args)
    suites = _build_suites(args)
    suite_statuses = {
        name: classify_cases_for_dataset(name, cases, set(corpus_by_code))
        for name, cases in suites.items()
    }

    model_specs = _available_model_specs(args)

    report: dict[str, Any] = {
        "canonical_dataset": str(args.canonical),
        "canonical_record_count": len(corpus["records"]),
        "candidate_pool_sources": {
            "cached_hybrid": [
                suite_name
                for suite_name, pools in cached_pools.items()
                if pools
            ],
            "fallback": "file_bm25",
        },
        "rerank_k": args.rerank_k,
        "return_k": args.return_k,
        "device": args.device,
        "models": {
            spec.key: {"label": spec.label, "path": str(spec.path)} for spec in model_specs
        },
        "suite_statuses": suite_statuses,
        "suite_results": {},
        "model_parameter_reports": {},
    }

    for spec in model_specs:
        print(f"Loading {spec.label}: {spec.path}")
        model = _load_cross_encoder(spec, args)
        report["model_parameter_reports"][spec.key] = _model_parameter_report(model)
        try:
            for suite_name, cases in suites.items():
                status = suite_statuses[suite_name]
                if status["status"] != COMPATIBLE:
                    continue
                print(f"Evaluating {spec.label} on {suite_name}...")
                started = perf_counter()
                try:
                    if suite_name in {
                        "product_aware_holdout",
                        "ambiguous_product_diagnostics",
                    }:
                        result = _evaluate_product_suite(
                            model,
                            cases,
                            suite_name=suite_name,
                            rerank_k=args.rerank_k,
                            return_k=args.return_k,
                            corpus_by_code=corpus_by_code,
                            bm25_index=bm25_index,
                            cached_pools=cached_pools,
                            batch_size=args.rerank_batch_size,
                            scored=any(case.expected_standard for case in cases),
                        )
                    else:
                        result = _evaluate_query_suite(
                            model,
                            cases,
                            suite_name=suite_name,
                            rerank_k=args.rerank_k,
                            return_k=args.return_k,
                            corpus_by_code=corpus_by_code,
                            bm25_index=bm25_index,
                            cached_pools=cached_pools,
                            batch_size=args.rerank_batch_size,
                        )
                    result["runtime_seconds"] = perf_counter() - started
                    report["suite_results"].setdefault(suite_name, {})[spec.key] = result
                except Exception as exc:  # noqa: BLE001 - suite isolation is intentional
                    report["suite_results"].setdefault(suite_name, {})[spec.key] = {
                        "status": ERROR,
                        "error": repr(exc),
                    }
        finally:
            _release_model(model)

    report["suite_statuses"] = suite_statuses
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.report.write_text(_human_report(report), encoding="utf-8")
    print(f"Wrote JSON report: {args.output}")
    print(f"Wrote human report: {args.report}")


def _evaluate_query_suite(
    model: Any,
    cases: list[EvalCase],
    *,
    suite_name: str,
    rerank_k: int,
    return_k: int,
    corpus_by_code: dict[str, dict[str, Any]],
    bm25_index: FileBm25Index,
    cached_pools: dict[str, dict[str, list[dict[str, Any]]]],
    batch_size: int,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        pool, retrieval_source = _candidate_pool(
            suite_name,
            case,
            case.query,
            rerank_k,
            bm25_index,
            cached_pools,
        )
        reranked_rows = _rerank_rows(
            model,
            case.query,
            pool,
            corpus_by_code,
            return_k=return_k,
            batch_size=batch_size,
        )
        expected_pool_rank = _expected_rank(pool, case.expected_standard, "rank")
        expected_rank = _expected_rank(reranked_rows, case.expected_standard, "rank")
        top1_standard = reranked_rows[0]["standard_code"] if reranked_rows else None
        rows.append(
            {
                **asdict(case),
                "expected_rank": expected_rank,
                "candidate_recall_rank": expected_pool_rank,
                "candidate_recall_at_k": expected_pool_rank is not None,
                "top1_standard": top1_standard,
                "retrieval_source": retrieval_source,
                "error_type": _classify_error(
                    case.expected_standard,
                    expected_pool_rank,
                    expected_rank,
                    top1_standard,
                    corpus_by_code,
                ),
                "candidates": reranked_rows,
                "candidate_pool": pool,
            }
        )
    return _suite_payload(rows, scored=True)


def _evaluate_product_suite(
    model: Any,
    cases: list[EvalCase],
    *,
    suite_name: str,
    rerank_k: int,
    return_k: int,
    corpus_by_code: dict[str, dict[str, Any]],
    bm25_index: FileBm25Index,
    cached_pools: dict[str, dict[str, list[dict[str, Any]]]],
    batch_size: int,
    scored: bool,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        query = build_product_query(case.product or "", case.description or "")
        pool, retrieval_source = _candidate_pool(
            suite_name,
            case,
            query,
            rerank_k,
            bm25_index,
            cached_pools,
        )
        canonical_product, product_confidence = canonicalize_product(case.product or "")
        compatibility_by_code = {
            row["standard_code"]: _product_compatibility(
                canonical_product if product_confidence == "high" else None,
                corpus_by_code.get(canonicalize_standard_id(row["standard_code"]), {}),
            )
            for row in pool
        }
        reranked_rows = _rerank_rows(
            model,
            query,
            pool,
            corpus_by_code,
            return_k=return_k,
            batch_size=batch_size,
            compatibility_by_code=compatibility_by_code,
            prioritize_function=_description_function(case.description or ""),
        )
        expected_pool_rank = _expected_rank(pool, case.expected_standard, "rank")
        expected_rank = _expected_rank(reranked_rows, case.expected_standard, "rank")
        top1_standard = reranked_rows[0]["standard_code"] if reranked_rows else None
        rows.append(
            {
                **asdict(case),
                "query": query,
                "canonical_product": canonical_product,
                "product_confidence": product_confidence,
                "expected_rank": expected_rank,
                "candidate_recall_rank": expected_pool_rank,
                "candidate_recall_at_k": expected_pool_rank is not None
                if case.expected_standard
                else None,
                "top1_standard": top1_standard,
                "top1_compatibility": reranked_rows[0]["product_compatibility"]
                if reranked_rows
                else None,
                "retrieval_source": retrieval_source,
                "error_type": _classify_error(
                    case.expected_standard,
                    expected_pool_rank,
                    expected_rank,
                    top1_standard,
                    corpus_by_code,
                )
                if scored and case.expected_standard
                else "UNSCORED",
                "candidates": reranked_rows,
                "candidate_pool": pool,
            }
        )
    payload = _suite_payload(rows, scored=scored)
    payload["cross_product_top1_errors"] = sum(
        row.get("top1_compatibility") == "incompatible"
        for row in rows
        if row.get("canonical_product")
    )
    return payload


def _suite_payload(rows: list[dict[str, Any]], *, scored: bool) -> dict[str, Any]:
    payload = {
        "status": COMPATIBLE,
        "query_count": len(rows),
        "scored": scored,
        "results": rows,
    }
    if scored:
        scored_rows = [row for row in rows if row.get("expected_standard")]
        payload["labeled_query_count"] = len(scored_rows)
        payload["metrics"] = _metrics(scored_rows)
        payload["candidate_recall"] = {
            "at_20_count": sum(row["candidate_recall_at_k"] is True for row in scored_rows),
            "at_20_rate": sum(row["candidate_recall_at_k"] is True for row in scored_rows)
            / len(scored_rows)
            if scored_rows
            else 0.0,
        }
        payload["error_counts"] = _error_counts(scored_rows)
    return payload


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_count = len(rows)
    ranks = [row["expected_rank"] for row in rows]
    return {
        "query_count": query_count,
        "hit_at_1": sum(rank == 1 for rank in ranks),
        "hit_at_1_rate": sum(rank == 1 for rank in ranks) / query_count if query_count else 0.0,
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
        "hit_at_3_rate": sum(rank is not None and rank <= 3 for rank in ranks) / query_count
        if query_count
        else 0.0,
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
        "hit_at_5_rate": sum(rank is not None and rank <= 5 for rank in ranks) / query_count
        if query_count
        else 0.0,
        "mrr": sum(1 / rank if rank else 0.0 for rank in ranks) / query_count
        if query_count
        else 0.0,
    }


def _classify_error(
    expected_standard: str,
    candidate_recall_rank: int | None,
    expected_rank: int | None,
    top1_standard: str | None,
    corpus_by_code: dict[str, dict[str, Any]],
) -> str:
    if not expected_standard:
        return "UNSCORED"
    if expected_rank == 1:
        return "PASS"
    if candidate_recall_rank is None:
        return "RETRIEVAL_MISS"
    if top1_standard is None:
        return "RERANKER_ERROR"

    expected_code = canonicalize_standard_id(expected_standard)
    top1_code = canonicalize_standard_id(top1_standard)
    expected_base = extract_base_code(expected_code)
    top1_base = extract_base_code(top1_code)
    if expected_base == top1_base or "PART" in expected_code or "PART" in top1_code:
        return "PART_OR_VARIANT_CONFUSION"

    expected_record = corpus_by_code.get(expected_code, {})
    top1_record = corpus_by_code.get(top1_code, {})
    if _different_nonempty(expected_record, top1_record, "function"):
        return "FUNCTION_CONFUSION"
    if _different_nonempty(expected_record, top1_record, "product_subtype"):
        return "SUBTYPE_CONFUSION"
    if _different_nonempty(expected_record, top1_record, "material") or _different_nonempty(
        expected_record,
        top1_record,
        "application",
    ):
        return "MATERIAL_APPLICATION_CONFUSION"
    return "RERANKER_ERROR"


def _different_nonempty(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    left_value = left.get(key)
    right_value = right.get(key)
    return bool(left_value and right_value and left_value != right_value)


def _error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row["error_type"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _expected_rank(candidates: list[dict[str, Any]], expected: str, rank_key: str) -> int | None:
    if not expected:
        return None
    expected_code = canonicalize_standard_id(expected)
    for candidate in candidates:
        if canonicalize_standard_id(str(candidate["standard_code"])) == expected_code:
            return int(candidate[rank_key])
    return None


class FileBm25Index:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._documents = [tokenize(record.get("retrieval_text") or "") for record in records]
        self._bm25 = _Bm25Okapi(self._documents, k1=1.5, b=0.75)

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        scores = self._bm25.score(query_tokens)
        ranked = [
            (record, score)
            for record, score in zip(self._records, scores, strict=True)
            if score > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]["standard_code_norm"]))
        return [
            {
                "rank": rank,
                "standard_code": record["standard_code"],
                "title": record["title"],
                "rrf_score": None,
                "semantic_rank": None,
                "semantic_score": None,
                "bm25_rank": rank,
                "bm25_score": score,
            }
            for rank, (record, score) in enumerate(ranked[:limit], start=1)
        ]


def _candidate_pool(
    suite_name: str,
    case: EvalCase,
    query: str,
    rerank_k: int,
    bm25_index: FileBm25Index,
    cached_pools: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], str]:
    cached = cached_pools.get(suite_name, {}).get(case.query_id)
    if cached is not None:
        return cached[:rerank_k], "cached_hybrid"
    return bm25_index.search(query, limit=rerank_k), "file_bm25"


def _rerank_rows(
    model: Any,
    query: str,
    pool: list[dict[str, Any]],
    corpus_by_code: dict[str, dict[str, Any]],
    *,
    return_k: int,
    batch_size: int,
    compatibility_by_code: dict[str, str] | None = None,
    prioritize_function: str | None = None,
) -> list[dict[str, Any]]:
    pairs = [
        [query, _candidate_text(corpus_by_code[canonicalize_standard_id(row["standard_code"])])]
        for row in pool
        if canonicalize_standard_id(row["standard_code"]) in corpus_by_code
    ]
    scored_pool = [
        row for row in pool if canonicalize_standard_id(row["standard_code"]) in corpus_by_code
    ]
    if not pairs:
        return []
    raw_scores = model.predict(pairs, batch_size=batch_size)
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, int | float):
        raw_scores = [float(raw_scores)]
    rows = []
    for row, score in zip(scored_pool, raw_scores, strict=True):
        reranked = {
            **row,
            "source_rank": row["rank"],
            "reranker_score": _as_float_score(score),
        }
        if compatibility_by_code is not None:
            reranked["product_compatibility"] = compatibility_by_code.get(
                row["standard_code"],
                "unknown_candidate_product",
            )
            record = corpus_by_code.get(canonicalize_standard_id(row["standard_code"]), {})
            reranked["canonical_product"] = record.get("canonical_product")
            reranked["standard_kind"] = record.get("standard_kind")
            reranked["family"] = record.get("family")
            reranked["function"] = record.get("function")
        rows.append(reranked)

    rows.sort(key=lambda row: (-row["reranker_score"], row["source_rank"]))
    rows = rows[:return_k]
    if prioritize_function is not None:
        matching = [row for row in rows if row.get("function") == prioritize_function]
        if matching:
            rows = matching + [row for row in rows if row.get("function") != prioritize_function]
    return [{**row, "rank": rank} for rank, row in enumerate(rows, start=1)]


def _candidate_text(record: dict[str, Any]) -> str:
    return record.get("retrieval_text") or "\n".join(
        line
        for line in [
            f"STANDARD: {record.get('standard_code')}",
            f"PRODUCT: {record.get('canonical_product')}",
            f"TYPE: {record.get('standard_kind')}",
            f"TITLE: {record.get('title')}",
            f"SCOPE: {record.get('scope')}",
        ]
        if line and not line.endswith(": None")
    )


def _product_compatibility(
    canonical_product: str | None,
    record: dict[str, Any],
) -> str:
    candidate_product = record.get("canonical_product")
    compatibility = product_compatibility(canonical_product, candidate_product)
    if compatibility != "incompatible" or canonical_product is None:
        return compatibility
    query_family = FAMILY_BY_PRODUCT.get(canonical_product)
    applies_to = set(record.get("applies_to_product_families") or [])
    if query_family and query_family in applies_to:
        return "applies_to_family"
    if query_family and record.get("family") == query_family:
        return "same_family"
    return compatibility


def _description_function(description: str) -> str | None:
    text = " ".join(description.casefold().split())
    if "reverse flow" in text or "non return" in text or "non-return" in text:
        return "prevent_reverse_flow"
    if "reduce pressure" in text or "pressure reducing" in text:
        return "reduce_pressure"
    if "release air" in text or "air relief" in text:
        return "release_air"
    return None


def _load_cross_encoder(spec: ModelSpec, args: argparse.Namespace) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        str(spec.path),
        device=args.device,
        local_files_only=args.local_files_only,
    )


def _model_parameter_report(cross_encoder: Any) -> dict[str, Any]:
    model = getattr(cross_encoder, "model", None)
    total = 0
    trainable = 0
    if model is not None:
        for parameter in model.parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_percentage": trainable / total if total else 0.0,
    }


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        return


def _as_float_score(score: Any) -> float:
    if isinstance(score, list | tuple):
        if not score:
            return 0.0
        return float(score[0])
    return float(score)


def _build_suites(args: argparse.Namespace) -> dict[str, list[EvalCase]]:
    suites = {
        "old_39_regression": [
            EvalCase(
                query_id=f"old39_{index:02d}",
                query=case.query,
                expected_standard=case.expected_standard,
                family=case.family,
            )
            for index, case in enumerate(PRIMARY_BENCHMARK, start=1)
        ],
        "natural_holdout": [
            EvalCase(
                query_id=f"natural_holdout_{index:02d}",
                query=case.query,
                expected_standard=case.expected_standard,
                family=case.family,
            )
            for index, case in enumerate(HOLDOUT_CASES, start=1)
        ],
        "unseen_holdout": [
            EvalCase(
                query_id=f"unseen_holdout_{index:02d}",
                query=case.query,
                expected_standard=case.expected_standard,
                family=case.family,
            )
            for index, case in enumerate(UNSEEN_HOLDOUT_CASES, start=1)
        ],
    }
    if args.full_eval.exists():
        suites["full_200_frozen"] = [
            EvalCase(
                query_id=record["query_id"],
                query=record["query"],
                expected_standard=record["expected_standard"],
                product=record.get("expected_product"),
                family=record.get("family"),
                standard_kind=record.get("standard_kind"),
            )
            for record in _read_jsonl(args.full_eval)
        ]
    if args.product_holdout.exists():
        suites["product_aware_holdout"] = [
            EvalCase(
                query_id=record["query_id"],
                query="",
                expected_standard=record.get("expected_standard") or "",
                product=record.get("product") or record.get("expected_product"),
                description=record.get("description") or "",
                family=record.get("family"),
                standard_kind=record.get("standard_kind"),
            )
            for record in _read_jsonl(args.product_holdout)
        ]
    if args.ambiguous_eval.exists():
        suites["ambiguous_product_diagnostics"] = [
            EvalCase(
                query_id=record["query_id"],
                query="",
                product=record.get("product"),
                description=record.get("description") or "",
            )
            for record in _read_jsonl(args.ambiguous_eval)
        ]
    return suites


def _load_cached_candidate_pools(
    args: argparse.Namespace,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if args.full_baseline.exists():
        baseline = json.loads(args.full_baseline.read_text(encoding="utf-8"))
        pools["full_200_frozen"] = _candidate_pools_from_result(
            baseline.get("hybrid", {}).get("results", [])
        )
    if args.regression_hybrid.exists():
        regression = json.loads(args.regression_hybrid.read_text(encoding="utf-8"))
        pools["old_39_regression"] = _candidate_pools_from_result(
            regression.get("results", [])
        )
    return pools


def _candidate_pools_from_result(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["query_id"]: [
            {
                "rank": candidate["rank"],
                "standard_code": candidate["standard_code"],
                "title": candidate.get("title") or "",
                "rrf_score": candidate.get("rrf_score"),
                "semantic_rank": candidate.get("semantic_rank"),
                "semantic_score": candidate.get("semantic_score"),
                "bm25_rank": candidate.get("bm25_rank"),
                "bm25_score": candidate.get("bm25_score"),
            }
            for candidate in row.get("candidates", [])
        ]
        for row in rows
    }


def _available_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    candidates = [
        ModelSpec("pretrained_bge_large", "pretrained BAAI/bge-reranker-large", args.pretrained),
        ModelSpec("old_head_only", "old head-only checkpoint", args.old_head),
        ModelSpec("partial_last4", "new partial-last-4 checkpoint", args.partial4),
    ]
    available = []
    for spec in candidates:
        if spec.path.exists():
            available.append(spec)
        else:
            print(f"Skipping unavailable model artifact: {spec.label} at {spec.path}")
    return available


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _human_report(report: dict[str, Any]) -> str:
    lines = [
        "# StandardWise Partial-Last-4 Post-Training Comparison",
        "",
        f"Canonical records: {report['canonical_record_count']}",
        f"Rerank K: {report['rerank_k']}",
        "",
        "## Suite Compatibility",
    ]
    for suite_name, status in report["suite_statuses"].items():
        line = f"- {suite_name}: {status['status']} ({status['query_count']} queries)"
        if status.get("missing_standard_codes"):
            line += f"; missing={', '.join(status['missing_standard_codes'])}"
        if status.get("error"):
            line += f"; error={status['error']}"
        lines.append(line)
    lines.extend(["", "## Metrics"])
    for suite_name, model_results in report["suite_results"].items():
        lines.append(f"### {suite_name}")
        if not model_results:
            lines.append("- No compatible model results.")
            continue
        for model_key, result in model_results.items():
            if result.get("status") == ERROR:
                lines.append(f"- {model_key}: ERROR {result.get('error')}")
                continue
            if not result.get("scored"):
                lines.append(
                    f"- {model_key}: unscored diagnostics, queries={result['query_count']}"
                )
                continue
            metrics = result["metrics"]
            recall = result.get("candidate_recall", {})
            extra = ""
            if "cross_product_top1_errors" in result:
                extra = f", cross_product_top1_errors={result['cross_product_top1_errors']}"
            lines.append(
                "- "
                f"{model_key}: Hit@1={metrics['hit_at_1']}/{metrics['query_count']} "
                f"Hit@3={metrics['hit_at_3']}/{metrics['query_count']} "
                f"Hit@5={metrics['hit_at_5']}/{metrics['query_count']} "
                f"MRR={metrics['mrr']:.3f} "
                f"candidate_recall@20={recall.get('at_20_count', 0)}/{metrics['query_count']}"
                f"{extra}"
            )
            error_counts = result.get("error_counts", {})
            if error_counts:
                lines.append(f"  Error types: {json.dumps(error_counts, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained, old head-only, and partial-last-4 rerankers."
    )
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--full-eval", type=Path, default=DEFAULT_FULL_EVAL_PATH)
    parser.add_argument("--product-holdout", type=Path, default=DEFAULT_PRODUCT_HOLDOUT_PATH)
    parser.add_argument("--ambiguous-eval", type=Path, default=DEFAULT_AMBIGUOUS_EVAL_PATH)
    parser.add_argument("--full-baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--regression-hybrid", type=Path, default=DEFAULT_REGRESSION_HYBRID_PATH)
    parser.add_argument("--pretrained", type=Path, default=Path("models/bge-reranker-large"))
    parser.add_argument("--old-head", type=Path, default=Path("models/standardwise-reranker/best"))
    parser.add_argument(
        "--partial4",
        type=Path,
        default=Path("models/full-corpus-reranker-partial4/best"),
    )
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--return-k", type=int, default=5)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
