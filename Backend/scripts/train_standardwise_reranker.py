from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import BACKEND_ROOT, get_settings  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.bm25_service import Bm25LexicalSearchService  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.hybrid_search_service import HybridSearchService  # noqa: E402
from app.services.reranker_service import (  # noqa: E402
    CrossEncoderRerankerService,
    _resolve_model_name,
)
from app.services.reranker_training import (  # noqa: E402
    CheckpointMetrics,
    RerankerPair,
    checkpoint_metrics_to_dict,
    configure_last_n_layers_trainable,
    frozen_query_leakage,
    group_pairs_by_query,
    load_pair_examples,
    ranking_metrics,
    score_pair_groups,
    select_best_checkpoint,
    standardwise_reranker_best_path,
)
from app.services.standard_vector_index import StandardVectorIndex  # noqa: E402
from scripts.evaluate_reranked_retrieval_50 import (  # noqa: E402
    AMBIGUOUS_DIAGNOSTIC_QUERIES,
    HOLDOUT_CASES,
    HYBRID_BASELINE,
    RETURN_K,
    SEMANTIC_BASELINE,
    UNSEEN_HOLDOUT_CASES,
    _evaluate_case,
    _family_by_code,
    _metrics,
    _print_candidates,
)
from scripts.evaluate_semantic_retrieval_50 import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    PRIMARY_BENCHMARK,
    _validate_benchmark,
)

DEFAULT_TRAIN_PATH = BACKEND_ROOT / "data" / "training" / "full_corpus_train_pairs.jsonl"
DEFAULT_VALIDATION_PATH = (
    BACKEND_ROOT / "data" / "training" / "full_corpus_validation_pairs.jsonl"
)
DEFAULT_AMBIGUOUS_PATH = (
    BACKEND_ROOT / "data" / "training" / "full_corpus_ambiguous_queries.jsonl"
)
DEFAULT_CANONICAL_PATH = (
    BACKEND_ROOT / "data" / "canonical" / "parsed_standards_canonical.json"
)
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "models" / "standardwise-reranker"
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-large"
KNOWN_FAILURE_QUERY = "Portland pozzolana cement made using fly ash"
KNOWN_FAILURE_EXPECTED = "IS 1489 (Part 1): 1991"


@dataclass(frozen=True)
class PairEvaluation:
    loss: float
    pair_count: int


@dataclass(frozen=True)
class PairwiseTrainingExample:
    query_id: str
    query: str
    positive_document: str
    negative_document: str


@dataclass(frozen=True)
class TrainingReport:
    base_model: str
    device: str
    gpu_name: str | None
    trainable_parameter_count: int
    total_parameter_count: int
    trainable_percentage: float
    unfreeze_last_n: int
    epochs: int
    learning_rate: float
    train_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    mixed_precision_mode: str
    per_epoch_metrics: list[dict[str, Any]]
    best_epoch: int
    best_checkpoint: str
    best_validation_mrr: float
    best_validation_hit1: float
    best_validation_loss: float


async def main(args: argparse.Namespace) -> None:
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _set_deterministic_seed(args.seed)
    best_path = standardwise_reranker_best_path(args.output_dir)
    if args.evaluate_only:
        print("StandardWise Step 5B frozen evaluation")
        print(f"Base model: {args.model}")
        print(f"Fine-tuned model: {best_path}")
        print(f"Device: {args.device or 'auto'}")
        if not best_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned checkpoint: {best_path}")
    else:
        _prepare_output_dir(args.output_dir, overwrite=args.overwrite_output)
        print("StandardWise Step 5B cross-encoder fine-tuning")
        print(f"Base model: {args.model}")
        print(f"Output dir: {args.output_dir}")
        print(f"Device: {args.device or 'auto'}")
        print(
            "Training config: "
            f"epochs={args.epochs}, lr={args.learning_rate}, "
            f"train_batch_size={args.train_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"eval_batch_size={args.eval_batch_size}, "
            f"weight_decay={args.weight_decay}, warmup_ratio={args.warmup_ratio}, "
            f"unfreeze_last_n={args.unfreeze_last_n}, seed={args.seed}"
        )

        train_pairs = load_pair_examples(args.train)
        validation_pairs = load_pair_examples(args.validation)
        train_groups = group_pairs_by_query(train_pairs)
        validation_groups = group_pairs_by_query(validation_pairs)
        _check_frozen_leakage(train_pairs + validation_pairs)
        print(
            f"Loaded {len(train_pairs)} training pairs across {len(train_groups)} queries "
            f"and {len(validation_pairs)} validation pairs across "
            f"{len(validation_groups)} queries."
        )

        _validate_device(args.device)
        cross_encoder = _load_cross_encoder(
            args.model,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        if args.freeze_base_model:
            _freeze_base_model(cross_encoder)
            trainable_report = _parameter_report(cross_encoder)
        else:
            trainable_report = configure_last_n_layers_trainable(
                cross_encoder,
                last_n=args.unfreeze_last_n,
            )
        _print_trainable_parameters(cross_encoder)
        checkpoint_results = _train_epochs(
            cross_encoder,
            train_pairs,
            train_groups,
            validation_groups,
            args,
        )
        best = select_best_checkpoint(checkpoint_results)
        _copy_best_checkpoint(best.checkpoint_path, best_path)
        _write_training_report(
            args.output_dir,
            args,
            checkpoint_results,
            best,
            trainable_report,
            _mixed_precision_mode(args.device, args.mixed_precision),
        )
        print()
        print(f"Best checkpoint: epoch {best.epoch} -> {best_path}")

    await _run_frozen_evaluation(args, best_path)


def _train_epochs(
    cross_encoder: Any,
    train_pairs: list[RerankerPair],
    train_groups: dict[str, list[RerankerPair]],
    validation_groups: dict[str, list[RerankerPair]],
    args: argparse.Namespace,
) -> list[CheckpointMetrics]:
    import torch
    from torch.utils.data import DataLoader

    device = _torch_device(args.device)
    cross_encoder.model.to(device)
    train_examples = _pairwise_examples(train_groups)
    validation_examples = _pairwise_examples(validation_groups)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=lambda batch: batch,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in cross_encoder.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _linear_warmup_decay_lambda(warmup_steps, total_optimizer_steps),
    )
    activation_fct = torch.nn.Identity()
    mixed_precision_mode = _mixed_precision_mode(args.device, args.mixed_precision)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision_mode == "fp16")
    checkpoint_results: list[CheckpointMetrics] = []

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Training epoch {epoch}/{args.epochs}...")
        train_eval = _train_one_epoch(
            cross_encoder,
            train_dataloader,
            optimizer,
            scheduler,
            scaler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision_mode=mixed_precision_mode,
        )
        train_eval = _evaluate_pairs(
            cross_encoder,
            train_examples,
            batch_size=args.eval_batch_size,
            device=device,
            mixed_precision_mode=mixed_precision_mode,
        )
        validation_eval = _evaluate_pairs(
            cross_encoder,
            validation_examples,
            batch_size=args.eval_batch_size,
            device=device,
            mixed_precision_mode=mixed_precision_mode,
        )
        scored_validation = score_pair_groups(
            cross_encoder,
            validation_groups,
            batch_size=args.eval_batch_size,
            activation_fn=activation_fct,
        )
        rank_metrics = ranking_metrics(scored_validation)
        checkpoint_path = args.output_dir / f"epoch-{epoch}"
        metrics = CheckpointMetrics(
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            train_loss=train_eval.loss,
            validation_loss=validation_eval.loss,
            validation_hit_at_1=rank_metrics["hit_at_1_rate"],
            validation_hit_at_3=rank_metrics["hit_at_3_rate"],
            validation_hit_at_5=rank_metrics["hit_at_5_rate"],
            validation_mrr=rank_metrics["mrr"],
        )
        checkpoint_results.append(metrics)
        _save_epoch_checkpoint(
            cross_encoder,
            checkpoint_path,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            metrics=metrics,
        )
        print(
            f"Epoch {epoch}: step_train_loss={train_eval.loss:.4f}, "
            f"train_loss={metrics.train_loss:.4f}, "
            f"val_loss={metrics.validation_loss:.4f}, "
            f"val_Hit@1={metrics.validation_hit_at_1:.3f}, "
            f"val_Hit@3={metrics.validation_hit_at_3:.3f}, "
            f"val_Hit@5={metrics.validation_hit_at_5:.3f}, "
            f"val_MRR={metrics.validation_mrr:.3f}"
        )
        _print_overfitting_warning(checkpoint_results)

    return checkpoint_results


def _train_one_epoch(
    cross_encoder: Any,
    train_dataloader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    *,
    device: Any,
    gradient_accumulation_steps: int,
    mixed_precision_mode: str,
) -> PairEvaluation:
    cross_encoder.model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    pair_count = 0
    for batch_index, batch in enumerate(train_dataloader, start=1):
        with _autocast_context(mixed_precision_mode):
            positive_scores = _score_documents(cross_encoder, batch, device=device, positive=True)
            negative_scores = _score_documents(cross_encoder, batch, device=device, positive=False)
            loss = _ranknet_loss_tensor(positive_scores, negative_scores)
            scaled_loss = loss / gradient_accumulation_steps
        if mixed_precision_mode == "fp16":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss += float(loss.detach().cpu()) * len(batch)
        pair_count += len(batch)
        if _should_optimizer_step(
            batch_index,
            total_batches=len(train_dataloader),
            gradient_accumulation_steps=gradient_accumulation_steps,
        ):
            if mixed_precision_mode == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return PairEvaluation(
        loss=total_loss / pair_count if pair_count else 0.0,
        pair_count=pair_count,
    )


def _evaluate_pairs(
    cross_encoder: Any,
    examples: list[PairwiseTrainingExample],
    *,
    batch_size: int,
    device: Any,
    mixed_precision_mode: str,
) -> PairEvaluation:
    import torch
    from torch.utils.data import DataLoader

    cross_encoder.model.eval()
    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)
    total_loss = 0.0
    pair_count = 0
    with torch.no_grad():
        for batch in dataloader:
            with _autocast_context(mixed_precision_mode):
                positive_scores = _score_documents(
                    cross_encoder,
                    batch,
                    device=device,
                    positive=True,
                )
                negative_scores = _score_documents(
                    cross_encoder,
                    batch,
                    device=device,
                    positive=False,
                )
                loss = _ranknet_loss_tensor(positive_scores, negative_scores)
            total_loss += float(loss.detach().cpu()) * len(batch)
            pair_count += len(batch)
    return PairEvaluation(
        loss=total_loss / pair_count if pair_count else 0.0,
        pair_count=pair_count,
    )


def _pairwise_examples(
    groups: dict[str, list[RerankerPair]],
) -> list[PairwiseTrainingExample]:
    examples = []
    for query_id, pairs in sorted(groups.items()):
        positives = [pair for pair in pairs if pair.label == 1]
        negatives = [pair for pair in pairs if pair.label == 0]
        if len(positives) != 1:
            raise ValueError(f"{query_id} must have exactly one positive pair")
        positive = positives[0]
        for negative in negatives:
            examples.append(
                PairwiseTrainingExample(
                    query_id=query_id,
                    query=positive.query,
                    positive_document=positive.document,
                    negative_document=negative.document,
                )
            )
    return examples


def _score_documents(
    cross_encoder: Any,
    batch: list[PairwiseTrainingExample],
    *,
    device: Any,
    positive: bool,
) -> Any:
    documents = [
        example.positive_document if positive else example.negative_document for example in batch
    ]
    queries = [example.query for example in batch]
    tokenizer_kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    max_length = getattr(cross_encoder, "max_length", None)
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length
    features = cross_encoder.tokenizer(queries, documents, **tokenizer_kwargs)
    features = {key: value.to(device) for key, value in features.items()}
    output = cross_encoder.model(**features)
    logits = getattr(output, "logits", output[0] if isinstance(output, tuple) else output)
    return logits.view(-1)


def _ranknet_loss_tensor(positive_scores: Any, negative_scores: Any) -> Any:
    import torch

    return torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()


def _should_optimizer_step(
    batch_index: int,
    *,
    total_batches: int,
    gradient_accumulation_steps: int,
) -> bool:
    return (
        batch_index % gradient_accumulation_steps == 0
        or batch_index == total_batches
    )


def _linear_warmup_decay_lambda(warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining_steps = max(1, total_steps - warmup_steps)
        completed_decay_steps = max(0, step - warmup_steps)
        return max(0.0, 1.0 - (completed_decay_steps / remaining_steps))

    return schedule


def _autocast_context(mixed_precision_mode: str):
    import torch

    if mixed_precision_mode == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if mixed_precision_mode == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def _mixed_precision_mode(device: str | None, enabled: bool) -> str:
    if not enabled:
        return "off"
    import torch

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(resolved).startswith("cuda"):
        return "off"
    if not torch.cuda.is_available():
        return "off"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _torch_device(device: str | None):
    import torch

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(resolved)


def _validate_device(device: str | None) -> None:
    if device is None:
        return
    if str(device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")


def _save_epoch_checkpoint(
    cross_encoder: Any,
    checkpoint_path: Path,
    *,
    epoch: int,
    optimizer: Any,
    scheduler: Any,
    args: argparse.Namespace,
    metrics: CheckpointMetrics,
) -> None:
    import torch

    cross_encoder.save(str(checkpoint_path))
    state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_config": _training_config_dict(args),
        "validation_metrics": checkpoint_metrics_to_dict(metrics),
    }
    torch.save(state, checkpoint_path / "training_state.pt")
    (checkpoint_path / "validation_metrics.json").write_text(
        json.dumps(checkpoint_metrics_to_dict(metrics), indent=2),
        encoding="utf-8",
    )


def _training_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "train": str(args.train),
        "validation": str(args.validation),
        "canonical": str(args.canonical),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "unfreeze_last_n": args.unfreeze_last_n,
        "freeze_base_model": args.freeze_base_model,
        "seed": args.seed,
    }


async def _run_frozen_evaluation(args: argparse.Namespace, best_path: Path) -> None:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    family_by_code = _family_by_code(dataset)
    _validate_benchmark(family_by_code)

    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    hybrid = HybridSearchService(
        StandardVectorIndex(
            qdrant.client,
            BgeM3EmbeddingService(settings.embedding_model),
            settings.qdrant_collection,
            dataset_name=str(dataset["dataset_name"]),
        ),
        Bm25LexicalSearchService(dataset_name=str(dataset["dataset_name"])),
        semantic_top_k=settings.hybrid_semantic_k,
        bm25_top_k=settings.hybrid_bm25_k,
        rrf_k=settings.rrf_k,
    )
    pretrained = CrossEncoderRerankerService(
        args.model,
        batch_size=args.rerank_batch_size,
    )

    try:
        async with database.session_factory() as session:
            print()
            print("Frozen 39-query benchmark side by side")
            print(
                "Semantic baseline: "
                f"Hit@1={SEMANTIC_BASELINE['hit_at_1']} | "
                f"Hit@3={SEMANTIC_BASELINE['hit_at_3']} | "
                f"Hit@5={SEMANTIC_BASELINE['hit_at_5']} | "
                f"MRR={SEMANTIC_BASELINE['mrr']}"
            )
            print(
                "Hybrid baseline: "
                f"Hit@1={HYBRID_BASELINE['hit_at_1']} | "
                f"Hit@3={HYBRID_BASELINE['hit_at_3']} | "
                f"Hit@5={HYBRID_BASELINE['hit_at_5']} | "
                f"MRR={HYBRID_BASELINE['mrr']}"
            )

            print()
            print("Evaluating pretrained reranker...")
            await _warmup(session, hybrid, pretrained)
            pretrained_results = await _evaluate_cases(
                session,
                hybrid,
                PRIMARY_BENCHMARK,
                args.rerank_k,
                pretrained,
            )
            pretrained_holdout = await _evaluate_cases(
                session,
                hybrid,
                HOLDOUT_CASES,
                args.rerank_k,
                pretrained,
            )
            pretrained_unseen_holdout = await _evaluate_cases(
                session,
                hybrid,
                UNSEEN_HOLDOUT_CASES,
                args.rerank_k,
                pretrained,
            )
            _release_reranker(pretrained)

            fine_tuned = CrossEncoderRerankerService(
                str(best_path),
                batch_size=args.rerank_batch_size,
            )
            print()
            print("Evaluating fine-tuned reranker...")
            await _warmup(session, hybrid, fine_tuned)
            fine_tuned_results = await _evaluate_cases(
                session,
                hybrid,
                PRIMARY_BENCHMARK,
                args.rerank_k,
                fine_tuned,
            )
            fine_tuned_holdout = await _evaluate_cases(
                session,
                hybrid,
                HOLDOUT_CASES,
                args.rerank_k,
                fine_tuned,
            )
            fine_tuned_unseen_holdout = await _evaluate_cases(
                session,
                hybrid,
                UNSEEN_HOLDOUT_CASES,
                args.rerank_k,
                fine_tuned,
            )

            _print_side_by_side_metrics("Pretrained", pretrained_results)
            _print_side_by_side_metrics("Fine-tuned", fine_tuned_results)
            _print_known_failure(pretrained_results, fine_tuned_results, family_by_code)

            _print_holdout_side_by_side(
                "Frozen natural holdout queries",
                pretrained_holdout,
                fine_tuned_holdout,
            )
            _print_holdout_side_by_side(
                "Newer unseen holdout checks",
                pretrained_unseen_holdout,
                fine_tuned_unseen_holdout,
            )
            await _print_ambiguous_diagnostics(
                session,
                hybrid,
                fine_tuned,
                args.ambiguous,
                args.rerank_k,
                family_by_code,
            )
    finally:
        await database.close()
        await qdrant.close()


async def _warmup(session, hybrid, reranker) -> None:
    print()
    print("Warming up reranker, excluded from latency metrics...")
    await _evaluate_case(
        session,
        hybrid,
        reranker,
        PRIMARY_BENCHMARK[0],
        rerank_k=5,
    )


async def _evaluate_cases(
    session,
    hybrid: HybridSearchService,
    cases: list[Any],
    rerank_k: int,
    reranker: CrossEncoderRerankerService,
) -> list[Any]:
    return [
        await _evaluate_case(session, hybrid, reranker, case, rerank_k=rerank_k)
        for case in cases
    ]


def _print_holdout_side_by_side(
    title: str,
    pretrained_results: list[Any],
    fine_tuned_results: list[Any],
) -> None:
    print()
    print(f"{title}, evaluation-only")
    _print_side_by_side_metrics("Pretrained", pretrained_results)
    _print_side_by_side_metrics("Fine-tuned", fine_tuned_results)


async def _print_ambiguous_diagnostics(
    session,
    hybrid: HybridSearchService,
    fine_tuned: CrossEncoderRerankerService,
    ambiguous_path: Path,
    rerank_k: int,
    family_by_code: dict[str, str],
) -> None:
    queries = [case.query for case in AMBIGUOUS_DIAGNOSTIC_QUERIES]
    queries.extend(_read_ambiguous_queries(ambiguous_path))
    deduped_queries = list(dict.fromkeys(queries))
    print()
    print("Ambiguous diagnostics, fine-tuned Top 5, excluded from metrics:")
    for query in deduped_queries:
        hybrid_result = await hybrid.search(session, query, limit=rerank_k)
        candidates = await fine_tuned.rerank(
            session,
            query,
            hybrid_result.candidates,
            limit=RETURN_K,
        )
        print()
        print(f"Query: {query}")
        _print_candidates(candidates, family_by_code)


def _print_side_by_side_metrics(label: str, results: list[Any]) -> None:
    metrics = _metrics(results)
    print(
        f"{label}: Hit@1={metrics['hit_at_1']}/{metrics['query_count']} | "
        f"Hit@3={metrics['hit_at_3']}/{metrics['query_count']} | "
        f"Hit@5={metrics['hit_at_5']}/{metrics['query_count']} | "
        f"MRR={metrics['mrr']:.3f} | "
        f"avg_reranker_ms={metrics['average_reranker_ms']:.1f} | "
        f"median_reranker_ms={metrics['median_reranker_ms']:.1f} | "
        f"p95_reranker_ms={metrics['p95_reranker_ms']:.1f} | "
        f"avg_total_ms={metrics['average_total_ms']:.1f}"
    )


def _print_known_failure(
    pretrained_results: list[Any],
    fine_tuned_results: list[Any],
    family_by_code: dict[str, str],
) -> None:
    pretrained_result = _find_result(pretrained_results, KNOWN_FAILURE_QUERY)
    fine_tuned_result = _find_result(fine_tuned_results, KNOWN_FAILURE_QUERY)
    print()
    print("Known failure check")
    print(f"Query: {KNOWN_FAILURE_QUERY}")
    print(f"Expected: {KNOWN_FAILURE_EXPECTED}")
    print(f"Pretrained rank: {_display_rank(pretrained_result.expected_rank)}")
    print(f"Fine-tuned rank: {_display_rank(fine_tuned_result.expected_rank)}")
    print("Fine-tuned Top 5:")
    _print_candidates(fine_tuned_result.candidates, family_by_code)


def _check_frozen_leakage(pairs: list[RerankerPair]) -> None:
    frozen_queries = [case.query for case in PRIMARY_BENCHMARK]
    frozen_queries.extend(case.query for case in HOLDOUT_CASES)
    frozen_queries.extend(case.query for case in UNSEEN_HOLDOUT_CASES)
    frozen_queries.extend(case.query for case in AMBIGUOUS_DIAGNOSTIC_QUERIES)
    leakage = frozen_query_leakage(pairs, frozen_queries)
    if leakage["exact"] or leakage["near"]:
        details = "\n".join(leakage["exact"] + leakage["near"])
        raise RuntimeError(f"Training data overlaps frozen evaluation queries:\n{details}")


def _load_cross_encoder(model_name: str, *, local_files_only: bool, device: str | None) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        _resolve_model_name(model_name),
        device=device,
        local_files_only=local_files_only,
    )


def _release_reranker(reranker: CrossEncoderRerankerService) -> None:
    reranker._model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        return


def _freeze_base_model(cross_encoder: Any) -> None:
    model = getattr(cross_encoder, "model", None)
    if model is None:
        raise RuntimeError("CrossEncoder does not expose its underlying torch model")
    trainable_markers = ("classifier", "score", "classification_head")
    trainable_count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(marker in name for marker in trainable_markers)
        trainable_count += int(parameter.requires_grad)
    if trainable_count == 0:
        raise RuntimeError("No classifier/head parameters were left trainable")


def _print_trainable_parameters(cross_encoder: Any) -> None:
    report = _parameter_report(cross_encoder)
    total = int(report["total_parameters"])
    trainable = int(report["trainable_parameters"])
    ratio = float(report["trainable_percentage"])
    print(f"Trainable parameters: {trainable:,}/{total:,} ({ratio:.2%})")


def _parameter_report(cross_encoder: Any) -> dict[str, int | float]:
    model = getattr(cross_encoder, "model", None)
    if model is None:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "trainable_percentage": 0.0,
        }
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = (trainable / total) if total else 0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": ratio,
    }


def _set_deterministic_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite-output to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_training_report(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_results: list[CheckpointMetrics],
    best: CheckpointMetrics,
    parameter_report: dict[str, int | float],
    mixed_precision_mode: str,
) -> None:
    report = TrainingReport(
        base_model=args.model,
        device=str(_torch_device(args.device)),
        gpu_name=_gpu_name(args.device),
        trainable_parameter_count=int(parameter_report["trainable_parameters"]),
        total_parameter_count=int(parameter_report["total_parameters"]),
        trainable_percentage=float(parameter_report["trainable_percentage"]),
        unfreeze_last_n=args.unfreeze_last_n,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.train_batch_size * args.gradient_accumulation_steps,
        mixed_precision_mode=mixed_precision_mode,
        per_epoch_metrics=[
            checkpoint_metrics_to_dict(metrics) for metrics in checkpoint_results
        ],
        best_epoch=best.epoch,
        best_checkpoint=str(best.checkpoint_path),
        best_validation_mrr=best.validation_mrr,
        best_validation_hit1=best.validation_hit_at_1,
        best_validation_loss=best.validation_loss,
    )
    payload = asdict(report)
    payload["best_model_path"] = str(standardwise_reranker_best_path(output_dir))
    payload["selection_metric"] = "validation_mrr, then validation_hit_at_1, then validation_loss"
    (output_dir / "training_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _copy_best_checkpoint(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _print_overfitting_warning(results: list[CheckpointMetrics]) -> None:
    if len(results) < 2:
        return
    previous = results[-2]
    current = results[-1]
    validation_decreased = (
        current.validation_mrr < previous.validation_mrr
        or current.validation_hit_at_1 < previous.validation_hit_at_1
    )
    if current.train_loss < previous.train_loss and validation_decreased:
        print(
            "Possible overfitting: train loss improved while validation ranking metrics decreased."
        )


def _gpu_name(device: str | None) -> str | None:
    import torch

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(resolved).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(resolved).index
        return torch.cuda.get_device_name(0 if index is None else index)
    return None


def _read_ambiguous_queries(path: Path) -> list[str]:
    if not path.exists():
        return []
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            query = record.get("query")
            if isinstance(query, str) and query.strip():
                queries.append(query)
    return queries


def _find_result(results: list[Any], query: str) -> Any:
    for result in results:
        if result.case.query == query:
            return result
    raise ValueError(f"Missing result for {query}")


def _display_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "not found"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate the StandardWise cross-encoder reranker."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIGUOUS_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=5108)
    parser.add_argument("--device", default=None)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument(
        "--freeze-base-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train only the cross-encoder classification head. By default, training "
            "uses partial fine-tuning: classification head plus --unfreeze-last-n "
            "final encoder layers."
        ),
    )
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--return-k", type=int, default=5)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.train_batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be at least 1")
    if args.unfreeze_last_n < 1:
        raise ValueError("unfreeze_last_n must be at least 1")
    if args.return_k != RETURN_K:
        raise ValueError(f"RETURN_K is frozen at {RETURN_K} for Step 5B evaluation")
    return args


if __name__ == "__main__":
    started = perf_counter()
    asyncio.run(main(_parse_args()))
    print()
    print(f"Completed Step 5B in {(perf_counter() - started) / 60:.1f} minutes.")
