# StandardWise Reranker Fit Audit

## Final Classification

- Fit classification: **INCONCLUSIVE**
- Fit confidence: **medium**
- Production decision: **DO_NOT_ENABLE_FINE_TUNED_RERANKER_IN_PRODUCTION**
- Production decision confidence: **high**

The preserved evidence does not show a useful production-quality improvement over the pretrained BGE reranker. It also does not cleanly prove classic overfitting or underfitting: validation metrics are high and stable, epoch 2 slightly worsens loss, and external suite metrics are unchanged versus pretrained. The best-supported classification is inconclusive fit with no demonstrated production value.

## Key Evidence

- Observed training metrics cover 2 epochs; best_epoch=1.
- Fine-tuned minus pretrained Hit@1 delta across compatible scored suites: 0 over 285 scored queries.
- Fine-tuned minus pretrained weighted MRR delta: 0.0.
- Validation loss last-minus-first: 0.0015304613829406621.
- Exact train/validation query or pair leakage count: 0.
- Validation standards overlap training standards, so the validation split primarily tests query paraphrase generalization rather than standard-level OOD generalization.

## Artifact Inventory

- Local model directories: bge-reranker-large, standardwise-reranker
- Post-training comparison references:
  - pretrained_bge_large: `models/bge-reranker-large` (exists here: True)
  - old_head_only: `models/standardwise-reranker/best` (exists here: True)
  - partial_last4: `models/full-corpus-reranker-partial4/best` (exists here: False)

## Dataset And Leakage

- Train pairs: 6965 (1393 positive, 5572 negative)
- Validation pairs: 1230 (246 positive, 984 negative)
- Train/validation query-id overlap: 0
- Train/validation normalized-query overlap: 0
- Train/validation exact pair overlap: 0
- Train/validation positive-standard overlap: 158 of 165 validation standards
- Negative-as-positive violations: 0

Positive-standard overlap is not counted as direct leakage, but it weakens claims about standard-level OOD generalization.

## Training Curve

- Selection metric: validation_mrr, then validation_hit_at_1
- Best epoch: 1
- Observed epoch count: 2

| Epoch | Train Loss | Validation Loss | Val Hit@1 | Val MRR | Selected | Checkpoint Exists Here |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.194346 | 0.231792 | 0.966667 | 0.980556 | True | True |
| 2 | 0.195807 | 0.233322 | 0.966667 | 0.980556 | False | True |

## Pretrained Vs Fine-Tuned

- Aggregate scored queries: 285
- Hit@1 delta: 0
- Weighted MRR delta: 0.0

| Suite | Status | Scored | Pretrained Hit@1 | Fine-Tuned Hit@1 | Hit@1 Delta | Pretrained MRR | Fine-Tuned MRR | MRR Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| natural_holdout | COMPATIBLE | True | 2 | 2 | 0 | 0.4666666666666667 | 0.4666666666666667 | 0.0 |
| unseen_holdout | COMPATIBLE | True | 2 | 2 | 0 | 0.7 | 0.7 | 0.0 |
| full_200_frozen | COMPATIBLE | True | 195 | 195 | 0 | 0.9851666666666666 | 0.9851666666666666 | 0.0 |
| product_aware_holdout | COMPATIBLE | True | 73 | 73 | 0 | 0.98 | 0.98 | 0.0 |
| ambiguous_product_diagnostics | COMPATIBLE | False | None | None | None | None | None | None |

## Domain Weaknesses

- Existing comparison errors remain concentrated in retrieval misses, function confusion, material/application confusion, and part/variant confusion.
- Product-aware holdout remains high, but the fine-tuned model did not improve the preserved product-aware metrics over pretrained BGE.
- Ambiguous product diagnostics are compatibility-checked but unscored, so they cannot prove ranking improvement.

## Score Distributions

Score distributions below are from returned top-5 candidates in the preserved comparison artifact, not a full-corpus score sweep.

| Model | Count | Min | Median | Mean | Max | Std Dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old_head_only | 1450 | 8.035521750571206e-05 | 0.5784608721733093 | 0.5341633922883996 | 0.9999322891235352 | 0.4048676728682164 |
| partial_last4 | 1450 | 8.873332262737677e-05 | 0.7606242895126343 | 0.6200127465908409 | 0.9999352693557739 | 0.36446397699043426 |
| pretrained_bge_large | 1450 | 0.0001110024459194392 | 0.752001941204071 | 0.616807487127351 | 0.9999150037765503 | 0.37031732759439234 |

## Integrity Report Signals

- final_conclusion: MODEL_CHANGED_AND_RANKINGS_CHANGED
- frozen_metrics_trustworthy: True
- compatible_scored_query_count: 285
- partial_last4_top1_changed_vs_pretrained: 0
- partial_last4_top3_ordering_changed_vs_pretrained: 3
- partial_last4_top5_ordering_changed_vs_pretrained: 7
- partial_last4_any_k20_ordering_changed_vs_pretrained: 30
- partial_last4_spearman_mean: 0.9975781559161061
- ranking_status: RANKING_CHANGED

## Required Decision

Do not enable the fine-tuned reranker in production from this evidence. The latency/runtime decision should remain separate from fit quality: this audit only says the preserved fine-tuning artifacts do not demonstrate ranking-quality improvement over pretrained BGE.
