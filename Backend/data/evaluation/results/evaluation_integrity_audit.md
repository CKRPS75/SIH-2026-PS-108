# StandardWise Reranker Evaluation Integrity Audit

Final conclusion: MODEL_CHANGED_AND_RANKINGS_CHANGED
Frozen metrics trustworthy: True

## Checkpoint Hashes
- pretrained_bge_large: model.safetensors size=2239618772 sha256=c5ae4e262c60ed2fb507ec587358285377ec36ee0a2e6da029f0534272b06d36
- old_head_only: model.safetensors size=2239614572 sha256=2f5bd99687e0f5e61dc35556d89e5af6d21205f5da0407aa5c682bd99fc2382a
- partial_last4: model.safetensors size=2239614572 sha256=0c0768c0d34028a6ef36e1bf26a510c37e3728a2fc89dd929a28edbd667fb447

## Parameter Delta Summary
- pretrained_bge_large_vs_old_head_only: tensors=393 nonzero=4 max=0.0020243302 mean=7.8413284e-07
  - classification_or_scoring_head: tensors=4 nonzero=4 max=0.0020243302 mean=0.00041787439
  - final_transformer_layer: tensors=16 nonzero=0 max=0 mean=0
  - fourth_from_last_transformer_layer: tensors=16 nonzero=0 max=0 mean=0
  - early_frozen_transformer_layer: tensors=16 nonzero=0 max=0 mean=0
- pretrained_bge_large_vs_partial_last4: tensors=393 nonzero=68 max=0.0013977811 mean=1.1150621e-05
  - classification_or_scoring_head: tensors=4 nonzero=4 max=0.001263828 mean=0.00012974623
  - final_transformer_layer: tensors=16 nonzero=16 max=0.0013977811 mean=0.00012933727
  - fourth_from_last_transformer_layer: tensors=16 nonzero=16 max=0.001300931 mean=0.00011814532
  - early_frozen_transformer_layer: tensors=16 nonzero=0 max=0 mean=0
- old_head_only_vs_partial_last4: tensors=393 nonzero=68 max=0.0022254493 mean=1.1740522e-05
  - classification_or_scoring_head: tensors=4 nonzero=4 max=0.0022254493 mean=0.00044411199
  - final_transformer_layer: tensors=16 nonzero=16 max=0.0013977811 mean=0.00012933727
  - fourth_from_last_transformer_layer: tensors=16 nonzero=16 max=0.001300931 mean=0.00011814532
  - early_frozen_transformer_layer: tensors=16 nonzero=0 max=0 mean=0

## Model Loading Verification
Fresh model instance per label: True
Score cache present: False
- pretrained_bge_large: requested=models/bge-reranker-large resolved=/home/barhatedigambar_05/standardwise-training/models/bge-reranker-large weight=models/bge-reranker-large/model.safetensors tokenizer=XLMRobertaTokenizerFast model=XLMRobertaForSequenceClassification object=127254446211488
- old_head_only: requested=models/standardwise-reranker/best resolved=/home/barhatedigambar_05/standardwise-training/models/standardwise-reranker/best weight=models/standardwise-reranker/best/model.safetensors tokenizer=XLMRobertaTokenizerFast model=XLMRobertaForSequenceClassification object=127254447502544
- partial_last4: requested=models/full-corpus-reranker-partial4/best resolved=/home/barhatedigambar_05/standardwise-training/models/full-corpus-reranker-partial4/best weight=models/full-corpus-reranker-partial4/best/model.safetensors tokenizer=XLMRobertaTokenizerFast model=XLMRobertaForSequenceClassification object=127254447505104

## Raw Score Deltas
Audited pair count: 80
- head_only_minus_pretrained: mean_abs=0.68678397 median_abs=0.65881705 max_abs=1.288928 identical=0
- partial_last4_minus_pretrained: mean_abs=0.35127499 median_abs=0.22357619 max_abs=1.2964115 identical=0
- partial_last4_minus_head_only: mean_abs=0.88148604 median_abs=0.83162531 max_abs=1.8540661 identical=0

## Rank Changes
- compatible_scored_query_count: 285
- head_only_top1_changed_vs_pretrained: 0
- partial_last4_top1_changed_vs_pretrained: 0
- partial_last4_top3_ordering_changed_vs_pretrained: 3
- partial_last4_top5_ordering_changed_vs_pretrained: 7
- partial_last4_any_k20_ordering_changed_vs_pretrained: 30
- partial_last4_spearman_mean: 0.9975781559161061
- head_only_spearman_mean: 0.9993932198918348
- ranking_status: RANKING_CHANGED

## Benchmark Compatibility
- old_39_regression: INCOMPATIBLE_DATASET (39 queries) missing=IS 14845: 2000
- natural_holdout: COMPATIBLE (5 queries)
- unseen_holdout: COMPATIBLE (5 queries)
- full_200_frozen: COMPATIBLE (200 queries)
- product_aware_holdout: COMPATIBLE (77 queries)
- ambiguous_product_diagnostics: COMPATIBLE (3 queries)

## Retrieval Misses (2)
- natural_holdout:natural_holdout_02 expected=IS 9842: 1994 cutoff=20 source=file_bm25
- natural_holdout:natural_holdout_03 expected=IS 8041: 1990 cutoff=20 source=file_bm25

## Current Error Audit
- part_variant_8008: expected=IS 8008 (Part 7): 2003 top1=IS 8008 (Part 4): 2003 pool_rank=1 pretrained_rank=5 head_rank=5 partial_rank=5 class=PART_VARIANT_CONFUSION
- function_12709_14402: expected=IS 12709: 1994 top1=IS 14402: 1996 pool_rank=2 pretrained_rank=2 head_rank=2 partial_rank=2 class=FUNCTION_CONFUSION
- material_6760_1365: expected=IS 6760: 1972 top1=IS 1365: 1978 pool_rank=1 pretrained_rank=2 head_rank=2 partial_rank=2 class=MATERIAL_APPLICATION_CONFUSION
- product_aware_1626: expected=IS 1626: 1984 top1=IS 459: 1992 pool_rank=1 pretrained_rank=2 head_rank=2 partial_rank=2 class=OTHER_RERANKER_ERROR
- product_aware_1703: expected=IS 1703: 2000 top1=IS 2963: 1979 pool_rank=1 pretrained_rank=6 head_rank=6 partial_rank=7 class=OTHER_RERANKER_ERROR
