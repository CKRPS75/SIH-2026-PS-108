# StandardWise Partial-Last-4 Post-Training Comparison

Canonical records: 558
Rerank K: 20

## Suite Compatibility
- old_39_regression: INCOMPATIBLE_DATASET (39 queries); missing=IS 14845: 2000
- natural_holdout: COMPATIBLE (5 queries)
- unseen_holdout: COMPATIBLE (5 queries)
- full_200_frozen: COMPATIBLE (200 queries)
- product_aware_holdout: COMPATIBLE (77 queries)
- ambiguous_product_diagnostics: COMPATIBLE (3 queries)

## Metrics
### natural_holdout
- pretrained_bge_large: Hit@1=2/5 Hit@3=3/5 Hit@5=3/5 MRR=0.467 candidate_recall@20=3/5
  Error types: {"MATERIAL_APPLICATION_CONFUSION": 1, "PASS": 2, "RETRIEVAL_MISS": 2}
- old_head_only: Hit@1=2/5 Hit@3=3/5 Hit@5=3/5 MRR=0.467 candidate_recall@20=3/5
  Error types: {"MATERIAL_APPLICATION_CONFUSION": 1, "PASS": 2, "RETRIEVAL_MISS": 2}
- partial_last4: Hit@1=2/5 Hit@3=3/5 Hit@5=3/5 MRR=0.467 candidate_recall@20=3/5
  Error types: {"MATERIAL_APPLICATION_CONFUSION": 1, "PASS": 2, "RETRIEVAL_MISS": 2}
### unseen_holdout
- pretrained_bge_large: Hit@1=2/5 Hit@3=5/5 Hit@5=5/5 MRR=0.700 candidate_recall@20=5/5
  Error types: {"FUNCTION_CONFUSION": 1, "PASS": 2, "RERANKER_ERROR": 2}
- old_head_only: Hit@1=2/5 Hit@3=5/5 Hit@5=5/5 MRR=0.700 candidate_recall@20=5/5
  Error types: {"FUNCTION_CONFUSION": 1, "PASS": 2, "RERANKER_ERROR": 2}
- partial_last4: Hit@1=2/5 Hit@3=5/5 Hit@5=5/5 MRR=0.700 candidate_recall@20=5/5
  Error types: {"FUNCTION_CONFUSION": 1, "PASS": 2, "RERANKER_ERROR": 2}
### full_200_frozen
- pretrained_bge_large: Hit@1=195/200 Hit@3=199/200 Hit@5=200/200 MRR=0.985 candidate_recall@20=200/200
  Error types: {"FUNCTION_CONFUSION": 1, "MATERIAL_APPLICATION_CONFUSION": 1, "PART_OR_VARIANT_CONFUSION": 1, "PASS": 195, "RERANKER_ERROR": 2}
- old_head_only: Hit@1=195/200 Hit@3=199/200 Hit@5=200/200 MRR=0.985 candidate_recall@20=200/200
  Error types: {"FUNCTION_CONFUSION": 1, "MATERIAL_APPLICATION_CONFUSION": 1, "PART_OR_VARIANT_CONFUSION": 1, "PASS": 195, "RERANKER_ERROR": 2}
- partial_last4: Hit@1=195/200 Hit@3=199/200 Hit@5=200/200 MRR=0.985 candidate_recall@20=200/200
  Error types: {"FUNCTION_CONFUSION": 1, "MATERIAL_APPLICATION_CONFUSION": 1, "PART_OR_VARIANT_CONFUSION": 1, "PASS": 195, "RERANKER_ERROR": 2}
### product_aware_holdout
- pretrained_bge_large: Hit@1=73/75 Hit@3=74/75 Hit@5=74/75 MRR=0.980 candidate_recall@20=75/75, cross_product_top1_errors=0
  Error types: {"PASS": 73, "RERANKER_ERROR": 2}
- old_head_only: Hit@1=73/75 Hit@3=74/75 Hit@5=74/75 MRR=0.980 candidate_recall@20=75/75, cross_product_top1_errors=0
  Error types: {"PASS": 73, "RERANKER_ERROR": 2}
- partial_last4: Hit@1=73/75 Hit@3=74/75 Hit@5=74/75 MRR=0.980 candidate_recall@20=75/75, cross_product_top1_errors=0
  Error types: {"PASS": 73, "RERANKER_ERROR": 2}
### ambiguous_product_diagnostics
- pretrained_bge_large: unscored diagnostics, queries=3
- old_head_only: unscored diagnostics, queries=3
- partial_last4: unscored diagnostics, queries=3
