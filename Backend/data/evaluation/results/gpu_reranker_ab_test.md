# StandardWise GPU Reranker A/B Test

## Environment
- Python: `3.14.4`
- Torch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- Candidate pool source: generalized_ranking_repair_after.json final candidates; candidate pool size is limited to the preserved top-5 per query

## Baseline
- Completed: 15/15
- Timeouts: 0
- Latency: mean 557.254 ms, median 178.075 ms, p95 1919.134 ms
- Explicit-label Top-1/Top-3/Top-5/MRR: 4/4/4/1.0
- Baseline-preservation Top-1/Top-3/Top-5/MRR: 15/15/15/1.0

## GPU Results
### pretrained_bge_large
- Checkpoint: `/home/barhatedigambar_05/standardwise-training/models/bge-reranker-large`
- SHA256: `c5ae4e262c60ed2fb507ec587358285377ec36ee0a2e6da029f0534272b06d36`
- Load time: 1735.076 ms
- Device / batch size: `cuda` / 8
- Candidate pool size: 5-5 candidates
- Explicit-label Top-1/Top-3/Top-5/MRR: 4/4/4/1.0
- Baseline-preservation Top-1/Top-3/Top-5/MRR: 14/15/15/0.955556
- Warm reranker inference: mean 135.511 ms, median 122.279 ms, p95 212.783 ms
- Regression-flagged queries: 1

### partial_last4
- Checkpoint: `/home/barhatedigambar_05/standardwise-training/models/full-corpus-reranker-partial4/best`
- SHA256: `0c0768c0d34028a6ef36e1bf26a510c37e3728a2fc89dd929a28edbd667fb447`
- Load time: 1458.126 ms
- Device / batch size: `cuda` / 8
- Candidate pool size: 5-5 candidates
- Explicit-label Top-1/Top-3/Top-5/MRR: 4/4/4/1.0
- Baseline-preservation Top-1/Top-3/Top-5/MRR: 14/15/15/0.955556
- Warm reranker inference: mean 128.456 ms, median 119.372 ms, p95 177.834 ms
- Regression-flagged queries: 1

## Per-Query Comparison

| # | Query | Baseline Top 1 | GPU Top 1 | Desired rank before -> after | Outcome | Regression flags |
|---:|---|---|---|---|---|---|
| 1 | plastic pipe for underground sewage line | IS 14333: 1996 | IS 14333: 1996 | None -> None | unchanged | - |
| 2 | HDPE pipe for municipal sewage | IS 14333: 1996 | IS 14333: 1996 | None -> None | unchanged | - |
| 3 | UPVC soil/waste/rainwater building pipe | IS 13592: 1992 | IS 13592: 1992 | 1 -> 1 | unchanged | - |
| 4 | GRP potable-water pipe | IS 12709: 1994 | IS 12709: 1994 | 1 -> 1 | unchanged | - |
| 5 | GRP industrial-waste pipe | IS 14402: 1996 | IS 14402: 1996 | None -> None | unchanged | - |
| 6 | ordinary Portland cement | IS 8112: 1989 | IS 8112: 1989 | None -> None | unchanged | - |
| 7 | PPC made using calcined clay | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | None -> None | unchanged | - |
| 8 | white cement for decorative architectural finish | IS 8042: 1989 | IS 8042: 1989 | None -> None | unchanged | - |
| 9 | calcium silicate insulation at 600 C | IS 8154: 1993 | IS 8154: 1993 | 1 -> 1 | unchanged | - |
| 10 | preformed fibrous insulation for hot-water pipe | IS 9842: 1994 | IS 9842: 1994 | 1 -> 1 | unchanged | - |
| 11 | hexagonal Grade C bolt | IS 1363 (Part 1): 2002 | IS 1363 (Part 1): 2002 | None -> None | unchanged | - |
| 12 | traditional clay roofing tile | IS 654: 1992 | IS 654: 1992 | None -> None | unchanged | - |
| 13 | tile for interior flooring, not roofing | IS 1478: 1992 | IS 1478: 1992 | None -> None | unchanged | - |
| 14 | reverse-flow water valve | IS 9338: 1984 | IS 778: 1984 | None -> None | changed_unlabeled | top1_changed_vs_deterministic_baseline |
| 15 | pressure-reducing water valve | IS 9739: 1981 | IS 9739: 1981 | None -> None | unchanged | - |

## Recommendation
- Decision: **USE DETERMINISTIC PIPELINE WITHOUT RERANKER**
- Reason: pretrained_bge_large changed deterministic Top-1 on 1 diagnostic queries; pretrained_bge_large has 1 regression-flagged queries; partial_last4 changed deterministic Top-1 on 1 diagnostic queries; partial_last4 has 1 regression-flagged queries
