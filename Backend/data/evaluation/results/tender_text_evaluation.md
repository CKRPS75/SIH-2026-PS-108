# Tender Text Evaluation

- Endpoint: `http://127.0.0.1:8000/api/v1/tenders/analyze-text`
- Cases: 8
- Completed: 8
- Failed: 0
- Mean latency ms: 3811.1275874980493
- Median latency ms: 3586.345399991842

| Case | Items | Expected Top-1 | Hit | Top-1 | Top-3 | Latency ms |
| --- | ---: | --- | --- | --- | --- | ---: |
| UPVC drainage tender | 1 | IS 13592 | True | IS 13592: 1992 | [["IS 13592: 1992", "IS 14182: 1994", "IS 12818: 1992"]] | 4144.896999991033 |
| OPC 43 tender | 1 | IS 8112 | True | IS 8112: 1989 | [["IS 8112: 1989", "IS 269: 1989", "IS 12269: 1987"]] | 2556.2279999721795 |
| PPC fly ash tender | 1 | IS 1489 (Part 1) | True | IS 1489 (Part 1): 1991 | [["IS 1489 (Part 1): 1991", "IS 12330: 1988", "IS 8112: 1989"]] | 5376.185000000987 |
| HDPE sewage pipe tender | 1 | None | None | IS 14333: 1996 | [["IS 14333: 1996", "IS 8008 (Part 6): 2003", "IS 4984: 1995"]] | 4555.760599963833 |
| Grade C bolt tender | 1 | IS 1363 (Part 1) | True | IS 1363 (Part 1): 2002 | [["IS 1363 (Part 1): 2002", "IS 1364 (Part 1): 2002", "IS 10238: 2001"]] | 2199.3770000408404 |
| Pressure-reducing valve tender | 1 | None | None | IS 9739: 1981 | [["IS 9739: 1981", "IS 9763: 2000", "IS 1703: 2000"]] | 3027.7937999926507 |
| Roofing flooring negative case | 1 | None | None | IS 1478: 1992 | [["IS 1478: 1992", "IS 1128: 1974", "IS 13317: 1992"]] | 2701.871700002812 |
| Multi-item tender | 3 | None | None | IS 13592: 1992, IS 8112: 1989, IS 1363 (Part 1): 2002 | [["IS 13592: 1992", "IS 14182: 1994", "IS 12818: 1992"], ["IS 8112: 1989", "IS 269: 1989", "IS 12269: 1987"], ["IS 1363 (Part 1): 2002", "IS 1364 (Part 1): 2002", "IS 10238: 2001"]] | 5926.907600020058 |
