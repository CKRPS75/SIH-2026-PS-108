# StandardWise Valve Diagnostic Before/After

## Summary
- Before: completed=9 failed=0 gemini_failures=0 passed=6 latency mean/median/p95=2762.717/2875.247/3570.261 ms
- After: completed=9 failed=0 gemini_failures=0 passed=9 latency mean/median/p95=5744.932/2969.89/18543.27 ms

## Per Query

| Query | Description | Before Top-5 | After Top-5 | Before failure | After failure | Latency after |
|---|---|---|---|---|---|---:|
| A1 | valve that prevents water from flowing backwards | IS 9338: 1984, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 5312 (Part 1): 2004, IS 778: 1984 | IS 9338: 1984, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 5312 (Part 1): 2004, IS 778: 1984 | [] | [] | 28602.371 |
| A2 | non-return valve for water line | IS 5312 (Part 1): 2004, IS 13114: 1991, IS 9338: 1984, IS 5312 (Part 2): 1986, IS 778: 1984 | IS 5312 (Part 1): 2004, IS 13114: 1991, IS 9338: 1984, IS 5312 (Part 2): 1986, IS 778: 1984 | [] | [] | 3454.618 |
| A3 | check valve for water pipeline | IS 9338: 1984, IS 5312 (Part 1): 2004, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 778: 1984 | IS 9338: 1984, IS 5312 (Part 1): 2004, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 778: 1984 | [] | [] | 3024.014 |
| A4 | reflux valve for water line | IS 9338: 1984, IS 5312 (Part 1): 2004, IS 5312 (Part 2): 1986, IS 13114: 1991, IS 778: 1984 | IS 5312 (Part 1): 2004, IS 9338: 1984, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 778: 1984 | [] | [] | 2776.619 |
| A5 | water should flow only one way and must not come back | IS 5312 (Part 1): 2004, IS 13114: 1991, IS 9338: 1984, IS 5312 (Part 2): 1986, IS 778: 1984 | IS 9338: 1984, IS 5312 (Part 1): 2004, IS 13114: 1991, IS 5312 (Part 2): 1986, IS 778: 1984 | ['NORMALIZATION_FAILURE'] | [] | 2464.707 |
| B1 | valve to reduce downstream pressure in water supply | IS 9739: 1981, IS 1703: 2000, IS 14846: 2000, IS 9763: 2000, IS 9758: 1981 | IS 9739: 1981, IS 1703: 2000, IS 14846: 2000, IS 9763: 2000, IS 9758: 1981 | ['NORMALIZATION_FAILURE'] | [] | 2969.89 |
| B2 | pressure reducing valve for water pipeline | IS 9739: 1981, IS 9763: 2000, IS 9758: 1981, IS 1703: 2000, IS 14846: 2000 | IS 9739: 1981, IS 9763: 2000, IS 9758: 1981, IS 1703: 2000, IS 14846: 2000 | [] | [] | 3025.895 |
| B3 | water pressure regulator valve for downstream pressure control | IS 9739: 1981, IS 9758: 1981, IS 1703: 2000, IS 9763: 2000, IS 14846: 2000 | IS 9739: 1981, IS 1703: 2000, IS 9763: 2000, IS 9758: 1981, IS 14846: 2000 | ['NORMALIZATION_FAILURE'] | [] | 2969.704 |
| C1 | valve for water pipeline | IS 9739: 1981, IS 9758: 1981, IS 1703: 2000, IS 14846: 2000, IS 9763: 2000 | IS 9739: 1981, IS 9758: 1981, IS 1703: 2000, IS 14846: 2000, IS 9763: 2000 | [] | [] | 2416.566 |

## Metadata Changes
- IS 5312 (Part 1): 2004: {'product_subtype': {'stored': None, 'effective': 'check_valve'}, 'primary_subject': {'stored': 'SWING CHECK TYPE REFLUX', 'effective': 'check valve'}}
- IS 5312 (Part 2): 1986: {'canonical_product': {'stored': 'door', 'effective': 'valve'}, 'product_subtype': {'stored': None, 'effective': 'check_valve'}, 'primary_subject': {'stored': 'SWING CHECK TYPE REFLUX', 'effective': 'check valve'}, 'family': {'stored': 'door_window', 'effective': 'water_valve'}}
- IS 9739: 1981: {'product_subtype': {'stored': None, 'effective': 'pressure_reducing_valve'}, 'primary_subject': {'stored': 'PRESSURE REDUCING VALVES FOR DOMESTIC WATER SUPPLY SYSTEM', 'effective': 'pressure reducing valve'}}

## Group Metrics
- reverse_flow: {'query_count': 5, 'passed_count': 5, 'top1_values': ['IS 9338: 1984', 'IS 5312 (Part 1): 2004', 'IS 9338: 1984', 'IS 5312 (Part 1): 2004', 'IS 9338: 1984'], 'top1_consistency_count': 3, 'top3_consistency_count': 2, 'equivalent_normalized_intent_count': 5, 'failure_categories': []}
- pressure_reducing: {'query_count': 3, 'passed_count': 3, 'top1_values': ['IS 9739: 1981', 'IS 9739: 1981', 'IS 9739: 1981'], 'top1_consistency_count': 3, 'top3_consistency_count': 1, 'equivalent_normalized_intent_count': 3, 'failure_categories': []}
- generic_ambiguity: {'query_count': 1, 'passed_count': 1, 'top1_values': ['IS 9739: 1981'], 'top1_consistency_count': 1, 'top3_consistency_count': 1, 'equivalent_normalized_intent_count': 1, 'failure_categories': []}
