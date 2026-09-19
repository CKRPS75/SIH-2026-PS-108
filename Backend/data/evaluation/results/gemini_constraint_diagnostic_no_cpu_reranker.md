# Gemini Constraint Diagnostic: No CPU Reranker

- Base URL: `http://127.0.0.1:8001`
- Mode: `replay`
- Interpretations from: `data/evaluation/results/gemini_constraint_diagnostic_before.json`
- Client timeout: 25.0 s
- Completed queries: 15/15
- Timeout count: 0
- Latency: mean 558.341 ms, median 171.826 ms, p95 1948.942 ms
- Before latency: mean 12341.667 ms, median 12493.0 ms, p95 13397.2 ms
- Top-1 changed vs before: 7/15

## 1. pipe | plastic pipe for underground sewage line

- Top 1: IS 3589: 2001
- Top 3: IS 3589: 2001, IS 7181: 1986, IS 13382: 2004
- Before Top 1: IS 3589: 2001
- Before Top 3: IS 3589: 2001, IS 7181: 1986, IS 13382: 2004
- Gemini interpretation: product=pipe; material=['plastic']; function=['carry_sewage']; application=['sewerage']
- Constraint flags: IS 3589: 2001=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]; IS 7181: 1986=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]; IS 13382: 2004=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]
- Ambiguity: True; missing=['plastic type or polymer grade', 'pipe dimensions or diameter', 'pressure rating or stiffness class']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 5749.096 ms

## 2. pipe | HDPE pipe for municipal sewage

- Top 1: IS 14333: 1996
- Top 3: IS 14333: 1996, IS 7181: 1986, IS 3589: 2001
- Before Top 1: IS 14333: 1996
- Before Top 3: IS 14333: 1996, IS 7181: 1986, IS 3589: 2001
- Gemini interpretation: product=pipe; material=['HDPE']; function=['carry_sewage']; application=['sewerage']
- Constraint flags: IS 14333: 1996=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]; IS 7181: 1986=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]; IS 3589: 2001=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH]
- Ambiguity: True; missing=['pipe dimensions', 'pressure rating', 'pe grade']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 202.056 ms

## 3. pipe | UPVC soil/waste/rainwater building pipe

- Top 1: IS 14402: 1996
- Top 3: IS 14402: 1996, IS 14735: 1999, IS 10124 (Part 2): 1988
- Before Top 1: IS 14735: 1999
- Before Top 3: IS 14735: 1999, IS 13592: 1992, IS 10124 (Part 2): 1988
- Gemini interpretation: product=pipe; material=['upvc']; application=['soil', 'waste', 'rainwater']
- Constraint flags: IS 14402: 1996=[PRODUCT_APPLIES_TO_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_EXACT_MATCH]; IS 14735: 1999=[PRODUCT_APPLIES_TO_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_UNKNOWN]; IS 10124 (Part 2): 1988=[PRODUCT_APPLIES_TO_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_UNKNOWN]
- Ambiguity: True; missing=['pipe dimensions or diameter', 'pressure rating or wall thickness']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 171.031 ms

## 4. pipe | GRP potable-water pipe

- Top 1: IS 8008 (Part 6): 2003
- Top 3: IS 8008 (Part 6): 2003, IS 3076: 1985, IS 4985: 2000
- Before Top 1: IS 10124 (Part 2): 1988
- Before Top 3: IS 10124 (Part 2): 1988, IS 8008 (Part 6): 2003, IS 10124 (Part 7): 1988
- Gemini interpretation: product=pipe; material=['glass fibre reinforced plastic']; application=['potable water supply']
- Constraint flags: IS 8008 (Part 6): 2003=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_EXACT_MATCH]; IS 3076: 1985=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_EXACT_MATCH]; IS 4985: 2000=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_EXACT_MATCH]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 172.621 ms

## 5. pipe | GRP industrial-waste pipe

- Top 1: IS 14402: 1996
- Top 3: IS 14402: 1996, IS 7319: 1974, IS 3589: 2001
- Before Top 1: IS 7319: 1974
- Before Top 3: IS 7319: 1974, IS 3589: 2001, IS 7181: 1986
- Gemini interpretation: product=pipe; material=['glass fibre reinforced plastic']; application=['industrial waste']
- Constraint flags: IS 14402: 1996=[PRODUCT_APPLIES_TO_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_EXACT_MATCH]; IS 7319: 1974=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_UNKNOWN]; IS 3589: 2001=[PRODUCT_EXACT_PRODUCT,MATERIAL_STRONG_COMPATIBLE,APPLICATION_UNKNOWN]
- Ambiguity: True; missing=['specific type or resin system of GRP', 'pressure rating', 'diameter', 'stiffness class']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 163.643 ms

## 6. cement | ordinary Portland cement

- Top 1: IS 8112: 1989
- Top 3: IS 8112: 1989, IS 269: 1989, IS 12269: 1987
- Before Top 1: IS 8112: 1989
- Before Top 3: IS 8112: 1989, IS 269: 1989, IS 12269: 1987
- Gemini interpretation: product=cement; subtype=ordinary_portland_cement
- Constraint flags: IS 8112: 1989=[PRODUCT_EXACT_PRODUCT,SUBTYPE_EXACT_MATCH]; IS 269: 1989=[PRODUCT_EXACT_PRODUCT,SUBTYPE_EXACT_MATCH]; IS 12269: 1987=[PRODUCT_EXACT_PRODUCT,SUBTYPE_EXACT_MATCH]
- Ambiguity: True; missing=['grade of cement', 'specific type or strength class']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 158.019 ms

## 7. cement | PPC made using calcined clay

- Top 1: IS 1489 (Part 2): 1991
- Top 3: IS 1489 (Part 2): 1991, IS 10360: 1982, IS 1542: 1992
- Before Top 1: IS 1489 (Part 2): 1991
- Before Top 3: IS 1489 (Part 2): 1991, IS 10360: 1982, IS 1542: 1992
- Gemini interpretation: product=cement; subtype=portland_pozzolana_cement; material=['clay']
- Constraint flags: IS 1489 (Part 2): 1991=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_EXACT_MATCH]; IS 10360: 1982=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_UNKNOWN]; IS 1542: 1992=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_UNKNOWN]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 156.155 ms

## 8. cement | white cement for decorative architectural finish

- Top 1: IS 8042: 1989
- Top 3: IS 8042: 1989, IS 459: 1992, IS 1237: 1980
- Before Top 1: IS 8042: 1989
- Before Top 3: IS 8042: 1989, IS 459: 1992, IS 1237: 1980
- Gemini interpretation: product=cement; subtype=white; application=['decorative architectural finish']
- Constraint flags: IS 8042: 1989=[PRODUCT_EXACT_PRODUCT,APPLICATION_UNKNOWN,SUBTYPE_EXACT_MATCH]; IS 459: 1992=[PRODUCT_EXACT_PRODUCT,APPLICATION_UNKNOWN,SUBTYPE_UNKNOWN]; IS 1237: 1980=[PRODUCT_EXACT_PRODUCT,APPLICATION_UNKNOWN,SUBTYPE_UNKNOWN]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 168.86 ms

## 9. thermal insulation | calcium silicate insulation at 600 C

- Top 1: IS 11128: 1984
- Top 3: IS 11128: 1984, IS 7509: 1993, IS 13204: 1991
- Before Top 1: IS 8154: 1993
- Before Top 3: IS 8154: 1993, IS 11128: 1984, IS 9428: 1993
- Gemini interpretation: product=thermal_insulation; subtype=calcium_silicate_insulation; material=['calcium silicate']; temperature_c=600.0
- Constraint flags: IS 11128: 1984=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_UNKNOWN,TEMPERATURE_UNKNOWN]; IS 7509: 1993=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_UNKNOWN,TEMPERATURE_UNKNOWN]; IS 13204: 1991=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,SUBTYPE_UNKNOWN,TEMPERATURE_UNKNOWN]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 198.017 ms

## 10. thermal insulation | preformed fibrous insulation for hot-water pipe

- Top 1: IS 9743: 1990
- Top 3: IS 9743: 1990, IS 6598: 1972, IS 9842: 1994
- Before Top 1: IS 9842: 1994
- Before Top 3: IS 9842: 1994, IS 9428: 1993, IS 12436: 1988
- Gemini interpretation: product=thermal_insulation; subtype=preformed_fibrous_insulation; material=['fibrous']; application=['water pipeline']
- Constraint flags: IS 9743: 1990=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_PARTIAL_MATCH,SUBTYPE_UNKNOWN]; IS 6598: 1972=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_PARTIAL_MATCH,SUBTYPE_UNKNOWN]; IS 9842: 1994=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,APPLICATION_UNKNOWN,SUBTYPE_UNKNOWN]
- Ambiguity: True; missing=['specific fibrous material type such as mineral wool, glass wool, or ceramic fiber', 'dimensions or thickness']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 228.0 ms

## 11. bolt | hexagonal Grade C bolt

- Top 1: IS 10238: 2001
- Top 3: IS 10238: 2001, IS 7540: 1974, IS 4621: 1975
- Before Top 1: IS 10238: 2001
- Before Top 3: IS 10238: 2001, IS 7540: 1974, IS 4621: 1975
- Gemini interpretation: product=bolt; subtype=hexagonal_bolt; function=['structural_fastening']; grade=HEXAGONAL
- Constraint flags: IS 10238: 2001=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,SUBTYPE_UNKNOWN,GRADE_UNKNOWN]; IS 7540: 1974=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,SUBTYPE_UNKNOWN,GRADE_UNKNOWN]; IS 4621: 1975=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,SUBTYPE_UNKNOWN,GRADE_UNKNOWN]
- Ambiguity: True; missing=['thread size', 'length', 'material']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 320.305 ms

## 12. tile | traditional clay roofing tile

- Top 1: IS 654: 1992
- Top 3: IS 654: 1992, IS 13317: 1992, IS 3622: 1977
- Before Top 1: IS 13317: 1992
- Before Top 3: IS 13317: 1992, IS 654: 1992, IS 3622: 1977
- Gemini interpretation: product=tile; subtype=roofing_tile; material=['clay']; function=['roofing']; application=['roofing']
- Constraint flags: IS 654: 1992=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_EXACT_MATCH]; IS 13317: 1992=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_UNKNOWN]; IS 3622: 1977=[PRODUCT_EXACT_PRODUCT,MATERIAL_EXACT_MATCH,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_UNKNOWN]
- Ambiguity: True; missing=['specific tile design or profile', 'dimensions', 'glazing or finish requirements']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 175.45 ms

## 13. tile | tile for interior flooring, not roofing

- Top 1: IS 1478: 1992
- Top 3: IS 1478: 1992, IS 1128: 1974, IS 653: 1992
- Before Top 1: IS 3622: 1977
- Before Top 3: IS 3622: 1977, IS 13317: 1992, IS 654: 1992
- Gemini interpretation: product=tile; application=['flooring']; excluded_function=['roofing']; excluded_application=['roofing']
- Constraint flags: IS 1478: 1992=[PRODUCT_EXACT_PRODUCT,APPLICATION_EXACT_MATCH,EXCLUDED_FUNCTION_CLEAR,EXCLUDED_APPLICATION_CLEAR,EXCLUDED_SUBTYPE_CLEAR]; IS 1128: 1974=[PRODUCT_EXACT_PRODUCT,APPLICATION_EXACT_MATCH,EXCLUDED_FUNCTION_CLEAR,EXCLUDED_APPLICATION_CLEAR,EXCLUDED_SUBTYPE_CLEAR]; IS 653: 1992=[PRODUCT_EXACT_PRODUCT,APPLICATION_EXACT_MATCH,EXCLUDED_FUNCTION_CLEAR,EXCLUDED_APPLICATION_CLEAR,EXCLUDED_SUBTYPE_CLEAR]
- Ambiguity: True; missing=['tile material', 'tile dimensions', 'tile finish']
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 168.909 ms

## 14. valve | reverse-flow water valve

- Top 1: IS 9338: 1984
- Top 3: IS 9338: 1984, IS 13114: 1991, IS 778: 1984
- Before Top 1: IS 9338: 1984
- Before Top 3: IS 9338: 1984, IS 13114: 1991, IS 5312 (Part 1): 2004
- Gemini interpretation: product=valve; subtype=check_valve; function=['prevent_reverse_flow']; application=['water']
- Constraint flags: IS 9338: 1984=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_EXACT_MATCH]; IS 13114: 1991=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_EXACT_MATCH]; IS 778: 1984=[PRODUCT_EXACT_PRODUCT,FUNCTION_EXACT_MATCH,APPLICATION_EXACT_MATCH,SUBTYPE_EXACT_MATCH]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 171.133 ms

## 15. valve | pressure-reducing water valve

- Top 1: IS 9739: 1981
- Top 3: IS 9739: 1981, IS 9338: 1984, IS 13114: 1991
- Before Top 1: IS 9739: 1981
- Before Top 3: IS 9739: 1981, IS 9338: 1984, IS 13114: 1991
- Gemini interpretation: product=valve; subtype=pressure-reducing_valve; function=['pressure-reducing']; application=['water']
- Constraint flags: IS 9739: 1981=[PRODUCT_EXACT_PRODUCT,FUNCTION_UNKNOWN,APPLICATION_EXACT_MATCH,SUBTYPE_UNKNOWN]; IS 9338: 1984=[PRODUCT_EXACT_PRODUCT,FUNCTION_UNKNOWN,APPLICATION_EXACT_MATCH,SUBTYPE_UNKNOWN]; IS 13114: 1991=[PRODUCT_EXACT_PRODUCT,FUNCTION_UNKNOWN,APPLICATION_EXACT_MATCH,SUBTYPE_UNKNOWN]
- Ambiguity: False; missing=[]
- Reranker: used=False success=False device=cpu fallback=cpu_reranker_disabled inference_ms=0.0
- Total latency: 171.826 ms
