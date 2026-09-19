# Gemini Constraint Diagnostic Before/After

## plastic pipe for underground sewage line

- Before: top3=['IS 3589: 2001 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 7181: 1986 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 13382: 2004 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None']; ambiguity=True missing=['plastic type or polymer grade', 'pipe dimensions or diameter', 'pressure rating or stiffness class']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=11749
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25055 ms)

## HDPE pipe for municipal sewage

- Before: top3=['IS 14333: 1996 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 7181: 1986 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 3589: 2001 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None']; ambiguity=True missing=['pipe dimensions', 'pressure rating', 'pe grade']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12700
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25005 ms)

## UPVC soil/waste/rainwater building pipe

- Before: top3=['IS 14735: 1999 flags=[PRODUCT_ADJACENT,MATERIAL_MATCH,APPLICATION_CONTRADICTION] ce=None final=None', 'IS 13592: 1992 flags=[PRODUCT_COMPATIBLE,MATERIAL_CONTRADICTION,APPLICATION_MATCH] ce=None final=None', 'IS 10124 (Part 2): 1988 flags=[PRODUCT_ADJACENT,MATERIAL_MATCH,APPLICATION_CONTRADICTION] ce=None final=None']; ambiguity=True missing=['pipe dimensions or diameter', 'pressure rating or wall thickness']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12809
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25003 ms)

## GRP potable-water pipe

- Before: top3=['IS 10124 (Part 2): 1988 flags=[PRODUCT_ADJACENT,MATERIAL_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 8008 (Part 6): 2003 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 10124 (Part 7): 1988 flags=[PRODUCT_ADJACENT,MATERIAL_MATCH,APPLICATION_MATCH] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=13046
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25014 ms)

## GRP industrial-waste pipe

- Before: top3=['IS 7319: 1974 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION] ce=None final=None', 'IS 3589: 2001 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION] ce=None final=None', 'IS 7181: 1986 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION] ce=None final=None']; ambiguity=True missing=['specific type or resin system of GRP', 'pressure rating', 'diameter', 'stiffness class']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12493
- After: top3=['IS 14402: 1996 flags=[] ce=None final=0.03278688524590164', 'IS 7319: 1974 flags=[] ce=None final=0.030117753623188408', 'IS 3989: 1984 flags=[] ce=None final=0.02904040404040404']; ambiguity=False missing=[]; gemini_success=False gemini_fallback=ValueError reranker_success=False reranker_timeout=True reranker_device=cpu latency_ms=10336

## ordinary Portland cement

- Before: top3=['IS 8112: 1989 flags=[PRODUCT_COMPATIBLE,SUBTYPE_MATCH] ce=None final=None', 'IS 269: 1989 flags=[PRODUCT_COMPATIBLE,SUBTYPE_MATCH] ce=None final=None', 'IS 12269: 1987 flags=[PRODUCT_COMPATIBLE,SUBTYPE_MATCH] ce=None final=None']; ambiguity=True missing=['grade of cement', 'specific type or strength class']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=10643
- After: top3=['IS 8112: 1989 flags=[] ce=None final=0.03252247488101534', 'IS 269: 1989 flags=[] ce=None final=0.032266458495966696', 'IS 8042: 1989 flags=[] ce=None final=0.03128054740957967']; ambiguity=False missing=[]; gemini_success=False gemini_fallback=ValueError reranker_success=False reranker_timeout=True reranker_device=cpu latency_ms=10459

## PPC made using calcined clay

- Before: top3=['IS 1489 (Part 2): 1991 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_MATCH] ce=None final=None', 'IS 10360: 1982 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 1542: 1992 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12387
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25005 ms)

## white cement for decorative architectural finish

- Before: top3=['IS 8042: 1989 flags=[PRODUCT_COMPATIBLE,APPLICATION_CONTRADICTION,SUBTYPE_MATCH] ce=None final=None', 'IS 459: 1992 flags=[PRODUCT_COMPATIBLE,APPLICATION_CONTRADICTION,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 1237: 1980 flags=[PRODUCT_COMPATIBLE,APPLICATION_CONTRADICTION,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=10749
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25013 ms)

## calcium silicate insulation at 600 C

- Before: top3=['IS 8154: 1993 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_MATCH] ce=None final=None', 'IS 11128: 1984 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 9428: 1993 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,SUBTYPE_MATCH] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=11641
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25015 ms)

## preformed fibrous insulation for hot-water pipe

- Before: top3=['IS 9842: 1994 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 9428: 1993 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 12436: 1988 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,APPLICATION_CONTRADICTION,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=True missing=['specific fibrous material type such as mineral wool, glass wool, or ceramic fiber', 'dimensions or thickness']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12504
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25003 ms)

## hexagonal Grade C bolt

- Before: top3=['IS 10238: 2001 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,SUBTYPE_CONTRADICTION,GRADE_UNKNOWN] ce=None final=None', 'IS 7540: 1974 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,SUBTYPE_CONTRADICTION,GRADE_UNKNOWN] ce=None final=None', 'IS 4621: 1975 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,SUBTYPE_CONTRADICTION,GRADE_UNKNOWN] ce=None final=None']; ambiguity=True missing=['thread size', 'length', 'material']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12421
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25008 ms)

## traditional clay roofing tile

- Before: top3=['IS 13317: 1992 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 654: 1992 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_MATCH] ce=None final=None', 'IS 3622: 1977 flags=[PRODUCT_COMPATIBLE,MATERIAL_MATCH,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=True missing=['specific tile design or profile', 'dimensions', 'glazing or finish requirements']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12609
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25009 ms)

## tile for interior flooring, not roofing

- Before: top3=['IS 3622: 1977 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 13317: 1992 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None', 'IS 654: 1992 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH] ce=None final=None']; ambiguity=True missing=['tile material', 'tile dimensions', 'tile finish']; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=12350
- After: ERROR The request was canceled due to the configured HttpClient.Timeout of 25 seconds elapsing. (25005 ms)

## reverse-flow water valve

- Before: top3=['IS 9338: 1984 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_MATCH] ce=None final=None', 'IS 13114: 1991 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_MATCH] ce=None final=None', 'IS 5312 (Part 1): 2004 flags=[PRODUCT_COMPATIBLE,FUNCTION_MATCH,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=13225
- After: top3=['IS 9338: 1984 flags=[] ce=None final=0.032266458495966696', 'IS 13114: 1991 flags=[] ce=None final=0.03225806451612903', 'IS 9739: 1981 flags=[] ce=None final=0.03131881575727918']; ambiguity=False missing=[]; gemini_success=False gemini_fallback=ValueError reranker_success=False reranker_timeout=True reranker_device=cpu latency_ms=16619

## pressure-reducing water valve

- Before: top3=['IS 9739: 1981 flags=[PRODUCT_COMPATIBLE,FUNCTION_CONTRADICTION,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 9338: 1984 flags=[PRODUCT_COMPATIBLE,FUNCTION_CONTRADICTION,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None', 'IS 13114: 1991 flags=[PRODUCT_COMPATIBLE,FUNCTION_CONTRADICTION,APPLICATION_MATCH,SUBTYPE_CONTRADICTION] ce=None final=None']; ambiguity=False missing=[]; gemini_success=True gemini_fallback=None reranker_success=None reranker_timeout=None reranker_device=None latency_ms=13799
- After: top3=['IS 9739: 1981 flags=[] ce=None final=0.03278688524590164', 'IS 13114: 1991 flags=[] ce=None final=0.03200204813108039', 'IS 9338: 1984 flags=[] ce=None final=0.031754032258064516']; ambiguity=False missing=[]; gemini_success=False gemini_fallback=ValueError reranker_success=False reranker_timeout=True reranker_device=cpu latency_ms=10748
