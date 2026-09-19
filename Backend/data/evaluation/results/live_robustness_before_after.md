# StandardWise Live Robustness Before/After

## Summary
- Before: completed 28, gemini_failures 4, top1 {'hits': 20, 'total': 28, 'rate': 0.7143}, top3 {'hits': 25, 'total': 28, 'rate': 0.8929}, latency mean/median/p95 4064.489/2847.142/5549.773 ms
- After: completed 28, gemini_failures 3, top1 {'hits': 28, 'total': 28, 'rate': 1.0}, top3 {'hits': 28, 'total': 28, 'rate': 1.0}, latency mean/median/p95 2348.405/2712.801/3079.51 ms

## Per Query

| Group | Query | Top-1 before | Top-1 after | Expected rank before | Expected rank after | Category after | Outcome | Latency after |
|---|---|---|---|---:|---:|---|---|---:|
| grp_potable_water | GRP pipe for potable drinking water | IS 14402: 1996 | IS 12709: 1994 | 2 | 1 | None | improved | 2967.343 |
| grp_potable_water | GFRP pipe for drinking water supply | IS 5382: 1985 | IS 12709: 1994 | 2 | 1 | None | improved | 2651.625 |
| grp_potable_water | glass fibre reinforced plastic pipe for potable water | IS 8008 (Part 6): 2003 | IS 12709: 1994 | 4 | 1 | None | improved | 2862.094 |
| grp_potable_water | glass reinforced plastic pipe carrying drinking water | IS 5382: 1985 | IS 12709: 1994 | None | 1 | None | improved | 3092.74 |
| grp_industrial_waste | GRP pipe for industrial waste and non potable water | IS 12709: 1994 | IS 14402: 1996 | 3 | 1 | None | improved | 2892.285 |
| grp_industrial_waste | GFRP pipe for industrial effluent | IS 14402: 1996 | IS 14402: 1996 | 1 | 1 | None | unchanged | 2472.766 |
| grp_industrial_waste | glass fibre reinforced pipe for non-potable industrial waste | IS 14402: 1996 | IS 14402: 1996 | 1 | 1 | None | unchanged | 2883.677 |
| ppc_fly_ash | Portland pozzolana cement made using fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2887.331 |
| ppc_fly_ash | PPC with fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2737.946 |
| ppc_fly_ash | fly ash based Portland pozzolana cement | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2726.923 |
| ppc_fly_ash | cement where the pozzolana source is fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2953.299 |
| ppc_calcined_clay | Portland pozzolana cement made using calcined clay | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 2698.679 |
| ppc_calcined_clay | PPC with calcined clay | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 2736.322 |
| ppc_calcined_clay | calcined clay based PPC | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 3868.353 |
| reverse_flow_valve | valve that prevents water from flowing backwards | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 1059.061 |
| reverse_flow_valve | non-return valve for water line | IS 13114: 1991 | IS 5312 (Part 1): 2004 | 2 | 1 | None | improved | 720.446 |
| reverse_flow_valve | check valve for water pipeline | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 734.329 |
| reverse_flow_valve | reflux valve for water line | IS 5312 (Part 1): 2004 | IS 5312 (Part 1): 2004 | 1 | 1 | None | unchanged | 714.83 |
| reverse_flow_valve | water should only flow one way through the valve | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 3048.152 |
| pressure_reducing_valve | pressure reducing valve for water supply | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 2618.272 |
| pressure_reducing_valve | valve to reduce downstream water pressure | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 2688.29 |
| pressure_reducing_valve | water pressure regulator valve | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 2599.019 |
| hdpe_sewage | HDPE pipe for municipal sewage system | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 2526.251 |
| hdpe_sewage | high density polyethylene sewage pipe | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 2816.365 |
| hdpe_sewage | municipal sewer pipe made from HDPE | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 3054.941 |
| opc_ambiguity | ordinary Portland cement | IS 8112: 1989 | IS 8112: 1989 | 1 | 1 | None | unchanged | 907.135 |
| interior_flooring_negation | tile for interior flooring, not roofing | IS 3622: 1977 | IS 1478: 1992 | 2 | 1 | None | improved | 774.796 |
| interior_flooring_negation | floor tile for indoor use and not roof application | IS 3622: 1977 | IS 1478: 1992 | 4 | 1 | None | improved | 1062.079 |

## Paraphrase Consistency
- grp_potable_water: 1/4 equivalent intents, 4/4 expected Top-1, 2/4 Top-3 consistency
- grp_industrial_waste: 1/3 equivalent intents, 3/3 expected Top-1, 1/3 Top-3 consistency
- ppc_fly_ash: 3/4 equivalent intents, 4/4 expected Top-1, 2/4 Top-3 consistency
- ppc_calcined_clay: 2/3 equivalent intents, 3/3 expected Top-1, 2/3 Top-3 consistency
- reverse_flow_valve: 2/5 equivalent intents, 5/5 expected Top-1, 2/5 Top-3 consistency
- pressure_reducing_valve: 1/3 equivalent intents, 3/3 expected Top-1, 1/3 Top-3 consistency
- hdpe_sewage: 1/3 equivalent intents, 3/3 expected Top-1, 2/3 Top-3 consistency
- opc_ambiguity: 1/1 equivalent intents, 1/1 expected Top-1, 1/1 Top-3 consistency
- interior_flooring_negation: 1/2 equivalent intents, 2/2 expected Top-1, 1/2 Top-3 consistency
