# StandardWise Live Robustness Before/After

## Summary
- Before: completed 28, gemini_failures 4, top1 {'hits': 20, 'total': 28, 'rate': 0.7143}, top3 {'hits': 25, 'total': 28, 'rate': 0.8929}, latency mean/median/p95 4064.489/2847.142/5549.773 ms
- After: completed 28, gemini_failures 7, top1 {'hits': 28, 'total': 28, 'rate': 1.0}, top3 {'hits': 28, 'total': 28, 'rate': 1.0}, latency mean/median/p95 2431.059/2743.341/3168.747 ms

## Per Query

| Group | Query | Top-1 before | Top-1 after | Expected rank before | Expected rank after | Category after | Outcome | Latency after |
|---|---|---|---|---:|---:|---|---|---:|
| grp_potable_water | GRP pipe for potable drinking water | IS 14402: 1996 | IS 12709: 1994 | 2 | 1 | None | improved | 3121.835 |
| grp_potable_water | GFRP pipe for drinking water supply | IS 5382: 1985 | IS 12709: 1994 | 2 | 1 | None | improved | 2894.611 |
| grp_potable_water | glass fibre reinforced plastic pipe for potable water | IS 8008 (Part 6): 2003 | IS 12709: 1994 | 4 | 1 | None | improved | 3069.114 |
| grp_potable_water | glass reinforced plastic pipe carrying drinking water | IS 5382: 1985 | IS 12709: 1994 | None | 1 | None | improved | 3155.748 |
| grp_industrial_waste | GRP pipe for industrial waste and non potable water | IS 12709: 1994 | IS 14402: 1996 | 3 | 1 | None | improved | 2867.204 |
| grp_industrial_waste | GFRP pipe for industrial effluent | IS 14402: 1996 | IS 14402: 1996 | 1 | 1 | None | unchanged | 2597.004 |
| grp_industrial_waste | glass fibre reinforced pipe for non-potable industrial waste | IS 14402: 1996 | IS 14402: 1996 | 1 | 1 | None | unchanged | 725.963 |
| ppc_fly_ash | Portland pozzolana cement made using fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2955.638 |
| ppc_fly_ash | PPC with fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2659.186 |
| ppc_fly_ash | fly ash based Portland pozzolana cement | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2956.47 |
| ppc_fly_ash | cement where the pozzolana source is fly ash | IS 1489 (Part 1): 1991 | IS 1489 (Part 1): 1991 | 1 | 1 | None | unchanged | 2829.869 |
| ppc_calcined_clay | Portland pozzolana cement made using calcined clay | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 2724.556 |
| ppc_calcined_clay | PPC with calcined clay | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 2834.142 |
| ppc_calcined_clay | calcined clay based PPC | IS 1489 (Part 2): 1991 | IS 1489 (Part 2): 1991 | 1 | 1 | None | unchanged | 2762.127 |
| reverse_flow_valve | valve that prevents water from flowing backwards | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 5380.409 |
| reverse_flow_valve | non-return valve for water line | IS 13114: 1991 | IS 5312 (Part 1): 2004 | 2 | 1 | None | improved | 3020.636 |
| reverse_flow_valve | check valve for water pipeline | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 2705.254 |
| reverse_flow_valve | reflux valve for water line | IS 5312 (Part 1): 2004 | IS 5312 (Part 1): 2004 | 1 | 1 | None | unchanged | 1276.937 |
| reverse_flow_valve | water should only flow one way through the valve | IS 9338: 1984 | IS 9338: 1984 | 1 | 1 | None | unchanged | 1199.134 |
| pressure_reducing_valve | pressure reducing valve for water supply | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 1042.666 |
| pressure_reducing_valve | valve to reduce downstream water pressure | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 1029.036 |
| pressure_reducing_valve | water pressure regulator valve | IS 9739: 1981 | IS 9739: 1981 | 1 | 1 | None | unchanged | 1030.533 |
| hdpe_sewage | HDPE pipe for municipal sewage system | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 1057.238 |
| hdpe_sewage | high density polyethylene sewage pipe | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 1047.312 |
| hdpe_sewage | municipal sewer pipe made from HDPE | IS 14333: 1996 | IS 14333: 1996 | 1 | 1 | None | unchanged | 3175.747 |
| opc_ambiguity | ordinary Portland cement | IS 8112: 1989 | IS 8112: 1989 | 1 | 1 | None | unchanged | 2794.89 |
| interior_flooring_negation | tile for interior flooring, not roofing | IS 3622: 1977 | IS 1478: 1992 | 2 | 1 | None | improved | 2571.45 |
| interior_flooring_negation | floor tile for indoor use and not roof application | IS 3622: 1977 | IS 1478: 1992 | 4 | 1 | None | improved | 2584.932 |

## Paraphrase Consistency
- grp_potable_water: 1/4 equivalent intents, 4/4 expected Top-1, 2/4 Top-3 consistency
- grp_industrial_waste: 1/3 equivalent intents, 3/3 expected Top-1, 1/3 Top-3 consistency
- ppc_fly_ash: 3/4 equivalent intents, 4/4 expected Top-1, 2/4 Top-3 consistency
- ppc_calcined_clay: 1/3 equivalent intents, 3/3 expected Top-1, 1/3 Top-3 consistency
- reverse_flow_valve: 3/5 equivalent intents, 5/5 expected Top-1, 2/5 Top-3 consistency
- pressure_reducing_valve: 2/3 equivalent intents, 3/3 expected Top-1, 1/3 Top-3 consistency
- hdpe_sewage: 2/3 equivalent intents, 3/3 expected Top-1, 2/3 Top-3 consistency
- opc_ambiguity: 1/1 equivalent intents, 1/1 expected Top-1, 1/1 Top-3 consistency
- interior_flooring_negation: 1/2 equivalent intents, 2/2 expected Top-1, 1/2 Top-3 consistency
