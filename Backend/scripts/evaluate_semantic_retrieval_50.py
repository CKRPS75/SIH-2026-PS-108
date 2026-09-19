from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.standards import canonicalize_standard_id  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.services.clients import QdrantClientService  # noqa: E402
from app.services.embedding_service import BgeM3EmbeddingService  # noqa: E402
from app.services.standard_vector_index import SemanticCandidate, StandardVectorIndex  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "data" / "standardwise_50_pilot_dataset.json"


@dataclass(frozen=True)
class BenchmarkCase:
    family: str
    query: str
    expected_standard: str


@dataclass(frozen=True)
class DiagnosticCase:
    query: str
    expected_standard: str | None = None


PRIMARY_BENCHMARK = [
    BenchmarkCase("tile", "tiles for lining an irrigation canal", "IS 3367: 1993"),
    BenchmarkCase("tile", "hand made burnt clay tiles for a terrace", "IS 2690 (Part 2): 1992"),
    BenchmarkCase("tile", "machine made burnt clay tiles for terracing", "IS 2690 (Part 1): 1993"),
    BenchmarkCase("tile", "hollow clay filler tiles for a roof", "IS 3951 (Part 1): 1975"),
    BenchmarkCase("tile", "structural hollow clay tiles for flooring", "IS 3951 (Part 2): 1975"),
    BenchmarkCase("tile", "clay floor tiles", "IS 1478: 1992"),
    BenchmarkCase("tile", "Mangalore pattern clay roofing tiles", "IS 654: 1992"),
    BenchmarkCase("tile", "clay tiles for a roof ridge", "IS 1464: 1992"),
    BenchmarkCase("tile", "handmade country roofing tiles", "IS 13317: 1992"),
    BenchmarkCase("pipe_water_drainage", "UPVC casing pipe for a borewell", "IS 12818: 1992"),
    BenchmarkCase("pipe_water_drainage", "steel casing tube for a water well", "IS 4270: 2001"),
    BenchmarkCase(
        "pipe_water_drainage",
        "plastic pipe for soil and waste discharge inside a building",
        "IS 13592: 1992",
    ),
    BenchmarkCase("pipe_water_drainage", "HDPE pipe for carrying sewage", "IS 14333: 1996"),
    BenchmarkCase(
        "pipe_water_drainage",
        "GRP pressure pipe for potable drinking water",
        "IS 12709: 1994",
    ),
    BenchmarkCase(
        "pipe_water_drainage",
        "GRP pipe for industrial wastewater and non potable water",
        "IS 14402: 1996",
    ),
    BenchmarkCase(
        "pipe_water_drainage",
        "large diameter welded steel pipe for water and sewage",
        "IS 3589: 2001",
    ),
    BenchmarkCase(
        "thermal_insulation",
        "calcium silicate insulation for equipment operating near 600 degrees C",
        "IS 8154: 1993",
    ),
    BenchmarkCase(
        "thermal_insulation",
        "calcium silicate insulation required for a surface near 900 degrees C",
        "IS 9428: 1993",
    ),
    BenchmarkCase("thermal_insulation", "bonded mineral wool insulation slabs", "IS 8183: 1993"),
    BenchmarkCase(
        "thermal_insulation",
        "loose unbonded rock wool for thermal insulation",
        "IS 3677: 1985",
    ),
    BenchmarkCase("thermal_insulation", "sprayed mineral wool thermal insulation", "IS 9742: 1993"),
    BenchmarkCase(
        "thermal_insulation",
        "expanded polystyrene material for thermal insulation",
        "IS 4671: 1984",
    ),
    BenchmarkCase(
        "thermal_insulation",
        "rigid polyurethane or polyisocyanurate foam thermal insulation",
        "IS 12436: 1988",
    ),
    BenchmarkCase("cement", "ordinary Portland cement 43 grade", "IS 8112: 1989"),
    BenchmarkCase("cement", "Portland slag cement", "IS 455: 1989"),
    BenchmarkCase(
        "cement",
        "Portland pozzolana cement made using fly ash",
        "IS 1489 (Part 1): 1991",
    ),
    BenchmarkCase(
        "cement",
        "PPC manufactured using calcined clay pozzolana",
        "IS 1489 (Part 2): 1991",
    ),
    BenchmarkCase(
        "cement",
        "cement for masonry mortar and not structural concrete",
        "IS 3466: 1988",
    ),
    BenchmarkCase("cement", "white cement for architectural decorative use", "IS 8042: 1989"),
    BenchmarkCase("fastener", "product grade C hexagon head bolt M12", "IS 1363 (Part 1): 2002"),
    BenchmarkCase("fastener", "product grade C hexagon head screw", "IS 1363 (Part 2): 2002"),
    BenchmarkCase("fastener", "product grade C hexagon nut", "IS 1363 (Part 3): 2002"),
    BenchmarkCase("fastener", "slotted countersunk head screw", "IS 1365: 1978"),
    BenchmarkCase("fastener", "high strength structural bolts", "IS 3757: 1985"),
    BenchmarkCase(
        "water_valve",
        "single door non return valve for a water pipeline",
        "IS 5312 (Part 1): 2004",
    ),
    BenchmarkCase(
        "water_valve",
        "multi door reflux valve for water works",
        "IS 5312 (Part 2): 1986",
    ),
    BenchmarkCase(
        "water_valve",
        "pressure reducing valve for a domestic water supply system",
        "IS 9739: 1981",
    ),
    BenchmarkCase("water_valve", "cast iron air relief valve for a water main", "IS 14845: 2000"),
    BenchmarkCase("water_valve", "non rising stem sluice valve for water supply", "IS 14846: 2000"),
]

DIAGNOSTIC_QUERIES = [
    DiagnosticCase(
        "I need tiles for the walking surface inside a residential room",
        "IS 1478: 1992",
    ),
    DiagnosticCase("I need tiles for house flooring", "IS 1478: 1992"),
    DiagnosticCase("I need a casing pipe for a borewell"),
    DiagnosticCase("I need insulation for equipment operating at about 800 C"),
    DiagnosticCase("I need cement for construction"),
    DiagnosticCase("I need a non-return valve for water supply"),
    DiagnosticCase("I need an M12 hexagonal fastener"),
]


async def main(dataset_path: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    family_by_code = _family_by_code(dataset)
    _validate_benchmark(family_by_code)

    settings = get_settings()
    database = Database(settings)
    qdrant = QdrantClientService(settings)
    index = StandardVectorIndex(
        qdrant.client,
        BgeM3EmbeddingService(settings.embedding_model),
        settings.qdrant_collection,
        dataset_name=str(dataset["dataset_name"]),
    )
    try:
        async with database.session_factory() as session:
            results = await _run_primary_benchmark(session, index, family_by_code)
            _print_aggregate_metrics(results)
            _print_non_hit_at_one(results, family_by_code)
            await _run_diagnostics(session, index, family_by_code)
    finally:
        await database.close()
        await qdrant.close()


async def _run_primary_benchmark(session, index, family_by_code: dict[str, str]) -> list[dict]:
    results: list[dict] = []
    print("Primary benchmark:")
    for case in PRIMARY_BENCHMARK:
        started_at = perf_counter()
        candidates = await index.semantic_search(session, case.query, limit=5)
        latency_ms = (perf_counter() - started_at) * 1000
        expected_rank = _expected_rank(candidates, case.expected_standard)
        expected_candidate = _expected_candidate(candidates, case.expected_standard)
        result = {
            "case": case,
            "candidates": candidates,
            "expected_rank": expected_rank,
            "expected_score": expected_candidate.score if expected_candidate else None,
            "latency_ms": latency_ms,
        }
        results.append(result)
        _print_case(result, family_by_code)
    return results


async def _run_diagnostics(session, index, family_by_code: dict[str, str]) -> None:
    print("Diagnostic queries (excluded from aggregate metrics):")
    for case in DIAGNOSTIC_QUERIES:
        started_at = perf_counter()
        candidates = await index.semantic_search(session, case.query, limit=5)
        latency_ms = (perf_counter() - started_at) * 1000
        print()
        print(f"Query: {case.query}")
        if case.expected_standard:
            expected_rank = _expected_rank(candidates, case.expected_standard)
            print(f"Expected: {case.expected_standard}")
            print(f"Expected rank: {_display_rank(expected_rank)}")
        else:
            print("Expected: exploratory; no forced result")
        print(f"Retrieval latency: {latency_ms:.1f} ms")
        _print_candidates(candidates, family_by_code)


def _print_case(result: dict, family_by_code: dict[str, str]) -> None:
    case: BenchmarkCase = result["case"]
    expected_rank: int | None = result["expected_rank"]
    print()
    print(f"Query: {case.query}")
    print(f"Expected: {case.expected_standard}")
    print(f"Expected family: {case.family}")
    print(f"Retrieval latency: {result['latency_ms']:.1f} ms")
    _print_candidates(result["candidates"], family_by_code)
    print(f"Expected rank: {_display_rank(expected_rank)}")
    print(f"Hit@1: {'PASS' if expected_rank == 1 else 'FAIL'}")
    print(f"Hit@3: {'PASS' if expected_rank is not None and expected_rank <= 3 else 'FAIL'}")


def _print_candidates(candidates: list[SemanticCandidate], family_by_code: dict[str, str]) -> None:
    for rank in range(1, 6):
        if rank > len(candidates):
            print(f"Rank {rank}: <none>")
            continue
        candidate = candidates[rank - 1]
        family = family_by_code.get(canonicalize_standard_id(candidate.standard_code), "unknown")
        print(
            f"Rank {rank}: {candidate.standard_code} | {candidate.title} | "
            f"family={family} | cosine_similarity={candidate.score:.4f}"
        )


def _print_aggregate_metrics(results: list[dict]) -> None:
    print()
    print("Overall metrics:")
    _print_metrics(results)
    print()
    print("Metrics by family:")
    by_family: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_family[result["case"].family].append(result)
    for family, family_results in by_family.items():
        print(f"{family}:")
        _print_metrics(family_results)


def _print_metrics(results: list[dict]) -> None:
    query_count = len(results)
    hit_at_1 = sum(result["expected_rank"] == 1 for result in results)
    hit_at_3 = sum(
        result["expected_rank"] is not None and result["expected_rank"] <= 3 for result in results
    )
    hit_at_5 = sum(result["expected_rank"] is not None for result in results)
    mrr = sum(1 / result["expected_rank"] if result["expected_rank"] else 0 for result in results)
    average_latency_ms = sum(result["latency_ms"] for result in results) / query_count
    print(f"  Queries: {query_count}")
    print(f"  Hit@1: {hit_at_1}/{query_count}")
    print(f"  Hit@3: {hit_at_3}/{query_count}")
    print(f"  Hit@5: {hit_at_5}/{query_count}")
    print(f"  MRR: {mrr / query_count:.3f}")
    print(f"  Average retrieval latency: {average_latency_ms:.1f} ms")


def _print_non_hit_at_one(results: list[dict], family_by_code: dict[str, str]) -> None:
    non_hits = [result for result in results if result["expected_rank"] != 1]
    print()
    print("Failed Hit@1 queries and cross-domain analysis:")
    if not non_hits:
        print("None")
        return
    for result in non_hits:
        case: BenchmarkCase = result["case"]
        candidates: list[SemanticCandidate] = result["candidates"]
        rank_one = candidates[0] if candidates else None
        rank_one_family = (
            family_by_code.get(canonicalize_standard_id(rank_one.standard_code), "unknown")
            if rank_one
            else "none"
        )
        classification = "CROSS-DOMAIN" if rank_one_family != case.family else "within-family"
        print()
        print(f"Classification: {classification}")
        print(f"Query: {case.query}")
        print(f"Expected standard: {case.expected_standard}")
        print(f"Rank 1 standard: {rank_one.standard_code if rank_one else '<none>'}")
        print(f"Expected rank: {_display_rank(result['expected_rank'])}")
        print(f"Expected score: {_display_score(result['expected_score'])}")
        print(f"Rank 1 score: {_display_score(rank_one.score if rank_one else None)}")
        print(f"Expected family: {case.family}")
        print(f"Rank 1 family: {rank_one_family}")


def _family_by_code(dataset: dict) -> dict[str, str]:
    return {
        canonicalize_standard_id(record["standard_code"]): record["family"]
        for record in dataset["records"]
    }


def _validate_benchmark(family_by_code: dict[str, str]) -> None:
    missing = [
        case.expected_standard
        for case in PRIMARY_BENCHMARK
        if canonicalize_standard_id(case.expected_standard) not in family_by_code
    ]
    if missing:
        raise ValueError(f"benchmark standards are absent from the selected dataset: {missing}")
    wrong_family = [
        case.expected_standard
        for case in PRIMARY_BENCHMARK
        if family_by_code[canonicalize_standard_id(case.expected_standard)] != case.family
    ]
    if wrong_family:
        raise ValueError(f"benchmark family metadata does not match cases: {wrong_family}")


def _expected_rank(candidates: list[SemanticCandidate], expected_standard: str) -> int | None:
    expected_code = canonicalize_standard_id(expected_standard)
    for rank, candidate in enumerate(candidates, start=1):
        if canonicalize_standard_id(candidate.standard_code) == expected_code:
            return rank
    return None


def _expected_candidate(
    candidates: list[SemanticCandidate], expected_standard: str
) -> SemanticCandidate | None:
    expected_code = canonicalize_standard_id(expected_standard)
    return next(
        (
            candidate
            for candidate in candidates
            if canonicalize_standard_id(candidate.standard_code) == expected_code
        ),
        None,
    )


def _display_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "not found in Top 5"


def _display_score(score: float | None) -> str:
    return f"{score:.4f}" if score is not None else "not found"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3 semantic retrieval on the 50-standard pilot."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.dataset))
