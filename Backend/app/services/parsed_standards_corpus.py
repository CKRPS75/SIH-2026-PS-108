from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.standards import canonicalize_standard_id

FULL_CORPUS_DATASET_NAME = "parsed_standards_full_corpus"
DEFAULT_PARSED_SOURCE = Path("data/parsed_standards.json")
DEFAULT_AUDIT_JSON = Path("data/reports/parsed_corpus_audit.json")
DEFAULT_AUDIT_CSV = Path("data/reports/parsed_corpus_audit.csv")
DEFAULT_CANONICAL_DATASET = Path("data/canonical/parsed_standards_canonical.json")
DEFAULT_PRODUCT_REVIEW_CSV = Path("data/reports/product_metadata_review.csv")
DEFAULT_METADATA_DIR = Path("data/metadata")
DEFAULT_PRODUCT_TAXONOMY_JSON = DEFAULT_METADATA_DIR / "product_taxonomy.json"
DEFAULT_NULL_PRODUCT_RESOLUTION_CSV = DEFAULT_METADATA_DIR / "null_product_resolution.csv"
DEFAULT_PRODUCT_METADATA_DIFF_CSV = DEFAULT_METADATA_DIR / "product_metadata_diff.csv"
DEFAULT_MERGED_ARTIFACT_REVIEW_CSV = DEFAULT_METADATA_DIR / "merged_artifact_review.csv"
DEFAULT_METADATA_REPORT_JSON = DEFAULT_METADATA_DIR / "metadata_enrichment_report.json"

STANDARD_KINDS = {
    "product_standard",
    "material_standard",
    "test_method",
    "code_of_practice",
    "terminology",
    "installation",
    "other",
}

PRODUCT_TAXONOMY: dict[str, dict[str, Any]] = {
    "valve": {
        "family": "water_valve",
        "aliases": [
            "valve",
            "valves",
            "sluice valve",
            "stop valve",
            "reflux valve",
            "check valve",
            "non return valve",
            "non-return valve",
            "pressure reducing valve",
            "air relief valve",
        ],
    },
    "pipe": {
        "family": "pipe_water_drainage",
        "aliases": [
            "pipe",
            "pipes",
            "tube",
            "tubes",
            "casing pipe",
            "water well tube",
            "grp pipe",
            "hdpe pipe",
            "pvc pipe",
            "steel tube",
        ],
    },
    "pipe_fitting": {
        "family": "pipe_water_drainage",
        "aliases": [
            "fitting",
            "fittings",
            "pipe fitting",
            "pipe fittings",
            "bend",
            "bends",
            "tee",
            "tees",
            "reducer",
            "reducers",
            "coupler",
            "couplers",
            "socket",
            "sockets",
            "saddle piece",
            "saddle pieces",
        ],
    },
    "cement": {
        "family": "cement",
        "aliases": [
            "cement",
            "portland cement",
            "ordinary portland cement",
            "masonry cement",
            "white cement",
            "slag cement",
            "pozzolana cement",
            "lime pozzolana mixture",
            "lime-pozzolana concrete",
        ],
    },
    "bolt": {
        "family": "fastener",
        "aliases": ["bolt", "bolts", "structural bolt", "hexagon head bolt"],
    },
    "screw": {
        "family": "fastener",
        "aliases": ["screw", "screws", "hexagon head screw", "countersunk screw"],
    },
    "nut": {"family": "fastener", "aliases": ["nut", "nuts", "hexagon nut"]},
    "washer": {"family": "fastener", "aliases": ["washer", "washers", "plain washer"]},
    "rivet": {"family": "fastener", "aliases": ["rivet", "rivets", "rivet bar", "rivet bars"]},
    "fastener": {"family": "fastener", "aliases": ["fastener", "fasteners", "stay", "stays"]},
    "tile": {
        "family": "tile",
        "aliases": [
            "tile",
            "tiles",
            "roofing tile",
            "flooring tile",
            "ridge tile",
            "terracing tile",
            "hollow clay tile",
            "lining tile",
            "clay roofing country tile",
        ],
    },
    "thermal_insulation": {
        "family": "thermal_insulation",
        "aliases": [
            "thermal insulation",
            "thermal insulating",
            "insulation",
            "mineral wool",
            "rock wool",
            "slag wool",
            "calcium silicate insulation",
            "fibrous pipe insulation",
            "expanded polystyrene",
        ],
    },
    "roofing_sheet": {
        "family": "roofing",
        "aliases": ["roofing sheet", "roofing sheets", "corrugated sheet", "corrugated sheets"],
    },
    "board": {
        "family": "wood_board",
        "aliases": [
            "board",
            "boards",
            "fibre board",
            "fibre boards",
            "fiber board",
            "particle board",
            "plywood",
            "veneered decorative plywood",
            "bamboo mat board",
            "coir board",
        ],
    },
    "brick": {
        "family": "masonry_unit",
        "aliases": ["brick", "bricks", "building brick", "building bricks", "fly ash brick"],
    },
    "block": {
        "family": "masonry_unit",
        "aliases": ["block", "blocks", "masonry block", "masonry blocks", "concrete block"],
    },
    "steel_sheet": {
        "family": "steel",
        "aliases": ["steel sheet", "steel sheets", "carbon steel sheet", "carbon steel sheets"],
    },
    "steel_strip": {
        "family": "steel",
        "aliases": ["steel strip", "steel strips", "carbon steel strip", "carbon steel strips"],
    },
    "steel_section": {
        "family": "steel",
        "aliases": [
            "steel section",
            "steel sections",
            "tee bar",
            "tee bars",
            "flange steel section",
        ],
    },
    "aggregate": {"family": "concrete_material", "aliases": ["aggregate", "aggregates"]},
    "admixture": {
        "family": "concrete_material",
        "aliases": ["admixture", "admixtures", "accelerating admixture"],
    },
    "sealant": {
        "family": "sealant",
        "aliases": ["sealant", "sealants", "polysulphide", "sealing compound"],
    },
    "glass": {
        "family": "glass",
        "aliases": ["glass", "glass fibre", "glass fiber", "glass fibre reinforced"],
    },
    "timber": {
        "family": "timber",
        "aliases": ["timber", "timbers", "sawn timber", "wooden sleeper", "wooden sleepers"],
    },
    "door": {
        "family": "door_window",
        "aliases": ["door", "doors", "door frame", "door frames", "shutter", "shutters"],
    },
    "window": {
        "family": "door_window",
        "aliases": ["window", "windows", "ventilator", "ventilators"],
    },
    "paint": {"family": "coating", "aliases": ["paint", "paints", "filler"]},
    "coating": {
        "family": "coating",
        "aliases": ["coating", "coatings", "epoxy coating", "water repellent"],
    },
    "adhesive": {"family": "adhesive", "aliases": ["adhesive", "adhesives"]},
    "hinge": {"family": "hardware", "aliases": ["hinge", "hinges", "butt hinge", "butt hinges"]},
    "latch": {"family": "hardware", "aliases": ["latch", "latches", "rim latch", "rim latches"]},
    "gate": {"family": "hardware", "aliases": ["gate", "gates", "collapsible gate"]},
    "handrail": {"family": "hardware", "aliases": ["hand rail", "handrail", "hand rail cover"]},
    "plug_socket": {
        "family": "electrical",
        "aliases": ["plug", "plugs", "socket-outlet", "socket"],
    },
    "sink": {"family": "sanitary_fixture", "aliases": ["sink", "sinks"]},
    "water_tank": {"family": "water_storage", "aliases": ["water tank", "water tanks", "tank"]},
    "stone": {
        "family": "stone",
        "aliases": ["stone", "stones", "building stone", "polished stone"],
    },
    "channel": {"family": "precast_concrete", "aliases": ["channel", "channels", "l-panel"]},
    "wire_fabric": {"family": "reinforcement", "aliases": ["wire fabric", "steel wire fabric"]},
    "rod": {"family": "welding", "aliases": ["rod", "rods", "filler rod", "filler rods"]},
    "electrode": {"family": "welding", "aliases": ["electrode", "electrodes"]},
    "waterproofing_membrane": {
        "family": "waterproofing",
        "aliases": ["bitumen felt", "bitumen felts", "water-proofing", "waterproofing"],
    },
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

PRODUCT_ALIASES: dict[str, set[str]] = {
    product: set(metadata["aliases"]) for product, metadata in PRODUCT_TAXONOMY.items()
}

FAMILY_BY_PRODUCT = {
    product: str(metadata["family"]) for product, metadata in PRODUCT_TAXONOMY.items()
}

LEGACY_PRODUCT_ALIASES: dict[str, set[str]] = {
    "valve": {
        "valve",
        "valves",
        "sluice valve",
        "stop valve",
        "reflux valve",
        "check valve",
        "non return valve",
        "non-return valve",
        "pressure reducing valve",
        "air relief valve",
    },
    "pipe": {
        "pipe",
        "pipes",
        "tube",
        "tubes",
        "casing pipe",
        "water well tube",
        "grp pipe",
        "hdpe pipe",
        "pvc pipe",
        "steel tube",
    },
    "pipe_fitting": {"fitting", "fittings", "pipe fitting", "pipe fittings"},
    "cement": {
        "cement",
        "portland cement",
        "ordinary portland cement",
        "masonry cement",
        "white cement",
        "slag cement",
        "pozzolana cement",
    },
    "bolt": {"bolt", "bolts", "structural bolt", "hexagon head bolt"},
    "screw": {"screw", "screws", "hexagon head screw", "countersunk screw"},
    "nut": {"nut", "nuts", "hexagon nut"},
    "washer": {"washer", "washers", "plain washer"},
    "tile": {
        "tile",
        "tiles",
        "roofing tile",
        "flooring tile",
        "ridge tile",
        "terracing tile",
        "hollow clay tile",
        "lining tile",
    },
    "thermal_insulation": {
        "thermal insulation",
        "insulation",
        "mineral wool",
        "rock wool",
        "slag wool",
        "calcium silicate insulation",
        "fibrous pipe insulation",
        "expanded polystyrene",
    },
}

NON_PRODUCT_KINDS = {"test_method", "code_of_practice", "terminology", "installation"}
DIRECT_PRODUCT_KINDS = {"product_standard", "material_standard", "component", "other"}

SUBTYPE_PATTERNS: list[tuple[str, str, list[str]]] = [
    (
        "valve",
        "check_valve",
        ["check valve", "check valves", "reflux valve", "non return valve", "non-return valve"],
    ),
    ("valve", "sluice_valve", ["sluice valve"]),
    ("valve", "pressure_reducing_valve", ["pressure reducing valve"]),
    ("valve", "air_relief_valve", ["air relief valve"]),
    ("pipe_fitting", "reducer", ["reducer", "reducers", "straight reducer"]),
    ("pipe_fitting", "tee", ["tee", "tees"]),
    ("pipe_fitting", "bend", ["bend", "bends"]),
    ("pipe_fitting", "coupler", ["coupler", "couplers"]),
    ("pipe_fitting", "socket", ["socket", "sockets"]),
    ("cement", "portland_pozzolana_fly_ash", ["fly ash", "part 1"]),
    ("cement", "portland_pozzolana_calcined_clay", ["calcined clay", "part 2"]),
    ("cement", "ordinary_portland_33_grade", ["33 grade ordinary portland"]),
    ("cement", "ordinary_portland_43_grade", ["43 grade ordinary portland"]),
    ("cement", "ordinary_portland_53_grade", ["53 grade ordinary portland"]),
    ("bolt", "hexagon_head_bolt", ["hexagon head bolt"]),
    ("screw", "hexagon_head_screw", ["hexagon head screw"]),
    ("nut", "hexagon_nut", ["hexagon nut"]),
    ("washer", "plain_washer", ["plain washer"]),
    ("tile", "roofing_tile", ["roofing tile", "country tile"]),
    ("tile", "flooring_tile", ["flooring tile"]),
    ("board", "medium_density_fibre_board", ["medium density fibre board", "mdf"]),
    ("board", "plywood", ["plywood"]),
    ("brick", "fly_ash_lime_brick", ["fly ash-lime brick", "fly ash lime brick"]),
    ("door", "steel_door", ["steel door"]),
    ("window", "steel_window", ["steel window"]),
]

MATERIAL_PATTERNS: list[tuple[str, list[str]]] = [
    ("HDPE", ["high density polyethylene", "hdpe"]),
    ("UPVC", ["unplasticized polyvinyl chloride", "unplasticised polyvinyl chloride", "upvc"]),
    ("PVC", ["polyvinyl chloride", "pvc"]),
    ("calcium silicate", ["calcium silicate"]),
    ("glass fibre reinforced plastic", ["glass fibre reinforced plastic", "grp", "gfrp"]),
    ("fibrous", ["fibrous", "fibre", "fiber"]),
    ("mineral wool", ["mineral wool"]),
    ("rock wool", ["rock wool"]),
    ("slag wool", ["slag wool"]),
    ("glass wool", ["glass wool"]),
    ("polyurethane", ["polyurethane", "pur"]),
    ("finishing cement", ["finishing cement"]),
    ("cement", ["cement"]),
    ("malleable cast iron", ["malleable cast iron"]),
    ("cast iron", ["cast iron", "ci "]),
    ("steel", ["mild steel", "carbon steel", "stainless steel", "structural steel", "steel"]),
    ("aluminium", ["aluminium", "aluminum"]),
    ("timber", ["timber", "wooden", "wood"]),
    ("bamboo", ["bamboo"]),
    ("coir", ["coir"]),
    ("bitumen", ["bitumen", "bituminous"]),
    ("rubber", ["rubber"]),
    ("polycarbonate", ["polycarbonate"]),
    ("polymethyl methacrylate", ["polymethyl methacrylate", "pmma"]),
    ("clay", ["burnt clay", "clay"]),
    ("concrete", ["concrete", "precast concrete"]),
    ("ferrocement", ["ferrocement"]),
]

APPLICATION_PATTERNS: list[tuple[str, list[str]]] = [
    ("potable water supply", ["potable water", "portable water"]),
    ("water supply", ["water supply", "water works"]),
    ("sewerage", ["sewerage", "sewage", "drainage"]),
    ("roofing", ["roofing", "roof"]),
    ("flooring", ["flooring", "floor"]),
    ("thermal insulation", ["thermal insulation", "insulating"]),
    ("railway track", ["railway track"]),
    ("waterproofing", ["water-proofing", "waterproofing"]),
    ("door and window construction", ["door", "window", "ventilator"]),
]

FUNCTION_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "prevent_reverse_flow",
        ["prevent reverse flow", "reflux", "check valve", "check valves", "non return"],
    ),
    ("reduce_pressure", ["pressure reducing", "reduce pressure"]),
    ("release_air", ["air relief", "release air"]),
    ("shut_off_flow", ["sluice valve", "stop valve", "shut off"]),
    ("carry_potable_water", ["potable water", "portable water"]),
    ("carry_sewage", ["sewerage", "sewage"]),
    ("thermal_insulation", ["thermal insulation", "insulating"]),
    ("roofing", ["roofing"]),
    ("flooring", ["flooring"]),
    ("structural_fastening", ["fastener", "bolt", "screw", "nut", "washer", "rivet"]),
    ("waterproofing", ["water-proofing", "waterproofing", "water repellent"]),
]


@dataclass(frozen=True)
class ProductMetadata:
    canonical_product: str | None
    product_subtype: str | None
    primary_subject: str
    product_aliases: list[str]
    material: str | None
    application: str | None
    function: str | None
    applies_to_product_families: list[str]
    product_confidence: str
    product_derivation_source: str
    metadata_confidence: str
    metadata_evidence: dict[str, list[str]]
    metadata_derivation_method: str
    review_required: bool
    review_reason: str | None
    standard_kind: str
    family: str | None


@dataclass(frozen=True)
class CanonicalStandard:
    standard_code: str
    standard_code_norm: str
    title: str | None
    revision: str | None
    page_start: int | None
    page_end: int | None
    scope: str | None
    full_text: str | None
    source_record_count: int
    source_provenance: list[dict[str, Any]]
    standard_kind: str
    canonical_product: str | None
    product_aliases: list[str]
    product_confidence: str
    product_derivation_source: str
    family: str | None
    retrieval_text: str
    quality_flags: list[str] = field(default_factory=list)
    raw_title: str | None = None
    canonical_title_expanded: str | None = None
    title_evidence: list[str] = field(default_factory=list)
    scope_reconstructed: str | None = None
    scope_source: str | None = None
    product_subtype: str | None = None
    primary_subject: str | None = None
    material: str | None = None
    application: str | None = None
    function: str | None = None
    applies_to_product_families: list[str] = field(default_factory=list)
    metadata_confidence: str = "low"
    metadata_evidence: dict[str, list[str]] = field(default_factory=dict)
    metadata_derivation_method: str = "deterministic_taxonomy_v1"
    review_required: bool = True
    review_reason: str | None = None


@dataclass(frozen=True)
class AuditIssue:
    issue_type: str
    source_index: int
    is_code: str | None
    is_code_norm: str | None
    title: str | None
    detail: str


@dataclass(frozen=True)
class CorpusAudit:
    total_records: int
    unique_normalized_standard_codes: int
    duplicate_codes: dict[str, int]
    missing_is_code: int
    missing_is_code_norm: int
    missing_title: int
    missing_scope: int
    missing_full_text: int
    empty_full_text: int
    suspiciously_short_records: int
    suspiciously_long_records: int
    possible_merged_record_artifacts: int
    duplicate_title_code_combinations: dict[str, int]
    page_range_anomalies: int
    known_duplicate_is_2116_records: int
    issues: list[AuditIssue]


def load_parsed_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("parsed standards source must be a JSON list or object with records")
    return [record if isinstance(record, dict) else {"raw_record": record} for record in data]


def audit_parsed_records(records: list[dict[str, Any]]) -> CorpusAudit:
    norm_counts: Counter[str] = Counter()
    title_code_counts: Counter[str] = Counter()
    issues: list[AuditIssue] = []

    for index, record in enumerate(records, start=1):
        is_code = _clean_text(record.get("is_code"))
        is_code_norm = normalize_source_code(record)
        title = _clean_text(record.get("title"))
        scope = _clean_text(record.get("scope"))
        full_text = _clean_text(record.get("full_text"))
        if is_code_norm:
            norm_counts[is_code_norm] += 1
        if title and is_code_norm:
            title_code_counts[f"{is_code_norm}|{_space_key(title)}"] += 1

        _append_missing_issue(issues, "missing_is_code", index, record, is_code, "is_code")
        _append_missing_issue(
            issues, "missing_is_code_norm", index, record, is_code_norm, "is_code_norm"
        )
        _append_missing_issue(issues, "missing_title", index, record, title, "title")
        _append_missing_issue(issues, "missing_scope", index, record, scope, "scope")
        _append_missing_issue(
            issues, "missing_full_text", index, record, full_text, "full_text"
        )
        if full_text == "":
            _append_issue(issues, "empty_full_text", index, record, "full_text is empty")
        if _record_length(record) < 80:
            _append_issue(
                issues,
                "suspiciously_short_record",
                index,
                record,
                "combined title/scope/full_text length is below 80 characters",
            )
        if _record_length(record) > 25_000:
            _append_issue(
                issues,
                "suspiciously_long_record",
                index,
                record,
                "combined title/scope/full_text length exceeds 25,000 characters",
            )
        if _possible_merged_record(record):
            _append_issue(
                issues,
                "possible_merged_record_artifact",
                index,
                record,
                "text contains repeated IS-code-like references or parser merge markers",
            )
        page_start = _as_int_or_none(record.get("page_start"))
        page_end = _as_int_or_none(record.get("page_end"))
        if page_start is not None and page_end is not None and page_end < page_start:
            _append_issue(
                issues,
                "page_range_anomaly",
                index,
                record,
                f"page_end {page_end} is before page_start {page_start}",
            )

    duplicate_codes = {code: count for code, count in norm_counts.items() if count > 1}
    duplicate_title_codes = {
        key: count for key, count in title_code_counts.items() if count > 1
    }
    return CorpusAudit(
        total_records=len(records),
        unique_normalized_standard_codes=len(norm_counts),
        duplicate_codes=duplicate_codes,
        missing_is_code=_issue_count(issues, "missing_is_code"),
        missing_is_code_norm=_issue_count(issues, "missing_is_code_norm"),
        missing_title=_issue_count(issues, "missing_title"),
        missing_scope=_issue_count(issues, "missing_scope"),
        missing_full_text=_issue_count(issues, "missing_full_text"),
        empty_full_text=_issue_count(issues, "empty_full_text"),
        suspiciously_short_records=_issue_count(issues, "suspiciously_short_record"),
        suspiciously_long_records=_issue_count(issues, "suspiciously_long_record"),
        possible_merged_record_artifacts=_issue_count(
            issues, "possible_merged_record_artifact"
        ),
        duplicate_title_code_combinations=duplicate_title_codes,
        page_range_anomalies=_issue_count(issues, "page_range_anomaly"),
        known_duplicate_is_2116_records=duplicate_codes.get(
            canonicalize_standard_id("IS 2116: 1980"),
            0,
        ),
        issues=issues,
    )


def write_audit_reports(audit: CorpusAudit, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_audit_to_dict(audit), indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["issue_type", "source_index", "is_code", "is_code_norm", "title", "detail"],
        )
        writer.writeheader()
        for issue in audit.issues:
            writer.writerow(asdict(issue))


def build_canonical_corpus(
    records: list[dict[str, Any]],
) -> tuple[list[CanonicalStandard], list[str]]:
    by_norm: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    warnings = []
    for index, record in enumerate(records, start=1):
        norm = normalize_source_code(record)
        if not norm:
            warnings.append(f"source index {index} missing normalized code; excluded")
            continue
        by_norm[norm].append((index, record))

    canonical_records = [
        _canonicalize_group(norm, grouped_records, warnings)
        for norm, grouped_records in sorted(by_norm.items())
    ]
    return canonical_records, warnings


def write_canonical_dataset(records: list[CanonicalStandard], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": FULL_CORPUS_DATASET_NAME,
        "record_count": len(records),
        "source_note": (
            "Parsed BIS building-material summary corpus; no current lifecycle/QCO status "
            "is inferred."
        ),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_canonical_dataset(path: Path) -> list[CanonicalStandard]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("canonical dataset must contain a records list")
    return [CanonicalStandard(**record) for record in records]


def write_product_taxonomy(path: Path = DEFAULT_PRODUCT_TAXONOMY_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        product: {
            "family": metadata["family"],
            "aliases": sorted(metadata["aliases"]),
        }
        for product, metadata in sorted(PRODUCT_TAXONOMY.items())
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metadata_enrichment_reports(
    old_records: list[CanonicalStandard | dict[str, Any]],
    new_records: list[CanonicalStandard],
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> dict[str, Any]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    old_by_code = {
        _record_value(record)["standard_code_norm"]: _record_value(record)
        for record in old_records
    }
    old_product_coverage = sum(
        1 for record in old_by_code.values() if record.get("canonical_product")
    )
    if len(old_by_code) != len(new_records) or old_product_coverage != 220:
        old_by_code = _legacy_baseline_by_code(new_records)
    new_by_code = {record.standard_code_norm: record for record in new_records}
    original_null_codes = {
        code for code, record in old_by_code.items() if not record.get("canonical_product")
    }

    _write_null_product_resolution(
        original_null_codes,
        old_by_code,
        new_by_code,
        metadata_dir / DEFAULT_NULL_PRODUCT_RESOLUTION_CSV.name,
    )
    _write_product_metadata_diff(
        old_by_code,
        new_records,
        metadata_dir / DEFAULT_PRODUCT_METADATA_DIFF_CSV.name,
    )
    merged_counts = _write_merged_artifact_review(
        new_records,
        metadata_dir / DEFAULT_MERGED_ARTIFACT_REVIEW_CSV.name,
    )
    write_product_taxonomy(metadata_dir / DEFAULT_PRODUCT_TAXONOMY_JSON.name)
    report = _metadata_report(old_by_code, new_records, original_null_codes, merged_counts)
    (metadata_dir / DEFAULT_METADATA_REPORT_JSON.name).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def write_product_review(records: list[CanonicalStandard], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "standard_code",
                "title",
                "scope_excerpt",
                "standard_kind",
                "canonical_product",
                "product_aliases",
                "product_confidence",
                "review_status",
                "review_notes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "standard_code": record.standard_code,
                    "title": record.title or "",
                    "scope_excerpt": _excerpt(record.scope or "", 220),
                    "standard_kind": record.standard_kind,
                    "canonical_product": record.canonical_product or "",
                    "product_aliases": "; ".join(record.product_aliases),
                    "product_confidence": record.product_confidence,
                    "review_status": "needs_review",
                    "review_notes": "",
                }
            )


def derive_product_metadata(
    title: str | None,
    scope: str | None,
    full_text: str | None = None,
) -> ProductMetadata:
    title_text = title or ""
    scope_text = scope or ""
    full_text = full_text or ""
    summary_heading = _extract_summary_heading(full_text)
    title_for_metadata = summary_heading or title_text
    scope_for_metadata = scope_text or _extract_scope_from_full_text(full_text) or ""
    text_by_source = {
        "title": title_for_metadata,
        "scope": scope_for_metadata,
        "full_text_summary_heading": summary_heading or "",
        "full_text_repeated_terms": full_text,
    }
    title_matches = _product_matches(title_for_metadata)
    scope_matches = _product_matches(scope_for_metadata)
    summary_matches = _product_matches(summary_heading or "")
    full_text_matches = _product_matches(full_text)
    combined_scores: Counter[str] = Counter()
    combined_scores.update({product: count * 3 for product, count in title_matches.items()})
    combined_scores.update({product: count * 2 for product, count in scope_matches.items()})
    combined_scores.update({product: count * 2 for product, count in summary_matches.items()})
    combined_scores.update(
        {product: min(count, 3) for product, count in full_text_matches.items()}
    )
    canonical_product = None
    confidence = "low"
    source = "title_and_scope"
    evidence: dict[str, list[str]] = {}
    if combined_scores:
        product, score = combined_scores.most_common(1)[0]
        second_score = combined_scores.most_common(2)[1][1] if len(combined_scores) > 1 else 0
        canonical_product = product
        evidence["canonical_product"] = _field_evidence(product, text_by_source)
        if (
            (title_matches.get(product, 0) or scope_matches.get(product, 0))
            and score >= second_score + 2
        ):
            confidence = "high"
            source = "title_scope_or_summary"
        elif summary_matches.get(product, 0) and score >= second_score + 2:
            confidence = "medium"
            source = "full_text_summary_heading"
        else:
            confidence = "low"
            source = "full_text_repeated_terms"

    standard_kind = derive_standard_kind(title_for_metadata, scope_for_metadata, full_text)
    fallback_product = None
    if canonical_product is None and standard_kind not in NON_PRODUCT_KINDS:
        fallback_product = _source_grounded_product_fallback(title_for_metadata, scope_for_metadata)
        if fallback_product:
            canonical_product = fallback_product
            confidence = "medium" if scope_for_metadata else "low"
            source = "source_grounded_title_phrase"
            evidence["canonical_product"] = ["title"] if title_for_metadata else ["scope"]

    heading_sources = {
        key: value
        for key, value in text_by_source.items()
        if key != "full_text_repeated_terms"
    }
    product_subtype, subtype_evidence = _first_pattern_match(
        SUBTYPE_PATTERNS,
        heading_sources,
        canonical_product=canonical_product,
    )
    if product_subtype:
        evidence["product_subtype"] = subtype_evidence
    material, material_evidence = _first_value_match(MATERIAL_PATTERNS, text_by_source)
    if material:
        evidence["material"] = material_evidence
    application, application_evidence = _first_value_match(APPLICATION_PATTERNS, text_by_source)
    if application:
        evidence["application"] = application_evidence
    function, function_evidence = _first_value_match(FUNCTION_PATTERNS, heading_sources)
    if function:
        evidence["function"] = function_evidence

    aliases = sorted(PRODUCT_ALIASES.get(canonical_product or "", set()))
    family = FAMILY_BY_PRODUCT.get(canonical_product or "")
    applies_to_families = sorted(
        {
            FAMILY_BY_PRODUCT[product]
            for product in set(title_matches) | set(scope_matches) | set(summary_matches)
            if product in FAMILY_BY_PRODUCT
        }
    )
    if family and standard_kind in NON_PRODUCT_KINDS:
        applies_to_families = sorted({*applies_to_families, family})
        canonical_product = None
        aliases = []
        confidence = "high" if title_matches or scope_matches else "medium"
        source = "non_product_family_link"
        family = None
    primary_subject = _derive_primary_subject(
        title_for_metadata,
        scope_for_metadata,
        standard_kind,
        canonical_product,
        product_subtype,
    )
    evidence["primary_subject"] = ["title"] if title_for_metadata else ["scope"]
    if applies_to_families:
        evidence["applies_to_product_families"] = ["title", "scope"]
    metadata_confidence = _metadata_confidence(confidence, evidence)
    review_reason = _review_reason(
        standard_kind,
        canonical_product,
        confidence,
        primary_subject,
        bool(fallback_product),
    )
    return ProductMetadata(
        canonical_product=canonical_product,
        product_subtype=product_subtype,
        primary_subject=primary_subject,
        product_aliases=aliases,
        material=material,
        application=application,
        function=function,
        applies_to_product_families=applies_to_families,
        product_confidence=confidence,
        product_derivation_source=source,
        metadata_confidence=metadata_confidence,
        metadata_evidence=evidence,
        metadata_derivation_method="deterministic_taxonomy_v1",
        review_required=review_reason is not None,
        review_reason=review_reason,
        standard_kind=standard_kind,
        family=family,
    )


def derive_standard_kind(title: str, scope: str, full_text: str | None = None) -> str:
    del full_text
    text = f"{title} {scope}".casefold()
    if any(term in text for term in ["method of test", "methods of test", "test method"]):
        return "test_method"
    if any(term in text for term in ["code of practice", "practice for"]):
        return "code_of_practice"
    if any(term in text for term in ["terminology", "glossary", "vocabulary"]):
        return "terminology"
    if any(term in text for term in ["installation", "laying", "fixing"]):
        return "installation"
    if "cement" in text or "material" in text or "steel" in text or "timber" in text:
        return "material_standard"
    if any(term in text for term in ["requirements", "specification", "covers"]):
        return "product_standard"
    return "other"


def canonicalize_product(product: str) -> tuple[str | None, str]:
    matches = _product_matches(product)
    if not matches:
        return None, "low"
    product_name, score = matches.most_common(1)[0]
    second_score = matches.most_common(2)[1][1] if len(matches) > 1 else 0
    confidence = "high" if score >= second_score + 1 else "low"
    return product_name, confidence


def product_compatibility(query_product: str | None, candidate_product: str | None) -> str:
    if query_product is None:
        return "unknown_query_product"
    if candidate_product is None:
        return "unknown_candidate_product"
    if query_product == candidate_product:
        return "compatible"
    if query_product == "pipe" and candidate_product == "pipe_fitting":
        return "adjacent"
    if query_product == "pipe_fitting" and candidate_product == "pipe":
        return "adjacent"
    if FAMILY_BY_PRODUCT.get(query_product) == FAMILY_BY_PRODUCT.get(candidate_product):
        return "same_family"
    return "incompatible"


def build_retrieval_text(record: CanonicalStandard | dict[str, Any]) -> str:
    value = asdict(record) if isinstance(record, CanonicalStandard) else record
    key_description = _key_source_description(value.get("full_text"), value.get("scope"))
    lines = [
        f"STANDARD: {value.get('standard_code') or ''}",
        f"PRODUCT: {value.get('canonical_product') or ''}",
        f"SUBTYPE: {value.get('product_subtype') or ''}",
        f"SUBJECT: {value.get('primary_subject') or ''}",
        f"FUNCTION: {value.get('function') or ''}",
        f"MATERIAL: {value.get('material') or ''}",
        f"APPLICATION: {value.get('application') or ''}",
        f"TYPE: {value.get('standard_kind') or 'other'}",
        f"TITLE: {value.get('canonical_title_expanded') or value.get('title') or ''}",
        f"SCOPE: {_single_line(value.get('scope') or value.get('scope_reconstructed') or '')}",
        f"KEY SOURCE DESCRIPTION: {_single_line(key_description)}",
    ]
    return "\n".join(line for line in lines if line.split(":", 1)[1].strip())


def build_product_query(product: str, description: str) -> str:
    return (
        f"PRODUCT: {_single_line(product)}\n"
        f"WORK DESCRIPTION: {_single_line(description)}"
    )


def normalize_source_code(record: dict[str, Any]) -> str:
    raw_norm = _clean_text(record.get("is_code_norm"))
    raw_code = _clean_text(record.get("is_code"))
    source = raw_norm or raw_code or ""
    try:
        return canonicalize_standard_id(source)
    except Exception:  # noqa: BLE001 - malformed source values are reported by audit
        return _single_line(source)


def source_provenance(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_index": index,
        "is_code": _clean_text(record.get("is_code")),
        "is_code_norm": _clean_text(record.get("is_code_norm")),
        "page_start": _as_int_or_none(record.get("page_start")),
        "page_end": _as_int_or_none(record.get("page_end")),
    }


def quality_flags(record: CanonicalStandard | dict[str, Any]) -> list[str]:
    value = asdict(record) if isinstance(record, CanonicalStandard) else record
    flags = []
    if not _clean_text(value.get("title")):
        flags.append("missing_title")
    if not _clean_text(value.get("scope")):
        flags.append("missing_scope")
    if not _clean_text(value.get("full_text")):
        flags.append("missing_full_text")
    if value.get("product_confidence") == "low":
        flags.append("low_product_confidence")
    if not _clean_text(value.get("primary_subject")):
        flags.append("missing_primary_subject")
    if value.get("review_required"):
        flags.append("metadata_review_required")
    if value.get("source_record_count", 1) > 1:
        flags.append("duplicate_source_records")
    return flags


def _canonicalize_group(
    norm: str,
    grouped_records: list[tuple[int, dict[str, Any]]],
    warnings: list[str],
) -> CanonicalStandard:
    chosen_index, chosen = _choose_canonical_source(grouped_records, warnings, norm)
    title = _clean_text(chosen.get("title")) or None
    scope = _clean_text(chosen.get("scope")) or None
    full_text = _clean_text(chosen.get("full_text")) or None
    canonical_title_expanded, title_evidence = _reconstruct_title(title, full_text)
    scope_reconstructed = None if scope else _extract_scope_from_full_text(full_text or "")
    scope_source = "reconstructed_from_full_text" if scope_reconstructed else None
    metadata = derive_product_metadata(
        canonical_title_expanded or title,
        scope or scope_reconstructed,
        full_text,
    )
    standard_code = _clean_text(chosen.get("is_code")) or norm
    canonical = CanonicalStandard(
        standard_code=standard_code,
        standard_code_norm=norm,
        title=title,
        revision=_clean_text(chosen.get("revision")) or None,
        page_start=_as_int_or_none(chosen.get("page_start")),
        page_end=_as_int_or_none(chosen.get("page_end")),
        scope=scope,
        full_text=full_text,
        source_record_count=len(grouped_records),
        source_provenance=[
            source_provenance(index, record) for index, record in grouped_records
        ],
        standard_kind=metadata.standard_kind,
        canonical_product=metadata.canonical_product,
        product_aliases=metadata.product_aliases,
        product_confidence=metadata.product_confidence,
        product_derivation_source=metadata.product_derivation_source,
        family=metadata.family,
        retrieval_text="",
        raw_title=title,
        canonical_title_expanded=canonical_title_expanded,
        title_evidence=title_evidence,
        scope_reconstructed=scope_reconstructed,
        scope_source=scope_source,
        product_subtype=metadata.product_subtype,
        primary_subject=metadata.primary_subject,
        material=metadata.material,
        application=metadata.application,
        function=metadata.function,
        applies_to_product_families=metadata.applies_to_product_families,
        metadata_confidence=metadata.metadata_confidence,
        metadata_evidence=metadata.metadata_evidence,
        metadata_derivation_method=metadata.metadata_derivation_method,
        review_required=metadata.review_required or (scope is None and scope_reconstructed is None),
        review_reason=metadata.review_reason
        or (
            "missing_scope_no_reliable_reconstruction"
            if scope is None and scope_reconstructed is None
            else None
        ),
    )
    return CanonicalStandard(
        **{
            **asdict(canonical),
            "retrieval_text": build_retrieval_text(canonical),
            "quality_flags": quality_flags(canonical),
        }
    )


def _choose_canonical_source(
    grouped_records: list[tuple[int, dict[str, Any]]],
    warnings: list[str],
    norm: str,
) -> tuple[int, dict[str, Any]]:
    scored = sorted(
        grouped_records,
        key=lambda item: (
            _record_quality_score(item[1]),
            -item[0],
        ),
        reverse=True,
    )
    best = scored[0]
    signatures = {_record_signature(record) for _, record in grouped_records}
    if len(signatures) > 1:
        warnings.append(
            f"{norm} has conflicting duplicate source records; chose source index {best[0]}"
        )
    return best


def _product_matches(text: str) -> Counter[str]:
    normalized = f" {_space_key(text)} "
    matches: Counter[str] = Counter()
    for product, aliases in PRODUCT_ALIASES.items():
        for alias in aliases:
            alias_key = _space_key(alias)
            if f" {alias_key} " in normalized:
                matches[product] += 1
    return matches


def _record_value(record: CanonicalStandard | dict[str, Any]) -> dict[str, Any]:
    return asdict(record) if isinstance(record, CanonicalStandard) else record


def _legacy_baseline_by_code(records: list[CanonicalStandard]) -> dict[str, dict[str, Any]]:
    return {
        record.standard_code_norm: {
            **asdict(record),
            **_legacy_product_metadata(record.raw_title or record.title, record.scope),
        }
        for record in records
    }


def _legacy_product_metadata(title: str | None, scope: str | None) -> dict[str, Any]:
    title_matches = _legacy_product_matches(title or "")
    scope_matches = _legacy_product_matches(scope or "")
    combined_scores: Counter[str] = Counter()
    combined_scores.update({product: count * 3 for product, count in title_matches.items()})
    combined_scores.update(scope_matches)
    canonical_product = None
    confidence = "low"
    if combined_scores:
        product, score = combined_scores.most_common(1)[0]
        second_score = combined_scores.most_common(2)[1][1] if len(combined_scores) > 1 else 0
        canonical_product = product
        if title_matches.get(product, 0) and score >= second_score + 2:
            confidence = "high"
        elif score >= second_score + 2:
            confidence = "medium"
    return {
        "canonical_product": canonical_product,
        "product_confidence": confidence,
    }


def _legacy_product_matches(text: str) -> Counter[str]:
    normalized = f" {_space_key(text)} "
    matches: Counter[str] = Counter()
    for product, aliases in LEGACY_PRODUCT_ALIASES.items():
        for alias in aliases:
            alias_key = _space_key(alias)
            if f" {alias_key} " in normalized:
                matches[product] += 1
    return matches


def _reconstruct_title(title: str | None, full_text: str | None) -> tuple[str | None, list[str]]:
    summary = _extract_summary_heading(full_text or "")
    if not summary or not title:
        return None, []
    title_key = _space_key(title)
    summary_key = _space_key(summary)
    if not title_key or title_key == summary_key:
        return None, []
    if summary_key.startswith(title_key) and len(summary_key) > len(title_key) + 2:
        return summary, ["full_text_summary_heading"]
    if title.rstrip().endswith(("AND", "FOR", "OF", "TO", ",")):
        return summary, ["full_text_summary_heading"]
    return None, []


def _extract_summary_heading(full_text: str) -> str | None:
    match = re.search(r"SUMMARY\s+OF\s*(.+)", full_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    heading_lines = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line and heading_lines:
            break
        if not line:
            continue
        if line.startswith("(") and heading_lines:
            break
        if re.match(
            r"^(for detailed information|section\s+\d+|sp\s+21|contents\b)",
            line,
            re.IGNORECASE,
        ):
            break
        if re.match(r"^[0-9]+[.)]\s", line) and heading_lines:
            break
        heading_lines.append(line)
        if len(heading_lines) >= 5:
            break
    heading = _single_line(" ".join(heading_lines))
    heading = re.sub(
        r"^IS\s*[0-9]{1,5}\s*(?:\([^)]*\))?\s*:?\s*[0-9]{4}\s*",
        "",
        heading,
        flags=re.IGNORECASE,
    )
    heading = re.sub(r"^\(PART\s*[0-9]+\)\s*:?\s*[0-9]{4}\s*", "", heading, flags=re.IGNORECASE)
    return heading[:500].strip(" -:") or None


def _extract_scope_from_full_text(full_text: str) -> str | None:
    if not full_text:
        return None
    patterns = [
        r"(?:^|\n)\s*1[.)]?\s*Scope\s*[—\-–:�]?\s*(.+?)(?=\n\s*2[.)]\s|\n\s*2\s+[A-Z]|\n\s*SUMMARY\s+OF|\Z)",
        r"(?:^|\n)\s*Scope\s*[—\-–:�]?\s*(.+?)(?=\n\s*2[.)]\s|\n\s*SUMMARY\s+OF|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            scope = _single_line(match.group(1))
            if 20 <= len(scope) <= 1200:
                return scope
    return None


def _field_evidence(product: str, text_by_source: dict[str, str]) -> list[str]:
    aliases = PRODUCT_ALIASES.get(product, set())
    evidence = []
    for source, text in text_by_source.items():
        if any(_phrase_in_text(alias, text) for alias in aliases):
            evidence.append(source)
    return evidence or ["full_text_repeated_terms"]


def _first_pattern_match(
    patterns: list[tuple[str, str, list[str]]],
    text_by_source: dict[str, str],
    *,
    canonical_product: str | None,
) -> tuple[str | None, list[str]]:
    for product, value, aliases in patterns:
        if canonical_product and product != canonical_product:
            continue
        evidence = _aliases_evidence(aliases, text_by_source)
        if evidence:
            return value, evidence
    return None, []


def _first_value_match(
    patterns: list[tuple[str, list[str]]],
    text_by_source: dict[str, str],
) -> tuple[str | None, list[str]]:
    for value, aliases in patterns:
        evidence = _aliases_evidence(aliases, text_by_source)
        if evidence:
            return value, evidence
    return None, []


def _aliases_evidence(aliases: list[str], text_by_source: dict[str, str]) -> list[str]:
    evidence = []
    for source, text in text_by_source.items():
        if source == "full_text_repeated_terms":
            matches = sum(1 for alias in aliases if _phrase_in_text(alias, text))
            if matches >= 1 and any(_phrase_in_text(alias, text) for alias in aliases):
                evidence.append(source)
        elif any(_phrase_in_text(alias, text) for alias in aliases):
            evidence.append(source)
    return evidence


def _phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    return f" {_space_key(phrase)} " in f" {_space_key(text)} "


def _source_grounded_product_fallback(title: str, scope: str) -> str | None:
    source = title or scope
    if not source:
        return None
    cleaned = re.sub(
        r"\b(specification|requirements?|method[s]? of test|code of practice)\b",
        "",
        source,
        flags=re.IGNORECASE,
    )
    tokens = _TOKEN_PATTERN.findall(cleaned.casefold())
    stop = {
        "for",
        "and",
        "of",
        "the",
        "part",
        "specific",
        "general",
        "requirements",
        "specification",
        "method",
        "methods",
        "test",
        "first",
        "second",
        "revision",
        "is",
        "iso",
    }
    subject_tokens = [token for token in tokens if token not in stop and not token.isdigit()]
    if not subject_tokens:
        return None
    return "_".join(subject_tokens[-4:])[:128]


def _derive_primary_subject(
    title: str,
    scope: str,
    standard_kind: str,
    canonical_product: str | None,
    product_subtype: str | None,
) -> str:
    base = title or _excerpt(scope, 140)
    base = _single_line(base).strip(" -:.,")
    if not base and canonical_product:
        base = canonical_product.replace("_", " ")
    if standard_kind in NON_PRODUCT_KINDS and base:
        return base[:240]
    if product_subtype:
        return product_subtype.replace("_", " ")
    return base[:240] if base else "source-described standard subject"


def _metadata_confidence(product_confidence: str, evidence: dict[str, list[str]]) -> str:
    if product_confidence == "high" or any(
        "title" in sources or "scope" in sources for sources in evidence.values()
    ):
        return "high"
    if product_confidence == "medium" or any(
        "full_text_summary_heading" in sources for sources in evidence.values()
    ):
        return "medium"
    return "low"


def _review_reason(
    standard_kind: str,
    canonical_product: str | None,
    confidence: str,
    primary_subject: str,
    used_fallback: bool,
) -> str | None:
    if not primary_subject:
        return "missing_primary_subject"
    if confidence == "low":
        return "low_confidence_metadata"
    if used_fallback:
        return "source_grounded_fallback_requires_review"
    if standard_kind not in NON_PRODUCT_KINDS and canonical_product is None:
        return "direct_standard_without_supported_product"
    return None


def _write_null_product_resolution(
    original_null_codes: set[str],
    old_by_code: dict[str, dict[str, Any]],
    new_by_code: dict[str, CanonicalStandard],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "standard_code",
            "raw_title",
            "canonical_title_expanded",
            "standard_kind",
            "old_canonical_product",
            "new_canonical_product",
            "product_subtype",
            "primary_subject",
            "material",
            "application",
            "function",
            "applies_to_product_families",
            "confidence",
            "evidence",
            "resolution_status",
            "review_required",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for code in sorted(original_null_codes):
            old = old_by_code[code]
            new = new_by_code[code]
            writer.writerow(
                {
                    "standard_code": new.standard_code,
                    "raw_title": new.raw_title or new.title or old.get("title") or "",
                    "canonical_title_expanded": new.canonical_title_expanded or "",
                    "standard_kind": new.standard_kind,
                    "old_canonical_product": old.get("canonical_product") or "",
                    "new_canonical_product": new.canonical_product or "",
                    "product_subtype": new.product_subtype or "",
                    "primary_subject": new.primary_subject or "",
                    "material": new.material or "",
                    "application": new.application or "",
                    "function": new.function or "",
                    "applies_to_product_families": "; ".join(new.applies_to_product_families),
                    "confidence": new.metadata_confidence,
                    "evidence": json.dumps(new.metadata_evidence, sort_keys=True),
                    "resolution_status": _resolution_status(new),
                    "review_required": new.review_required,
                    "notes": new.review_reason or "",
                }
            )


def _write_product_metadata_diff(
    old_by_code: dict[str, dict[str, Any]],
    new_records: list[CanonicalStandard],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "standard_code",
            "old_product",
            "new_product",
            "old_confidence",
            "new_confidence",
            "changed",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for new in sorted(new_records, key=lambda record: record.standard_code_norm):
            old = old_by_code.get(new.standard_code_norm, {})
            old_product = old.get("canonical_product")
            old_confidence = old.get("product_confidence")
            changed = (
                old_product != new.canonical_product
                or old_confidence != new.product_confidence
            )
            writer.writerow(
                {
                    "standard_code": new.standard_code,
                    "old_product": old_product or "",
                    "new_product": new.canonical_product or "",
                    "old_confidence": old_confidence or "",
                    "new_confidence": new.product_confidence,
                    "changed": changed,
                    "reason": new.product_derivation_source if changed else "unchanged",
                }
            )


def _write_merged_artifact_review(records: list[CanonicalStandard], path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "standard_code",
            "raw_title",
            "summary_heading_present",
            "scope_present",
            "is_mentions",
            "classification",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            text = "\n".join([record.title or "", record.scope or "", record.full_text or ""])
            is_mentions = len(re.findall(r"\bIS\s*\d{2,5}\b", text, flags=re.IGNORECASE))
            if is_mentions < 5 and "merged" not in text.casefold():
                continue
            summary_present = bool(_extract_summary_heading(record.full_text or ""))
            scope_present = bool(record.scope or record.scope_reconstructed)
            classification = _merged_artifact_classification(record, summary_present, scope_present)
            counts[classification] += 1
            writer.writerow(
                {
                    "standard_code": record.standard_code,
                    "raw_title": record.raw_title or record.title or "",
                    "summary_heading_present": summary_present,
                    "scope_present": scope_present,
                    "is_mentions": is_mentions,
                    "classification": classification,
                    "notes": _merged_artifact_notes(classification),
                }
            )
    return dict(counts)


def _merged_artifact_classification(
    record: CanonicalStandard,
    summary_present: bool,
    scope_present: bool,
) -> str:
    if record.standard_code and record.title and summary_present and scope_present:
        return "SAFE"
    if record.standard_code and (summary_present or scope_present):
        return "REVIEW_REQUIRED"
    return "UNSAFE_FOR_TRAINING"


def _merged_artifact_notes(classification: str) -> str:
    return {
        "SAFE": (
            "own code, title/summary, and scope are identifiable; "
            "IS mentions treated as references"
        ),
        "REVIEW_REQUIRED": (
            "some own-record anchors are present but source boundaries need human review"
        ),
        "UNSAFE_FOR_TRAINING": "own subject cannot be reliably separated by deterministic checks",
    }[classification]


def _metadata_report(
    old_by_code: dict[str, dict[str, Any]],
    new_records: list[CanonicalStandard],
    original_null_codes: set[str],
    merged_counts: dict[str, int],
) -> dict[str, Any]:
    before_products = [record.get("canonical_product") for record in old_by_code.values()]
    before_confidence = Counter(
        record.get("product_confidence") or "missing" for record in old_by_code.values()
    )
    after_confidence = Counter(record.product_confidence for record in new_records)
    new_by_code = {record.standard_code_norm: record for record in new_records}
    null_statuses = Counter(
        _resolution_status(record)
        for code, record in new_by_code.items()
        if code in original_null_codes
    )
    return {
        "total_corpus": len(new_records),
        "original_null_products": len(original_null_codes),
        "null_resolution": {
            "resolved_product": null_statuses["RESOLVED_PRODUCT"],
            "resolved_non_product": null_statuses["RESOLVED_NON_PRODUCT"],
            "review_required": null_statuses["REVIEW_REQUIRED"],
        },
        "product_coverage": {
            "before": sum(1 for product in before_products if product),
            "after": sum(1 for record in new_records if record.canonical_product),
        },
        "confidence": {
            "before": dict(before_confidence),
            "after": dict(after_confidence),
        },
        "coverage": {
            "primary_subject": sum(1 for record in new_records if record.primary_subject),
            "product_subtype": sum(1 for record in new_records if record.product_subtype),
            "function": sum(1 for record in new_records if record.function),
            "material": sum(1 for record in new_records if record.material),
            "application": sum(1 for record in new_records if record.application),
            "applies_to_product_families": sum(
                1 for record in new_records if record.applies_to_product_families
            ),
            "review_required": sum(1 for record in new_records if record.review_required),
        },
        "merged_artifact_review": merged_counts,
    }


def _resolution_status(record: CanonicalStandard) -> str:
    if record.review_required:
        return "REVIEW_REQUIRED"
    if record.canonical_product:
        return "RESOLVED_PRODUCT"
    if record.standard_kind in NON_PRODUCT_KINDS and record.primary_subject:
        return "RESOLVED_NON_PRODUCT"
    if record.applies_to_product_families and record.primary_subject:
        return "RESOLVED_NON_PRODUCT"
    return "REVIEW_REQUIRED"


def _append_missing_issue(
    issues: list[AuditIssue],
    issue_type: str,
    index: int,
    record: dict[str, Any],
    value: str | None,
    field_name: str,
) -> None:
    if value is None:
        _append_issue(issues, issue_type, index, record, f"{field_name} is missing")


def _append_issue(
    issues: list[AuditIssue],
    issue_type: str,
    index: int,
    record: dict[str, Any],
    detail: str,
) -> None:
    issues.append(
        AuditIssue(
            issue_type=issue_type,
            source_index=index,
            is_code=_clean_text(record.get("is_code")),
            is_code_norm=normalize_source_code(record) or _clean_text(record.get("is_code_norm")),
            title=_clean_text(record.get("title")),
            detail=detail,
        )
    )


def _audit_to_dict(audit: CorpusAudit) -> dict[str, Any]:
    return {
        **asdict(audit),
        "issues": [asdict(issue) for issue in audit.issues],
    }


def _issue_count(issues: Iterable[AuditIssue], issue_type: str) -> int:
    return sum(issue.issue_type == issue_type for issue in issues)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def _space_key(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(" ".join(value.casefold().split())))


def _as_int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_length(record: dict[str, Any]) -> int:
    return sum(
        len(_clean_text(record.get(field)) or "") for field in ["title", "scope", "full_text"]
    )


def _possible_merged_record(record: dict[str, Any]) -> bool:
    text = "\n".join(
        _clean_text(record.get(field)) or "" for field in ["title", "scope", "full_text"]
    )
    is_mentions = len(re.findall(r"\bIS\s*\d{2,5}\b", text, flags=re.IGNORECASE))
    return is_mentions >= 5 or "merged" in text.casefold()


def _record_quality_score(record: dict[str, Any]) -> int:
    score = 0
    for field_name in ["is_code", "is_code_norm", "title", "scope", "full_text"]:
        if _clean_text(record.get(field_name)):
            score += 1
    score += min(_record_length(record) // 500, 10)
    return score


def _record_signature(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _space_key(_clean_text(record.get("title")) or ""),
        _space_key(_clean_text(record.get("scope")) or ""),
        _space_key(_excerpt(_clean_text(record.get("full_text")) or "", 1000)),
    )


def _key_source_description(full_text: str | None, scope: str | None) -> str:
    source = _single_line(full_text or scope or "")
    if not source:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", source)
    return _excerpt(" ".join(sentences[:2]) if sentences else source, 240)


def _excerpt(text: str, limit: int) -> str:
    text = _single_line(text)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
