from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from app.db.models import Standard
from app.schemas.tenders import (
    StandardSourceMetadata,
    StandardTechnicalDetails,
    TechnicalFact,
    TechnicalStandardDetail,
)

_MAX_EXCERPT_CHARS = 700
_MAX_FACT_EVIDENCE_CHARS = 240


def build_technical_standard_detail(standard: Standard) -> TechnicalStandardDetail:
    return TechnicalStandardDetail(
        standard_code=standard.standard_id,
        standard_id=str(standard.id),
        title=standard.title,
        canonical_title_expanded=standard.canonical_title_expanded,
        scope=standard.scope,
        scope_text=standard.scope_text,
        canonical_product=standard.canonical_product,
        product_subtype=standard.product_subtype,
        primary_subject=standard.primary_subject,
        material=standard.material,
        application=standard.application,
        function=standard.function,
        family=standard.family,
        standard_kind=standard.standard_kind,
        metadata_confidence=standard.metadata_confidence,
        metadata_evidence=standard.metadata_evidence or {},
        source_dataset=standard.source_dataset,
        source_revision=standard.source_revision,
        source_page_start=standard.source_page_start,
        source_page_end=standard.source_page_end,
        source_provenance=standard.source_provenance or [],
        publication_year=standard.publication_year,
        lifecycle_status=standard.lifecycle_status,
        lifecycle_status_note=standard.lifecycle_status_note,
        full_text_excerpt=_short_excerpt(standard.full_text, max_chars=500),
    )


def build_readable_technical_details(standard: Standard) -> StandardTechnicalDetails:
    text = _source_text(standard)
    normalized_text = _normalize(text)
    overview = _first_present(standard.scope, standard.scope_text, standard.scope_reconstructed)
    nominal_sizes = _nominal_size_facts(normalized_text)
    dimensional_requirements = _dimensional_requirement_facts(normalized_text)
    return StandardTechnicalDetails(
        overview=overview,
        scope=overview,
        product=_product_name(standard),
        subtype=standard.product_subtype,
        material=_validated_material(standard),
        application=standard.application,
        function=standard.function,
        primary_subject=standard.primary_subject,
        family=standard.family,
        nominal_sizes=nominal_sizes,
        classification=_classification_facts(normalized_text),
        dimensional_requirements=dimensional_requirements,
        dimensions=nominal_sizes,
        performance_requirements=_performance_facts(normalized_text),
        workmanship_requirements=_workmanship_facts(normalized_text),
        test_requirements=_test_requirement_facts(normalized_text),
        grades=_grade_facts(normalized_text),
        temperature_ranges=_temperature_facts(normalized_text),
        other_requirements=_other_requirement_facts(normalized_text),
        source_excerpt=_source_excerpt(standard),
    )


def build_source_metadata(standard: Standard) -> StandardSourceMetadata:
    return StandardSourceMetadata(
        source_dataset=standard.source_dataset,
        source_revision=standard.source_revision,
        source_page_start=standard.source_page_start,
        source_page_end=standard.source_page_end,
        source_provenance=standard.source_provenance or [],
        metadata_confidence=standard.metadata_confidence,
        metadata_evidence=standard.metadata_evidence or {},
        publication_year=standard.publication_year,
        lifecycle_status=standard.lifecycle_status,
        lifecycle_status_note=standard.lifecycle_status_note,
    )


def _source_text(standard: Standard) -> str:
    evidence = _jsonish_text(standard.metadata_evidence)
    search_profile = _jsonish_text(standard.search_profile)
    return "\n".join(
        value
        for value in [
            standard.full_text,
            standard.scope,
            standard.scope_text,
            standard.scope_reconstructed,
            standard.retrieval_text,
            evidence,
            search_profile,
        ]
        if value
    )


def _jsonish_text(value: Any) -> str:
    if not value:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _source_excerpt(standard: Standard) -> str | None:
    return _short_excerpt(
        _first_present(
            standard.scope,
            standard.scope_text,
            standard.scope_reconstructed,
            standard.full_text,
            standard.retrieval_text,
        ),
        max_chars=_MAX_EXCERPT_CHARS,
    )


def _short_excerpt(value: str | None, *, max_chars: int) -> str | None:
    if not value:
        return None
    text = _normalize(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _product_name(standard: Standard) -> str | None:
    for value in [
        standard.primary_subject,
        standard.canonical_title_expanded,
        standard.title,
        standard.canonical_product,
    ]:
        if value:
            return _display_text(value)
    return None


def _validated_material(standard: Standard) -> str | None:
    title_scope = _normalize(
        " ".join(
            value
            for value in [
                standard.title,
                standard.canonical_title_expanded,
                standard.scope,
                standard.scope_text,
                standard.scope_reconstructed,
                standard.primary_subject,
            ]
            if value
        )
    ).casefold()
    if "clay" in title_scope:
        return "clay"
    if "calcium silicate" in title_scope:
        return "calcium silicate"
    if "high density polyethylene" in title_scope or "hdpe" in title_scope:
        return "HDPE"
    if (
        "unplasticized polyvinyl chloride" in title_scope
        or "unplasticised polyvinyl chloride" in title_scope
        or "upvc" in title_scope
    ):
        return "UPVC"
    if "ordinary portland cement" in title_scope or "portland cement" in title_scope:
        return "cement"
    material = standard.material
    if material and _is_test_equipment_material(material, standard.full_text or ""):
        return None
    return material


def _is_test_equipment_material(material: str, source_text: str) -> bool:
    lowered = material.casefold()
    if lowered not in {"steel", "glass", "metal"}:
        return False
    source = source_text.casefold()
    equipment_phrases = [
        f"{lowered} ball",
        f"{lowered} mandrel",
        f"{lowered} test",
        f"{lowered} fixture",
        f"{lowered} container",
    ]
    return any(phrase in source for phrase in equipment_phrases)


def _nominal_size_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    size_pattern = (
        r"\b(\d+(?:\.\d+)?\s*(?:x|X|\u00d7)\s*\d+(?:\.\d+)?"
        r"(?:\s*(?:x|X|\u00d7)\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|m))\b"
    )
    for match in re.finditer(size_pattern, text):
        context = _window(text, match.span(), before=120, after=20).casefold()
        if _is_tolerance_context(context) or _is_test_row_context(context):
            continue
        if not any(token in context for token in ["dimension", "size", "nominal"]):
            continue
        facts.append(
            _fact(
                "Nominal size",
                _normalize_size(match.group(1)),
                text,
                match.span(),
                fact_type="NOMINAL_SIZE",
                name="Nominal size",
                unit=_unit_from_value(match.group(1)),
            )
        )

    diameter_patterns = [
        r"\b(\d+(?:\.\d+)?\s*(?:mm|cm|m))\s+nominal\s+diameter\b",
        r"\bnominal\s+diameter[^.;]{0,40}?(\d+(?:\.\d+)?\s*(?:mm|cm|m))\b",
        r"\bdiameter[^.;]{0,40}?(\d+(?:\.\d+)?\s*(?:mm|cm|m))\b",
    ]
    for pattern in diameter_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context = _window(text, match.span(), before=100, after=80).casefold()
            if _is_tolerance_context(context):
                continue
            facts.append(
                _fact(
                    "Nominal diameter",
                    _normalize(match.group(1)),
                    text,
                    match.span(),
                    fact_type="NOMINAL_SIZE",
                    name="Nominal diameter",
                    unit=_unit_from_value(match.group(1)),
                )
            )
    return _dedupe(facts, limit=12)


def _dimensional_requirement_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    for match in re.finditer(
        r"(?:grooves?\s+or\s+frogging|frogging|groove)[^.]{0,80}?"
        r"(?:shall\s+)?not\s+exceed\s+(\d+(?:\.\d+)?)\s*(mm|cm|m)\b",
        text,
        flags=re.IGNORECASE,
    ):
        facts.append(
            _fact(
                "Groove/frogging depth",
                f"maximum {match.group(1)} {match.group(2)}",
                text,
                match.span(),
                fact_type="MAXIMUM_LIMIT",
                name="Groove/frogging depth",
                qualifier="maximum",
                unit=match.group(2),
            )
        )

    tolerance_patterns = [
        (
            "Length/breadth tolerance",
            r"Length\s+and\s+breadth[^.;]{0,80}?Average\s*([+\-\u00b1]?\s*\d+(?:\.\d+)?\s*mm)"
            r"[^.;]{0,80}?individual\s*([+\-\u00b1]?\s*\d+(?:\.\d+)?\s*mm)",
        ),
        (
            "Thickness tolerance",
            r"Thickness[^.;]{0,80}?Average\s*([+\-\u00b1]?\s*\d+(?:\.\d+)?\s*mm)"
            r"[^.;]{0,80}?individual\s*([+\-\u00b1]?\s*\d+(?:\.\d+)?\s*mm)",
        ),
    ]
    for label, pattern in tolerance_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = (
                f"Average {_clean_sign(match.group(1))}; "
                f"Individual {_clean_sign(match.group(2))}"
            )
            facts.append(
                _fact(
                    label,
                    value,
                    text,
                    match.span(),
                    fact_type="DIMENSIONAL_TOLERANCE",
                    name=label,
                    qualifier="tolerance",
                    unit="mm",
                )
            )

    warpage = re.search(
        r"Warpage[^.;]{0,120}?not\s+exceed\s+(\d+(?:\.\d+)?)\s*percent\s+along\s+edges"
        r"[^.;]{0,80}?(\d+(?:\.\d+)?)\s*percent\s+along\s+diagonals",
        text,
        flags=re.IGNORECASE,
    )
    if warpage:
        facts.append(
            _fact(
                "Warpage",
                (
                    f"maximum {warpage.group(1)} percent along edges; "
                    f"maximum {warpage.group(2)} percent along diagonals"
                ),
                text,
                warpage.span(),
                fact_type="MAXIMUM_LIMIT",
                name="Warpage",
                qualifier="maximum",
                unit="percent",
            )
        )
    return _dedupe(facts, limit=12)


def _temperature_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    patterns = [
        (
            "Maximum temperature",
            r"\b(?:operating|service|maximum|max)?\s*(?:temperature|temperatures)?"
            r"[^.;]{0,40}?\b(?:up\s*to|upto|not exceeding|maximum|max)\s+"
            r"(\d+(?:\.\d+)?)\s*(?:\u00b0\s*C|\u00ba\s*C|0C|degrees?\s+Celsius|deg\s*C)\b",
        ),
        (
            "Temperature range",
            r"\b(\d+(?:\.\d+)?\s*(?:\u00b0\s*C|\u00ba\s*C|0C|degrees?\s+Celsius|deg\s*C)"
            r"\s*(?:to|-)\s*\d+(?:\.\d+)?\s*"
            r"(?:\u00b0\s*C|\u00ba\s*C|0C|degrees?\s+Celsius|deg\s*C))\b",
        ),
        (
            "Temperature",
            r"\b(\d+(?:\.\d+)?\s*(?:\u00b0\s*C|\u00ba\s*C|0C|degrees?\s+Celsius|deg\s*C))\b",
        ),
    ]
    for label, pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context = _window(text, match.span(), before=80, after=60).casefold()
            if not any(token in context for token in ["temperature", "temperatures", "heat"]):
                continue
            facts.append(
                _fact(
                    label,
                    _normalize_temperature(match.group(1)),
                    text,
                    match.span(),
                    fact_type="TEMPERATURE_LIMIT",
                    name=label,
                    unit="C",
                )
            )
    return _dedupe(facts, limit=6)


def _grade_facts(text: str) -> list[TechnicalFact]:
    patterns = [
        r"\b(\d{2,3}\s+grade)\b",
        r"\b(grade\s+[A-Z0-9]+)\b",
        r"\b(class\s+[0-9A-Z]+)\b",
        r"\b(product\s+grade\s+[A-Z])\b",
        r"\b(property\s+class\s+[0-9.]+)\b",
    ]
    facts = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            facts.append(
                _fact(
                    "Grade/Class",
                    _display_text(match.group(1)),
                    text,
                    match.span(),
                    fact_type="GRADE",
                    name="Grade/Class",
                )
            )
    return _dedupe(facts, limit=8)


def _classification_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    match = re.search(
        r"Classification[^.;]{0,120}?(Class\s+1)[^.;]{0,40}?(Class\s+2)[^.;]{0,40}?(Class\s+3)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        for group in match.groups():
            facts.append(
                _fact(
                    "Classification",
                    _display_text(group),
                    text,
                    match.span(),
                    fact_type="CLASSIFICATION",
                    name="Classification",
                )
            )
    return _dedupe(facts, limit=10)


def _performance_facts(text: str) -> list[TechnicalFact]:
    table_facts = _classification_table_facts(text)
    if table_facts:
        return table_facts
    keywords = {
        "compressive strength": "Compressive strength",
        "flexural strength": "Flexural strength",
        "impact": "Impact requirement",
        "water absorption": "Water absorption",
        "thermal conductivity": "Thermal conductivity",
        "bulk density": "Bulk density",
        "setting time": "Setting time",
        "soundness": "Soundness",
        "fineness": "Fineness",
        "leakage": "Leakage",
        "water tightness": "Water tightness",
        "pressure": "Pressure/rating",
        "rating": "Pressure/rating",
        "strength": "Strength",
    }
    return _keyword_sentence_facts(text, keywords, limit=10)


def _other_requirement_facts(text: str) -> list[TechnicalFact]:
    keywords = {
        "shall be free": "Workmanship",
        "shall not exceed": "Limit",
        "not more than": "Maximum",
        "not less than": "Minimum",
        "shall be between": "Range",
        "tolerance": "Tolerance",
        "tolerances": "Tolerance",
        "smooth": "Finish",
        "watertight": "Water tightness",
        "no leakage": "No leakage",
    }
    return _keyword_sentence_facts(text, keywords, limit=8)


def _workmanship_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    for match in re.finditer(
        r"(?:General\s+Quality|Workmanship)[^.;]{0,80}?shall\s+be\s+free\s+from[^.]{20,220}\.",
        text,
        flags=re.IGNORECASE,
    ):
        sentence = _normalize(match.group(0))
        facts.append(
            TechnicalFact(
                label="Workmanship",
                name="Workmanship",
                type="WORKMANSHIP_REQUIREMENT",
                value=_truncate(sentence, 220),
                evidence=_truncate(sentence, _MAX_FACT_EVIDENCE_CHARS),
            )
        )
    return _dedupe(facts, limit=6)


def _test_requirement_facts(text: str) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    for pattern, label in [
        (r"Impact\s+maximum\s+height[^.]{20,260}", "Impact test"),
        (r"Note\s+[—-]\s+For\s+methods?\s+of\s+tests?[^.]{20,240}", "Test method"),
    ]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _truncate(match.group(0), 220)
            facts.append(
                TechnicalFact(
                    label=label,
                    name=label,
                    type="TEST_REQUIREMENT",
                    value=value,
                    evidence=_truncate(value, _MAX_FACT_EVIDENCE_CHARS),
                )
            )
    return _dedupe(facts, limit=6)


def _classification_table_facts(text: str) -> list[TechnicalFact]:
    if not all(
        token in text.casefold()
        for token in ["water absorption", "class 1", "class 2", "class 3"]
    ):
        return []
    facts: list[TechnicalFact] = []
    patterns = [
        (
            "Water absorption maximum",
            r"Water\s+absorption\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+percent,\s*Max",
            "percent",
        ),
        (
            "Flexural strength average",
            r"Flexural\s+strength[^.]{0,180}?Average\s+"
            r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
            None,
        ),
        (
            "Flexural strength individual",
            r"Individual\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+iii\)\s+Impact",
            None,
        ),
    ]
    for name, pattern, unit in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _class_values(match.groups(), unit)
            facts.append(
                _fact(
                    name,
                    value,
                    text,
                    match.span(),
                    fact_type="PERFORMANCE_REQUIREMENT",
                    name=name,
                    unit=unit,
                )
            )

    impact_patterns = [
        ("Impact test - 15 mm tile", r"15\s+mm\s+thick\s+(\d+)\s+(\d+)\s+(\d+)"),
        ("Impact test - 20 mm tile", r"20\s+mm\s+thick\s+(\d+)\s+(\d+)\s+(\d+)"),
        ("Impact test - 25 mm tile", r"25\s+mm\s+thick\s+(\d+)\s+(\d+)\s+(\d+)"),
        ("Impact test - 30 mm tile", r"30\s+mm\s+thick\s+(\d+)\s+(\d+)\s+(\d+)"),
    ]
    for name, pattern in impact_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            facts.append(
                _fact(
                    name,
                    _class_values(match.groups(), "mm"),
                    text,
                    match.span(),
                    fact_type="PERFORMANCE_REQUIREMENT",
                    name=name,
                    qualifier="impact maximum height",
                    unit="mm",
                )
            )
    return _dedupe(facts, limit=12)


def _keyword_sentence_facts(
    text: str,
    keywords: dict[str, str],
    *,
    limit: int,
) -> list[TechnicalFact]:
    facts: list[TechnicalFact] = []
    for sentence in _sentences(text):
        low = sentence.casefold()
        if not any(token in low for token in ["shall", "not ", "min", "max", "between", "%"]):
            continue
        for keyword, label in keywords.items():
            if keyword in low:
                facts.append(
                    TechnicalFact(
                        label=label,
                        name=label,
                        type="OTHER_REQUIREMENT",
                        value=_short_value(sentence),
                        evidence=_truncate(sentence, _MAX_FACT_EVIDENCE_CHARS),
                    )
                )
                break
        if len(facts) >= limit:
            break
    return _dedupe(facts, limit=limit)


def _sentences(text: str) -> Iterable[str]:
    for chunk in re.split(r"(?<=[.;])\s+|\n+", text):
        normalized = _normalize(chunk)
        if 24 <= len(normalized) <= 500:
            yield normalized


def _fact(
    label: str,
    value: str,
    text: str,
    span: tuple[int, int],
    *,
    fact_type: str | None = None,
    name: str | None = None,
    qualifier: str | None = None,
    unit: str | None = None,
) -> TechnicalFact:
    return TechnicalFact(
        label=label,
        value=_normalize(value),
        evidence=_evidence(text, span),
        type=fact_type,
        name=name,
        qualifier=qualifier,
        unit=unit,
    )


def _evidence(text: str, span: tuple[int, int]) -> str:
    start = max(0, span[0] - 100)
    end = min(len(text), span[1] + 100)
    return _truncate(_normalize(text[start:end]), _MAX_FACT_EVIDENCE_CHARS)


def _dedupe(facts: list[TechnicalFact], *, limit: int) -> list[TechnicalFact]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TechnicalFact] = []
    for fact in facts:
        key = (fact.label.casefold(), fact.value.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
        if len(deduped) >= limit:
            break
    return deduped


def _short_value(sentence: str) -> str:
    return _truncate(sentence, 180)


def _window(text: str, span: tuple[int, int], *, before: int, after: int) -> str:
    start = max(0, span[0] - before)
    end = min(len(text), span[1] + after)
    return text[start:end]


def _is_tolerance_context(context: str) -> bool:
    lowered = context.casefold()
    return any(
        token in lowered
        for token in [
            "tolerance",
            "tolerances",
            "variation",
            "average",
            "individual",
            "+",
            "\u00b1",
            "plus",
            "minus",
        ]
    )


def _is_test_row_context(context: str) -> bool:
    lowered = context.casefold()
    return any(token in lowered for token in ["impact", "drop of steel ball", "test"])


def _normalize_size(value: str) -> str:
    text = _normalize(value)
    text = re.sub(r"\s*(?:x|X|\u00d7)\s*", " × ", text)
    return text


def _normalize_temperature(value: str) -> str:
    text = _normalize(value)
    text = re.sub(r"\s*(?:\u00b0\s*C|\u00ba\s*C|0C|degrees?\s+Celsius|deg\s*C)\b", " C", text)
    return text


def _unit_from_value(value: str) -> str | None:
    match = re.search(r"\b(mm|cm|m|percent|%)\b", value, flags=re.IGNORECASE)
    if match:
        return "percent" if match.group(1) == "%" else match.group(1)
    return None


def _clean_sign(value: str) -> str:
    compact = re.sub(r"\s+", "", value).replace("+-", "\u00b1")
    match = re.match(r"([+\-\u00b1]?)(\d+(?:\.\d+)?)([A-Za-z]+)", compact)
    if match:
        return f"{match.group(1)}{match.group(2)} {match.group(3)}"
    return compact


def _display_text(value: str) -> str:
    text = _normalize(value).replace("_", " ")
    if text.isupper():
        return text.title()
    return text[:1].upper() + text[1:]


def _class_values(values: Iterable[str], unit: str | None) -> str:
    suffix = f" {unit}" if unit else ""
    class_values = [
        f"Class {index} = {_normalize(value)}{suffix}"
        for index, value in enumerate(values, start=1)
    ]
    return "; ".join(class_values)


def _truncate(value: str, max_chars: int) -> str:
    text = _normalize(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def _first_present(*values: str | None) -> str | None:
    for value in values:
        cleaned = _short_excerpt(value, max_chars=_MAX_EXCERPT_CHARS)
        if cleaned:
            return cleaned
    return None
