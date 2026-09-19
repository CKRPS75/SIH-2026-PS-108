from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from difflib import get_close_matches
from time import perf_counter, time
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.services.parsed_standards_corpus import (
    APPLICATION_PATTERNS,
    FUNCTION_PATTERNS,
    MATERIAL_PATTERNS,
    PRODUCT_ALIASES,
    SUBTYPE_PATTERNS,
    canonicalize_product,
)

SYSTEM_INSTRUCTION = """You are a procurement-requirement semantic parser.

Extract only information stated or strongly implied by the user's product and description.
Do not invent technical requirements.
Do not output Indian Standard numbers.
Do not recommend BIS/IS standards.
Separate attributes of the procured product from attributes of surrounding equipment/context.

Examples:

"valve on steel water pipeline"

means:
product = valve
context_material = steel
application = water pipeline

NOT:
valve material = steel

"plastic pipe for underground sewage"

means:
product = pipe
material = plastic
application = sewage
installation_context = underground

If the description does not contain enough information to distinguish a subtype,
set ambiguity=true and explain what discriminator is missing.

Examples:

"valve for water pipeline"

should NOT invent check valve / sluice valve / pressure reducing valve.
Instead:
ambiguity=true
missing_information includes valve function.

"Portland pozzolana cement"

without fly ash/calcined clay information should be ambiguous.

Represent negative constraints separately. For example, "tile for interior flooring,
not roofing" means application=flooring and excluded_application=roofing.
"""


class SemanticQueryIntent(BaseModel):
    normalized_product: str | None = None
    product_aliases: list[str] = Field(default_factory=list)
    true_product_aliases: list[str] = Field(default_factory=list)
    known_subtypes: list[str] = Field(default_factory=list)
    query_subtype: str | None = None
    subtype: str | None = None
    subtype_family: str | None = None
    cement_type: str | None = None
    pozzolana_source: str | None = None
    head_shape: str | None = None
    form: str | None = None

    material: list[str] = Field(default_factory=list)
    context_material: list[str] = Field(default_factory=list)
    excluded_material: list[str] = Field(default_factory=list)

    function: list[str] = Field(default_factory=list)
    application: list[str] = Field(default_factory=list)
    medium: list[str] = Field(default_factory=list)
    installation_context: list[str] = Field(default_factory=list)
    excluded_function: list[str] = Field(default_factory=list)
    excluded_application: list[str] = Field(default_factory=list)
    excluded_subtype: list[str] = Field(default_factory=list)
    excluded_medium: list[str] = Field(default_factory=list)
    excluded_installation_context: list[str] = Field(default_factory=list)

    grade: str | None = None
    temperature_c: float | None = None
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    pressure: str | None = None

    explicit_constraints: list[str] = Field(default_factory=list)
    context_only_terms: list[str] = Field(default_factory=list)
    attribute_evidence: list[str] = Field(default_factory=list)

    ambiguity: bool = False
    missing_information: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "product_aliases",
        "true_product_aliases",
        "known_subtypes",
        "material",
        "context_material",
        "excluded_material",
        "function",
        "application",
        "medium",
        "installation_context",
        "excluded_function",
        "excluded_application",
        "excluded_subtype",
        "excluded_medium",
        "excluded_installation_context",
        "explicit_constraints",
        "context_only_terms",
        "missing_information",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]


@dataclass(frozen=True)
class QueryInterpretationResult:
    intent: SemanticQueryIntent | None
    gemini_used: bool
    gemini_success: bool
    gemini_latency_ms: float
    fallback_reason: str | None


class GeminiQueryInterpreter:
    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        timeout_s: float = 5.0,
        cache_ttl_s: float = 300.0,
        cache_max_size: int = 128,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._cache_ttl_s = cache_ttl_s
        self._cache_max_size = cache_max_size
        self._client = client
        self._cache: OrderedDict[tuple[str, str], tuple[float, SemanticQueryIntent]] = OrderedDict()

    async def interpret(self, *, product: str, description: str) -> QueryInterpretationResult:
        started_at = perf_counter()
        cache_key = (_space_key(product), _space_key(description))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return QueryInterpretationResult(
                intent=cached,
                gemini_used=True,
                gemini_success=True,
                gemini_latency_ms=(perf_counter() - started_at) * 1000,
                fallback_reason=None,
            )
        if not self._api_key and self._client is None:
            return QueryInterpretationResult(
                intent=None,
                gemini_used=False,
                gemini_success=False,
                gemini_latency_ms=(perf_counter() - started_at) * 1000,
                fallback_reason="missing_api_key",
            )

        try:
            raw_intent = await asyncio.wait_for(
                asyncio.to_thread(self._generate, product, description),
                timeout=self._timeout_s,
            )
            intent = normalize_intent(raw_intent, product=product, description=description)
        except TimeoutError:
            return self._failure(started_at, "timeout")
        except ValidationError:
            return self._failure(started_at, "invalid_structured_response")
        except Exception as exc:  # noqa: BLE001 - external SDK failures become fallback reasons
            reason = f"{exc.__class__.__name__}: {str(exc)[:160]}"
            return self._failure(started_at, reason)

        self._put_cached(cache_key, intent)
        return QueryInterpretationResult(
            intent=intent,
            gemini_used=True,
            gemini_success=True,
            gemini_latency_ms=(perf_counter() - started_at) * 1000,
            fallback_reason=None,
        )

    def _generate(self, product: str, description: str) -> SemanticQueryIntent:
        client = self._get_client()
        prompt = (
            "Parse this procurement requirement into the response schema.\n\n"
            f"Product:\n{product.strip()}\n\n"
            f"Description:\n{description.strip()}"
        )
        try:
            from google.genai import types
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("google-genai is required for Gemini query interpretation") from exc

        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SemanticQueryIntent,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SemanticQueryIntent):
            return parsed
        if isinstance(parsed, dict):
            return SemanticQueryIntent.model_validate(parsed)
        text = getattr(response, "text", None)
        if text:
            return SemanticQueryIntent.model_validate_json(text)
        raise ValueError("Gemini returned no structured intent")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "google-genai is required for Gemini query interpretation"
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _failure(self, started_at: float, reason: str) -> QueryInterpretationResult:
        return QueryInterpretationResult(
            intent=None,
            gemini_used=True,
            gemini_success=False,
            gemini_latency_ms=(perf_counter() - started_at) * 1000,
            fallback_reason=reason,
        )

    def _get_cached(self, key: tuple[str, str]) -> SemanticQueryIntent | None:
        value = self._cache.get(key)
        if value is None:
            return None
        cached_at, intent = value
        if time() - cached_at > self._cache_ttl_s:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return intent

    def _put_cached(self, key: tuple[str, str], intent: SemanticQueryIntent) -> None:
        self._cache[key] = (time(), intent)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)


def normalize_intent(
    intent: SemanticQueryIntent,
    *,
    product: str,
    description: str,
) -> SemanticQueryIntent:
    text = f"{product} {description}"
    normalized_product = _normalize_product(intent.normalized_product or product)
    material = _normalize_values(intent.material, MATERIAL_PATTERNS)
    context_material = _normalize_values(intent.context_material, MATERIAL_PATTERNS)

    material, context_material, context_only_terms = _separate_context_material(
        product=normalized_product,
        text=text,
        material=material,
        context_material=context_material,
        context_only_terms=intent.context_only_terms,
    )

    function = _normalize_values(intent.function, FUNCTION_PATTERNS)
    function = _grounded_functions(function, description)
    application = _normalize_values(intent.application, APPLICATION_PATTERNS)
    subtype = _normalize_subtype(intent.query_subtype or intent.subtype, normalized_product)
    subtype_family = _normalize_subtype_family(intent.subtype_family or subtype, normalized_product)
    cement_type = _normalize_cement_type(intent.cement_type, description, subtype)
    pozzolana_source = _normalize_pozzolana_source(
        intent.pozzolana_source,
        description,
        subtype,
    )
    head_shape = _normalize_head_shape(intent.head_shape, description)
    form = _normalize_form(intent.form, description)
    medium = _normalize_medium(intent.medium, text)
    installation = _normalize_installation(intent.installation_context, text)
    excluded_material = _normalize_values(intent.excluded_material, MATERIAL_PATTERNS)
    excluded_function = _normalize_values(intent.excluded_function, FUNCTION_PATTERNS)
    excluded_application = _normalize_values(intent.excluded_application, APPLICATION_PATTERNS)
    excluded_subtype = []
    for item in intent.excluded_subtype:
        normalized_subtype = _normalize_subtype(item, normalized_product)
        if normalized_subtype:
            excluded_subtype.append(normalized_subtype)
    excluded_medium = _normalize_medium(intent.excluded_medium, "")
    excluded_installation = _normalize_installation(intent.excluded_installation_context, "")

    negatives = _extract_negatives(description, normalized_product)
    excluded_material = _merge_unique(excluded_material, negatives["material"])
    excluded_function = _merge_unique(excluded_function, negatives["function"])
    excluded_application = _merge_unique(excluded_application, negatives["application"])
    excluded_subtype = _merge_unique(excluded_subtype, negatives["subtype"])
    excluded_medium = _merge_unique(excluded_medium, negatives["medium"])
    excluded_installation = _merge_unique(excluded_installation, negatives["installation_context"])
    excluded_medium = _sanitize_excluded_medium(excluded_medium, description)

    inferred = _infer_grounded_terms(
        product=normalized_product,
        description=description,
        function=function,
        application=application,
        subtype=subtype,
        material=material,
        medium=medium,
        installation=installation,
    )
    function = _merge_unique(function, inferred["function"])
    application = _merge_unique(application, inferred["application"])
    material = _merge_unique(material, inferred["material"])
    medium = _merge_unique(medium, inferred["medium"])
    installation = _merge_unique(installation, inferred["installation_context"])
    if inferred["subtype"] and subtype in {None, "portland_pozzolana_cement"}:
        subtype = inferred["subtype"]
    subtype_family = subtype_family or inferred["subtype_family"]
    cement_type = cement_type or inferred["cement_type"]
    pozzolana_source = pozzolana_source or inferred["pozzolana_source"]
    head_shape = head_shape or inferred["head_shape"]
    form = form or inferred["form"]
    material = _without_excluded(material, excluded_material)
    function = _without_excluded(function, excluded_function)
    application = _without_excluded(application, excluded_application)
    medium = _without_excluded(medium, excluded_medium)
    installation = _without_excluded(installation, excluded_installation)
    if subtype in set(excluded_subtype):
        subtype = None

    ambiguity = intent.ambiguity
    missing = list(intent.missing_information)
    if normalized_product == "valve" and not function and not subtype:
        ambiguity = True
        missing.append("valve function")
    if normalized_product == "cement" and cement_type == "ppc":
        if subtype not in {"portland_pozzolana_fly_ash", "portland_pozzolana_calcined_clay"}:
            ambiguity = True
            missing.append("pozzolana type: fly ash or calcined clay")
        else:
            missing = [
                item
                for item in missing
                if "pozzolana" not in item.casefold()
                and "fly ash" not in item.casefold()
                and "calcined clay" not in item.casefold()
            ]
            ambiguity = bool(missing) and ambiguity
    if normalized_product == "cement" and subtype in {
        "portland_pozzolana_fly_ash",
        "portland_pozzolana_calcined_clay",
    }:
        material = [
            value for value in material if value not in {"clay", "fly ash", "calcined clay"}
        ]
    if (
        normalized_product == "cement"
        and cement_type == "opc"
        and not _normalize_grade(None, description)
    ):
        ambiguity = True
        missing.append("cement grade")

    product_aliases = _true_product_aliases(normalized_product, intent.product_aliases)
    known_subtypes = _known_subtypes(normalized_product)
    return SemanticQueryIntent(
        normalized_product=normalized_product,
        product_aliases=product_aliases,
        true_product_aliases=product_aliases,
        known_subtypes=known_subtypes,
        query_subtype=subtype,
        subtype=subtype,
        subtype_family=subtype_family,
        cement_type=cement_type,
        pozzolana_source=pozzolana_source,
        head_shape=head_shape,
        form=form,
        material=material,
        context_material=context_material,
        excluded_material=excluded_material,
        function=function,
        application=application,
        medium=medium,
        installation_context=installation,
        excluded_function=excluded_function,
        excluded_application=excluded_application,
        excluded_subtype=excluded_subtype,
        excluded_medium=excluded_medium,
        excluded_installation_context=excluded_installation,
        grade=_normalize_grade(intent.grade, description),
        temperature_c=intent.temperature_c or _extract_temperature_c(description),
        temperature_min_c=intent.temperature_min_c,
        temperature_max_c=intent.temperature_max_c,
        pressure=_clean_optional(intent.pressure),
        explicit_constraints=_clean_list(intent.explicit_constraints),
        context_only_terms=context_only_terms,
        attribute_evidence=_attribute_evidence(description, product, subtype=subtype),
        ambiguity=ambiguity,
        missing_information=_dedupe(missing),
        confidence=max(0.0, min(intent.confidence, 1.0)),
    )


def build_enriched_query(product: str, description: str, intent: SemanticQueryIntent | None) -> str:
    if intent is None:
        from app.services.parsed_standards_corpus import build_product_query

        return build_product_query(product, description)
    fields = [f"product: {product.strip()}"]
    if intent.normalized_product and intent.normalized_product != _space_key(product):
        fields.append(f"normalized product: {intent.normalized_product.replace('_', ' ')}")
    for label, values in [
        ("subtype", [intent.subtype] if intent.subtype else []),
        ("subtype family", [intent.subtype_family] if intent.subtype_family else []),
        ("cement type", [intent.cement_type] if intent.cement_type else []),
        ("pozzolana source", [intent.pozzolana_source] if intent.pozzolana_source else []),
        ("material", intent.material),
        ("function", intent.function),
        ("application", intent.application),
        ("medium", intent.medium),
        ("installation", intent.installation_context),
    ]:
        if values:
            fields.append(f"{label}: {', '.join(value.replace('_', ' ') for value in values)}")
    if intent.grade:
        fields.append(f"grade: {intent.grade}")
    fields.append(f"original requirement: {description.strip()}")
    return "; ".join(fields)


def interpretation_to_dict(intent: SemanticQueryIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return intent.model_dump()


def _normalize_product(value: str | None) -> str | None:
    if not value:
        return None
    product, confidence = canonicalize_product(value)
    if product and confidence == "high":
        return product
    return _fuzzy_product(value)


def _fuzzy_product(value: str) -> str | None:
    normalized = _space_key(value)
    choices: dict[str, str] = {}
    for product, aliases in PRODUCT_ALIASES.items():
        choices[product] = product
        for alias in aliases:
            choices[_space_key(alias)] = product
    match = get_close_matches(normalized, list(choices), n=1, cutoff=0.82)
    return choices[match[0]] if match else None


def _normalize_values(values: list[str], patterns: list[tuple[str, list[str]]]) -> list[str]:
    normalized = []
    for value in values:
        matched = _match_pattern_value(value, patterns)
        normalized.append(matched or _space_key(value))
    return _dedupe(value for value in normalized if value)


def _normalize_subtype(value: str | None, product: str | None) -> str | None:
    if not value:
        return None
    text = _space_key(value)
    for pattern_product, subtype, aliases in SUBTYPE_PATTERNS:
        if product and pattern_product != product:
            continue
        if any(_contains(text, alias) for alias in aliases):
            return subtype
    return text.replace(" ", "_")[:128] if text else None


def _normalize_medium(values: list[str], text: str) -> list[str]:
    medium = _clean_list(values)
    if _contains(text, "water"):
        medium.append("water")
    if _contains(text, "sewage"):
        medium.append("sewage")
    return _dedupe(medium)


def _normalize_installation(values: list[str], text: str) -> list[str]:
    installation = _clean_list(values)
    if _contains(text, "underground"):
        installation.append("underground")
    return _dedupe(installation)


def _infer_grounded_terms(
    *,
    product: str | None,
    description: str,
    function: list[str],
    application: list[str],
    subtype: str | None,
    material: list[str],
    medium: list[str],
    installation: list[str],
) -> dict[str, Any]:
    inferred = {
        "function": [],
        "application": [],
        "material": [],
        "medium": [],
        "installation_context": [],
        "subtype": None,
        "subtype_family": None,
        "cement_type": None,
        "pozzolana_source": None,
        "head_shape": None,
        "form": None,
    }
    text = _space_key(description)
    for value, aliases in FUNCTION_PATTERNS:
        if value == "structural_fastening":
            continue
        if any(_contains(text, alias) for alias in aliases):
            inferred["function"].append(value)
    if (
        "come backward" in text
        or "flowing backwards" in text
        or "flow backwards" in text
        or "flow one way" in text
        or "only flow one way" in text
        or "back flow" in text
        or "backflow" in text
        or "backward" in text
    ):
        inferred["function"].append("prevent_reverse_flow")
    for value, aliases in APPLICATION_PATTERNS:
        if any(_contains(text, alias) for alias in aliases):
            inferred["application"].append(value)
    for value, aliases in MATERIAL_PATTERNS:
        if any(_contains(text, alias) for alias in aliases):
            inferred["material"].append(value)
    if "plastic" in text:
        inferred["material"].append("plastic")
    if "water" in text:
        inferred["medium"].append("water")
        if not application:
            inferred["application"].append("water pipeline")
    if "drinking water" in text or "potable water" in text:
        inferred["application"].append("potable water supply")
        inferred["function"].append("carry_potable_water")
    if "industrial waste" in text or "industrial effluent" in text:
        inferred["application"].append("industrial waste")
    if "non potable water" in text:
        inferred["application"].append("industrial waste")
    if "sewage" in text or "sewerage" in text:
        inferred["medium"].append("sewage")
        inferred["application"].append("sewerage")
    if "underground" in text:
        inferred["installation_context"].append("underground")
    if product == "valve" and "prevent_reverse_flow" in {*function, *inferred["function"]}:
        inferred["subtype"] = subtype or "check_valve"
        inferred["subtype_family"] = "check_valve"
    if product == "valve" and "reduce_pressure" in {*function, *inferred["function"]}:
        inferred["subtype"] = subtype or "pressure_reducing_valve"
    if product == "cement" and "calcined clay" in text:
        inferred["subtype"] = "portland_pozzolana_calcined_clay"
        inferred["cement_type"] = "ppc"
        inferred["pozzolana_source"] = "calcined_clay"
    if product == "cement" and ("fly ash" in text or "flyash" in text):
        inferred["subtype"] = "portland_pozzolana_fly_ash"
        inferred["cement_type"] = "ppc"
        inferred["pozzolana_source"] = "fly_ash"
    if product == "cement" and ("ordinary portland cement" in text or "opc" in text):
        inferred["cement_type"] = "opc"
    if product == "cement" and ("portland pozzolana" in text or "ppc" in text):
        inferred["cement_type"] = "ppc"
    if product in {"bolt", "screw", "nut"} and ("hexagonal" in text or "hexagon" in text):
        inferred["subtype"] = subtype or f"hexagon_{product}"
        inferred["head_shape"] = "hexagonal"
    if "preformed" in text or "pre formed" in text:
        inferred["form"] = "preformed"
    return inferred


def _separate_context_material(
    *,
    product: str | None,
    text: str,
    material: list[str],
    context_material: list[str],
    context_only_terms: list[str],
) -> tuple[list[str], list[str], list[str]]:
    normalized_text = _space_key(text)
    if product == "valve" and (
        "ms water line" in normalized_text
        or "steel water line" in normalized_text
        or "steel water pipeline" in normalized_text
        or "ms pipeline" in normalized_text
    ):
        material = [value for value in material if value != "steel"]
        context_material = _merge_unique(context_material, ["steel"])
        context_only_terms = _merge_unique(context_only_terms, ["steel pipeline context"])
    return _dedupe(material), _dedupe(context_material), _dedupe(context_only_terms)


def _match_pattern_value(value: str, patterns: list[tuple[str, list[str]]]) -> str | None:
    text = _space_key(value)
    for canonical, aliases in patterns:
        if _space_key(canonical) == text or any(_contains(text, alias) for alias in aliases):
            return canonical
    return None


def _normalize_grade(value: str | None, description: str) -> str | None:
    source = " ".join(item for item in [value, description] if item)
    grade_match = re.search(r"\bgrade\s+([a-z0-9]+)\b", source, re.I)
    if grade_match:
        return grade_match.group(1).upper()
    grade_match = re.search(r"\b([a-z0-9]+)\s+grade\b", source, re.I)
    if grade_match:
        return grade_match.group(1).upper()
    for grade in ["33", "43", "53"]:
        if _contains(source, f"{grade} grade") or _contains(source, f"grade {grade}"):
            return grade
    return _clean_optional(value)


def _normalize_subtype_family(value: str | None, product: str | None) -> str | None:
    if product != "valve" or not value:
        return None
    subtype = _normalize_subtype(value, product)
    return "check_valve" if subtype == "check_valve" else None


def _normalize_cement_type(value: str | None, description: str, subtype: str | None) -> str | None:
    text = _space_key(" ".join(item for item in [value, description, subtype] if item))
    if "portland pozzolana" in text or "ppc" in text or "pozzolana" in text:
        return "ppc"
    if "ordinary portland" in text or "opc" in text:
        return "opc"
    return None


def _normalize_pozzolana_source(
    value: str | None,
    description: str,
    subtype: str | None,
) -> str | None:
    text = _space_key(" ".join(item for item in [value, description, subtype] if item))
    if "fly ash" in text or "flyash" in text:
        return "fly_ash"
    if "calcined clay" in text:
        return "calcined_clay"
    return None


def _normalize_head_shape(value: str | None, description: str) -> str | None:
    text = _space_key(" ".join(item for item in [value, description] if item))
    if "hexagonal" in text or "hexagon" in text:
        return "hexagonal"
    return _clean_optional(value)


def _normalize_form(value: str | None, description: str) -> str | None:
    text = _space_key(" ".join(item for item in [value, description] if item))
    if "preformed" in text or "pre formed" in text:
        return "preformed"
    return _clean_optional(value)


def _grounded_functions(functions: list[str], description: str) -> list[str]:
    text = _space_key(description)
    grounded = []
    for function in functions:
        if function != "structural_fastening":
            grounded.append(function)
            continue
        if any(term in text for term in ["fasten", "fastening", "anchor", "connect", "joint"]):
            grounded.append(function)
    return _dedupe(grounded)


def _extract_temperature_c(description: str) -> float | None:
    match = re.search(r"\b(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", description, re.I)
    return float(match.group(1)) if match else None


def _extract_negatives(description: str, product: str | None) -> dict[str, list[str]]:
    text = _space_key(description)
    result = {
        "material": [],
        "function": [],
        "application": [],
        "subtype": [],
        "medium": [],
        "installation_context": [],
    }
    negated_phrases = re.findall(
        r"\b(?:not|no|except|excluding|without)\s+([a-z0-9\-/ ]{1,40})",
        text,
    )
    for phrase in negated_phrases:
        phrase = phrase.strip()
        material = _match_pattern_value(phrase, MATERIAL_PATTERNS)
        if material:
            result["material"].append(material)
        function = _match_pattern_value(phrase, FUNCTION_PATTERNS)
        if function:
            result["function"].append(function)
        application = _match_pattern_value(phrase, APPLICATION_PATTERNS)
        if application:
            result["application"].append(application)
        subtype = _normalize_subtype(phrase, product)
        if subtype:
            result["subtype"].append(subtype)
        if "roofing" in phrase or "roof" in phrase:
            result["application"].append("roofing")
            result["function"].append("roofing")
            result["subtype"].append("roofing_tile")
        if "underground" in phrase:
            result["installation_context"].append("underground")
    if "non potable" in text:
        result["application"].append("potable water supply")
        result["medium"].append("potable")
    return {key: _dedupe(values) for key, values in result.items()}


def _sanitize_excluded_medium(values: list[str], description: str) -> list[str]:
    text = _space_key(description)
    sanitized = []
    for value in values:
        normalized = _space_key(value)
        if normalized == "water":
            continue
        if normalized == "potable" and "non potable" not in text:
            sanitized.append("potable water")
            continue
        sanitized.append(normalized)
    return _dedupe(sanitized)


def _without_excluded(values: list[str], excluded: list[str]) -> list[str]:
    excluded_set = set(excluded)
    return [value for value in values if value not in excluded_set]


def _true_product_aliases(product: str | None, supplied: list[str]) -> list[str]:
    if product is None:
        return _clean_list(supplied)
    blocked_subtypes = set(_known_subtypes(product))
    aliases = [
        alias
        for alias in PRODUCT_ALIASES.get(product, set())
        if _normalize_subtype(alias, product) not in blocked_subtypes
    ]
    aliases = [
        alias
        for alias in aliases
        if " valve" not in alias and " tile" not in alias and " cement" not in alias
    ]
    return _merge_unique(_clean_list(supplied), sorted(aliases)[:5])


def _known_subtypes(product: str | None) -> list[str]:
    if product is None:
        return []
    return sorted(
        subtype
        for pattern_product, subtype, _aliases in SUBTYPE_PATTERNS
        if pattern_product == product
    )


def _attribute_evidence(description: str, product: str, *, subtype: str | None) -> list[str]:
    evidence: list[str] = []
    source = f"{product} {description}"
    if subtype:
        for _product, value, aliases in SUBTYPE_PATTERNS:
            if value == subtype:
                for alias in aliases:
                    if _contains(source, alias):
                        evidence.append(f"subtype={alias}")
                        break
    if "fly ash" in _space_key(description):
        evidence.append("pozzolana_source=fly ash")
    if "calcined clay" in _space_key(description):
        evidence.append("pozzolana_source=calcined clay")
    grade = _normalize_grade(None, description)
    if grade:
        evidence.append(f"grade=Grade {grade}")
    temperature = _extract_temperature_c(description)
    if temperature is not None:
        evidence.append(f"temperature_c={temperature:g} C")
    return evidence


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _clean_list(values: list[str]) -> list[str]:
    return _dedupe(_space_key(value) for value in values if _space_key(value))


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    return _dedupe([*left, *right])


def _dedupe(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        result.append(cleaned)
        seen.add(cleaned)
    return result


def _contains(text: str, phrase: str) -> bool:
    return f" {_space_key(phrase)} " in f" {_space_key(text)} "


def _space_key(value: str) -> str:
    return " ".join(
        str(value).casefold().replace("_", " ").replace("-", " ").replace("/", " ").split()
    )


def intent_from_json(value: str) -> SemanticQueryIntent:
    return SemanticQueryIntent.model_validate(json.loads(value))
