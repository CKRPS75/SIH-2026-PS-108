from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.schemas.tenders import AttributeEvidence, ExtractedTenderItem, TenderExtractionResult
from app.services.parsed_standards_corpus import canonicalize_product

SYSTEM_INSTRUCTION = """You are a tender text extraction engine for StandardWise.

Extract procurement line items and their technical requirements from raw tender text.
Return structured procurement items only.

You must:
- identify procurement items
- preserve each item's raw tender text
- extract explicit technical attributes with evidence from the tender text
- distinguish primary product from accessories/components
- normalize a concise search description for standard selection
- preserve negative constraints such as "not roofing"

You must not:
- recommend Indian Standards
- output IS codes
- invent standards
- perform compliance decisions
- turn every accessory into a separate item unless the tender clearly procures it separately
- allow contractual boilerplate to dominate the normalized search description

Use null/empty lists when information is missing. Do not silently infer unsupported hard
constraints. If a field is inferred from wording, include evidence and mark it inferred.
"""

IS_CODE_PATTERN = re.compile(r"\bIS\s*\d{1,5}(?:\s*\([^)]*\))?(?:\s*:\s*\d{4})?\b", re.I)


class GeminiTenderExtractionPayload(BaseModel):
    items: list[ExtractedTenderItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GeminiTenderExtractor:
    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        timeout_s: float = 8.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._client = client

    async def extract(self, text: str) -> TenderExtractionResult:
        started_at = perf_counter()
        cleaned_text = text.strip()
        if not cleaned_text:
            return TenderExtractionResult(
                warnings=["empty_tender_text"],
                gemini_latency_ms=(perf_counter() - started_at) * 1000,
                fallback_reason="empty_input",
            )

        if not self._api_key and self._client is None:
            fallback = _deterministic_extract(cleaned_text)
            fallback.gemini_latency_ms = (perf_counter() - started_at) * 1000
            fallback.fallback_reason = "missing_api_key_deterministic_fallback"
            return fallback

        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(self._generate, cleaned_text),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            fallback = _deterministic_extract(cleaned_text)
            fallback.gemini_used = True
            fallback.gemini_success = False
            fallback.gemini_latency_ms = (perf_counter() - started_at) * 1000
            fallback.fallback_reason = "timeout_deterministic_fallback"
            fallback.warnings.insert(0, "Gemini tender extraction timed out")
            return fallback
        except (ValidationError, ValueError, TypeError) as exc:
            fallback = _deterministic_extract(cleaned_text)
            fallback.gemini_used = True
            fallback.gemini_success = False
            fallback.gemini_latency_ms = (perf_counter() - started_at) * 1000
            fallback.fallback_reason = "invalid_structured_response_deterministic_fallback"
            fallback.warnings.insert(0, f"Gemini returned invalid tender extraction: {exc}")
            return fallback
        except Exception as exc:  # noqa: BLE001 - SDK failures become diagnostics
            fallback = _deterministic_extract(cleaned_text)
            fallback.gemini_used = True
            fallback.gemini_success = False
            fallback.gemini_latency_ms = (perf_counter() - started_at) * 1000
            fallback.fallback_reason = f"{exc.__class__.__name__}_deterministic_fallback"
            fallback.warnings.insert(0, f"Gemini tender extraction unavailable: {str(exc)[:160]}")
            return fallback

        items = [_sanitize_item(item) for item in payload.items]
        warnings = [*payload.warnings]
        if not items:
            warnings.append("Gemini returned no procurement items")
        return TenderExtractionResult(
            items=items,
            warnings=warnings,
            gemini_used=True,
            gemini_success=True,
            gemini_latency_ms=(perf_counter() - started_at) * 1000,
        )

    def _generate(self, text: str) -> GeminiTenderExtractionPayload:
        client = self._get_client()
        prompt = (
            "Extract structured procurement items from this tender text. "
            "Do not output Indian Standard codes. Return only JSON with this shape:\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "item_no": string|null,\n'
            '      "raw_text": string,\n'
            '      "product": string,\n'
            '      "normalized_product": string|null,\n'
            '      "normalized_description": string,\n'
            '      "material": [string],\n'
            '      "application": [string],\n'
            '      "function": [string],\n'
            '      "installation_context": [string],\n'
            '      "associated_components": [string],\n'
            '      "dimensions": object,\n'
            '      "technical_requirements": [string],\n'
            '      "excluded_attributes": [string],\n'
            '      "attribute_evidence": [\n'
            '        {"field": string, "value": string, "evidence": string, "inferred": boolean}\n'
            "      ],\n"
            '      "ambiguity": boolean,\n'
            '      "missing_information": [string],\n'
            '      "confidence": number\n'
            "    }\n"
            "  ],\n"
            '  "warnings": [string]\n'
            "}\n\n"
            f"Tender text:\n{text}"
        )
        kwargs: dict[str, Any] = {"model": self._model_name, "contents": prompt}
        if client.__class__.__module__.startswith("google."):
            try:
                from google.genai import types
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("google-genai is required for Gemini tender extraction") from exc
            kwargs["config"] = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            )
        response = client.models.generate_content(**kwargs)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, GeminiTenderExtractionPayload):
            return parsed
        if isinstance(parsed, dict):
            return GeminiTenderExtractionPayload.model_validate(parsed)
        text = getattr(response, "text", None)
        if text:
            return GeminiTenderExtractionPayload.model_validate_json(text)
        raise ValueError("Gemini returned no structured tender extraction")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("google-genai is required for Gemini tender extraction") from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client


def _deterministic_extract(text: str) -> TenderExtractionResult:
    blocks = _split_items(text)
    items = [_extract_item(block, index) for index, block in enumerate(blocks, start=1)]
    items = [item for item in items if item is not None]
    warnings = ["deterministic_grounded_fallback_used"]
    if not items:
        warnings.append("no_grounded_procurement_item_detected")
    return TenderExtractionResult(
        items=items,
        warnings=warnings,
        gemini_used=False,
        gemini_success=False,
    )


def _split_items(text: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:Tender\s+)?Item\s+(?:No\.?\s*)?[A-Za-z0-9()./-]+[\s:—-]+",
            text,
        )
    )
    if len(matches) <= 1:
        numbered = list(re.finditer(r"(?m)^\s*\d+[.)]\s+", text))
        matches = numbered if len(numbered) > 1 else matches
    if not matches:
        return [text.strip()]
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start() : end].strip())
    return [block for block in blocks if block]


def _extract_item(block: str, index: int) -> ExtractedTenderItem | None:
    raw = block.strip()
    if not raw:
        return None
    item_no = _extract_item_no(raw) or str(index)
    lower = raw.casefold()
    product = _detect_product(raw)
    if not product:
        product = _first_nounish_product(raw)
    normalized_product, _confidence = canonicalize_product(product)
    normalized_product = normalized_product or product.casefold()
    material = _detect_materials(lower)
    applications = _detect_applications(lower)
    functions = _detect_functions(lower)
    installation = _detect_installation(lower)
    associated = _detect_components(lower)
    excluded = _detect_exclusions(lower)
    dimensions = _detect_dimensions(raw)
    technical = _detect_technical_requirements(raw)
    description = _build_normalized_description(
        product=product,
        material=material,
        applications=applications,
        functions=functions,
        installation=installation,
        dimensions=dimensions,
        raw=raw,
    )
    evidence = _build_evidence(raw, material, applications, functions, installation, dimensions)
    ambiguity = normalized_product in {"valve", "cement"} and not (functions or applications)
    missing = ["product subtype/function"] if ambiguity else []
    return _sanitize_item(
        ExtractedTenderItem(
            item_no=item_no,
            raw_text=raw,
            product=product,
            normalized_product=normalized_product,
            normalized_description=description,
            material=material,
            application=applications,
            function=functions,
            installation_context=installation,
            associated_components=associated,
            dimensions=dimensions,
            technical_requirements=technical,
            excluded_attributes=excluded,
            attribute_evidence=evidence,
            ambiguity=ambiguity,
            missing_information=missing,
            confidence=0.72 if product else 0.35,
        )
    )


def _extract_item_no(text: str) -> str | None:
    match = re.search(r"(?i)\bItem\s+(?:No\.?\s*)?([A-Za-z0-9()./-]+)", text)
    return match.group(1).strip(" .:-—") if match else None


def _detect_product(text: str) -> str | None:
    lower = text.casefold()
    if "upvc" in lower and "pipe" in lower:
        return "UPVC drainage pipe" if "drain" in lower else "UPVC pipe"
    if "hdpe" in lower and "pipe" in lower:
        return "HDPE pipe"
    if "ordinary portland cement" in lower or re.search(r"\bopc\b", lower):
        return "ordinary Portland cement"
    if "portland pozzolana" in lower or re.search(r"\bppc\b", lower):
        return "Portland pozzolana cement"
    if "cement" in lower:
        return "cement"
    if "hexagonal" in lower and "bolt" in lower:
        return "hexagonal bolt"
    if "bolt" in lower:
        return "bolt"
    if "pressure" in lower and "valve" in lower:
        return "pressure reducing valve" if "reducing" in lower or "reduce" in lower else "valve"
    if "valve" in lower:
        return "valve"
    if "tile" in lower:
        return "tile"
    if "pipe" in lower:
        return "pipe"
    return None


def _first_nounish_product(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    blocked = {
        "supply",
        "delivery",
        "installation",
        "contractor",
        "provide",
        "new",
        "material",
        "item",
    }
    for word in words:
        if word.casefold() not in blocked:
            return word
    return "procurement item"


def _detect_materials(lower: str) -> list[str]:
    materials = []
    for label, aliases in {
        "UPVC": ["upvc", "unplasticized polyvinyl chloride", "unplasticised polyvinyl chloride"],
        "HDPE": ["hdpe", "high density polyethylene"],
        "PVC": [" pvc ", "polyvinyl chloride"],
        "fly ash": ["fly ash", "flyash"],
        "calcined clay": ["calcined clay"],
        "clay": ["clay"],
        "steel": ["steel", "mild steel"],
    }.items():
        if any(alias in f" {lower} " for alias in aliases):
            materials.append(label)
    return _dedupe(materials)


def _detect_applications(lower: str) -> list[str]:
    applications = []
    if "soil" in lower:
        applications.append("soil drainage")
    if "waste" in lower or "wastewater" in lower:
        applications.append("waste drainage")
    if "rainwater" in lower:
        applications.append("rainwater drainage")
    if "sewage" in lower or "sewerage" in lower:
        applications.append("sewerage")
    if "potable" in lower or "drinking water" in lower:
        applications.append("potable water supply")
    if "flooring" in lower:
        applications.append("flooring")
    if "roofing" in lower and "not roofing" not in lower and "excluding roofing" not in lower:
        applications.append("roofing")
    if "building" in lower and any("drainage" in item for item in applications):
        applications.append("building drainage")
    return _dedupe(applications)


def _detect_functions(lower: str) -> list[str]:
    functions = []
    if "gravity drainage" in lower:
        functions.append("gravity drainage")
    if "wastewater" in lower or "soil discharge" in lower:
        functions.append("wastewater conveyance")
    if "pressure reducing" in lower or "reduce pressure" in lower:
        functions.append("reduce_pressure")
    if "hexagonal" in lower and "bolt" in lower:
        functions.append("fastening")
    return _dedupe(functions)


def _detect_installation(lower: str) -> list[str]:
    installation = []
    if "building" in lower:
        installation.append("building drainage")
    if "internal" in lower:
        installation.append("internal")
    if "external" in lower:
        installation.append("external")
    if "underground" in lower:
        installation.append("underground")
    return _dedupe(installation)


def _detect_components(lower: str) -> list[str]:
    components = []
    for component in ["fittings", "bends", "tees", "couplers", "clamps", "supports"]:
        if component in lower:
            components.append(component)
    return components


def _detect_exclusions(lower: str) -> list[str]:
    exclusions = []
    for match in re.finditer(r"\b(?:not|excluding|except|without)\s+([a-z0-9\-/ ]{1,40})", lower):
        phrase = " ".join(match.group(1).split())
        if phrase:
            exclusions.append(phrase)
    return _dedupe(exclusions)


def _detect_dimensions(text: str) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    diameter = re.search(r"\b(\d+(?:\.\d+)?)\s*mm\s+(?:nominal\s+)?diameter\b", text, re.I)
    if not diameter:
        diameter = re.search(r"\bdiameter\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mm\b", text, re.I)
    if diameter:
        dimensions["diameter"] = {"value": float(diameter.group(1)), "unit": "mm"}
    return dimensions


def _detect_technical_requirements(text: str) -> list[str]:
    requirements = []
    patterns = [
        "watertight",
        "smooth internal surfaces",
        "resistant to normal domestic wastewater",
        "moisture",
        "corrosion",
        "leakage",
        "satisfactory flow",
    ]
    lower = text.casefold()
    for pattern in patterns:
        if pattern in lower:
            requirements.append(pattern)
    return _dedupe(requirements)


def _build_normalized_description(
    *,
    product: str,
    material: list[str],
    applications: list[str],
    functions: list[str],
    installation: list[str],
    dimensions: dict[str, Any],
    raw: str,
) -> str:
    parts = [product]
    parts.extend(material)
    parts.extend(applications)
    parts.extend(functions)
    if installation:
        parts.append(" ".join(installation))
    diameter = dimensions.get("diameter")
    if isinstance(diameter, dict) and diameter.get("value"):
        value = diameter["value"]
        parts.append(f"{value:g} mm nominal diameter")
    if not applications and not functions:
        parts.append(_contractual_noise_removed(raw)[:320])
    return ", ".join(_dedupe(parts))


def _contractual_noise_removed(text: str) -> str:
    cleaned = re.sub(
        (
            r"\b(?:supply|delivery|installation|contractor|payment|completion|"
            r"new material|free from defects)\b"
        ),
        " ",
        text,
        flags=re.I,
    )
    return " ".join(cleaned.split())


def _build_evidence(
    raw: str,
    material: list[str],
    applications: list[str],
    functions: list[str],
    installation: list[str],
    dimensions: dict[str, Any],
) -> list[AttributeEvidence]:
    evidence = []
    for field, values in [
        ("material", material),
        ("application", applications),
        ("function", functions),
        ("installation_context", installation),
    ]:
        for value in values:
            snippet = _evidence_snippet(raw, value)
            evidence.append(
                AttributeEvidence(
                    field=field,
                    value=value,
                    evidence=snippet or value.replace("_", " "),
                    inferred=snippet is None,
                )
            )
    diameter = dimensions.get("diameter")
    if isinstance(diameter, dict):
        value = f"{diameter.get('value'):g} {diameter.get('unit', '')}".strip()
        evidence.append(
            AttributeEvidence(
                field="dimensions.diameter",
                value=value,
                evidence=_evidence_snippet(raw, value) or value,
            )
        )
    return evidence


def _evidence_snippet(raw: str, value: str) -> str | None:
    tokens = [token for token in re.split(r"[_\s]+", value) if len(token) > 2]
    if not tokens:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    for sentence in sentences:
        lower = sentence.casefold()
        if any(token.casefold() in lower for token in tokens):
            return sentence.strip()[:240]
    return None


def _sanitize_item(item: ExtractedTenderItem) -> ExtractedTenderItem:
    data = item.model_dump()
    sanitized = ExtractedTenderItem.model_validate(_sanitize_value(data))
    canonical_product, confidence = canonicalize_product(
        sanitized.normalized_product or sanitized.product
    )
    if canonical_product and confidence == "high":
        sanitized.normalized_product = canonical_product
    if sanitized.normalized_product == "cement":
        sanitized.normalized_description = _cement_description_from_grounded_terms(sanitized)
    if sanitized.normalized_product == "pipe":
        sanitized.normalized_description = _pipe_description_from_grounded_terms(sanitized)
    if sanitized.normalized_product == "bolt":
        sanitized.normalized_description = _strip_generic_fastener_noise(
            sanitized.normalized_description
        )
    return sanitized


def _strip_generic_fastener_noise(description: str) -> str:
    cleaned = re.sub(r"\s+for\s+fastening\b", "", description, flags=re.I)
    cleaned = re.sub(r"\bfastening\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bbolts\b", "bolt", cleaned, flags=re.I)
    return " ".join(cleaned.split()) or description


def _pipe_description_from_grounded_terms(item: ExtractedTenderItem) -> str:
    raw = item.raw_text.casefold()
    if (
        ("upvc" in raw or "unplasticized polyvinyl chloride" in raw)
        and "soil" in raw
        and "waste" in raw
        and "rainwater" in raw
        and ("building" in raw or "buildings" in raw)
    ):
        return (
            "UPVC pipe for soil and waste discharge inside and outside buildings "
            "including rainwater drainage"
        )
    return item.normalized_description


def _cement_description_from_grounded_terms(item: ExtractedTenderItem) -> str:
    raw = item.raw_text.casefold()
    if "43 grade" in raw and "ordinary portland cement" in raw:
        return "43 grade ordinary Portland cement"
    return item.normalized_description


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return IS_CODE_PATTERN.sub("", value).strip()
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if not cleaned or cleaned.casefold() in seen:
            continue
        result.append(cleaned)
        seen.add(cleaned.casefold())
    return result
