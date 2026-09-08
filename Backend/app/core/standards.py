import re

_WHITESPACE = re.compile(r"\s+")
_COLON = re.compile(r"\s*:\s*")
_PART = re.compile(r"\bpart\s*(\d+)\b", re.IGNORECASE)
_SECTION = re.compile(r"\bsec(?:tion)?\s*(\d+)\b", re.IGNORECASE)


def canonicalize_standard_id(value: str) -> str:
    """Create a stable join key without removing meaningful part or section data."""
    normalized = _WHITESPACE.sub(" ", value.strip())
    normalized = _COLON.sub(":", normalized)
    normalized = _PART.sub(lambda match: f"PART {match.group(1)}", normalized)
    normalized = _SECTION.sub(lambda match: f"SEC {match.group(1)}", normalized)
    return normalized.upper()


def extract_base_code(canonical_id: str) -> str:
    match = re.search(r"\bIS(?:\s*/\s*IEC)?\s*\d{2,6}\b", canonical_id)
    return _WHITESPACE.sub(" ", match.group(0)).upper() if match else canonical_id
