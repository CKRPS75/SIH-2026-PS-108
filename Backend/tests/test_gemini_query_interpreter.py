import asyncio
from types import SimpleNamespace

from app.services.gemini_query_interpreter import (
    GeminiQueryInterpreter,
    SemanticQueryIntent,
    normalize_intent,
)


def test_valve_reverse_flow_keeps_ms_as_context_material() -> None:
    result = normalize_intent(
        SemanticQueryIntent(
            normalized_product="valve",
            material=["mild steel"],
            function=["prevent reverse flow"],
            application=["water pipeline"],
            medium=["water"],
            confidence=0.9,
        ),
        product="valve",
        description="need something on MS water line so water should not come backward",
    )

    assert result.normalized_product == "valve"
    assert "prevent_reverse_flow" in result.function
    assert "water pipeline" in result.application
    assert "water" in result.medium
    assert "steel" in result.context_material
    assert "steel" not in result.material


def test_plastic_pipe_for_underground_sewage() -> None:
    result = normalize_intent(
        SemanticQueryIntent(normalized_product="pipe", material=["plastic"]),
        product="pipe",
        description="plastic pipe for underground sewage",
    )

    assert result.normalized_product == "pipe"
    assert "plastic" in result.material
    assert "sewerage" in result.application
    assert "underground" in result.installation_context


def test_white_cement_decorative_architectural_finish() -> None:
    result = normalize_intent(
        SemanticQueryIntent(
            normalized_product="cement",
            application=["decorative architectural finish"],
        ),
        product="cement",
        description="white decorative architectural finish",
    )

    assert result.normalized_product == "cement"
    assert any("decorative" in value for value in result.application)


def test_portland_pozzolana_cement_requires_pozzolana_discriminator() -> None:
    result = normalize_intent(
        SemanticQueryIntent(normalized_product="cement", subtype="pozzolana cement"),
        product="cement",
        description="Portland pozzolana cement",
    )

    assert result.ambiguity is True
    assert any("fly ash" in item and "calcined clay" in item for item in result.missing_information)


def test_valve_for_water_pipeline_does_not_invent_subtype_or_function() -> None:
    result = normalize_intent(
        SemanticQueryIntent(normalized_product="valve"),
        product="valve",
        description="for water pipeline",
    )

    assert result.ambiguity is True
    assert result.subtype is None
    assert result.function == []
    assert "valve function" in result.missing_information


def test_negated_roofing_stays_out_of_positive_fields() -> None:
    result = normalize_intent(
        SemanticQueryIntent(normalized_product="tile", application=["flooring", "roofing"]),
        product="tile",
        description="tile for interior flooring, not roofing",
    )

    assert "flooring" in result.application
    assert "roofing" not in result.application
    assert "roofing" in result.excluded_application
    assert "roofing_tile" in result.excluded_subtype


def test_noisy_cemet_flyash_ppc_normalizes_when_supported() -> None:
    result = normalize_intent(
        SemanticQueryIntent(
            normalized_product="cemet",
            subtype="fly ash PPC",
            application=["normal civil works"],
        ),
        product="cemet",
        description="flyash ppc for normal civil works",
    )

    assert result.normalized_product == "cement"
    assert result.subtype == "portland_pozzolana_fly_ash"
    assert any("civil" in value for value in result.application)


def test_ppc_calcined_clay_is_not_ambiguous_for_pozzolana_type() -> None:
    result = normalize_intent(
        SemanticQueryIntent(normalized_product="cement", subtype="portland pozzolana cement"),
        product="cement",
        description="Portland pozzolana cement made using calcined clay",
    )

    assert result.subtype == "portland_pozzolana_calcined_clay"
    assert "clay" not in result.material
    assert not any("pozzolana type" in item for item in result.missing_information)


def test_hyphenated_pressure_reducing_valve_normalizes() -> None:
    result = normalize_intent(
        SemanticQueryIntent(
            normalized_product="valve",
            subtype="pressure-reducing_valve",
            function=["pressure-reducing"],
        ),
        product="valve",
        description="pressure-reducing water valve",
    )

    assert result.subtype == "pressure_reducing_valve"
    assert result.function == ["reduce_pressure"]


def test_hexagonal_grade_c_bolt_keeps_only_grounded_attributes() -> None:
    result = normalize_intent(
        SemanticQueryIntent(
            normalized_product="bolt",
            function=["structural_fastening"],
        ),
        product="bolt",
        description="hexagonal Grade C bolt",
    )

    assert result.subtype == "hexagon_bolt"
    assert result.grade == "C"
    assert "structural_fastening" not in result.function


def test_gemini_unavailable_returns_safe_fallback_diagnostics() -> None:
    async def run() -> None:
        interpreter = GeminiQueryInterpreter(api_key=None, model_name="gemini-2.5-flash-lite")

        result = await interpreter.interpret(product="valve", description="for water pipeline")

        assert result.intent is None
        assert result.gemini_success is False
        assert result.fallback_reason == "missing_api_key"

    asyncio.run(run())


def test_interpreter_uses_structured_response_and_cache() -> None:
    async def run() -> None:
        client = _FakeGeminiClient(
            SemanticQueryIntent(
                normalized_product="valve",
                function=["prevent reverse flow"],
                confidence=0.8,
            )
        )
        interpreter = GeminiQueryInterpreter(
            api_key="test-key",
            model_name="gemini-test",
            client=client,
        )

        first = await interpreter.interpret(
            product="valve",
            description="need something so water should not come backward",
        )
        second = await interpreter.interpret(
            product="valve",
            description="need something so water should not come backward",
        )

        assert first.gemini_success is True
        assert second.gemini_success is True
        assert first.intent is not None
        assert "prevent_reverse_flow" in first.intent.function
        assert client.calls == 1

    asyncio.run(run())


class _FakeGeminiClient:
    def __init__(self, intent: SemanticQueryIntent) -> None:
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self._intent = intent

    def _generate_content(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(parsed=self._intent)
