from __future__ import annotations

from structured_output_policy import (
    strict_tool_schema_options,
    supports_strict_json_schema,
    textual_schema_fallback,
    with_provider_structured_output,
)
from pydantic import BaseModel, Field


class ExampleSchema(BaseModel):
    value: str = Field(description="field guidance survives in the fallback")


class RecordingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[object, dict[str, object]]] = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return self


def test_qwen_and_probed_glm_use_native_strict_json_schema() -> None:
    for model_name in (
        "qwen3.7-flash",
        "qwen3.8-flash",
        "ZHIPU/GLM-5.3-Flash",
    ):
        model = RecordingModel(model_name)
        result = with_provider_structured_output(
            model,
            dict,
            include_raw=True,
            fallback_method="json_mode",
        )

        assert result is model
        assert supports_strict_json_schema(model)
        assert model.calls == [
            (
                dict,
                {
                    "method": "json_schema",
                    "strict": True,
                    "include_raw": True,
                },
            )
        ]
        assert strict_tool_schema_options(model) == {"strict": True}


def test_unverified_provider_keeps_the_call_site_fallback() -> None:
    model = RecordingModel("other-model")

    with_provider_structured_output(
        model,
        dict,
        include_raw=True,
        fallback_method="json_mode",
    )

    assert not supports_strict_json_schema(model)
    assert model.calls == [
        (dict, {"include_raw": True, "method": "json_mode"})
    ]
    assert strict_tool_schema_options(model) == {}

    assert not supports_strict_json_schema(RecordingModel("qwen3.8-omni-flash"))
    assert not supports_strict_json_schema(RecordingModel("ZHIPU/GLM-5.3-FlashX"))


def test_existing_default_method_can_be_preserved_for_other_providers() -> None:
    model = RecordingModel("other-model")

    with_provider_structured_output(
        model,
        dict,
        include_raw=False,
        fallback_method=None,
    )

    assert model.calls == [(dict, {})]


def test_text_schema_is_only_repeated_for_unverified_providers() -> None:
    strict_model = RecordingModel("qwen3.8-flash")
    legacy_model = RecordingModel("other-model")

    assert textual_schema_fallback(strict_model, ExampleSchema) == ""
    fallback = textual_schema_fallback(legacy_model, ExampleSchema)
    assert fallback.startswith("Schema:")
    assert "field guidance survives in the fallback" in fallback
