"""Chat Completions adapter for user-selected OpenAI-compatible providers."""
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import Field


class CompatibleChatModel(ChatOpenAI):
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = Field(
        default="max_tokens", exclude=True,
    )

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump(
            mode="json", exclude={"choices": {"__all__": {"message": {"parsed"}}}})
        # Capture response-body fields only, never clients, request headers or API keys.
        captured = {k: data[k] for k in ("id", "object", "created", "model", "choices", "usage",
                    "system_fingerprint", "service_tier") if k in data}
        result.llm_output = {**(result.llm_output or {}), "provider_response": captured}
        for generation, choice in zip(result.generations, data.get("choices", [])):
            message = choice.get("message") or {}
            if "reasoning_content" in message:
                generation.message.additional_kwargs["reasoning_content"] = message["reasoning_content"]
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if result is not None and choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta.get("reasoning_content"), str):
                result.message.additional_kwargs["reasoning_content"] = delta["reasoning_content"]
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.token_limit_parameter == "max_tokens" and "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return self._compatible_tool_parameters(payload)

    def _get_invocation_params(self, stop=None, **kwargs):
        # Keep callback-visible request settings consistent with the wire payload.
        return self._compatible_tool_parameters(super()._get_invocation_params(stop=stop, **kwargs))

    def _compatible_tool_parameters(self, payload):
        payload = dict(payload)
        # Qwen 3.8 rejects forced tool choice in thinking mode. Preserve the
        # caller's mandatory handoff, disabling thinking only for this request.
        choice = payload.get("tool_choice")
        if self.model_name.startswith("qwen3.8-") and (
            choice == "required" or isinstance(choice, dict)
        ):
            extra = dict(payload.get("extra_body") or {})
            extra["enable_thinking"] = False
            for key in ("reasoning_effort", "thinking_budget"):
                extra.pop(key, None)
                payload.pop(key, None)
            payload["extra_body"] = extra
        return payload
