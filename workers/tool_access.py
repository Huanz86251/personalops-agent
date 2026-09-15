"""Model-facing tool visibility policies for specialized Workers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse


def _tool_name(tool) -> str:
    if isinstance(tool, dict):
        function = tool.get("function", {})
        return str(tool.get("name") or function.get("name") or "").strip()
    return str(getattr(tool, "name", "")).strip()


class WorkerToolAllowlistMiddleware(AgentMiddleware):
    """Expose only an explicit tool-name set to a specialized Worker model."""

    def __init__(self, allowed_tool_names: Iterable[str]) -> None:
        self.allowed_tool_names = frozenset(
            name
            for raw_name in allowed_tool_names
            if (name := str(raw_name).strip())
        )
        if not self.allowed_tool_names:
            raise ValueError("Worker tool allowlist cannot be empty.")

    def _filter(self, request: ModelRequest) -> ModelRequest:
        return request.override(
            tools=[
                tool
                for tool in request.tools
                if _tool_name(tool) in self.allowed_tool_names
            ]
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ) -> ModelResponse:
        return await handler(self._filter(request))


__all__ = ["WorkerToolAllowlistMiddleware"]
