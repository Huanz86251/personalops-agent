"""Generate one complete local Phoenix trace without calling any provider."""

from __future__ import annotations

import json
import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from .adapter import UsageMeter, make_discover_tool, make_execute_tool


from trace_presentation import run_name
PROJECT_NAME = run_name("Learning Preview")
ROOT_NAME = "EVAL / Guided learning trace"


class GuidedWorld:
    def execute(self, code: str) -> str:
        if "show_api_doc" in code:
            return json.dumps({
                "app": "spotify",
                "api": "show_playlist_library",
                "parameters": {"access_token": "string"},
                "returns": "the simulated user's playlists",
            }, ensure_ascii=False)
        if "show_playlist_library" in code:
            return json.dumps({
                "playlists": [
                    {"title": "Focus", "song_count": 12},
                    {"title": "Interview Prep", "song_count": 8},
                ]
            }, ensure_ascii=False)
        return "Execution failed. Unknown guided-demo code."


def _complete_model_call(
    meter: UsageMeter,
    *,
    role: str,
    messages: list,
    response: AIMessage,
    input_tokens: int,
    output_tokens: int,
) -> None:
    run_id = uuid4()
    meter.on_chat_model_start(
        {"name": "guided-local-model"},
        [messages],
        run_id=run_id,
        tags=["appworld:" + role],
        invocation_params={
            "model_name": "guided-local-no-provider",
            "temperature": 0,
            "tools": [{
                "name": "appworld_execute",
                "description": (
                    "Execute Python against the persistent simulated AppWorld."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }],
        },
    )
    response.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    meter.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=response)]]),
        run_id=run_id,
    )


def generate_learning_trace() -> dict:
    os.environ["PHOENIX_TRACING_ENABLED"] = "true"
    os.environ["PHOENIX_TELEMETRY_ENABLED"] = "false"
    os.environ["PHOENIX_PROJECT"] = PROJECT_NAME
    os.environ["PHOENIX_TRACE_PROFILE"] = "curated"
    os.environ["PHOENIX_OTEL_PROTOCOL"] = "http/protobuf"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = (
        "http://127.0.0.1:6007/v1/traces"
    )
    os.environ["PHOENIX_UI_URL"] = "http://127.0.0.1:6007"

    from observability import (
        set_span_output,
        setup_observability,
        shutdown_observability,
        trace_context,
        trace_span,
    )

    provider = setup_observability()
    if provider is None:
        raise RuntimeError("Phoenix tracing failed to initialize")

    session_id = "guided_" + uuid4().hex
    meter = UsageMeter(max_calls=8)
    world = GuidedWorld()
    discover_tool = make_discover_tool(world)
    execute_tool = make_execute_tool(world)
    trace_id = None

    try:
        with trace_context(
            session_id=session_id,
            metadata={
                "benchmark": "guided_local_fixture",
                "external_model_calls": 0,
                "contains_complete_content": True,
            },
        ):
            with trace_span(
                ROOT_NAME,
                kind="agent",
                input_value={
                    "purpose": "learn how to read a PersonalOps trace",
                    "external_model_calls": 0,
                },
            ) as root:
                if root is not None and root.get_span_context().is_valid:
                    trace_id = format(root.get_span_context().trace_id, "032x")

                with trace_span(
                    "1 / INPUT / AppWorld task + toolset",
                    kind="chain",
                    input_value={"source": "local guided fixture"},
                ) as span:
                    set_span_output(span, {
                        "reading_guide": (
                            "Start here: this is everything the agent receives "
                            "before planning."
                        ),
                        "appworld_task": {
                            "instruction": (
                                "Inspect the documented Spotify playlist API "
                                "and report the simulated playlist names."
                            ),
                            "datetime": "2026-09-01T10:00:00+08:00",
                            "supervisor": {
                                "name": "Guided Demo User",
                                "account_type": "simulated",
                            },
                        },
                        "agent_tool": {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": (
                                tool.args_schema.model_json_schema()
                            ),
                        },
                    })

                with trace_span(
                    "2 / RUN / PersonalOps Agent",
                    kind="agent",
                    input_value={
                        "graph": (
                            "supervisor -> executor -> reporter -> reviewer"
                        ),
                    },
                ) as run_span:
                    with trace_span(
                        "hard_supervisor",
                        kind="chain",
                        input_value={
                            "responsibility": (
                                "turn the benchmark instruction into "
                                "verifiable steps"
                            ),
                        },
                    ) as plan_span:
                        plan = {
                            "action": "PLAN",
                            "plan_objective": (
                                "Return the simulated Spotify playlist names."
                            ),
                            "plan_success_criteria": [
                                "Inspect the documented API.",
                                "Retrieve the playlist library.",
                                "Ground the answer in the tool observation.",
                            ],
                            "steps": [{
                                "step_id": 1,
                                "objective": (
                                    "Inspect the API and retrieve playlists."
                                ),
                            }],
                        }
                        _complete_model_call(
                            meter,
                            role="hard",
                            messages=[
                                SystemMessage(
                                    content=(
                                        "You are the PersonalOps hard "
                                        "supervisor. Produce a JSON plan."
                                    )
                                ),
                                HumanMessage(
                                    content=(
                                        "Inspect the documented Spotify "
                                        "playlist API and report the "
                                        "simulated playlist names."
                                    )
                                ),
                            ],
                            response=AIMessage(
                                content=json.dumps(plan, ensure_ascii=False)
                            ),
                            input_tokens=180,
                            output_tokens=90,
                        )
                        set_span_output(plan_span, plan)

                    with trace_span(
                        "main_agent.run",
                        kind="agent",
                        input_value={
                            "step_id": 1,
                            "step_objective": (
                                "Inspect the API and retrieve playlists."
                            ),
                        },
                    ) as execute_span:
                        messages = [
                            SystemMessage(
                                content=(
                                    "Use appworld_discover for API docs, then "
                                    "appworld_execute for business calls."
                                )
                            ),
                            HumanMessage(
                                content=(
                                    "Step 1: inspect the Spotify playlist "
                                    "API and retrieve the playlists."
                                )
                            ),
                        ]
                        first_call = AIMessage(
                            content="I will inspect the documented signature.",
                            tool_calls=[{
                                "name": "appworld_discover",
                                "args": {
                                    "source_refs": ["MODEL"],
                                    "reason": "Need the exact playlist API signature.",
                                    "code": (
                                        'print(apis.api_docs.show_api_doc('
                                        'app_name="spotify", '
                                        'api_name="show_playlist_library"))'
                                    )
                                },
                                "id": "guided-tool-01",
                                "type": "tool_call",
                            }],
                        )
                        _complete_model_call(
                            meter,
                            role="simple",
                            messages=messages,
                            response=first_call,
                            input_tokens=220,
                            output_tokens=55,
                        )
                        observation_1 = discover_tool.invoke({
                            "name": "appworld_discover",
                            "args": first_call.tool_calls[0]["args"],
                            "id": "guided-tool-01",
                            "type": "tool_call",
                        })
                        messages.extend([first_call, observation_1])

                        second_call = AIMessage(
                            content=(
                                "The signature is now known; I will call it."
                            ),
                            tool_calls=[{
                                "name": "appworld_execute",
                                "args": {
                                    "source_refs": ["E1"],
                                    "reason": "The documented signature was returned by E1.",
                                    "code": (
                                        "print(apis.spotify."
                                        "show_playlist_library("
                                        'access_token="SIMULATED_TOKEN"))'
                                    )
                                },
                                "id": "guided-tool-02",
                                "type": "tool_call",
                            }],
                        )
                        _complete_model_call(
                            meter,
                            role="simple",
                            messages=messages,
                            response=second_call,
                            input_tokens=410,
                            output_tokens=65,
                        )
                        observation_2 = execute_tool.invoke({
                            "name": "appworld_execute",
                            "args": second_call.tool_calls[0]["args"],
                            "id": "guided-tool-02",
                            "type": "tool_call",
                        })
                        messages.extend([second_call, observation_2])

                        final_message = AIMessage(
                            content=(
                                "The observed playlists are Focus and "
                                "Interview Prep."
                            )
                        )
                        _complete_model_call(
                            meter,
                            role="simple",
                            messages=messages,
                            response=final_message,
                            input_tokens=620,
                            output_tokens=30,
                        )
                        set_span_output(execute_span, {
                            "step_id": 1,
                            "final_answer": final_message.content,
                            "model_calls": 3,
                            "tool_calls": 2,
                        })

                    with trace_span(
                        "step_report.step_1",
                        kind="chain",
                        input_value={
                            "step_id": 1,
                            "expected": "documented and observed playlists",
                        },
                    ) as report_span:
                        set_span_output(report_span, {
                            "status": "COMPLETED",
                            "evidence": [
                                "API signature observed.",
                                "Tool returned Focus and Interview Prep.",
                            ],
                            "unresolved_items": [],
                        })

                    with trace_span(
                        "hard_final_reviewer",
                        kind="chain",
                        input_value={
                            "plan_success_criteria": plan[
                                "plan_success_criteria"
                            ],
                        },
                    ) as review_span:
                        set_span_output(review_span, {
                            "action": "FINAL",
                            "status": "COMPLETED",
                            "final_answer": (
                                "The simulated playlists are Focus and "
                                "Interview Prep."
                            ),
                        })

                    set_span_output(run_span, {
                        "self_reported_status": "COMPLETED",
                        "model_calls": 4,
                        "tool_calls": 2,
                    })

                with trace_span(
                    "3 / GRADE / official AppWorld evaluator",
                    kind="chain",
                    input_value={
                        "agent_cannot_call_grader": True,
                        "fixture": True,
                    },
                ) as grade_span:
                    set_span_output(grade_span, {
                        "reading_guide": (
                            "For a real run, this node contains the complete "
                            "official AppWorld grader result."
                        ),
                        "success": True,
                        "num_tests": 2,
                        "tests": [
                            {
                                "name": "used documented API",
                                "passed": True,
                            },
                            {
                                "name": "reported observed playlist names",
                                "passed": True,
                            },
                        ],
                    })

                with trace_span(
                    "4 / SAVE / local evidence",
                    kind="chain",
                    input_value={"fixture": True},
                ) as save_span:
                    set_span_output(save_span, {
                        "trace_stored_in_local_phoenix": True,
                        "project": PROJECT_NAME,
                        "trace_id": trace_id,
                    })

                set_span_output(root, {
                    "official_task_success": True,
                    "self_reported_status": "COMPLETED",
                    "external_model_calls": 0,
                    "model_calls_simulated_for_learning": 4,
                    "tool_calls": 2,
                })
    finally:
        shutdown_observability()

    return {
        "kind": "guided_trace_preview",
        "trace_id": trace_id,
        "phoenix": {
            "project": PROJECT_NAME,
            "root_span_name": ROOT_NAME,
        },
        "external_model_calls": 0,
        "model_cost": 0,
    }


if __name__ == "__main__":
    print(json.dumps(generate_learning_trace(), ensure_ascii=False, indent=2))
