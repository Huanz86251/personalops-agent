"""Offline graph construction against the installed DeepAgents API."""
from unittest.mock import patch
import pytest
import deepagents.graph
from test_worker_submission import SubmissionScriptModel, web_search
from workers.web_worker import create_web_worker
from workers.code_worker import create_code_worker
from workers.code_reviewer import create_code_reviewer
from workers.default_middleware import DisabledDefaultSummarizationMiddleware


@pytest.mark.parametrize("factory", [create_web_worker, create_code_worker, create_code_reviewer])
def test_default_summarizer_is_replaced_in_real_compiled_graph(factory):
    stacks = []
    original = deepagents.graph.create_agent

    def capture(*args, **kwargs):
        stacks.append(kwargs["middleware"])
        return original(*args, **kwargs)

    with patch.object(deepagents.graph, "create_agent", side_effect=capture):
        graph = factory(SubmissionScriptModel(), tools=[web_search] if factory is create_web_worker else [])
    assert callable(graph.ainvoke)
    assert stacks
    for stack in stacks:
        summaries = [mw for mw in stack if mw.name == "SummarizationMiddleware"]
        assert len(summaries) == 1
        assert isinstance(summaries[0], DisabledDefaultSummarizationMiddleware)
    if factory is create_web_worker:
        assert any(getattr(tool, "name", None) == "write_file" for mw in stacks[-1] for tool in getattr(mw, "tools", []))
