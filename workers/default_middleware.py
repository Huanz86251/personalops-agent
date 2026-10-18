"""Explicit replacement for the SDK summarizer; project compaction owns policy."""
from langchain.agents.middleware import AgentMiddleware


class DisabledDefaultSummarizationMiddleware(AgentMiddleware):
    # DeepAgents merges custom middleware by public name, including its GP stack.
    # No hooks: retain original messages and never invoke an SDK summary model.
    name = "SummarizationMiddleware"
