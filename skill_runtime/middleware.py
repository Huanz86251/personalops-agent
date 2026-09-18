"""A checkpointed preparation node inside each execution role's graph."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig

from skill_runtime import prepare_skills, prepare_skills_sync, skill_prompt
from workers.progress import WorkerProgressState


class RoleSkillsMiddleware(AgentMiddleware[WorkerProgressState]):
    state_schema = WorkerProgressState

    def __init__(self, model, role, tool_names, *, mode=None, policies=(), backend=None):
        self.model = model
        self.role = role
        available = set(tool_names)
        for policy in policies:
            allowlist = getattr(policy, "allowed_tool_names", None)
            if allowlist is not None:
                available.intersection_update(allowlist)
        if backend is not None and not callable(getattr(backend, "execute", None)):
            available.discard("execute")
        self.tool_names = tuple(sorted(available))
        self.mode = mode

    def _options(self, state):
        saved = state.get("role_skill_snapshot")
        used = int(state.get("executor_model_calls_used", 0) or 0) + int(state.get("skill_preparation_calls_used", 0) or 0)
        used += int(state.get("worker_compaction_calls_used", 0) or 0)
        limit = state.get("executor_model_run_limit")
        # Reserve a useful execution turn and its submission, rather than
        # spend the last available call selecting optional guidance.
        allowed = limit is None or int(limit) - used >= 3
        messages = state.get("messages", [])
        # Initial assignment plus role-specific evidence, never another
        # role's hidden reasoning. On resume the saved snapshot wins.
        task = {key: state.get(key) for key in (
            "code_task", "code_candidate", "code_worker_submission",
            "memory_context", "execution_instructions",
        ) if state.get(key)}
        task["assignment"] = [
            m if isinstance(m, dict) else {"role": m.type, "content": m.content}
            for m in messages
            if (m.get("role") if isinstance(m, dict) else m.type) in {"user", "human"}
            and not (m.get("additional_kwargs", {}) if isinstance(m, dict) else m.additional_kwargs).get("personalops_runtime_event")
        ][-1:]
        if self.role != "reviewer" and state.get("skill_reselection_context"):
            task = {
                "task_contract": state["skill_reselection_context"],
                "recent_attempt_outcomes": [],
            }
        if self.role == "reviewer":
            from knowledge_rag.query import task_query
            assignment = "\n".join(str(m.get("content", "")) for m in task.get("assignment", []))
            contract = state.get("code_task") or {}
            task = {"task_contract": {"step_assignment": task_query(assignment, contract)[:1800]},
                    "recent_attempt_outcomes": [{"has_artifacts": bool(state.get("code_worker_submission"))}]}
        from workers.evidence_refs import display, registry
        task = display(task, registry(state))
        return dict(role=self.role, task=task, catalog=state.get("skill_catalog"),
                    topics=[*state.get("skill_topics", []), *(["appworld"] if {"appworld_discover", "appworld_execute", "appworld_verify"}.intersection(self.tool_names) else [])], tools=self.tool_names,
                    mode=state.get("skill_mode") or self.mode,
                    fixed_ids=state.get("skill_fixed_ids", {}).get(self.role, []),
                    saved=saved, allow_model=allowed)

    def _update(self, state, snapshot):
        if state.get("role_skill_snapshot") is not None:
            return None
        update = {"role_skill_snapshot": snapshot.model_dump(mode="json"),
                  "skill_preparation_calls_used": snapshot.model_calls}
        return update

    def before_agent(self, state, runtime, config: RunnableConfig):
        return self._update(state, prepare_skills_sync(self.model, config=config, **self._options(state)))

    async def abefore_agent(self, state, runtime, config: RunnableConfig):
        return self._update(state, await prepare_skills(self.model, config=config, **self._options(state)))

    @staticmethod
    def _with_skills(request):
        addition = skill_prompt(request.state.get("role_skill_snapshot"))
        if not addition:
            return request
        message = request.system_message
        if message is None:
            from langchain.messages import SystemMessage
            message = SystemMessage(content="")
        content = message.content
        content = (content.rstrip() + "\n\n" + addition if isinstance(content, str)
                   else [*content, {"type": "text", "text": addition}])
        return request.override(system_message=message.model_copy(update={"content": content}))

    def wrap_model_call(self, request, handler):
        return handler(self._with_skills(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._with_skills(request))
