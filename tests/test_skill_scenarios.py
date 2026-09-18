"""Offline scenario plumbing; scripted choices are not model-quality scores."""

import copy
import json
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from skill_runtime import load_catalog, prepare_skills_sync, skill_prompt
from skill_runtime.preparation import SkillChoice
from step_execution import _reporter_skill_task, run_step_reporter
from reporting import build_step_review_packet
from tests.test_skill_preparation import ProbeModel
from tests.test_step_reporter_agent import sample_step, sample_submission_record


def failed_packet():
    record = sample_submission_record().model_dump(mode="json")
    record["resolved_evidence"][0]["result"] = "HTTP 403: map access denied"
    record["submission"]["summary"] = "Map failed repeatedly; cinema showtimes still missing."
    return build_step_review_packet(
        user_request="查今晚影院场次", plan_objective="核实场次", current_step=sample_step(),
        current_attempt={"attempt": 1, "worker_submission": record,
                         "finish_reason": "READY_FOR_REVIEW", "stop_reason": "Map blocked",
                         "messages": ["PRIVATE_WORKER_TRAJECTORY"]},
        stop_reason="Map blocked",
    )


class ScenarioCatalogTests(unittest.TestCase):
    def test_every_scenario_is_selectable_only_with_its_role_and_tools(self):
        catalog = load_catalog()
        for asset in catalog:
            for role in asset["roles"]:
                with self.subTest(skill=asset["name"], role=role):
                    model = ProbeModel(selected=[asset["name"]])
                    selected = prepare_skills_sync(
                        model, role=role, task=asset["description"], catalog=catalog,
                        topics=asset["topics"], tools=asset["required_tools"], mode="dynamic")
                    self.assertEqual([s.name for s in selected.selected], [asset["name"]])
                    request = json.loads(model.selection_requests[0][0]["content"])
                    self.assertTrue(all(set(item) == {"id", "description", "exclusive", "conflicts_with"}
                                        for item in request["available_skills"]))
                    for other in catalog:
                        if role not in other["roles"]:
                            self.assertNotIn(other["name"], [s["id"] for s in request["available_skills"]])
                    if asset["required_tools"]:
                        with self.assertRaises(ValueError):
                            prepare_skills_sync(model, role=role, task="no tool", catalog=catalog,
                                topics=asset["topics"], tools=[], mode="fixed", fixed_ids=[asset["name"]])

    def test_specialized_bodies_do_not_load_all_domains(self):
        catalog = load_catalog()
        for role, names, tools in [
            ("web", ["web-local-services"], ["web_search"]),
            ("web", ["web-venue-visits"], ["web_search"]),
            ("web", ["web-cinema-showtimes"], ["web_search"]),
            ("web", ["web-map-routes"], ["web_search"]),
            ("web", ["web-shop-prices"], ["web_search"]),
            ("web", ["web-map-routes"], ["web_search"]),
            ("code", ["code-validate-schema"], ["read_file", "execute"]),
            ("reviewer", ["review-code-state"], ["read_file", "execute"]),
            ("step_reporter", ["report-web-evidence"], []),
            ("general", ["general-save-email-draft"], ["email_create_draft"]),
        ]:
            with self.subTest(role=role):
                snapshot = prepare_skills_sync(None, role=role, task="scenario", catalog=catalog,
                    mode="fixed", tools=tools, fixed_ids=names)
                prompt = skill_prompt(snapshot)
                for asset in catalog:
                    body = asset["content"].split("---", 2)[2].strip()
                    self.assertEqual(body in prompt, asset["name"] in names)

    def test_reporter_directory_is_discovered_without_tools_or_topics(self):
        expected = {"report-web-evidence", "report-execution-failure", "report-artifact-delivery"}
        model = ProbeModel(selected=["report-web-evidence"])
        result = prepare_skills_sync(model, role="step_reporter", task="check evidence", mode="dynamic")
        self.assertEqual({s.name for s in result.selected}, {"report-web-evidence"})
        available = {s["id"] for s in json.loads(model.selection_requests[0][0]["content"])["available_skills"]}
        self.assertTrue(expected.issubset(available))
        self.assertTrue(all(s.source == "step_reporter" and not s.required_tools for s in result.selected))

    def test_business_skill_requirements_match_declarations_without_connecting(self):
        from tools import ALL_TOOLS
        from mcp_runtime import EMAIL_MCP_TOOL_NAMES, EmailMCPRuntime, EmailDraftInput
        native_names = {tool.name for tool in ALL_TOOLS}
        # Building this tool does not start IMAP or read account configuration.
        draft = EmailMCPRuntime._build_draft_tool(object())
        declared_names = native_names | set(EMAIL_MCP_TOOL_NAMES.values()) | {draft.name}
        assets = {a["name"]: a for a in load_catalog()}
        for name in ("general-read-email", "general-save-email-draft", "general-manage-reminders"):
            self.assertTrue(set(assets[name]["required_tools"]).issubset(declared_names))
        self.assertFalse(draft.metadata["send_capability"])
        self.assertEqual(set(EmailDraftInput.model_fields), {"to", "cc", "bcc", "subject", "body_text"})

    def test_outcomes_are_bounded_and_do_not_include_raw_messages_or_arguments(self):
        packet = failed_packet()
        attempt = packet.attempts[0]
        attempt.summary = "s" * 20000
        attempt.stop_reason = "r" * 20000
        attempt.unresolved_items = ["u" * 10000] * 20
        attempt.resolved_evidence[0].result = "HTTP 403 " + "x" * 10000 + " END"
        attempt.resolved_evidence[0].arguments = {"excluded": "PRIVATE_TOOL_ARGUMENTS"}
        packet.attempts = [copy.deepcopy(attempt) for _ in range(10)]
        task = _reporter_skill_task(packet)
        text = json.dumps(task, ensure_ascii=False)
        self.assertEqual(len(task["recent_attempt_outcomes"]), 2)
        self.assertLess(len(text), 9000)
        self.assertNotIn("HTTP 403", text)
        self.assertNotIn("END", text)
        self.assertNotIn("PRIVATE_WORKER_TRAJECTORY", text)
        self.assertNotIn("PRIVATE_TOOL_ARGUMENTS", text)
        self.assertEqual(len(packet.attempts), 10)  # Selection never mutates review evidence.


class ScenarioReporterModel(ProbeModel):
    def with_structured_output(self, schema, **kwargs):
        if schema is SkillChoice:
            return super().with_structured_output(schema, **kwargs)

        def respond(messages):
            self.execution_requests.append(copy.deepcopy(messages))
            return {"parsed": {"step_id": 1, "status": "BLOCKED", "summary": "Evidence missing",
                "stop_reason": "Map blocked", "criterion_results": [{
                    "criterion": sample_step().success_criteria[0], "status": "NOT_MET",
                    "evidence": ["call-1"]}], "unresolved_items": ["Showtimes missing"]},
                "parsing_error": None}
        return RunnableLambda(respond)


class ScenarioReporterTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_reporter_selects_with_outcomes_injects_only_selected_and_resumes(self):
        catalog = load_catalog()
        names = ["report-web-evidence"]
        model = ScenarioReporterModel(selected=names)
        result = await run_step_reporter(model, current_step=sample_step(), review_packet=failed_packet(),
            max_model_rounds=5, model_output_max_tokens=4096, skill_catalog=catalog, skill_mode="dynamic")
        self.assertEqual(result.report.status, "BLOCKED")
        self.assertEqual(len(model.selection_requests), 1)
        self.assertNotIn("HTTP 403", str(model.selection_requests))
        self.assertNotIn("PRIVATE_WORKER_TRAJECTORY", str(model.selection_requests))
        first = model.execution_requests[0][0]
        injected = first["content"] if isinstance(first, dict) else first.content
        self.assertIn(skill_prompt(result.skill_snapshot), injected)
        self.assertNotIn("SELECTOR_ONLY_MARKER", str(model.execution_requests))
        self.assertNotIn("HTTP 403", injected)  # Outcome clues stay out of stable system prefix.
        for asset in catalog:
            self.assertEqual(asset["content"].split("---", 2)[2].strip() in injected, asset["name"] in names)
        with patch("skill_runtime.preparation.load_catalog", side_effect=AssertionError("no reload")):
            resumed = await run_step_reporter(model, current_step=sample_step(), review_packet=failed_packet(),
                max_model_rounds=2, model_output_max_tokens=4096,
                saved_skill_snapshot=result.skill_snapshot, skill_mode="dynamic")
        self.assertEqual(len(model.selection_requests), 1)
        self.assertEqual(result.skill_snapshot, resumed.skill_snapshot)

    async def test_reporter_small_budget_skips_selection_not_review(self):
        model = ScenarioReporterModel(selected=["report-web-evidence"])
        result = await run_step_reporter(model, current_step=sample_step(), review_packet=failed_packet(),
            max_model_rounds=2, model_output_max_tokens=4096, skill_mode="dynamic")
        self.assertEqual(model.selection_requests, [])
        self.assertEqual(len(model.execution_requests), 1)
        self.assertEqual(result.skill_snapshot["selection_method"], "budget_skip")


if __name__ == "__main__":
    unittest.main()
