"""Regression cases for reporting correctness; no Phoenix server or API access."""
from copy import deepcopy
import unittest

from scripts.audit_phoenix import summarize


def cloud_span(response_id, name="ChatCompletion", trace=1):
    output = {"id": response_id}
    if name == "ChatDeepSeek":
        output = {"generations": [[{"message": {"kwargs": {
            "response_metadata": {"id": response_id}}}}]]}
    return {
        "span_id": "unused", "parent_id": "root", "trace_rowid": trace,
        "name": name, "span_kind": "LLM", "status_code": "OK",
        "start_time": "2026-01-01 00:00:00", "end_time": "2026-01-01 00:00:01",
        "llm_token_count_prompt": 100, "llm_token_count_completion": 20,
        "attrs": {"llm": {"provider": "example", "model_name": "model"},
                  "output": {"value": output}},
    }


class AuditTests(unittest.TestCase):
    def test_two_instrumentation_layers_count_one_provider_call(self):
        usage = summarize([cloud_span("one"), cloud_span("one", "ChatDeepSeek")])["cloud_usage"]
        self.assertEqual(usage["identified_provider_calls"], 1)
        self.assertEqual(usage["prompt_tokens"], 100)

    def test_distinct_calls_with_identical_usage_are_not_merged(self):
        usage = summarize([cloud_span("one"), cloud_span("two")])["cloud_usage"]
        self.assertEqual(usage["identified_provider_calls"], 2)
        self.assertEqual(usage["prompt_tokens"], 200)

    def test_unknown_identity_and_conflicting_usage_refuse_a_total(self):
        first = cloud_span("one")
        conflict = deepcopy(first)
        conflict["llm_token_count_prompt"] = 101
        for rows in ([cloud_span(None)], [first, conflict],
                     [cloud_span("one", trace=1), cloud_span("one", trace=2)]):
            with self.subTest(rows=len(rows)):
                usage = summarize(rows)["cloud_usage"]
                self.assertFalse(usage["total_is_valid_for_identified_cloud_calls"])
                self.assertIsNone(usage["prompt_tokens"])

    def test_missing_local_usage_is_not_claimed_as_measured_zero(self):
        local = cloud_span(None)
        local["attrs"] = {"local_model": {"name": "local"}}
        local["llm_token_count_prompt"] = local["llm_token_count_completion"] = 0
        self.assertEqual(summarize([local])["local_llm_usage"],
                         {"spans": 1, "spans_with_explicit_usage": 0})

    def test_error_envelope_is_detected_even_if_span_is_ok(self):
        tool = cloud_span(None)
        tool.update(name="browser_navigate", span_kind="TOOL")
        tool["attrs"] = {"output": {"value": {"data": {"status": "error"}}}}
        report = summarize([tool])
        self.assertEqual(report["tool_message_errors_with_ok_span"], 1)

    def test_error_word_in_retrieved_content_is_not_a_tool_failure(self):
        tool = cloud_span(None)
        tool.update(name="web_search", span_kind="TOOL")
        tool["attrs"] = {"output": {"value": {
            "data": {"status": "success", "content": "Article about an error"}}}}
        self.assertEqual(summarize([tool])["tool_message_errors"], 0)

    def test_self_reported_completion_does_not_become_independent_success(self):
        root = cloud_span(None)
        root.update(name="conversation_turn", span_kind="CHAIN", parent_id=None)
        root["attrs"] = {"output": {"value": {
            "planning": {"final_status": "COMPLETED"}}}}
        self.assertIsNone(summarize([root])["conversation_turns"][0]["independent_task_success"])


if __name__ == "__main__":
    unittest.main()
