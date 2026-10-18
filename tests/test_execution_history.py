"""Offline checks for step call listing and stable short-reference result reads."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage
from workers.history_archive import (history_scope, capture, catalog,
    read_execution_history, ExecutionHistoryMiddleware, execution_records)


def messages(call_id, name="lookup", args=None, result="observed", status="success"):
    return [AIMessage(content="private model prose", tool_calls=[{
        "name": name, "args": args or {}, "id": call_id}],
        additional_kwargs={"reasoning_content": "private reasoning"}),
        ToolMessage(content=result, tool_call_id=call_id, name=name, status=status)]


class HistoryTests(unittest.TestCase):
    def test_multiple_steps_list_calls_without_results_and_read_selected_results(self):
        with tempfile.TemporaryDirectory() as root:
            for step in (1, 2, 3):
                with history_scope("task", f"C:R:step_{step}:attempt_1", root):
                    capture(messages(f"call-{step}", args={"brand": str(step)}, result="large-result-" + str(step)))
            with history_scope("task", "C:R:step_4:attempt_1", root):
                listing = read_execution_history.invoke({"step_ids": [1,2,3]})
                self.assertEqual([x["reference"] for x in listing["calls"]], ["S1-1","S2-1","S3-1"])
                self.assertNotIn("large-result", json.dumps(listing))
                self.assertNotIn("private", json.dumps(listing))
                self.assertEqual(listing["calls"][0]["arguments"], {"brand":"1"})
                selected = read_execution_history.invoke({"step_ids":[1,2], "references":["S1-1","S2-1"]})
                self.assertEqual(selected["mode"], "results")
                self.assertEqual([x["tool_results"][0]["content"] for x in selected["results"]],
                                 ["large-result-1","large-result-2"])
                self.assertNotIn("large-result-3", json.dumps(selected))

    def test_long_and_failed_results_are_returned_whole(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            capture(messages("c", result="x"*60000, status="error"))
            response = read_execution_history.invoke({"step_ids":[1], "references":["S1-1"]})
            msg = response["results"][0]["tool_results"][0]
            self.assertEqual(len(msg["content"]), 60000)
            self.assertEqual(msg["status"], "error")
            self.assertNotIn("private reasoning", json.dumps(response))

    def test_all_invalid_references_fall_back_to_complete_listing(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            capture(messages("c", result="do not return"))
            for refs in (["S1-999"], ["S2-1"], ["../secret"], [""]):
                response = read_execution_history.invoke({"step_ids":[1], "references":refs})
                self.assertEqual(response["mode"], "calls")
                self.assertTrue(response["invalid_references"])
                self.assertEqual(response["calls"][0]["reference"], "S1-1")
                self.assertNotIn("do not return", json.dumps(response))
            for refs in (None, []):
                self.assertEqual(read_execution_history.invoke({"step_ids":[1],"references":refs})["mode"], "calls")

    def test_references_override_steps_and_partial_matches_return_only_results(self):
        with tempfile.TemporaryDirectory() as root:
            for step in (1,2):
                with history_scope("task", f"C:R:step_{step}:attempt_1", root):
                    capture(messages(f"c{step}",result=f"result-{step}"))
            with history_scope("task", "C:R:step_4:attempt_1", root):
                for steps in (None,[],[1],[99]):
                    query={"references":["S2-1","S1-99"]}
                    if steps is not None:query["step_ids"]=steps
                    response=read_execution_history.invoke(query)
                    self.assertEqual(response["status"],"OK")
                    self.assertEqual(response["mode"],"results")
                    self.assertEqual(response["step_ids"],[2])
                    self.assertEqual(response["invalid_references"],["S1-99"])
                    self.assertNotIn("calls",response)
                    self.assertEqual([r["reference"] for r in response["results"]],["S2-1"])
                    self.assertEqual(response["results"][0]["tool_results"][0]["content"],"result-2")
                for query in ({},{"step_ids":[]},{"references":["bad"]},{"references":[]}):
                    response=read_execution_history.invoke(query)
                    self.assertEqual(response["mode"],"calls")
                    self.assertEqual([c["reference"] for c in response["calls"]],["S1-1","S2-1"])
                    self.assertNotIn("result-",json.dumps(response))

    def test_aliases_survive_recapture_new_calls_and_new_scope(self):
        with tempfile.TemporaryDirectory() as root:
            with history_scope("task", "C:R:step_1:attempt_1", root):
                capture(messages("c", args={"id":7}))
                original = read_execution_history.invoke({"step_ids":[1]})["calls"]
                before=catalog(); capture(messages("c", args={"id":7}))
                self.assertEqual(before,catalog())
                capture(messages("d", args={"id":8}))
            with history_scope("task", "C:R:step_4:attempt_1", root):
                after=read_execution_history.invoke({"step_ids":[1]})["calls"]
                self.assertEqual(after[0],original[0])
                self.assertEqual(after[1]["reference"],"S1-2")
                self.assertIsNotNone(ExecutionHistoryMiddleware().before_agent({},None))
            with history_scope("other-task", "C:R2:step_1:attempt_1", root):
                isolated=read_execution_history.invoke({"step_ids":[1],"references":["S1-1"]})
                self.assertEqual(isolated["status"],"NOT_FOUND")
                self.assertEqual(isolated["calls"],[])

    def test_retries_parallel_workers_and_reviewer_do_not_merge_equal_call_ids(self):
        with tempfile.TemporaryDirectory() as root:
            for suffix in ("attempt_1:slot_a","attempt_1:slot_b","attempt_2:worker","attempt_2:reviewer"):
                with history_scope("task", "C:R:step_2:"+suffix, root):
                    capture(messages("same-id", result=suffix))
            with history_scope("task", "C:R:step_4:attempt_1", root):
                calls=read_execution_history.invoke({"step_ids":[2]})["calls"]
                self.assertEqual(len(calls),4)
                self.assertEqual([c["reference"] for c in calls],["S2-1","S2-2","S2-3","S2-4"])
                response=read_execution_history.invoke({"step_ids":[2],"references":[c["reference"] for c in calls]})
                self.assertEqual(len({r["tool_results"][0]["content"] for r in response["results"]}),4)

    def test_control_calls_excluded_without_losing_business_call_in_same_message(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            capture([AIMessage(content="",tool_calls=[
                {"id":"b","name":"lookup","args":{}},
                {"id":"r","name":"complete_web_step","args":{"summary":"REPORT_ONLY"}}]),
                ToolMessage(content="business",tool_call_id="b"),
                ToolMessage(content="REPORT_ONLY",tool_call_id="r")])
            response=read_execution_history.invoke({"step_ids":[1]})
            self.assertEqual(len(response["calls"]),1)
            self.assertNotIn("REPORT_ONLY",json.dumps(execution_records()))
            response=read_execution_history.invoke({"step_ids":[1],"references":["S1-1"]})
            self.assertEqual(response["results"][0]["tool_results"][0]["content"],"business")

    def test_pending_result_is_explicit_and_later_result_uses_same_alias(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            pair=messages("c")
            capture(pair[:1])
            response=read_execution_history.invoke({"step_ids":[1],"references":["S1-1"]})
            self.assertEqual(response["results"][0]["status"],"PENDING_RESULT")
            capture(pair[1:])
            response=read_execution_history.invoke({"step_ids":[1],"references":["S1-1"]})
            self.assertEqual(response["results"][0]["status"],"AVAILABLE")

    def test_explicit_step_metadata_supports_nonstandard_worker_and_legacy_thread_archive(self):
        with tempfile.TemporaryDirectory() as root:
            with history_scope("task", "opaque-worker", root):
                ExecutionHistoryMiddleware().before_model({"step_id":"3","messages":messages("c")},None)
            with history_scope("task", "C:R:step_1:attempt_1", root):
                # Existing immutable archives without step metadata remain indexable.
                from workers.history_archive import digest, ACTIVE
                body={"worker":"C:R:step_2:attempt_1","message":messages("legacy")[0].model_dump(mode="json")}
                raw=json.dumps(body,ensure_ascii=False,sort_keys=True)
                (ACTIVE.get()[0]/("H"+digest(raw)+".json")).write_text(raw,encoding="utf-8")
                listing=read_execution_history.invoke({"step_ids":[2,3]})
                self.assertEqual({c["step_id"] for c in listing["calls"]},{2,3})

    def test_duplicate_steps_refs_and_unknown_steps(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            capture(messages("c"))
            self.assertEqual(len(read_execution_history.invoke({"step_ids":[1,1]})["calls"]),1)
            self.assertEqual(len(read_execution_history.invoke({"step_ids":[1],"references":["S1-1","S1-1"]})["results"]),1)
            self.assertEqual(read_execution_history.invoke({"step_ids":[9]})["available_step_ids"],[1])

    def test_concurrent_index_allocation_keeps_one_alias_per_call(self):
        with tempfile.TemporaryDirectory() as root:
            with history_scope("task", "C:R:step_1:attempt_1", root):
                capture(messages("c"));capture(messages("d"))
            def read(_):
                with history_scope("task", "C:R:step_4:attempt_1", root):
                    return read_execution_history.invoke({"step_ids":[1]})["calls"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs=list(pool.map(read,range(8)))
            self.assertTrue(all(out==outputs[0] for out in outputs))
            self.assertEqual([c["reference"] for c in outputs[0]],["S1-1","S1-2"])

    def test_corrupted_archive_still_rejected(self):
        with tempfile.TemporaryDirectory() as root, history_scope("task", "C:R:step_1:attempt_1", root):
            capture(messages("c"))
            ref=catalog()[0]["reference"]
            next(Path(root).rglob(ref+".json")).write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError): read_execution_history.invoke({"step_ids":[1]})

    def test_tool_schema_supports_reference_only_queries_and_optional_steps(self):
        schema=read_execution_history.args_schema.model_json_schema()
        self.assertFalse(schema.get("required"))
        self.assertEqual(set(schema["properties"]),{"step_ids","references"})
        for valid in ({},{"step_ids":[]},{"references":["S1-1"]}):
            read_execution_history.args_schema.model_validate(valid)
        for invalid in ({"step_ids":[0]},{"step_ids":[True]}):
            with self.assertRaises(ValidationError):read_execution_history.args_schema.model_validate(invalid)
