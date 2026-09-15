import unittest
from pydantic import ValidationError
from evals.appworld.conversation import TaskTools
from evals.appworld.adapter import make_discover_tool, make_execute_tool

class World:
    def __init__(self): self.codes = []
    def execute(self, code): self.codes.append(code); return "ok"

class ReasonTests(unittest.TestCase):
    def test_contract_and_execution_boundary(self):
        world = World()
        task = TaskTools(world)
        for tool in (*task.build(), make_discover_tool(world), make_execute_tool(world)):
            schema = tool.args_schema.model_json_schema()
            expected = (
                ["source_refs", "reason", "code"]
                if tool.name == "appworld_discover"
                else ["source_refs", "reason", "action_phase", "binding_ref", "binding_checks", "set_bindings", "code"]
            )
            self.assertEqual(list(schema["properties"]), expected)
            self.assertIn("source_refs", schema["required"])
            self.assertIn("reason", schema["required"])
            if tool.name != "appworld_discover":
                self.assertIn("action_phase", schema["required"])
                self.assertIn("set_bindings", schema["required"])
                phases = [item["action_phase"] for item in schema["examples"]]
                self.assertEqual(
                    phases,
                    ["PREREQUISITE", "TARGET_READ", "TARGET_WRITE", "TARGET_VERIFY", "FINALIZE"],
                )
            for args in ({"code": "print(1)"}, {"source_refs": ["MODEL"], "reason": "", "code": "print(1)"}):
                before = len(world.codes)
                with self.assertRaises(ValidationError): tool.invoke(args)
                self.assertEqual(len(world.codes), before)
            code = ("print(apis.api_docs.show_app_descriptions())"
                    if tool.name == "appworld_discover" else "print(1)")
            args = {"source_refs": ["MODEL"],
                    "reason": "API from docs; value from observed E1.", "code": code}
            if tool.name != "appworld_discover":
                args.update(action_phase="OTHER", set_bindings=[])
            tool.invoke(args)
        self.assertEqual(len(world.codes), 5)
        self.assertTrue(all(r["reason"] for r in task.calls))

    def test_string_null_binding_ref_is_normalized_but_bad_ids_still_fail(self):
        from evals.appworld.tool_descriptions import AppWorldCallInput

        for sentinel in (None, False, "", "null", "NULL", "none", "false"):
            value = AppWorldCallInput.model_validate({
                "source_refs": ["U1"],
                "reason": "普通动作没有集合绑定。",
                "action_phase": "OTHER",
                "binding_ref": sentinel,
                "binding_checks": [],
                "set_bindings": [],
                "code": "print(1)",
            })
            self.assertIsNone(value.binding_ref)
        with self.assertRaises(ValidationError):
            AppWorldCallInput.model_validate({
                "source_refs": ["U1"],
                "reason": "错误编号仍应拒绝。",
                "action_phase": "OTHER",
                "binding_ref": "missing",
                "binding_checks": [],
                "set_bindings": [],
                "code": "print(1)",
            })

    def test_discover_rejects_business_api(self):
        world = World()
        discover = TaskTools(world).build()[0]
        with self.assertRaises(Exception):
            discover.invoke({"source_refs": ["U1"], "reason": "Wrong boundary",
                             "code": "print(apis.spotify.login(username='x'))"})
        self.assertEqual(world.codes, [])
