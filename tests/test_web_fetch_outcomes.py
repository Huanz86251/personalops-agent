import json
import unittest
from unittest.mock import patch
import httpx
from tools import local_native as local
from planning_models import SupervisorDecision, ReplanDecision, StepArtifactOutput, PlanStep
from pydantic import ValidationError


class FetchOutcomeTests(unittest.TestCase):
    def fetch(self, handler, **kwargs):
        factory = httpx.Client
        with patch.object(local, "_public_url"), patch("httpx.Client", side_effect=lambda **kw: factory(transport=httpx.MockTransport(handler), **kw)):
            return local.fetch_webpage.func("https://example.com/page", **kwargs)

    def test_empty_and_pagination_are_distinct(self):
        for text in ("", "   ", "<script>hello()</script>"):
            result = self.fetch(lambda r: httpx.Response(200, text=text, headers={"content-type":"text/html"}))
            self.assertEqual(result["fetch_status"], "EMPTY_CONTENT")
            self.assertEqual(result["http_status"], 200)
            self.assertFalse(result["evidence_available"])
        result = self.fetch(lambda r: httpx.Response(200, text="hello", headers={"content-type":"text/plain"}), start_index=99)
        self.assertEqual(result["fetch_status"], "PAGINATION_EXHAUSTED")

    def test_http_errors_and_retry_hint(self):
        for code, status in ((401,"AUTH_REQUIRED"),(403,"ACCESS_DENIED"),(429,"RATE_LIMITED"),(500,"HTTP_ERROR")):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(code, headers={"retry-after":"60"})
            result = self.fetch(handler)
            self.assertEqual(result["fetch_status"], status)
            self.assertEqual(result["http_status"], code)
            self.assertFalse(result["evidence_available"])
            self.assertEqual(len(calls), 1)
            if code == 429:
                self.assertEqual(result["retry_after"], "60")

    def test_timeout_is_not_http_error(self):
        def handler(request):
            raise httpx.ReadTimeout("fixture", request=request)
        result = self.fetch(handler)
        self.assertEqual(result["fetch_status"], "NETWORK_ERROR")
        self.assertIsNone(result["http_status"])

    def test_valid_body_and_redirect_keep_provenance(self):
        def handler(request):
            return httpx.Response(302, headers={"location":"/final"}) if request.url.path == "/page" else httpx.Response(200, text="verified source", headers={"content-type":"text/plain"})
        result = self.fetch(handler)
        self.assertEqual(result["fetch_status"], "SUCCESS")
        self.assertTrue(result["final_url"].endswith("/final"))
        self.assertTrue(result["source_url"].endswith("/page"))
        self.assertNotIn("_has_text", result)

    def test_schema_rules_reach_both_planning_entrypoints(self):
        for schema in (SupervisorDecision, ReplanDecision):
            definitions = schema.model_json_schema()["$defs"]
            self.assertIn("CODE must use an empty list", definitions["PlanStep"]["properties"]["artifact_outputs"]["description"])
            self.assertIn("Must be null for INTERNAL_HANDOFF", definitions["StepArtifactOutput"]["properties"]["target_path"]["description"])
        with self.assertRaises(ValidationError):
            StepArtifactOutput(output_id="notes", description="notes", target_path="notes.md")
        self.assertIsNone(StepArtifactOutput(output_id="notes", description="notes").target_path)
