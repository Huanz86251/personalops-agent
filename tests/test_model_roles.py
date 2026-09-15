"""Provider-free validation of independent model credentials and payloads."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
# Load TLS-backed SDK before tests clear environment for configuration isolation.
import model_clients

from agent import build_role_model
from model_roles import load_role_models


def defaults():
    return SimpleNamespace(
        llm_provider="deepseek", llm_model="deepseek-v4-flash", llm_thinking_enabled=False,
        hard_llm_provider="deepseek", hard_llm_model="deepseek-v4-pro", scheduler_thinking_enabled=False,
        summary_llm_provider="openai", summary_llm_model="gpt-5-nano",
        extraction_llm_provider="openai", extraction_llm_model="gpt-5-nano",
        cloud_llm_max_tokens=5000, memory_extraction_max_tokens=16000,
        memory_extraction_enabled=True,
    )


class RoleConfigurationTests(unittest.TestCase):
    def test_zero_budget_overrides_legacy_and_extra_flags(self):
        settings = defaults()
        settings.llm_provider = 'qwen'
        settings.llm_model = 'qwen3.7-flash'
        env = {'DASHSCOPE_API_KEY':'test', 'DASHSCOPE_BASE_URL':'https://qwen.invalid/v1',
               'OPENAI_API_KEY':'test', 'DEEPSEEK_API_KEY':'test',
               'GENERAL_LLM_THINKING_ENABLED':'true',
               'GENERAL_LLM_EXTRA_BODY_JSON':'{"enable_thinking":true,"thinking_budget":9999}'}
        with patch.dict(os.environ,env,clear=True), patch('model_roles.Path.read_text',return_value='{"qwen3.7-flash":{"thinking_budget":0}}'):
            roles = load_role_models(settings)
        self.assertFalse(roles['general'].thinking_enabled)
        self.assertEqual(roles['general'].extra_body, {'enable_thinking':False})
        self.assertEqual(roles['skill_selector'].max_tokens,512)

    def test_budget_validation_and_selector_answer_headroom(self):
        settings=defaults(); settings.llm_provider='qwen'; settings.llm_model='qwen3.7-flash'
        env={'DASHSCOPE_API_KEY':'test','DASHSCOPE_BASE_URL':'https://qwen.invalid/v1',
             'OPENAI_API_KEY':'test','DEEPSEEK_API_KEY':'test'}
        with patch.dict(os.environ,env,clear=True):
            self.assertEqual(load_role_models(settings)['skill_selector'].max_tokens,1536)
            self.assertTrue(load_role_models(settings)['skill_selector'].thinking_enabled)
            with patch.dict(os.environ,{'GENERAL_LLM_MAX_TOKENS':'1024'}):
                with self.assertRaisesRegex(ValueError,'leave room'):
                    load_role_models(settings)
            for invalid in (-1, True, '1024'):
                with patch('model_roles.Path.read_text',return_value=json.dumps({'qwen3.7-flash':{'thinking_budget':invalid}})):
                    with self.assertRaises(ValueError):load_role_models(settings)

    def test_qwen_role_defaults_and_actual_request_payloads(self):
        settings = defaults()
        settings.llm_provider = settings.hard_llm_provider = "qwen"
        settings.summary_llm_provider = settings.extraction_llm_provider = "qwen"
        settings.llm_model = settings.summary_llm_model = settings.extraction_llm_model = "qwen3.7-flash"
        settings.hard_llm_model = "qwen3.8-flash"
        settings.scheduler_thinking_enabled = True
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-key",
                                    "DASHSCOPE_BASE_URL": "https://qwen.invalid/v1"}, clear=True):
            settings.role_models = load_role_models(settings)
        strong = {"scope_resolver", "code", "code_reviewer", "scheduler", "code_scheduler", "replanner", "final_reviewer", "worker_leader"}
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 1,
                "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "OK"}}]})
        for role, config in settings.role_models.items():
            expected = role in strong
            self.assertEqual(config.model, "qwen3.8-flash" if expected else "qwen3.7-flash", role)
            self.assertTrue(config.thinking_enabled, role)
            self.assertEqual(config.reasoning_effort, "medium" if role == "scope_resolver" else "low" if expected else None, role)
            self.assertEqual(config.max_tokens, 4096 if role == "scope_resolver" else 16000 if role == "extraction" else 1536 if role == "skill_selector" else 5000, role)
            model = build_role_model(settings, role)
            model.root_client._client.close()
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                model.root_client._client = client
                model.invoke("offline configuration check")
            payload = requests[-1]
            self.assertTrue(payload["enable_thinking"], role)
            self.assertEqual(payload.get("reasoning_effort"), "medium" if role == "scope_resolver" else "low" if expected else None, role)
            self.assertEqual(payload.get("thinking_budget"), None if expected else 1024)
        self.assertEqual(len(requests), len(settings.role_models))

    def test_qwen_override_changes_only_one_role(self):
        env = {"OPENAI_API_KEY": "test-openai", "DEEPSEEK_API_KEY": "test-deepseek",
               "WEB_LLM_PROVIDER": "qwen", "WEB_LLM_MODEL": "qwen3.7-flash",
               "WEB_LLM_API_KEY_ENV": "WEB_ACCOUNT", "WEB_ACCOUNT": "test-qwen",
               "WEB_LLM_BASE_URL": "https://qwen.invalid/v1", "WEB_LLM_MAX_TOKENS": "9000"}
        with patch.dict(os.environ, env, clear=True):
            roles = load_role_models(defaults())
        self.assertEqual(roles["web"].provider, "qwen")
        self.assertEqual(roles["web"].max_tokens, 9000)
        self.assertEqual(roles["web"].api_key, "test-qwen")
        self.assertEqual(roles["general"].model, "deepseek-v4-flash")
        self.assertEqual(roles["web_reporter"].model, "deepseek-v4-flash")
        self.assertEqual(roles["code_reviewer"].model, "deepseek-v4-pro")
        self.assertEqual(roles["extraction"].max_tokens, 16000)
        self.assertNotIn("test-qwen", repr(roles["web"]))

    def test_missing_custom_key_does_not_fall_back_to_openai_key(self):
        env = {"OPENAI_API_KEY": "test-openai", "DEEPSEEK_API_KEY": "test-deepseek",
               "CODE_LLM_API_KEY_ENV": "MISSING_CODE_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "MISSING_CODE_KEY"):
                load_role_models(defaults())

    def test_all_roles_can_leave_the_old_global_provider(self):
        from model_roles import ROLE_DEFAULTS
        env = {"OPENAI_API_KEY": "test-only-openai"}
        for role in ROLE_DEFAULTS:
            env[role.upper() + "_LLM_PROVIDER"] = "openai"
            env[role.upper() + "_LLM_MODEL"] = "gpt-5-nano"
        with patch.dict(os.environ, env, clear=True):
            roles = load_role_models(defaults())
        self.assertTrue(all(c.provider == "openai" for c in roles.values()))

    def test_invalid_options_fail_before_client_construction(self):
        base = {"OPENAI_API_KEY": "test-openai", "DEEPSEEK_API_KEY": "test-deepseek"}
        for override in [{"CODE_LLM_PROVIDER": "qwen"},
                         {"CODE_LLM_MAX_TOKENS": "zero"},
                         {"CODE_LLM_EXTRA_BODY_JSON": '{"model":"wrong"}'},
                         {"CODE_LLM_BASE_URL": "https://user:password@example.invalid/v1"}]:
            with self.subTest(override=override), patch.dict(os.environ, {**base, **override}, clear=True):
                with self.assertRaises(ValueError):
                    load_role_models(defaults())

    def test_actual_mock_http_requests_keep_keys_and_endpoints_separate(self):
        env = {"OPENAI_API_KEY": "test-openai", "DEEPSEEK_API_KEY": "test-deepseek",
               "WEB_LLM_PROVIDER": "qwen", "WEB_LLM_MODEL": "qwen3.7-flash",
               "WEB_LLM_BASE_URL": "https://web.invalid/v1", "DASHSCOPE_API_KEY": "test-web",
               "CODE_LLM_PROVIDER": "compatible", "CODE_LLM_MODEL": "custom-code",
               "CODE_LLM_BASE_URL": "https://code.invalid/v1", "COMPATIBLE_API_KEY": "test-code"}
        with patch.dict(os.environ, env, clear=True):
            settings = defaults()
            settings.role_models = load_role_models(settings)
            models = {role: build_role_model(settings, role) for role in ("web", "code")}
        requests = []
        def handler(request):
            requests.append((str(request.url), request.headers["authorization"], json.loads(request.content)))
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 1,
                "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "offline OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        for model in models.values():
            model.root_client._client.close()
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                model.root_client._client = client
                self.assertEqual(model.invoke("synthetic").content, "offline OK")
        self.assertEqual(requests[0][0], "https://web.invalid/v1/chat/completions")
        self.assertEqual(requests[0][1], "Bearer test-web")
        self.assertTrue(requests[0][2]["enable_thinking"])
        self.assertEqual(requests[0][2]["max_completion_tokens"], 5000)
        self.assertEqual(requests[1][0], "https://code.invalid/v1/chat/completions")
        self.assertEqual(requests[1][1], "Bearer test-code")
        self.assertEqual(requests[1][2]["max_tokens"], 5000)
        self.assertNotIn("enable_thinking", requests[1][2])


if __name__ == "__main__":
    unittest.main()
