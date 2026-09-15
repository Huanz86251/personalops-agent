"""Run a provider-free smoke evaluation of the local toolset Cross-Encoder."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from retrieval_models import RetrievalModelManager
from toolset_router import ToolsetRouter
from toolsets import DEFAULT_TOOLSET_REGISTRY


CASES = [
    ("查一下今天英伟达的最新新闻并核对来源", {"WEB_RESEARCH"}),
    ("登录网页邮箱，给指定联系人发送邮件", {"BROWSER_AUTOMATION"}),
    ("提取这个 PDF 第五页的表格并总结", {"LOCAL_DOCUMENTS"}),
    ("找出项目中所有引用 ToolsetRouter 的文件，只分析，不要修改", {"FILE_INSPECTION"}),
    ("修改配置文件里的模型名并回读确认", {"FILE_EDITING"}),
    ("实现这个功能并运行 pytest 验证", {"SOFTWARE_DEVELOPMENT"}),
    ("用符号计算解方程 x**2 - 5*x + 6 = 0", {"LOCAL_ANALYSIS"}),
    ("截一张当前Windows桌面图并识别上面的错误文字", {"DESKTOP_OBSERVATION"}),
    ("明天上午九点提醒我给客户回电话", {"SCHEDULED_AUTOMATION"}),
    ("列出我的本地提醒并取消周报提醒", {"SCHEDULED_AUTOMATION"}),
    ("明天上午九点在飞书提醒我提交材料", {"SCHEDULED_AUTOMATION"}),
    ("十分钟后帮我查询上海天气并把结果告诉我", {"SCHEDULED_AUTOMATION"}),
    ("帮我看看QQ邮箱最近十封邮件并总结重点", {"EMAIL_READING"}),
    ("把UID为12345的邮件中那份PDF附件下载下来", {"EMAIL_READING"}),
    ("根据刚才那封邮件写一份回复并保存到草稿箱，不要发送", {"EMAIL_READING"}),
    ("把刚生成的报告目录打包回传到当前飞书", {"FEISHU_FILE_EXPORT"}),
    ("在 AppWorld 中查 API 并发送一封模拟邮件", {"APPWORLD"}),
    ("解释一下什么是递归", {"NO_TOOL"}),
    ("查询官方说明，然后把结论写进本地 Markdown", {"WEB_RESEARCH", "FILE_EDITING"}),
]


def _all_declared_tools():
    names = {
        name
        for spec_name in DEFAULT_TOOLSET_REGISTRY.toolset_names
        for name in (
            *DEFAULT_TOOLSET_REGISTRY.get(spec_name).required_tool_names,
            *DEFAULT_TOOLSET_REGISTRY.get(spec_name).optional_tool_names,
        )
    }
    return [SimpleNamespace(name=name) for name in sorted(names)]


async def main():
    settings = load_settings()
    manager = RetrievalModelManager(
        embedding_model_name=settings.memory_embedding_model,
        reranker_model_name=settings.memory_reranker_model,
        cache_dir=settings.memory_model_cache_dir,
        device=settings.memory_model_device,
        router_enabled=False,
    )
    await manager.aload()
    router = ToolsetRouter(manager)
    tools = _all_declared_tools()
    rows = []
    try:
        for query, expected in CASES:
            decision = await router.route(query, tools)
            actual = set(decision.selected_toolset_names) if decision else {"FALLBACK_ALL"}
            rows.append({
                "query": query,
                "expected": sorted(expected),
                "actual": sorted(actual),
                "expected_is_subset": expected.issubset(actual),
                "scores": (
                    {
                        item.name: item.score
                        for item in decision.route_scores
                    }
                    if decision
                    else {}
                ),
            })
    finally:
        await manager.aclose()
    print(json.dumps({
        "model": settings.memory_reranker_model,
        "device": manager.device,
        "passed": sum(row["expected_is_subset"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
