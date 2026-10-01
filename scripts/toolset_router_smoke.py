"""Run a provider-free smoke evaluation of production toolset recall.

Business-tool cases exercise the real first stage used in production:
BM25 top-3 plus Cross-Encoder top-5, fused with RRF.  The dedicated Qwen
selector is intentionally not called here, so this script remains free and
repeatable.  The direct-answer case separately checks the local NO_TOOL guard.
"""

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


RECALL_CASES = [
    ("查一下今天英伟达的最新新闻并核对来源", {"WEB_RESEARCH"}),
    ("登录网页邮箱，给指定联系人发送邮件", {"BROWSER_AUTOMATION"}),
    ("提取这个 PDF 第五页的表格并总结", {"DOCUMENT_READING"}),
    ("找出项目中所有引用 ToolsetRouter 的文件，只分析，不要修改", {"FILE_INSPECTION"}),
    ("修改配置文件里的模型名并回读确认", {"FILE_EDITING"}),
    ("实现这个功能并运行 pytest 验证", {"SOFTWARE_DEVELOPMENT"}),
    ("用符号计算解方程 x**2 - 5*x + 6 = 0", {"LOCAL_ANALYSIS"}),
    ("截一张当前Windows桌面图并识别上面的错误文字", {"DESKTOP_OBSERVATION"}),
    ("明天上午九点提醒我给客户回电话", {"SCHEDULE_CREATION"}),
    ("列出我的本地提醒并取消周报提醒", {"SCHEDULE_MANAGEMENT"}),
    ("明天上午九点在飞书提醒我提交材料", {"SCHEDULE_CREATION"}),
    ("十分钟后帮我查询上海天气并把结果告诉我", {"SCHEDULE_CREATION"}),
    ("帮我看看QQ邮箱最近十封邮件并总结重点", {"EMAIL_READING"}),
    ("把UID为12345的邮件中那份PDF附件下载下来", {"EMAIL_READING"}),
    ("根据刚才那封邮件写一份回复并保存到草稿箱，不要发送", {"EMAIL_DRAFTING"}),
    ("在山东省青岛市莱西市找几家健身房候选", {"AMAP_POI"}),
    ("查青岛北站到莱西市政府的驾车距离和路线", {"AMAP_ROUTE"}),
    ("把这个地址转换成坐标并查询莱西天气", {"AMAP_LOCATION_CONTEXT"}),
    ("给我生成一个去青岛北站的高德导航入口", {"AMAP_ACTION_LINKS"}),
    ("把这份PPTX转换成DOCX文件", {"DOCUMENT_CONVERSION"}),
    ("修改Excel公式、设置金额格式并生成柱状图", {"SPREADSHEET_WORK"}),
    ("把刚生成的报告目录打包回传到当前飞书", {"FEISHU_FILE_EXPORT"}),
    ("在 AppWorld 中查 API 并发送一封模拟邮件", {"APPWORLD"}),
    ("查询官方说明，然后把结论写进本地 Markdown", {"WEB_RESEARCH", "FILE_EDITING"}),
]

NO_TOOL_CASES = [
    ("解释一下什么是递归", {"NO_TOOL"}),
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
        for query, expected in RECALL_CASES:
            candidates = await router.hybrid_candidates(
                primary_task=query,
                user_request_window=query,
                available_tools=tools,
            )
            actual = set(candidates.names) if candidates else set()
            rows.append({
                "stage": "bm25_cross_encoder_rrf_recall",
                "query": query,
                "expected": sorted(expected),
                "actual_candidates": sorted(actual),
                "expected_is_subset": expected.issubset(actual),
                "bm25_top3": list(candidates.bm25_names) if candidates else [],
                "cross_encoder_top5": (
                    list(candidates.cross_encoder_names) if candidates else []
                ),
                "rrf_union": list(candidates.names) if candidates else [],
            })

        for query, expected in NO_TOOL_CASES:
            decision = await router.route(query, tools)
            actual = set(decision.selected_toolset_names) if decision else set()
            rows.append({
                "stage": "local_no_tool_guard",
                "query": query,
                "expected": sorted(expected),
                "actual_candidates": sorted(actual),
                "expected_is_subset": expected.issubset(actual),
                "bm25_top3": [],
                "cross_encoder_top5": [],
                "rrf_union": [],
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
