from __future__ import annotations

import asyncio
import os
from urllib.parse import (
    urlsplit,
)

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from langchain.tools import tool

from path import PROJECT_ROOT


# 即使单独运行web_tools.py测试，
# 也能正确读取项目根目录中的.env。
load_dotenv(
    dotenv_path=(
        PROJECT_ROOT
        / ".env"
    ),
    override=False,
)


# 这些配置不暴露到.env，
# 保持工具配置简单。
SEARCH_PAGE_SIZE = 5
SEARCH_TIMEOUT_SECONDS = 15
SEARCH_SAFESEARCH = "moderate"


def _get_search_backend() -> str:
    """读取当前搜索后端。"""

    backend = os.getenv(
        "WEB_SEARCH_BACKEND",
        "yandex",
    ).strip()

    return (
        backend
        or "yandex"
    )


def _get_search_region() -> str:
    """读取当前搜索区域。"""

    region = os.getenv(
        "WEB_SEARCH_REGION",
        "wt-wt",
    ).strip()

    return (
        region
        or "wt-wt"
    )


def _run_web_search(
    query: str,
    page: int,
    max_results: int,
) -> list[
    dict[
        str,
        str,
    ]
]:
    """在线程中执行同步DDGS搜索。"""

    return DDGS(
        timeout=(
            SEARCH_TIMEOUT_SECONDS
        )
    ).text(
        query=query,

        region=(
            _get_search_region()
        ),

        safesearch=(
            SEARCH_SAFESEARCH
        ),

        max_results=max_results,

        page=page,

        backend=(
            _get_search_backend()
        ),
    )


async def web_search(
    query: str,
    page: int = 1,
) -> dict:
    """搜索互联网中的公开网页。

    每次返回最多5条标题、URL和搜索摘要。
    当前结果不足时，应保持相同query，
    并把page增加为2、3、4以继续搜索。

    本工具只负责搜索，不会打开、读取或点击网页。
    需要深入读取网页时，应把选中的URL交给浏览器工具。
    除非用户明确要求，否则优先选择可直接阅读的文本网页，而非视频网站。
    如果中文搜索质量差可换用英文关键词。

    Args:
        query: 清晰具体的中文或英文搜索关键词。
        page: 搜索结果页码，从1开始。

    Returns:
        当前搜索后端、页码和搜索结果。
    """

    normalized_query = (
        " ".join(
            query
            .strip()
            .split()
        )
    )

    if not normalized_query:
        return {
            "error": (
                "query不能为空。"
            )
        }

    if page < 1:
        return {
            "error": (
                "page必须从1开始。"
            )
        }

    try:
        results = await asyncio.to_thread(
            _run_web_search,

            normalized_query,

            page,

            SEARCH_PAGE_SIZE,
        )

    except Exception as error:
        return {
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),

            "backend": (
                _get_search_backend()
            ),

            "query": normalized_query,

            "page": page,
        }

    return {
        "backend": (
            _get_search_backend()
        ),

        "region": (
            _get_search_region()
        ),

        "query": normalized_query,

        "page": page,

        "results": results,

        # 返回完整5条时，
        # 提示模型可以继续请求下一页。
        "next_page": (
            page + 1

            if len(results)
            >= SEARCH_PAGE_SIZE

            else None
        ),
    }


def _parse_github_repository(
    github_url: str,
) -> tuple[
    str,
    str,
    str,
]:
    """解析GitHub仓库URL。"""

    normalized_url = (
        github_url
        .strip()
    )

    if not normalized_url:
        raise ValueError(
            "github_url不能为空。"
        )

    # 同时支持：
    # github.com/owner/repository
    if "://" not in normalized_url:
        normalized_url = (
            "https://"
            + normalized_url
        )

    parsed_url = urlsplit(
        normalized_url
    )

    hostname = (
        parsed_url.hostname
        or ""
    ).lower()

    if hostname not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError(
            "只支持github.com仓库地址。"
        )

    path_parts = [
        part
        for part in (
            parsed_url.path
            .strip("/")
            .split("/")
        )
        if part
    ]

    if len(path_parts) < 2:
        raise ValueError(
            "GitHub地址必须包含"
            "owner和repository。"
        )

    owner = path_parts[0]

    repository = (
        path_parts[1]
        .removesuffix(
            ".git"
        )
    )

    original_url = (
        "https://github.com/"
        f"{owner}/{repository}"
    )

    return (
        owner,
        repository,
        original_url,
    )

def _build_gitcode_mirror_url(
    repository: str,
) -> str:
    """根据GitCode常见规则生成镜像候选URL。"""

    prefix = (
        repository[:2]
        .casefold()
    )

    return (
        "https://gitcode.com/"
        "gh_mirrors/"
        f"{prefix}/"
        f"{repository}"
    )

async def _verify_gitcode_page(
    url: str,
    owner: str,
    repository: str,
) -> bool:
    """确认GitCode镜像页面可以访问且包含仓库信息。"""

    try:
        async with httpx.AsyncClient(
            timeout=10,

            follow_redirects=True,

            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64)"
                ),
            },
        ) as client:
            response = await client.get(
                url
            )

    except Exception:
        return False

    if response.status_code != 200:
        return False

    page_text = (
        response.text
        .casefold()
    )

    invalid_markers = (
        "404 not found",
        "页面不存在",
        "项目不存在",
        "仓库不存在",
    )

    if any(
        marker in page_text
        for marker in invalid_markers
    ):
        return False

    repository_name = (
        repository
        .casefold()
    )

    original_signature = (
        "github.com/"
        f"{owner}/{repository}"
    ).casefold()

    return (
        repository_name in page_text
        or original_signature in page_text
    )

async def find_github_mirror(
    github_url: str,
) -> dict:
    """查找适合中国网络访问的GitHub仓库镜像。

    工具会根据GitCode常见镜像路径生成候选地址，
    然后真实访问该页面进行基础验证。

    找到时返回GitCode镜像URL；
    不存在时返回not_found。

    镜像只适合公开仓库的只读浏览。

    Args:
        github_url: GitHub公开仓库URL。

    Returns:
        原始仓库地址、镜像地址和验证状态。
    """

    try:
        (
            owner,
            repository,
            original_url,
        ) = _parse_github_repository(
            github_url
        )

    except ValueError as error:
        return {
            "status": "invalid",

            "error": str(
                error
            ),
        }

    mirror_url = (
        _build_gitcode_mirror_url(
            repository
        )
    )

    is_valid = await (
        _verify_gitcode_page(
            mirror_url,

            owner,

            repository,
        )
    )

    if not is_valid:
        return {
            "status": "not_found",

            "github_url": (
                original_url
            ),

            "mirror_url_checked": (
                mirror_url
            ),

            "message": (
                "没有找到能够验证的"
                "GitCode镜像。"
            ),
        }

    return {
        "status": "found",

        "github_url": (
            original_url
        ),

        "mirror_url": (
            mirror_url
        ),

        "note": (
            "镜像页面已经通过基础可用性验证。"
        ),
    }

web_search_tool = tool(
    parse_docstring=True
)(web_search)


github_mirror_tool = tool(
    parse_docstring=True
)(find_github_mirror)