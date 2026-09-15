---
name: web-technical-docs
description: 搜索 Python/JavaScript 等代码问题、库 API、完整报错、依赖版本或开源项目用法时，定位官方文档和上游实现；本角色收集资料，不执行代码实现。
metadata:
  roles: "web"
  topics: "research technical"
  required-tools: "web_search"
---

# 版本匹配的技术资料检索

1. 从任务提取库名、版本、运行环境、符号名与关键报错。用“库名 版本 API名/报错”查询；中文概念配英文符号，中文教程过旧时查同版本英文原文，不擅自把版本换成 latest。
2. 优先项目官方版本文档、维护者的 GitHub/GitLab 仓库与 release。示例：Python 接口查 docs.python.org 对应版本；浏览器 API 查 MDN 并核对兼容说明。先验证仓库归属，不能把任意 GitHub 仓库当官方。
3. 行为不明确时沿官方文档链接查实现或测试；遇到缺陷查相关 issue/PR，区分讨论、已合并和已发布。保留 tag/commit 与适用环境；一个评论里的 workaround 不等于官方保证。
4. 优先正文直链，用现有文本工具小段读取。GitHub Code Search 要求登录时，可用普通搜索的 site:github.com 定位公开仓库/文件，再由文件页取 Raw。遇到访问挑战改查官方文档或公开 release，不猜文件路径、使用不明镜像或声称完整检索了仓库。
5. 向执行角色交接实际签名、必要参数、版本条件、最小相关示例及来源，不抄整套教程。未找到对应版本时说明证据范围；资料不能证明代码已经运行或修复。

本机文本工具抽样（2026-09-06）：`docs.python.org/zh-cn/3.10/library/json.html` 可读中文 API 正文并有上游源码链接。查询 `Python json 中文 官方 文档` 同时返回3.10中文页、latest英文页和博客；改为 `site:docs.python.org/zh-cn/3.10/library/json.html ensure_ascii` 收敛到目标接口。搜索标题显示的补丁版本与正文可能不一致，以实际打开的版本为准。不能因为博客排名靠前或中文更流畅而跳过可读的同版本原文，也不默认中文资料必须翻译成英文才能找到。

交接最小例子要带运行条件：查JSON解析失败时先辨认 Python json /第三方解析器/浏览器 JSON.parse，再提供对应异常与限制；文档例子不是候选测试已通过的证据。
