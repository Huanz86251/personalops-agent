# 独立选择各角色的模型

在本机 `.env` 修改对应角色的配置，下次重新启动应用时生效。当前没有重启正式服务。不同角色可以选择不同厂商、账户和 API 地址；同一角色在本次应用生命周期内使用固定客户端。

## 角色列表

| 配置前缀 | 用途 | 本机当前模型 |
| --- | --- | --- |
| `GENERAL_LLM` | General Agent | ZHIPU/GLM-5.3-Flash（思考 low） |
| `WEB_LLM` | Web Search / 浏览器 Agent | qwen3.7-flash（思考1024） |
| `CODE_LLM` | Code Agent | ZHIPU/GLM-5.3-Flash（思考 low） |
| `CODE_REVIEWER_LLM` | Code 独立审核 | ZHIPU/GLM-5.3-Flash（思考 low） |
| `REPORTER_LLM` | 通用独立 Reporter，包括 General 文件结果 | qwen3.8-flash（思考 low） |
| `WEB_REPORTER_LLM` | Web 独立 Reporter | qwen3.8-flash（思考 low） |
| `SCHEDULER_LLM` | 普通规划 Scheduler / Supervisor | qwen3.8-flash（思考 low） |
| `CODE_SCHEDULER_LLM` | Code 审核升级后的调度决定 | qwen3.8-flash（思考 low） |
| `REPLANNER_LLM` | 重规划 | qwen3.8-flash（思考 low） |
| `FINAL_REVIEWER_LLM` | 最终结果审核 | qwen3.8-flash（思考 low） |
| `WORKER_LEADER_LLM` | Worker 进度决策 | qwen3.8-flash（思考 low） |
| `TITLE_LLM` | 会话标题 | qwen3.8-flash（思考 low） |
| `SUMMARY_LLM` | 会话历史摘要 | qwen3.8-flash（思考 low） |
| `GENERAL_SUMMARY_LLM` | General 历史压缩 | qwen3.8-flash（思考 low） |
| `WEB_SUMMARY_LLM` | Web 历史压缩 | qwen3.8-flash（思考 low） |
| `CODE_SUMMARY_LLM` | Code 历史压缩 | qwen3.8-flash（思考 low） |
| `CODE_REVIEWER_SUMMARY_LLM` | Code Reviewer 历史压缩 | qwen3.8-flash（思考 low） |
| `EXTRACTION_LLM` | 长期记忆两轮提取 | qwen3.8-flash（思考 low） |

配置只决定已有调用使用谁，不新增调用。General 纯文本仍可自报告；Code 审核后由宿主确定性生成报告，不多调一个 Reporter。动态 Skills 选择使用独立 skill_selector，当前为 Qwen3.8 Flash low。本地 embedding、reranker、Write Gate 和注入检测不属于这些云端角色，继续用原来的本地配置。

当前生效配置（2026-09-11）：General、Code、Code Reviewer 使用百炼 ZHIPU/GLM-5.3-Flash；Web Search 使用 Qwen3.7 Flash，沿用 config/model_thinking.json 的1024思考预算；Web Reporter、Final Reviewer 与其余云角色保留 Qwen3.8 Flash low。共19角色。GLM 保留 low 档，使用 max_tokens 与 thinking.clear_thinking=true，不发送 Qwen 专用 preserve_thinking。密钥与百炼地址沿用现有配置。未重启服务、未付费验证；账号需已开通智谱直供服务。GLM 尚未加入本地定价表，费用估算应显示 unknown。

code_scheduler 是主规划图的代码审核升级分支：收到 Reviewer 上报后决定继续、重启或停止，并非第二套独立总调度器。记忆冲突目前由确定性规则建组，不增加云端冲突模型调用。

## 每个角色可以改什么

把下面的 `<前缀>` 换成上表任意前缀。

| 配置 | 含义 |
| --- | --- |
| `<前缀>_PROVIDER` | `openai`、`deepseek`、`anthropic`、`qwen` 或 `compatible` |
| `<前缀>_MODEL` | 供应商给出的准确模型 ID |
| `<前缀>_API_KEY_ENV` | 存放密钥的环境变量名；这里填变量名，不填密钥 |
| `<前缀>_BASE_URL` | 该角色 API 基础地址；千问可统一使用 DASHSCOPE_BASE_URL；其他兼容服务必须配置 |
| `<前缀>_THINKING_ENABLED` | `true` / `false`，用于 DeepSeek、千问及 GPT-5 nano 的已适配控制 |
| `<前缀>_REASONING_EFFORT` | 可选，直接指定供应商支持的推理档位 |
| `<前缀>_MAX_TOKENS` | 单次输出上限；默认为原来的 5000，提取为 16000 |
| `<前缀>_TIMEOUT_SECONDS` | 单次超时；普通默认 120，调度/审核默认 180 |
| `<前缀>_MAX_RETRIES` | SDK 网络重试次数，默认 2，填 0 可禁用；不改变业务层结构化修复次数 |
| `<前缀>_TOKEN_LIMIT_PARAMETER` | 兼容服务选 `max_tokens` 或 `max_completion_tokens` |
| `<前缀>_EXTRA_BODY_JSON` | 可选供应商专用 JSON 参数；不能覆盖模型、消息、流式开关和输出上限 |

`API_KEY_ENV` 留空时，默认使用 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`、`DASHSCOPE_API_KEY` 或 `COMPATIBLE_API_KEY`。同一厂商多个账户可以分别填写自己的变量名。配置指定了一个密钥变量但没有提供其值时，启动检查会报错，不会悄悄借用别的厂商密钥。

`BASE_URL` 留空时读取对应厂商的 `OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL`、`ANTHROPIC_BASE_URL`、`DASHSCOPE_BASE_URL` 或 `COMPATIBLE_BASE_URL`；前三种可以继续使用 SDK 官方默认地址。

本机已经给每个角色写了显式模型配置。代码仍保留旧 `LLM_*` / `HARD_LLM_*` 作为缺省配置来源；已有显式角色配置不受它们改变影响。旧的 `SUMMARY_LLM_*` 也是尚未单独指定的 Worker 摘要模型的缺省来源。

## 示例：只把 Web Agent 换成千问

在本机 `.env` 修改现有同名项，避免添加重复配置。以下密钥是占位符，需要替换为你自己的值：

```dotenv
WEB_LLM_PROVIDER=qwen
WEB_LLM_MODEL=qwen3.7-flash
WEB_LLM_API_KEY_ENV=MY_WEB_QWEN_KEY
MY_WEB_QWEN_KEY=replace_with_your_key
WEB_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
WEB_LLM_THINKING_ENABLED=false
WEB_LLM_MAX_TOKENS=5000
```

这只改变 Web Agent，不改变 Web Reporter 或其他角色。北京旧域名仍可使用；也可填写百炼控制台给出的业务空间专属地址，密钥与地域需匹配。参见[千问官方兼容 API 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。

## 示例：Code 使用另一家兼容 API

```dotenv
CODE_LLM_PROVIDER=compatible
CODE_LLM_MODEL=your_model_id
CODE_LLM_API_KEY_ENV=MY_CODE_API_KEY
MY_CODE_API_KEY=replace_with_your_key
CODE_LLM_BASE_URL=https://your-provider.example/v1
CODE_LLM_TOKEN_LIMIT_PARAMETER=max_tokens
```

服务必须提供 Chat Completions 接口；Agent 角色还需支持工具调用。供应商专用参数可以通过 `EXTRA_BODY_JSON` 添加。`compatible` 不会自动加入千问的 `enable_thinking` 或 DeepSeek 的 `thinking` 参数。

千问默认发送 `enable_thinking`。较新千问使用 `max_completion_tokens`；旧 `qwen-flash*` 默认使用 `max_tokens`，其他服务可自行配置参数名。具体模型支持哪些推理参数由该供应商决定，不保证任意模型都能关掉推理。

GPT-5 nano 的 `false` 对应 `minimal`，仍可能计费推理 token；不能把它称为完全非推理模型。OpenAI 其他型号可显式填写其官方支持的 `REASONING_EFFORT`，不要假设所有型号都支持 `none`。

## 查看配置与验证范围

```text
C:/ProgramData/anaconda3/envs/agent/python.exe scripts/show_model_roles.py
```

该命令只读配置，不构造模型客户端、不发送网络请求、不显示密钥。缺密钥时会报告缺少哪个变量。

本轮对千问和兼容 API 使用本地模拟 HTTP，验证独立地址、密钥和请求参数；未进行供应商真实质量测试。用户唯一授权的真实调用已用于此前 GPT-5 nano 的合成摘要/提取样例，成功，输入 156、输出 68、推理 0 token。后续角色拆分没有再调用 API。
