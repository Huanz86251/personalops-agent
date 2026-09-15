<div align="center">

# PersonalOps Agent

**A personal tool-using agent that resolves scope before it plans, executes, verifies, and repairs.**

[中文](README.md) · [Benchmark protocol](docs/benchmark.md) · [Models and datasets on Hugging Face](https://huggingface.co/chris0809)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Feishu](https://img.shields.io/badge/Feishu-Personal_Agent-3370FF?logo=lark&logoColor=white)](https://open.feishu.cn/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Workflow-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![AppWorld Pass@1](https://img.shields.io/badge/AppWorld_Normal_Pass%401-74.4%25-27AE60)](docs/benchmark.md)

![Synthetic PersonalOps workflow demo](docs/assets/personalops-demo.gif)

</div>

## Why this project is evaluated end to end

A tool-using agent can sound correct while selecting the wrong object, calling a nearby API,
skipping read-after-write verification, or repeating work after a worker handoff. PersonalOps
therefore evaluates final world state instead of treating an agent self-report as success.

On **168 AppWorld Test-N Normal tasks**, with one attempt per task and no failed-task reruns,
the system passed **125/168 (74.4% Pass@1)**. Cloud planning and execution used only
Qwen3.8-Flash and GLM-5.3-Flash, without Max/Pro tiers or high-reasoning mode. The arithmetic
mean of per-task prompt-cache hit rates was **78.9%**; cost per task was **¥0.112 P50** and
**¥0.355 P90**; end-to-end latency was **215.1 s P50** and **645.9 s P90**.

See the [full aggregate protocol and percentile table](docs/benchmark.md). Raw tasks, simulated
accounts, traces, runtime archives, private conversations, and credentials are never published.

## Architecture

PersonalOps uses LangGraph and DeepAgents to implement:

- durable Feishu event ingestion, isolated conversations, checkpoints, interruption, and recovery;
- a local Scope Router plus a narrow Scope Resolver for requests with complex object relations;
- a Scheduler and specialized General, Web, and Code workers;
- structured Scope Contracts, Action Cards, evidence receipts, and cross-worker handoffs;
- specialized review where required, followed by a bounded Final Reviewer repair loop;
- Docker-isolated code and file processing, plus browser, web, OCR, and simulated AppWorld tools;
- Phoenix traces for planning, model calls, tool calls, review, token use, cache use, and cost.

The worker tool catalog is selected dynamically. A local Cross Encoder ranks capability groups
using the user request and current step; only low-confidence routes use a short selection call with
the current worker model. API documentation and ordinary documents use hierarchical RAG with
dense retrieval, BM25, Cross Encoder reranking, and explicit on-demand expansion.

## Feishu RAG admission

The control card contains a **Submit RAG** action. The next message is admitted to the local
knowledge base and does not enter the agent task queue.

- TXT, Markdown, JSON/JSONL, CSV, HTML, PDF, DOCX, XLSX, and PPTX are supported.
- OpenAPI and compatible API JSON become an app/resource/read-write/endpoint hierarchy.
- Ordinary documents retain headings and structural paths.
- PDFs are parsed directly first; unreadable scanned PDFs fall back to an isolated OCR process.
- Unsupported, invalid, or unreadable input receives a clear rejection message.

Uploads, indexes, and receipts live under the Git-ignored `.agent/rag/` directory.

## Local setup

Python 3.10 is the primary development environment.

```bash
git clone https://github.com/Huanz86251/personalops-agent.git
cd personalops-agent
conda env create -f environment.yml
conda activate agent

# Download the two public fine-tuned local classifiers.
bash scripts/setup_local_models.sh

# Optional: also prepare RAG and OCR dependencies/models.
WITH_RAG_MODELS=1 WITH_OCR=1 bash scripts/setup_local_models.sh
```

On Windows PowerShell:

```powershell
.\scripts\setup_local_models.ps1
$env:WITH_RAG_MODELS="1"; $env:WITH_OCR="1"; .\scripts\setup_local_models.ps1
```

The scripts download
[`chris0809/scope-intent-distilmbert`](https://huggingface.co/chris0809/scope-intent-distilmbert)
and
[`chris0809/memoperator-0.6b-memory-write-gate`](https://huggingface.co/chris0809/memoperator-0.6b-memory-write-gate)
into the Git-ignored `.models/` directory. They do not read or write API keys.

Copy the environment template and replace its placeholders locally:

```bash
cp .env.example .env
python main.py
```

Never commit `.env`. The repository includes only empty or clearly fake configuration values.

## Local classifiers and public data

Two bilingual synthetic datasets contain 20,000 examples each. Seed families are split across
train, validation, and test sets to avoid evaluating on rewrites of a training seed.

- DistilBERT Scope Router: decides whether a narrow Scope Resolver call is needed.
- MemOperator-0.6B with a LoRA classification head: decides whether a user message should enter
  the long-term-memory extraction queue.

Generation, training, blind-evaluation, and Hugging Face publishing scripts are retained under
`scripts/`; local weights are excluded from Git. Synthetic-set metrics are reported as such and
do not replace human-labelled production evaluation.

## Security and evidence boundary

The public repository excludes API keys, `.env`, local model weights, runtime databases,
personal messages, raw AppWorld tasks, execution archives, Phoenix traces, simulated credentials,
and unpacked benchmark images. Code sandboxes run without network access, with a read-only root
filesystem and bounded resources. AppWorld official grading, agent self-reports, infrastructure
health, and trace status remain separate signals.

This is a single-owner local agent research project, not a multi-tenant SaaS service.
