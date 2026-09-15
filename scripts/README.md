# Script inventory

Every script in this directory was reviewed before the public repository release. Scripts do
not embed credentials. Commands that can call a paid provider or a private service are opt-in and
say so in their module docstring; setup and analysis commands are safe by default.

## Supported setup and daily operation

| Script | Purpose |
| --- | --- |
| `setup_local_models.sh` / `setup_local_models.ps1` | Install local dependencies and download the two public fine-tuned classifiers. |
| `setup_local_classifiers.py` | Download only the Scope Router and Memory Write Gate into `.models/`. |
| `setup_rag.py` | Pre-download public embedding and reranker models. |
| `setup_ocr.py` | Install the isolated OCR runtime. |
| `setup_prompt_injection_guard.py` | Pre-download the two-stage injection guard. |
| `setup_code_sandbox.py` | Build or verify the Docker code sandbox. |
| `setup_local_tools.py` | Install document tools without modifying the Agent environment. |
| `rag_documents.py` | Index or query the local document knowledge base. |
| `show_model_roles.py` | Print resolved model roles without keys. |
| `tool_inventory.py` | Print role/tool exposure without starting MCP or a model. |
| `appworld_batch_stats.py` | Aggregate private batch records without reading task text. |
| `resume_status.py` | Compare the local checkpoint with current source and private evidence. |

## Reproducible classifiers and public artifacts

- `generate_scope_intent_dataset.py`, `expand_scope_intent_dataset.py`, and
  `train_scope_intent_classifier.py` reproduce the Scope Router experiment.
- `generate_memory_write_gate_dataset.py`, `expand_memory_write_gate_long_dataset.py`,
  `train_memory_write_gate_memoperator.py`, and `evaluate_memory_write_gate_blind.py` reproduce
  the Memory Write Gate experiment.
- `publish_huggingface_classifiers.py` prints its upload manifest by default and only publishes
  with an explicit `--push` flag.
- `generate_readme_demo.py` regenerates the synthetic README GIF. It does not read traces.

Generated JSONL corpora are excluded from Git because their canonical copies are on Hugging Face.
Seed files, prompts, metrics, model cards, and training code remain in this repository. Model
weights stay under `.models/` and are also excluded.

## Read-only analysis

`analyze_chain_audit.py`, `analyze_scripted_routes.py`, `audit_phoenix.py`,
`summarize_runtime_traces.py`, `verify_full_conversation_evidence.py`, and
`verify_trace_hierarchy.py` inspect local evidence. `preview_appworld_trace.py` can copy an existing
local trace into a clearly labelled local Phoenix preview; it does not publish the trace.

## Provider-free probes

The `probe_*`, `audit_scripted_routes.py`, `replay_plan_repair.py`, `toolset_router_smoke.py`,
`verify_rag_expansion.py`, `verify_rag_integration.py`, and `verify_rag_local.py` files are focused
diagnostics. Their module docstrings state whether they use local Docker, public network access,
cached models, or synthetic fixtures. They are retained because they exercise real integration
boundaries that unit tests mock.

## Explicitly armed operations

- `run_chain_audit.py` can make a real provider call only after explicit arming and stores private
  evidence locally.
- `probe_email_attachment_read.py` performs a bounded read from a configured private mailbox; it
  never sends mail.
- `publish_huggingface_classifiers.py --push` writes to the configured Hugging Face account.

The historical one-shot `validate_nano_once.py` entry was removed: it duplicated current model
client validation and its “run once” state made it unsuitable as a reusable public command.
