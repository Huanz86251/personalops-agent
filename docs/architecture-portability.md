# Runtime boundaries and reuse outside AppWorld

PersonalOps has one planning/execution core and separate entry points. The normal Feishu path
enters `ConversationRuntime`; the AppWorld evaluator subclasses it in
`evals/appworld/conversation.py` and supplies simulated time, tools, environment instructions,
and the official grader. The general Scope Router/Resolver, Scheduler, Worker roles, review,
checkpoint, memory, and retrieval code are not AppWorld-only. AppWorld-specific tool calls,
API-document discovery, task-world clock, and grading remain in `evals/appworld/` and
`skills/appworld/`. To use another environment, it needs its own trustworthy tool adapter,
documentation source, clock semantics, and outcome verifier; an AppWorld score does not
establish accuracy there.

For a multi-turn Feishu conversation, the checkpoint retains all user instructions verbatim
and the last user-facing answer verbatim. Older user-facing answers may enter a rolling
summary. After a completed run, a bounded projection of Scheduler, Worker reports, replans,
and review status is summarized separately with the configured `summary` role (default
Qwen3.7-Flash). The next planning run sees this handoff but does not replay the full prior
internal trace. Original run checkpoints remain the audit source. Worker compaction stays off
by default; it governs **within-step** tool history, not the **between-turn** Feishu handoff.
Summaries are advisory, not a replacement for the original user request or verification.

RAG is also available on the normal runtime path, not just in AppWorld. The Feishu **Submit
RAG** action admits supported files to the owner's local knowledge base. Retrieval uses the
local index and reranker only after content is admitted; an empty index does not magically
provide API documentation. AppWorld's `appworld_discover` reads its own simulated API docs,
which are distinct from arbitrary user-uploaded RAG documents. Real-world integrations must
provide their own current interface documentation and permissions.

Private runtime state, RAG uploads, model weights, evaluation traces, and simulated accounts
stay under ignored local directories. The public repository contains code, generic examples,
offline tests, and aggregate evaluation results only.
