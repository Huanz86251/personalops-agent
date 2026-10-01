# Structured-output prompt audit

Date: 2026-09-23

## Rule used by the runtime

For verified Qwen 3.7/3.8 and `ZHIPU/GLM-5.3-Flash`, the provider receives the
complete JSON Schema exactly once through `response_format=json_schema` with
`strict=true`, or through a strict control-tool definition. The text prompt does
not repeat the full schema. The Harness still parses, normalizes and validates
the returned object.

For an unverified provider, the legacy compact text schema remains in the
prompt. This is a compatibility fallback, not a second validation contract.

Field-local meaning, limits and null/empty behavior belong in Pydantic `Field`
descriptions. Cross-field policy, evidence precedence, stopping rules and small
positive/negative examples remain in prompts because JSON Schema cannot express
them clearly enough for a Flash model.

## Production paths checked

| Role/path | Provider contract | Prompt result |
|---|---|---|
| Scheduler initial plan | `SupervisorDecision` | Full schema removed for strict providers; planning policy and Step examples retained. |
| Scheduler replan | `ReplanDecision` | Full schema removed for strict providers; downgrade, immutable-history and action policy retained. |
| Worker leader | leadership decision schema | Uses the same Scheduler disclosure rule; no repeated schema for strict providers. |
| Final Reviewer | `FinalReviewDecision` | Full schema removed for strict providers. Three copied full JSON objects were replaced by three short cross-field action examples. |
| Scope Resolver | strict `ScopeResolutionContinue` / `ScopeContract` tools | The prompt no longer embeds `ScopeContract.model_json_schema()`. The stale request saying “only output ScopeContract” now requires choosing exactly one action. Semantic set-algebra examples remain. |
| Skill Selector | `SkillChoice` | No full schema was embedded. Candidate matching examples remain; critical `skill_ids` rules now also live in the field description. |
| Initial capability selector | `ModelToolsetSelection` / `WebModelToolsetSelection` | Full schema removed for strict providers; browser and progress-checkpoint positive/negative policy retained. |
| Capability expansion selector | `CapabilityExpansionSelection` | Full schema removed for strict providers; one-shot gap policy retained. |
| General Step Reporter | `ReferencedStepReport` | Deleted the schema-only `step_report_output.md`; strict providers receive the contract only through the API. Evidence policy and schema-repair feedback remain. |
| Web completion | `WorkerSubmission` control tool | The schema is registered as the terminal tool, not pasted into the prompt. Web Worker policy remains because it governs evidence and stopping behavior. |
| General completion | General report control tool | The schema is registered as a control tool; no full text copy was found. |
| Code Worker and Code Reviewer | code submission/review control tools | Schemas are registered with their tools; prompts contain validation policy rather than copied JSON Schema. |
| Recovery proposal | `GeneralRecoveryProposal` / `CodeRecoveryProposal` | Full schema removed for strict providers; downgrade examples remain because they explain a cross-field business decision. |
| Host-file Judge | `HostFileJudgeOutput` | No full schema was embedded. `decision` and bounded-reason behavior are now explicit field descriptions. |
| Memory frame detection | `MemoryFramePlan` | Migrated verified Qwen/GLM models to native strict schema; the text schema remains only for unverified providers. |
| Memory typed extraction | dynamic records schema | Migrated to native strict schema with the same unknown-provider fallback. |
| Memory conflict resolution | `MemoryResolution` | Migrated to native strict schema; action examples remain and the repeated full schema was removed. |
| Ask User | strict function tool | No text schema copy; the tool description remains the user-interaction policy. |

## Largest duplicate copies eliminated

Measured compact provider schemas in the current tree:

- `SupervisorDecision`: about 16.8k characters (roughly 5.6k tokens).
- `ReplanDecision`: about 17.3k characters (roughly 5.8k tokens).
- `ScopeContract`: about 6.2k characters (roughly 2.1k tokens).
- `ReferencedStepReport`: about 6.1k characters (roughly 2.0k tokens).
- `FinalReviewDecision`: about 3.5k characters (roughly 1.2k tokens).

These copies are no longer repeated in the visible prompt for verified strict
providers. The API schema is still present and the Harness validators and repair
fallbacks are unchanged.

## Intentionally retained text

- Cross-field examples where one choice changes several nullable fields.
- Scheduler object-scope, set-algebra, mutation/readback and Step-merging rules.
- Reviewer evidence precedence and stopping rules.
- Repair prompts shown only after an actual validation failure. They use a
  bounded minimal shape instead of replaying the full schema.
- Unknown-provider text schemas, because those providers have not demonstrated
  native strict-schema support.

