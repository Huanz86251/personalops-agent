"""Deterministic General review routing; no routing model call."""
def general_review_reasons(step, trace, packet, stop_reason=''):
    result=trace.get('general_result') or {}
    reasons=[]
    submission = trace.get('worker_submission') or {}
    if result.get('plan_challenge') or (submission.get('submission') or {}).get('plan_challenge'):
        reasons.append('plan_challenge')
    if result.get('status')!='COMPLETED' or result.get('unresolved_items'):
        reasons.append('not_completed')
    schema_repair_only = trace.get('worker_finalize_reason') == 'SCHEMA_REPAIR'
    if (trace.get('error') or trace.get('finish_reason') in {'ERROR','BUDGET_EXHAUSTED','LEADERSHIP_CANCEL','LEADERSHIP_REPLACE'}
        or result.get('forced_finalization')
        or (trace.get('worker_finalize_requested') and not schema_repair_only)
        or '预算' in str(stop_reason) or 'budget' in str(stop_reason).lower()):
        reasons.append('interrupted_or_budget_exhausted')
    if result.get('files') or step.artifact_outputs or any(a.resolved_artifacts for a in packet.attempts):
        reasons.append('artifact_review')
    return reasons
