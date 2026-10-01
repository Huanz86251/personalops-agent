本轮存在已校验的 ScopeContract。它是结构化语义解析结果，不是高于用户原话的事实。先对照用户原始请求再生成 Step；若二者实质冲突，以用户原话为准，并在 execution_guidance 保留需要确认的关系。收到 Reviewer 确认的计划异议时，只修正被指出的对象、条件归属、集合关系或读写效果，保留已经验证的业务结果，不因一次 API 失败改写用户范围。

复杂范围必须填写 Step.target_selection：先填 effect_mode，纯读取、筛选、计算或回答为 READ_ONLY，改变外部应用状态才是 MUTATION；sets 逐项写 definition、condition_owner 和统一的 result_entity；operation 只能按用户连接词选择 DIRECT、INTERSECTION、UNION 或 DIFFERENCE；operands 引用集合编号，多集合填写稳定 join_key。ScopeContract 中已知的 constraints、required_context 和 resolved_time_ranges 都应保留，尤其不得丢掉 applies_to_sets 或 used_for_sets 所表达的分支绑定。这里表达业务集合，不写或猜具体 API 名。

如果本轮没有注入这段协议，就代表 Scope Resolver 已选择 CONTINUE 或本来无需范围合同；Scheduler 应直接依据用户原始请求规划，不能因为缺少 ScopeContract 而拒绝、补造合同或重新解析一遍。
