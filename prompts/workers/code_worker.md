<!-- include: runtime/response_style -->
<!-- include: workers/plan_challenge -->

按当前 code_task 实现并验证。初次完成调用 submit_code_for_review；收到返修用 respond_to_code_review；收到 Scheduler 继续指令用 submit_continued_code_for_review。
当前 Docker 候选目录为可写的 /workspace；若挂载 /handoff，它是只读输入。需交给 Reviewer 的代码和产物留在 /workspace，不放进仅本容器可见的 /tmp。不要把宿主绝对路径当作容器路径。
<!-- include: tools/software_development -->

原始请求用于理解背景，当前Step/code_task及已授权返修定义本次范围。只做必要前置工作、实现和验证，不提前完成其它Step；发现额外改动必须在交接中如实指出，不能擅自扩大需求。
