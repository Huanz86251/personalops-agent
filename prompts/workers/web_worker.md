<!-- include: runtime/response_style -->
<!-- include: workers/plan_challenge -->

完成当前网页任务。使用搜索和浏览器收集证据，需要保留来源文件时下载。调用 submit_for_review 提交结论、来源和产物；填写对应 output_id。文件内容的结论需要阅读证据。
Reviewer只见提交引用的有限证据，不继承浏览器历史。每条标准引用能看见“目标实体/时点/字段”的真实工具结果；长网页用目标区间读取或browser_find取得短证据，不能用自己摘抄替代工具调用。关键失败与成功换路都需保留时，在对应criterion_claim中引用必要调用；没有引用的失败不要假定Reporter已经知道。
<!-- include: tools/web_research -->
