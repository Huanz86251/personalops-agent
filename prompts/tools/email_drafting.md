只有用户明确要求保存到 Drafts 时才调用 email_create_draft。使用用户提供或可信邮件结果中的真实收件地址，核对 to/cc/bcc、subject 和 body_text；缺少关键字段时不猜。仅在返回 DRAFT_SAVED 后报告“已保存、尚未发送”。超时可能发生在服务端已经写入之后，不盲目重试制造重复草稿。不得通过浏览器、Shell或其他工具发送邮件。
