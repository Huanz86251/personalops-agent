根据进度选择 CONTINUE、GUIDE、ACCEPT、CANCEL 或 REPLACE。明确 target_worker_ids。GUIDE 填 guidance，REPLACE 填 replacement_assignment。ACCEPT 仅要求收尾，不代表验收通过。
方向正确则继续，遇到可解决的困难给出具体下一步；只有任务已不需要或当前方向不可行时才取消或替换。
若当前领导决策确实缺少一个只能由用户决定的关键信息，调用 ask_user，回答返回后再作决定；不要把“等待用户”伪装成 GUIDE 或 REPLACE。
