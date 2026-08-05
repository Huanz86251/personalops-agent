LangChain 的动态 Prompt Middleware 设计提示词

顺序	工具	文件	实现方式
1	get_current_time	time_tools.py	自己写
2	calculate	calculator_tools.py	自己写，不能直接 eval
3	list_directory	file_tools.py	自己写
4	read_file	file_tools.py	自己写
5	write_file	file_tools.py	自己写
6	replace_in_file	file_tools.py	自己写
7	search_files	file_tools.py	自己写
8	run_shell	shell_tools.py	自己写
9	run_python	python_tools.py	自己写
10	web_search	web_tools.py	Tavily 官方集成
11	web_fetch	web_tools.py	自己封装 httpx
12	get_weather	weather_tools.py	调天气 API
13	get_system_info	system_tools.py	自己写
14	list_available_tools	registry.py	自己写


自动记忆抽取
向量数据库
CE 重排
知识图谱
PostgreSQL
复杂任务队列
自动修改用户画像

pdf ocr 不同种类问题




第一步
拆分 prompts 文件
把 SYSTEM_PROMPT 和标题 Prompt 移出去

第二步
新增 context_middlewares.py
接入 SummarizationMiddleware

第三步
新增 AgentContext
建立稳定 user_id="owner"

第四步
ConversationRuntime 接入 AsyncSqliteStore

第五步
增加动态 Prompt Middleware
每次调用读取用户画像和相关记忆

第六步
增加结构化 Memory Consolidator
从摘要中提取长期记忆候选

第七步
再做记忆合并、替代、retired 和语义检索

最后一步
需要让其他 Agent 使用记忆时，再封装 MCP