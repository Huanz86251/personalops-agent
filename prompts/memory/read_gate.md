判断记忆对当前问题是否有帮助，只输出 RELEVANT 或 NOT_RELEVANT。
记忆能补充回答所需的偏好、条件或项目背景时才相关；仅出现同一个人或一个相同词语不够。
示例：用户问“晚饭做什么”，记忆为“用户对花生过敏” → RELEVANT。
反例：用户问“修复 Python 报错”，记忆为“用户喜欢跑步” → NOT_RELEVANT。

待判断：
用户：{{user_message}}
记忆：{{memory_text}}
