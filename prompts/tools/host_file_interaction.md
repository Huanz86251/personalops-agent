`host_list_files` 列出一层目录；查找Downloads、Desktop等根目录里的文件时先用它，避免无意义地递归扫描整棵子树。只有明确需要跨子目录查找时才用 `host_search_files`，并先采用较小的 `max_depth`；工具达到扫描上限会保留已找到结果并返回 `truncated=true`。

找到候选后，使用 `host_import_file` 把精确文件只读复制到当前Worker隔离工作区。查看、搜索和复制到隔离工作区默认允许，不需要Judge，也不得把它声称成已经获得对外发送或修改原文件的权限。只有 `IMPORTED` 才表示文件进入当前Worker工作区，后续读取使用返回的 `/downloads/...` 路径；若用户要求发回文件，使用文件发送能力提交该候选，发送会单独经过独立Judge。
