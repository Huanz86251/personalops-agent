{
  "ls": "列出目录内容。",
  "read_file": "读取任务文件，也接受当前 Worker 的 /downloads/<candidate_id>。文本按行使用 offset（从0开始）和 limit；PDF/Office 在本地解析，不发送原始附件给模型，offset 为起始页减1，limit 每次最多20页。扫描页自动OCR；查看返回的页数和警告。完整文字按返回路径继续读取。单文件最多20MiB。图片只传 file_path。",
  "write_file": "创建文件或完整覆盖已有内容；局部修改用 edit_file。",
  "edit_file": "先读文件，再精确替换 old_string；保留缩进，不带显示行号。",
  "delete": "删除指定文件，或递归删除目录及其内容。",
  "glob": "按 glob 模式查找文件。无 / 的模式匹配各层文件名，含 / 则匹配相对路径；隐藏路径需显式匹配。",
  "grep": "按字面文本搜索，不支持正则；output_mode 决定返回文件名或内容。",
  "execute": "执行命令，返回输出和退出码；大输出可能截断。使用绝对路径，含空格的路径加引号。"
}
