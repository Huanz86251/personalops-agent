# LOCAL_DOCUMENTS 能力卡

## 选择它

任务需要读取或转换 PDF、PPTX、图片文字、表格等文档型文件，或需要 OCR、Excel 单元格读写、格式设置和图表操作。

## 不选择它

- 普通源码、配置文件、日志和 Markdown 的查找或读取。
- 运行 Python、pytest、Git、Shell 或静态检查。
- 只搜索互联网上的文档而不处理本地文件。

## 典型命令

- 提取这个 PDF 第 5 页的表格并总结。
- 识别这张中文截图里的文字。
- 修改 Excel 的公式和单元格格式，再生成图表。
- 把 PPTX 转成 PDF。

## 易混淆边界

FILE_INSPECTION 面向普通文本与目录；LOCAL_DOCUMENTS 面向需要解析器、OCR、格式转换或电子表格语义的文件。若先查找文件再处理 PDF，可以同时选择 FILE_INSPECTION。
