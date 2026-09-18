---
name: appworld-csv-export
description: 在 AppWorld 中通过 file_system 创建、覆盖、更新或导出 CSV 文件时选择；按任务指定的表头、列序和分隔符构造兼容文本，并在删除账户等后续不可逆动作前回读原始内容验收。不用于普通本地 CSV 编程，也不用于只读取且不修改现有 CSV 的任务。
metadata:
  roles: "general"
  topics: "appworld csv file-export"
  required-tools: "appworld_discover appworld_execute"
---

# AppWorld CSV 导出

`file_system.create_file` 和 `update_file` 保存传入的原始字符串，不会替你生成或校验 CSV。先按用户原话固定路径、表头、列序、列分隔符和列内分隔符；未指定的格式细节采用最小必要引号：字段不含逗号、双引号、回车或换行时直接写字面值，只有包含这些字符时才按 CSV 规则加双引号并把内部双引号写成两个。不要为了“保险”给每个字段统一加引号。

接口签名已有可信依据时直接复用；否则先用 `appworld_discover` 查询 `file_system.create_file`、`update_file` 或 `show_file` 的真实文档。典型调用是：

```python
apis.file_system.create_file(
    file_path="~/backups/export.csv",
    content=content,
    overwrite=False,
    access_token=file_token,
)
apis.file_system.show_file(
    file_path="~/backups/export.csv",
    access_token=file_token,
)
```

## 正例：最小必要引号

下面是格式方法示例，不是当前任务的数据、路径或固定答案。`rows` 必须来自本轮真实读取结果；`file_token` 必须来自已确认的登录回执。

```python
def csv_cell(value):
    text = str(value)
    if any(char in text for char in (',', '"', '\r', '\n')):
        return '"' + text.replace('"', '""') + '"'
    return text

header = ["Title", "Artists"]  # 替换成用户实际要求的表头和列序
lines = [",".join(csv_cell(value) for value in header)]
for title, artists in rows:
    lines.append(",".join(csv_cell(value) for value in (title, artists)))
content = "\n".join(lines) + "\n"

created = apis.file_system.create_file(
    file_path=output_path,
    content=content,
    overwrite=False,
    access_token=file_token,
)
print(created)
```

若安全字段为 `Example Song` 和 `Alice|Bob`，原始数据行应是：

```text
Example Song,Alice|Bob
```

若字段确实含逗号，例如 `Example, Live`，该字段才写成：

```text
"Example, Live",Alice|Bob
```

## 反例：统一加引号并用反向解析自证

不要把每个安全字段都固定写成：

```python
lines.append(f'"{title}","{artists}"')
```

也不要在验收时再用 `lstrip('"')`、`rstrip('"')` 等操作主动抹掉这些引号后宣布格式通过。归一化后的字段相同只能证明部分业务内容相同，不能证明保存的原始 CSV 符合要求。

## 回读验收

写入成功后用 `show_file` 回读同一路径，并分别保留两类证据：

1. 原始格式：精确表头、原始首行和末行、换行与分隔符、总行数；安全字段不应出现无必要的外层引号。
2. 业务内容：每个来源对象恰好对应一行，去重键、列值和列内分隔符与用户要求一致。

```python
shown = apis.file_system.show_file(
    file_path=output_path,
    access_token=file_token,
)
raw = shown["content"]
raw_lines = raw.splitlines()
assert raw_lines and raw_lines[0] == ",".join(header)
assert raw == content
assert len(raw_lines) == len(rows) + 1
print({
    "header": raw_lines[0],
    "row_count": len(raw_lines) - 1,
    "first_raw_row": raw_lines[1] if len(raw_lines) > 1 else None,
    "last_raw_row": raw_lines[-1] if len(raw_lines) > 1 else None,
})
```

若解析器或验收代码自身抛异常，本轮只能得出“验收未完成，文件结论未知”。修复验收代码后检查同一格式要求；不能换成会删除引号、忽略分隔符或改写原始文本的宽松解析方式。只有原始格式和业务内容都通过，才继续删除账户或执行其他不可逆的后续动作。

验收代码补充：不要把**验收器自己的类型错误**当成文件写错。若 `rows` 的 `Artists` 已是 `"Alice|Bob"` 这样的字符串，从回读原文实际解析得到的 `parsed_rows` 也应保留该列为字符串，再逐行比较；不要只在一边做 `.split("|")`，拿名单与字符串比。若确需按名单比较，两边都拆。以下比较不代替上面的原始表头、引号和行数检查：

```python
# parsed_rows 须来自本轮回读的实际解析，不能凭预期内容伪造。
expected_pairs = [(title, artists_text) for title, artists_text in rows]
actual_pairs = [(title, artists_text) for title, artists_text in parsed_rows]
assert all(isinstance(artists_text, str) for _, artists_text in actual_pairs)
assert actual_pairs == expected_pairs
```

当前固定 AppWorld 执行沙箱禁止导入 `io`，也禁止调用 `csv.reader`；不要把它们写成必需的验收路径。使用环境允许的检查方式，并继续保持原始格式与业务内容两项独立验收。
