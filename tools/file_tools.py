from itertools import islice
from pathlib import Path
from shutil import copy2

from langchain.tools import tool

from path import WORKSPACE_ROOT

PAGE_SIZE = 20

def _resolve_path(
    path: str,
) -> Path:
    """把工具收到的路径转换成最终绝对路径。"""

    input_path = (
        Path(path)
        .expanduser()
    )

    # 用户或模型明确传入绝对路径时，
    # 直接使用该路径。
    if input_path.is_absolute():
        return input_path.resolve()

    # 相对路径默认放到 workspace 中。
    resolved_path = (
        WORKSPACE_ROOT
        / input_path
    ).resolve()

    # 防止通过 ../ 离开 workspace。
    try:
        resolved_path.relative_to(
            WORKSPACE_ROOT
        )

    except ValueError as error:
        raise ValueError(
            "相对路径不能离开workspace目录。"
            "如需访问外部文件，请明确提供绝对路径。"
        ) from error

    return resolved_path

def _is_in_workspace(
    path: Path,
) -> bool:
    """判断解析后的路径是否位于workspace内部。"""

    try:
        path.resolve().relative_to(
            WORKSPACE_ROOT.resolve()
        )

        return True

    except ValueError:
        return False

def _build_copy_path(
    source_path: Path,
) -> Path:
    """为外部文件生成workspace内不冲突的副本路径。"""

    WORKSPACE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "".join(
        source_path.suffixes
    )

    if suffix:
        base_name = source_path.name[
            :-len(suffix)
        ]

    else:
        base_name = source_path.name

    candidate = (
        WORKSPACE_ROOT
        / f"{base_name}_copy{suffix}"
    )

    copy_number = 2

    while candidate.exists():
        candidate = (
            WORKSPACE_ROOT
            / (
                f"{base_name}_copy_"
                f"{copy_number}{suffix}"
            )
        )

        copy_number += 1

    return candidate.resolve()

def read_file(
    path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> str:
    """读取UTF-8文本文件的指定行。

    Args:
        path: 文件路径；相对路径默认位于项目workspace目录，绝对路径直接使用。
        start_line: 开始读取的行号，从1开始。
        max_lines: 最多读取的行数，范围1到1000。
    """

    if start_line < 1:
        return (
            "读取失败："
            "start_line不能小于1。"
        )

    if not 1 <= max_lines <= 1000:
        return (
            "读取失败："
            "max_lines必须在1到1000之间。"
        )

    try:
        file_path = _resolve_path(
            path
        )

    except ValueError as error:
        return f"读取失败：{error}"

    if not file_path.exists():
        return (
            "读取失败：文件不存在。"
            f"\n文件：{file_path}"
        )

    if not file_path.is_file():
        return (
            "读取失败：目标不是文件。"
            f"\n路径：{file_path}"
        )

    try:
        with file_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:

            # 跳过开始行之前的内容。
            for _ in range(
                start_line - 1
            ):
                if next(file, None) is None:
                    return (
                        "读取失败：start_line"
                        "超过文件总行数。"
                    )

            # 多读取一行，用来判断后面是否还有内容。
            lines = list(
                islice(
                    file,
                    max_lines + 1,
                )
            )

    except UnicodeDecodeError:
        return (
            "读取失败：文件不是UTF-8文本，"
            "或者它是二进制文件。"
        )

    except OSError as error:
        return f"读取失败：{error}"

    has_more = (
        len(lines) > max_lines
    )

    lines = lines[:max_lines]

    if not lines:
        return (
            f"文件：{file_path}\n"
            "指定位置之后没有内容。"
        )

    numbered_content = "\n".join(
        (
            f"{start_line + index}: "
            f"{line.rstrip(chr(13) + chr(10))}"
        )
        for index, line in enumerate(lines)
    )

    result = (
        f"文件：{file_path}\n"
        f"{numbered_content}"
    )

    if has_more:
        next_line = (
            start_line
            + max_lines
        )

        result += (
            "\n\n内容尚未读完，"
            f"下一次可从第{next_line}行继续。"
        )

    return result


def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """将UTF-8文本内容写入文件。

    Args:
        path: 文件路径；相对路径默认写入项目workspace目录，绝对路径直接使用。
        content: 需要写入文件的完整文本内容。
        overwrite: 文件已存在时是否允许整体覆盖，默认为False。
    """

    try:
        file_path = _resolve_path(
            path
        )

    except ValueError as error:
        return f"写入失败：{error}"

    if file_path.exists():

        if not file_path.is_file():
            return (
                "写入失败：目标路径存在，"
                "但不是文件。"
                f"\n路径：{file_path}"
            )

        if not overwrite:
            return (
                "写入失败：文件已经存在。"
                "如需覆盖，请将overwrite设为True。"
                f"\n文件：{file_path}"
            )

    try:
        # 自动创建中间文件夹。
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding="utf-8",
        )

    except OSError as error:
        return f"写入失败：{error}"

    return (
        "写入成功。"
        f"\n文件：{file_path}"
        f"\n字符数：{len(content)}"
    )


def list_directory(
    path: str = ".",
    offset: int = 0,
) -> str:
    """列出目录当前层级中的文件和子文件夹。

    Args:
        path: 目录路径；相对路径默认位于workspace，"."表示workspace根目录。
        offset: 跳过前面的结果，用于继续读取下一批。
    """

    if offset < 0:
        return "列出失败：offset不能小于0。"

    try:
        directory_path = _resolve_path(
            path
        )

    except ValueError as error:
        return f"列出失败：{error}"

    if not directory_path.exists():
        return (
            "列出失败：目录不存在。"
            f"\n目录：{directory_path}"
        )

    if not directory_path.is_dir():
        return (
            "列出失败：目标不是目录。"
            f"\n路径：{directory_path}"
        )

    try:
        entries = sorted(
            directory_path.iterdir(),
            key=lambda entry: (
                not entry.is_dir(),
                entry.name.casefold(),
            ),
        )

    except OSError as error:
        return f"列出失败：{error}"

    results = entries[
        offset:offset + PAGE_SIZE + 1
    ]

    has_more = (
        len(results) > PAGE_SIZE
    )

    results = results[:PAGE_SIZE]

    if not results:
        if offset == 0:
            return (
                "目录为空。"
                f"\n目录：{directory_path}"
            )

        return "没有更多目录内容。"

    lines = [
        f"目录：{directory_path}",
        "",
    ]

    for entry in results:
        entry_type = (
            "[DIR]"
            if entry.is_dir()
            else "[FILE]"
        )

        lines.append(
            f"{entry_type} {entry.resolve()}"
        )

    if has_more:
        next_offset = (
            offset
            + PAGE_SIZE
        )

        lines.extend(
            [
                "",
                "还有更多内容，"
                f"可将offset设为{next_offset}继续列出。",
            ]
        )

    return "\n".join(lines)

def grep_files(
    query: str,
    root: str = ".",
    offset: int = 0,
) -> str:
    """递归搜索UTF-8文本文件中的内容。

    Args:
        query: 要搜索的文本。
        root: 搜索路径；相对路径默认位于workspace，"."表示整个workspace。
        offset: 跳过前面的匹配结果，用于继续读取下一批。
    """

    if not query:
        return "搜索失败：query不能为空。"

    if offset < 0:
        return "搜索失败：offset不能小于0。"

    try:
        root_path = _resolve_path(
            root
        )
    except ValueError as error:
        return f"搜索失败：{error}"

    if not root_path.exists():
        return (
            "搜索失败：路径不存在。"
            f"\n路径：{root_path}"
        )

    if root_path.is_file():
        files = [
            root_path
        ]
    else:
        files = sorted(
            path
            for path in root_path.rglob("*")
            if path.is_file()
        )

    query_lower = query.casefold()

    results = []
    matched_count = 0
    page_size = PAGE_SIZE

    for file_path in files:
        try:
            with file_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):
                    if query_lower not in line.casefold():
                        continue

                    if matched_count < offset:
                        matched_count += 1
                        continue

                    line_text = line.rstrip(
                        "\r\n"
                    )

                    results.append(
                        f"{file_path}:"
                        f"{line_number}: "
                        f"{line_text}"
                    )

                    matched_count += 1

                    # 多取一条，用来判断是否还有下一页。
                    if len(results) > page_size:
                        break

        except (
            UnicodeDecodeError,
            OSError,
        ):
            # 自动跳过二进制、非UTF-8或无法读取的文件。
            continue

        if len(results) > page_size:
            break

    has_more = (
        len(results) > page_size
    )

    results = results[:page_size]

    if not results:
        return "没有找到匹配内容。"

    response = "\n".join(
        results
    )

    if has_more:
        next_offset = (
            offset
            + page_size
        )

        response += (
            "\n\n还有更多结果，"
            f"可将offset设为{next_offset}继续搜索。"
        )

    return response

def replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """精确替换文本文件中的一处内容。

    workspace内的文件会被直接修改。workspace之外的文件不会被修改，
    工具会先在workspace根目录生成带_copy后缀的安全副本，
    然后只修改这个副本。

    Args:
        path: 文件路径；相对路径默认位于workspace，外部文件必须使用绝对路径。
        old_text: 文件中需要替换的原始文本，必须唯一出现。
        new_text: 替换后的文本。
    """

    if not old_text:
        return "修改失败：old_text不能为空。"

    try:
        source_path = _resolve_path(
            path
        )

    except ValueError as error:
        return f"修改失败：{error}"

    if not source_path.is_file():
        return (
            "修改失败：文件不存在。"
            f"\n文件：{source_path}"
        )

    try:
        content = source_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return "修改失败：文件不是UTF-8文本。"

    except OSError as error:
        return f"修改失败：{error}"

    match_count = content.count(
        old_text
    )

    if match_count == 0:
        return "修改失败：没有找到old_text。"

    if match_count > 1:
        return (
            f"修改失败：old_text出现了{match_count}次。"
            "\n请提供包含更多上下文的old_text，"
            "确保它只出现一次。"
        )

    updated_content = content.replace(
        old_text,
        new_text,
        1,
    )

    is_external_file = not _is_in_workspace(
        source_path
    )

    if is_external_file:
        target_path = _build_copy_path(
            source_path
        )

        try:
            copy2(
                source_path,
                target_path,
            )

        except OSError as error:
            return (
                "修改失败：外部原文件未被修改，"
                "但创建workspace安全副本时失败。"
                f"\n原文件：{source_path}"
                f"\n错误：{error}"
            )

    else:
        target_path = source_path

    try:
        target_path.write_text(
            updated_content,
            encoding="utf-8",
        )

    except OSError as error:
        return f"修改失败：{error}"

    if is_external_file:
        return (
            "原文件位于workspace之外，"
            "因此没有直接修改。"
            f"\n原文件：{source_path}"
            "\n已创建并修改安全副本。"
            f"\n副本：{target_path}"
        )

    return (
        "修改成功。"
        f"\n文件：{target_path}"
    )

def find_files(
    pattern: str = "*",
    root: str = ".",
    offset: int = 0,
) -> str:
    """按照文件名查找文件。

    Args:
        pattern: 文件名匹配规则，例如"*.py"或"agent*.py"。
        root: 搜索目录；相对路径默认位于workspace。
        offset: 跳过前面的结果，用于继续读取下一批。
    """

    if offset < 0:
        return "查找失败：offset不能小于0。"

    try:
        root_path = _resolve_path(
            root
        )
    except ValueError as error:
        return f"查找失败：{error}"

    if not root_path.is_dir():
        return (
            "查找失败：搜索目录不存在。"
            f"\n目录：{root_path}"
        )

    try:
        matched_files = sorted(
            path.resolve()
            for path in root_path.rglob(
                pattern
            )
            if path.is_file()
        )
    except (
        OSError,
        ValueError,
    ) as error:
        return f"查找失败：{error}"

    page_size = PAGE_SIZE

    results = matched_files[
        offset:offset + page_size + 1
    ]

    has_more = (
        len(results) > page_size
    )

    results = results[:page_size]

    if not results:
        return "没有找到匹配的文件。"

    response = "\n".join(
        str(path)
        for path in results
    )

    if has_more:
        next_offset = (
            offset
            + page_size
        )

        response += (
            "\n\n还有更多结果，"
            f"可将offset设为{next_offset}继续查找。"
        )

    return response



read_file_tool = tool(
    parse_docstring=True
)(read_file)

write_file_tool = tool(
    parse_docstring=True
)(write_file)

list_directory_tool = tool(
    parse_docstring=True
)(list_directory)

grep_files_tool = tool(
    parse_docstring=True
)(grep_files)

replace_in_file_tool = tool(
    parse_docstring=True
)(replace_in_file)

find_files_tool = tool(
    parse_docstring=True
)(find_files)