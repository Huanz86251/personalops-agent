from functools import lru_cache
from pathlib import Path
import re
from path import PROJECT_ROOT


PROMPT_ROOT = (
    PROJECT_ROOT
    / "prompts"
)
PROMPT_VARIABLE_PATTERN = re.compile(
    r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}"
)
@lru_cache(
    maxsize=None,
)
def load_prompt(
    prompt_name: str,
) -> str:
    """读取prompts目录中的Markdown提示词。"""

    normalized_name = (
        prompt_name
        .strip()
    )

    if not normalized_name:
        raise ValueError(
            "prompt_name不能为空。"
        )

    # 只允许传入简单名称，
    # 防止出现 ../ 或子目录路径。
    if (
        Path(normalized_name).name
        != normalized_name
    ):
        raise ValueError(
            "prompt_name只能是提示词名称，"
            "不能包含路径。"
        )

    prompt_path = (
        PROMPT_ROOT
        / f"{normalized_name}.md"
    )

    try:
        prompt_text = (
            prompt_path
            .read_text(
                encoding="utf-8",
            )
            .strip()
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "没有找到提示词文件："
            f"{prompt_path}"
        ) from error

    except OSError as error:
        raise RuntimeError(
            "读取提示词文件失败："
            f"{prompt_path}"
        ) from error

    if not prompt_text:
        raise RuntimeError(
            "提示词文件内容为空："
            f"{prompt_path}"
        )

    return prompt_text
def render_prompt(
    prompt_name: str,
    **values: object,
) -> str:
    """读取提示词，并替换其中的双大括号变量。"""

    template = load_prompt(
        prompt_name
    )

    variable_names = set(
        PROMPT_VARIABLE_PATTERN.findall(
            template
        )
    )

    missing_names = sorted(
        variable_names
        - values.keys()
    )

    if missing_names:
        missing_text = ", ".join(
            missing_names
        )

        raise RuntimeError(
            f"提示词{prompt_name!r}"
            "缺少变量："
            f"{missing_text}"
        )

    def replace_variable(
        match: re.Match,
    ) -> str:
        variable_name = (
            match.group(1)
        )

        return str(
            values[variable_name]
        )

    rendered_prompt = (
        PROMPT_VARIABLE_PATTERN
        .sub(
            replace_variable,
            template,
        )
        .strip()
    )

    if not rendered_prompt:
        raise RuntimeError(
            "渲染后的提示词为空："
            f"{prompt_name}"
        )

    return rendered_prompt


