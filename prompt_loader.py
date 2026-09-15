from functools import lru_cache
import json
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
PROMPT_NAME_PART_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]+$"
)
PROMPT_INCLUDE_PATTERN = re.compile(r"<!-- include: ([A-Za-z0-9_/-]+) -->")
RUNTIME_CONTEXT_MARKER = "<!-- runtime-context -->"


def render_structured_prompt(prompt_name: str, output_schema) -> str:
    """Use the same model for the visible output format and response validation."""
    return render_prompt(
        prompt_name,
        schema=json.dumps(output_schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")),
    )


def _read_composed_prompt(prompt_name: str, stack: tuple[str, ...] = ()) -> str:
    if prompt_name in stack:
        raise ValueError(f"Prompt include cycle: {' -> '.join((*stack, prompt_name))}")
    text = _resolve_prompt_path(prompt_name).read_text(encoding="utf-8").strip()
    return PROMPT_INCLUDE_PATTERN.sub(
        lambda match: _read_composed_prompt(match[1], (*stack, prompt_name)), text
    )


def split_prompt(prompt: str) -> tuple[str, str]:
    """Keep role/schema/skill prefixes independent of task values."""
    fixed, marker, context = prompt.partition(RUNTIME_CONTEXT_MARKER)
    return fixed.strip(), context.strip() if marker else ""


def _resolve_prompt_path(
    prompt_name: str,
) -> Path:
    """Resolve a prompt name inside PROMPT_ROOT without allowing escape."""

    normalized_name = prompt_name.strip().replace(
        "\\",
        "/",
    )

    if not normalized_name:
        raise ValueError(
            "prompt_name不能为空。"
        )

    parts = normalized_name.split("/")
    if any(
        not part
        or part in {".", ".."}
        or not PROMPT_NAME_PART_PATTERN.fullmatch(part)
        for part in parts
    ):
        raise ValueError(
            "prompt_name只能包含安全的提示词名称和子目录。"
        )

    prompt_path = PROMPT_ROOT.joinpath(
        *parts
    ).with_suffix(".md")
    resolved_root = PROMPT_ROOT.resolve()
    resolved_path = prompt_path.resolve()

    if resolved_path.parent != resolved_root and resolved_root not in (
        resolved_path.parents
    ):
        raise ValueError(
            "prompt_name不能逃出prompts目录。"
        )

    return resolved_path


@lru_cache(
    maxsize=None,
)
def load_prompt(
    prompt_name: str,
) -> str:
    """读取prompts目录中的Markdown提示词。"""

    prompt_path = _resolve_prompt_path(
        prompt_name
    )

    try:
        prompt_text = _read_composed_prompt(prompt_name)

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
