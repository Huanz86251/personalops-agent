"""Generate the synthetic README animation; it contains no private run data."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "personalops-demo.gif"
WIDTH, HEIGHT = 1120, 620
BG = "#07111f"
PANEL = "#101d30"
MUTED = "#91a4bf"
WHITE = "#f5f8ff"
BLUE = "#62a8ff"
GREEN = "#53d69d"
ORANGE = "#ffbc66"


def font(size: int, *, bold: bool = False):
    names = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


TITLE = font(30, bold=True)
BODY = font(23)
SMALL = font(19)


def box(draw, xy, title, body, color=BLUE):
    draw.rounded_rectangle(xy, radius=16, fill=PANEL, outline=color, width=2)
    x1, y1, x2, _ = xy
    draw.text((x1 + 20, y1 + 16), title, fill=color, font=BODY)
    y = y1 + 58
    for line in body:
        draw.text((x1 + 20, y), line, fill=WHITE, font=SMALL)
        y += 29


def frame(stage: int):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((42, 28), "PersonalOps Agent · Evidence-first workflow", fill=WHITE, font=TITLE)
    draw.text((42, 72), "合成演示 · Synthetic demo · No private task data", fill=MUTED, font=SMALL)
    box(draw, (42, 116, 1078, 225), "飞书请求 / Feishu request", [
        "“把项目笔记中标星且尚未归档的条目移到归档区。”",
        '"Archive starred project notes that are not already archived."',
    ], ORANGE)
    if stage >= 1:
        box(draw, (42, 258, 340, 438), "1  Scope Contract", [
            "目标：项目笔记条目", "条件：标星 ∩ 未归档", "效果：MUTATION",
        ])
    if stage >= 2:
        box(draw, (411, 258, 709, 438), "2  Worker + Action Card", [
            "读取候选集合", "绑定写入对象", "执行修改并回读",
        ], GREEN)
        draw.line((340, 348, 411, 348), fill=MUTED, width=3)
    if stage >= 3:
        box(draw, (780, 258, 1078, 438), "3  Final Review", [
            "逐条核对条件", "核验写后回执", "COMPLETED",
        ], ORANGE)
        draw.line((709, 348, 780, 348), fill=MUTED, width=3)
    if stage >= 4:
        draw.rounded_rectangle((42, 480, 1078, 570), radius=14, fill="#0d332a", outline=GREEN, width=2)
        draw.text((66, 500), "✓ 目标范围、工具证据与最终结果贯通；失败会返回原 Worker 返修。", fill=GREEN, font=BODY)
        draw.text((66, 537), "Scope, evidence and verification remain traceable across workers.", fill=WHITE, font=SMALL)
    return image


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(i) for i in range(5)] + [frame(4)]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=[700, 850, 850, 850, 1800, 600],
        loop=0,
        optimize=True,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
