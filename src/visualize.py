"""用 PIL 在检测框上绘制中文标签（OpenCV 的 putText 不支持中文）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 常见中文字体候选（Windows / Linux 各留了几个）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

BOX_COLORS = [
    (66, 186, 255), (92, 219, 109), (64, 156, 255),
    (255, 157, 41), (255, 102, 178), (108, 99, 255),
]

_font_cache: dict = {}


def _load_font(size: int = 22):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, color) -> None:
    """在 (x, y) 上方画一个带背景色的标签，避免超出图片顶部。"""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    x = max(0, x)
    y = max(0, y - h - 6)
    draw.rectangle([x, y, x + w + 8, y + h + 6], fill=color)
    draw.text((x + 4, y + 2), text, fill=(20, 20, 20), font=font)


def draw_detections(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
) -> np.ndarray:
    """在图像上绘制检测框与中文标签，返回 BGR 数组。boxes 与 labels 一一对应。"""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_font(max(18, pil.width // 40))

    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if label:
            _draw_label(draw, x1, y1, label, font, color)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
