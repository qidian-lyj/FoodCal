"""Unicode 安全的图像读写。

Windows 下 cv2.imread / cv2.imwrite 依赖本地代码页（GBK），
遇到中文路径会静默失败返回 None。改用 np.fromfile / tofile 中转即可支持任意路径。
项目内所有脚本读图请用本模块，不要直接用 cv2.imread。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path):
    """读取图片为 BGR 数组；失败返回 None（与 cv2.imread 行为一致）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, image, ext: str = ".jpg") -> bool:
    """写图片到任意路径，成功返回 True。"""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
