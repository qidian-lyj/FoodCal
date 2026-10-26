"""云端视觉大模型识别（智谱 GLM 视觉系列）——本地 YOLO 的兜底与增强方案。

- recognize(): 识别整张图里的所有食物（GLM 引擎 / 混合引擎兜底用）
- name_dish(): 判断一块裁剪区域是什么菜（混合引擎用）
- find_bbox(): 给定菜名找边界框（数据集自动预标注用）

默认使用完全免费的 glm-4.6v-flash 模型。
API Key 在 https://open.bigmodel.cn 免费申请，放到项目根目录 .env 的 ZHIPU_API_KEY。
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import requests

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.6v-flash"

_MAX_SIDE = 1024  # 整图识别前把最长边压到该值，省流量且更快


def _hints_text(dish_hints: list[str] | None) -> str:
    if not dish_hints:
        return ""
    return (
        "\n如果是食堂菜品，优先从这个清单里选名称："
        + "、".join(dish_hints)
        + "。清单之外的菜也照常识别命名。"
    )


RECOGNIZE_PROMPT = (
    "你是食堂菜品识别助手。请识别这张图片里的所有食物和菜品。\n{hints}\n"
    "严格按照下面的 JSON 格式输出，不要输出任何其他内容：\n"
    '{{"items": [{{"名称": "菜品名", "份量克": 200, "置信度": 0.9}}]}}\n'
    "要求：\n"
    "1. 份量克 是你估计的这份菜的克数（整数）\n"
    "2. 置信度 是 0 到 1 的小数\n"
    "3. 最多输出 {max_items} 项，按置信度从高到低排列\n"
    "4. 餐盘、桌子、人等非食物不要输出"
)

NAME_PROMPT = (
    "这张图片里是一道菜（可能装在餐盘或碗里）。\n{hints}\n"
    "请判断这道菜的名称。严格按照下面的 JSON 格式输出，不要输出任何其他内容：\n"
    '{{"名称": "菜品名", "份量克": 200, "置信度": 0.9}}'
)

BBOX_PROMPT = (
    "这张图片里是否有「{dish}」这道菜？\n"
    "如果有，请给出它的边界框。严格按照下面的 JSON 格式输出，不要输出任何其他内容：\n"
    '{{"found": true, "bbox": [x, y, width, height], "confidence": 0.9}}\n'
    "bbox 基于图片左上角，单位是像素。如果没有找到，输出 {{\"found\": false}}"
)


def _prepare(image_bgr, max_side: int):
    """按最长边缩放图片，返回 (图片, 缩放比例)。"""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))))
    return image_bgr, scale


def _to_data_url(image_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("图片编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _clamp_float(value, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f > 1.0:  # 模型偶尔返回百分数 85 而不是 0.85
        f /= 100.0
    return min(1.0, max(0.0, f))


def _extract_json(text: str) -> dict | None:
    """从模型回复里提取第一个 JSON 对象，容忍代码块和多余说明文字。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # 逐字符扫描，收集所有大括号配平的顶层对象
    chunks, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                chunks.append(text[start : i + 1])
    for chunk in reversed(chunks):  # 通常 JSON 在回复末尾
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_item(obj: dict) -> dict | None:
    name = str(obj.get("名称") or obj.get("name") or "").strip()
    if not name:
        return None
    return {
        "name_zh": name,
        "grams": _clamp_int(obj.get("份量克") or obj.get("grams"), 10, 1500, 150),
        "conf": _clamp_float(obj.get("置信度") or obj.get("confidence"), 0.8),
    }


class VisionRecognizer:
    """调用智谱视觉大模型完成食物识别。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ):
        if not api_key:
            raise ValueError(
                "缺少智谱 API Key：请在项目根目录 .env 里配置 ZHIPU_API_KEY"
                "（https://open.bigmodel.cn 免费申请），或在网页侧边栏直接输入"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, image_bgr, prompt: str, max_side: int = _MAX_SIDE):
        """发送图片+提示词，返回 (回复文本, 图片缩放比例)。"""
        prepared, scale = _prepare(image_bgr, max_side)
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _to_data_url(prepared)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"网络请求失败：{exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"API 返回 {resp.status_code}：{resp.text[:200]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"API 响应格式异常：{resp.text[:200]}") from exc
        return content, scale

    def recognize(self, image_bgr, dish_hints: list[str] | None = None, max_items: int = 8) -> list[dict]:
        """识别整张图中的所有食物，返回 [{name_zh, grams, conf}]。"""
        prompt = RECOGNIZE_PROMPT.format(hints=_hints_text(dish_hints), max_items=max_items)
        content, _ = self._chat(image_bgr, prompt)
        obj = _extract_json(content)
        items = []
        if obj and isinstance(obj.get("items"), list):
            for raw in obj["items"][:max_items]:
                if isinstance(raw, dict):
                    item = _parse_item(raw)
                    if item:
                        items.append(item)
        return items

    def name_dish(self, image_bgr, dish_hints: list[str] | None = None) -> dict | None:
        """判断一块裁剪区域是什么菜，返回 {name_zh, grams, conf} 或 None。"""
        prompt = NAME_PROMPT.format(hints=_hints_text(dish_hints))
        content, _ = self._chat(image_bgr, prompt)
        obj = _extract_json(content)
        return _parse_item(obj) if obj else None

    def find_bbox(self, image_bgr, dish: str) -> tuple[tuple[int, int, int, int], float] | None:
        """给指定菜名找边界框，返回 ((x1, y1, x2, y2), conf) 或 None。坐标基于原图。"""
        prompt = BBOX_PROMPT.format(dish=dish)
        content, scale = self._chat(image_bgr, prompt, max_side=1280)
        obj = _extract_json(content)
        if not obj or not obj.get("found"):
            return None
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        h, w = image_bgr.shape[:2]
        try:
            x, y, bw, bh = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        x1, y1 = max(0.0, x / scale), max(0.0, y / scale)          # 缩放比例还原到原图坐标
        x2, y2 = min(float(w), (x + bw) / scale), min(float(h), (y + bh) / scale)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (int(x1), int(y1), int(x2), int(y2)), _clamp_float(obj.get("confidence"), 0.7)
