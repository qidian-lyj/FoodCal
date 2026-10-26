"""营养数据库加载与份量、卡路里估算。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "calories.csv"


@dataclass(frozen=True)
class FoodInfo:
    """一种食物的营养信息。"""

    name_en: str
    name_zh: str
    kcal_per_100g: float      # 每 100 克的热量（千卡）
    typical_serving_g: float  # 常见一次食用量（克）
    category: str


# 数据库未收录时使用的兜底值：按 150 千卡/100克、150 克粗略估计
UNKNOWN = FoodInfo("unknown", "未知食物", 150.0, 150.0, "其他")


def estimate_grams(
    area_ratio: float,
    typical_g: float,
    ref_ratio: float = 0.25,
    lo: float = 0.5,
    hi: float = 2.5,
) -> float:
    """按检测框占整张图的比例粗估份量（克）。

    约定：食物框占画面 25%（ref_ratio）时视为一份“标准份量”。
    面积比换算成线性尺寸比取平方根，并限制在 [lo, hi] 倍之间。
    单目照片无法真正测距测体积，因此这只是启发式估计，网页端提供滑块手动修正。
    """
    ratio = area_ratio / ref_ratio if ref_ratio > 0 else 1.0
    factor = min(hi, max(lo, ratio**0.5))
    return typical_g * factor


def kcal_of(grams: float, kcal_per_100g: float) -> float:
    """由份量与每 100 克热量计算总热量。"""
    return grams / 100.0 * kcal_per_100g


class NutritionDB:
    """食物 → 营养信息 查询表，同时支持英文/中文类名。"""

    def __init__(self, rows: list[FoodInfo]):
        self._by_key: dict[str, FoodInfo] = {}
        for row in rows:
            self._by_key[row.name_en.lower()] = row
            self._by_key[row.name_zh] = row

    @classmethod
    def load(cls, path: Path = DB_PATH) -> "NutritionDB":
        df = pd.read_csv(path, encoding="utf-8-sig")
        rows = [
            FoodInfo(
                name_en=str(r.name_en).strip(),
                name_zh=str(r.name_zh).strip(),
                kcal_per_100g=float(r.kcal_per_100g),
                typical_serving_g=float(r.typical_serving_g),
                category=str(r.category).strip(),
            )
            for r in df.itertuples()
        ]
        return cls(rows)

    def lookup(self, name: str) -> FoodInfo:
        """按类名查询；未收录时返回 UNKNOWN（按 150 千卡/100克 计）。"""
        key = name.strip().lower()
        if key in self._by_key:
            return self._by_key[key]
        # 模糊匹配：类名包含数据库词条，或反之（如 "club_sandwich" 命中 "sandwich"）。
        # 单字词条不参与模糊匹配，避免「凤梨」误命中「梨」这类歧义。
        for k, info in self._by_key.items():
            if len(k) >= 2 and (k in key or key in k):
                return info
        return UNKNOWN

    def zh_names(self) -> list[str]:
        """全部中文名（不含兜底条目），用作 GLM 识别时的取名提示。

        往 calories.csv 里加新条目，GLM 的取名提示会自动跟着变，无需改代码。
        """
        return sorted({
            info.name_zh for info in self._by_key.values() if info.name_en != "unknown"
        })

    def __len__(self) -> int:
        return len({id(v) for v in self._by_key.values()})
