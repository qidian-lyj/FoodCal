"""YOLO 食物检测的轻量封装。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """一次检测结果。"""

    name: str
    conf: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)，像素坐标
    area_ratio: float               # 检测框面积 / 整张图面积，用于粗估份量


class FoodDetector:
    """加载 YOLO 模型并检测图片中的食物目标。"""

    # COCO 预训练模型中与食物相关的类别（其余类别默认过滤掉）
    COCO_FOOD_CLASSES = frozenset({
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake",
    })

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        food_only: bool = True,
    ):
        from ultralytics import YOLO  # 延迟导入，加快页面首屏速度

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

        names = set(self.model.names.values())
        is_coco = "dining table" in names  # COCO 特有类别，用来判断是否为预训练模型
        self.allowed = self.COCO_FOOD_CLASSES if (food_only and is_coco) else None

    def detect(self, image_bgr) -> list[Detection]:
        """输入 BGR 图像（numpy 数组），返回按置信度降序排列的检测结果。"""
        h, w = image_bgr.shape[:2]
        detections: list[Detection] = []

        for result in self.model.predict(image_bgr, conf=self.conf, iou=self.iou, verbose=False):
            if result.boxes is None:
                continue
            for b in result.boxes:
                name = result.names[int(b.cls[0])]
                if self.allowed is not None and name not in self.allowed:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / float(w * h))
                detections.append(Detection(
                    name=name,
                    conf=float(b.conf[0]),
                    box=(x1, y1, x2, y2),
                    area_ratio=area_ratio,
                ))

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections
