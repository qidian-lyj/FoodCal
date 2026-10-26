"""命令行预测：识别一张图片中的食物并估算卡路里。

用法：
    python scripts/predict.py 某张图.jpg
    python scripts/predict.py 某张图.jpg --weights models/food.pt --conf 0.3
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from detector import FoodDetector  # noqa: E402
from nutrition import NutritionDB, estimate_grams, kcal_of  # noqa: E402
from visualize import draw_detections  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="识别图片中的食物并估算卡路里")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--weights", default="yolov8n.pt", help="模型权重")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"读取图片失败: {args.image}")

    detector = FoodDetector(weights=args.weights, conf=args.conf, food_only=True)
    db = NutritionDB.load()
    detections = detector.detect(image)

    if not detections:
        print("未检测到食物，可尝试调低 --conf（如 0.15）。")
        return

    rows, boxes, labels = [], [], []
    for det in detections:
        info = db.lookup(det.name)
        zh = det.name if info.name_en == "unknown" else info.name_zh
        grams = estimate_grams(det.area_ratio, info.typical_serving_g)
        kcal = round(kcal_of(grams, info.kcal_per_100g))
        rows.append({
            "食物": zh,
            "置信度": f"{det.conf:.0%}",
            "份量(克)": round(grams),
            "热量(千卡)": kcal,
        })
        boxes.append(det.box)
        labels.append(f"{zh} {det.conf:.0%} 约{kcal}千卡")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"合计约 {sum(r['热量(千卡)'] for r in rows)} 千卡")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predict_{Path(args.image).stem}.jpg"
    cv2.imwrite(str(out_path), draw_detections(image, boxes, labels))
    print(f"标注图已保存: {out_path}")


if __name__ == "__main__":
    main()
