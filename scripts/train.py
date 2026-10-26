"""在自定义食物数据集上微调 YOLOv8。

用法：
    python scripts/train.py --data data/datasets/food/data.yaml --epochs 60

训练完成后，把 runs/detect/<name>/weights/best.pt 复制到 models/ 目录，
刷新网页即可在侧边栏选择使用。
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练自定义食物检测模型")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8n.pt", help="用于微调的预训练权重")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="food_yolov8n", help="本次训练的名称")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"找不到数据集配置: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    best = Path("runs") / "detect" / args.name / "weights" / "best.pt"
    print("\n训练完成！最优权重:", best.resolve() if best.exists() else "(见上方 ultralytics 输出)")
    print("把它复制到 models/ 目录（如 models/food.pt），刷新网页即可在侧边栏选择。")


if __name__ == "__main__":
    main()
