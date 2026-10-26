"""把按菜名分文件夹的照片整理成 YOLO 训练数据集，可用 GLM 大模型自动预标注。

照片目录结构（每个子文件夹 = 一道菜，文件夹名即类别名，中文即可）：
    photos/
    ├── 番茄炒蛋/
    │   ├── 001.jpg
    │   └── ...
    ├── 宫保鸡丁/
    │   └── ...
    └── 米饭/
        └── ...

用法：
    # 1) GLM 自动预标注 + 切分 + 生成 data.yaml（需要 .env 里的 ZHIPU_API_KEY）
    python scripts/build_dataset.py --src photos --out data/datasets/food --auto-label

    # 2) 图片旁边已放好同名 .txt 标注（人工核对过）时，只做切分，不调用大模型
    python scripts/build_dataset.py --src photos --out data/datasets/food

推荐流程：
    每道菜拍 20~50 张 → --auto-label 自动预标注 → 用 X-AnyLabeling 抽查修正
    （https://github.com/CVHub520/X-AnyLabeling，把修正后的 txt 放回图片旁）
    → 重新跑本脚本（不带 --auto-label）→ scripts/train.py 开始训练
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from vision import VisionRecognizer  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def yolo_line(cls_id: int, box: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    """把 (x1,y1,x2,y2) 像素框转成 YOLO 归一化格式的一行标签。"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 YOLO 食物数据集")
    parser.add_argument("--src", required=True, help="照片根目录（子文件夹=类别）")
    parser.add_argument("--out", default="data/datasets/food", help="输出数据集目录")
    parser.add_argument("--auto-label", action="store_true", help="用 GLM 大模型自动预标注")
    parser.add_argument("--api-key", default="", help="智谱 API Key（默认读 .env）")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(f"照片目录不存在: {src}\n结构要求: photos/番茄炒蛋/*.jpg, photos/宫保鸡丁/*.jpg ...")

    classes = sorted(d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise SystemExit(f"{src} 下没有子文件夹。每个子文件夹是一道菜，文件夹名即类别名。")

    print(f"类别 {len(classes)} 个：")
    for i, name in enumerate(classes):
        print(f"  {i}: {name}")

    recognizer = None
    if args.auto_label:
        api_key = args.api_key or os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            raise SystemExit(
                "--auto-label 需要 API Key：请在项目根目录 .env 配置 ZHIPU_API_KEY"
                "（https://open.bigmodel.cn 免费申请），或用 --api-key 传入"
            )
        recognizer = VisionRecognizer(api_key=api_key)
        print("\n已启用 GLM 自动预标注（免费模型，注意每张图会调用一次 API）")

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)  # 固定种子保证可复现
    total_train, total_val, skipped = 0, 0, 0

    for cls_id, cls in enumerate(classes):
        imgs = sorted(p for p in (src / cls).iterdir() if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * args.val_ratio)) if len(imgs) >= 2 else 0
        n_train = 0
        print(f"\n[{cls}] {len(imgs)} 张照片")

        for idx, img_path in enumerate(imgs):
            split = "val" if idx < n_val else "train"
            label_path = img_path.with_suffix(".txt")

            if label_path.exists():  # 已有人工标注，直接用
                label_text = label_path.read_text(encoding="utf-8").strip() + "\n"
            elif recognizer is not None:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  [跳过] {img_path.name}: 图片读取失败")
                    skipped += 1
                    continue
                result = recognizer.find_bbox(image, cls)
                if result is None:
                    print(f"  [跳过] {img_path.name}: 未找到「{cls}」")
                    skipped += 1
                    continue
                box, conf = result
                label_text = yolo_line(cls_id, box, image.shape[1], image.shape[0])
                print(f"  [自动] {split}/{img_path.name}: conf={conf:.2f}")
            else:
                print(f"  [跳过] {img_path.name}: 无同名 .txt 标注（可加 --auto-label）")
                skipped += 1
                continue

            shutil.copy2(img_path, out / split / "images" / img_path.name)
            (out / split / "labels" / f"{img_path.stem}.txt").write_text(label_text, encoding="utf-8")
            if split == "train":
                n_train += 1
            else:
                total_val += 1
        total_train += n_train
        print(f"  → train {n_train} 张, val {min(n_val, len(imgs))} 张")

    data_yaml = {
        "path": ".",  # 相对数据集根目录
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n完成 ✅ train {total_train} 张 / val {total_val} 张 / 跳过 {skipped} 张")
    print(f"数据集已写入: {out.resolve()}")
    print("下一步:")
    print(f"  1. python scripts/check_dataset.py {out}")
    print(f"  2. python scripts/train.py --data {out / 'data.yaml'} --epochs 80")
    print("  3. 训练完把 runs/detect/*/weights/best.pt 复制到 models/ 即可在网页使用")
    print("  4. 记得在 data/calories.csv 里核对新类别的热量条目")


if __name__ == "__main__":
    main()
