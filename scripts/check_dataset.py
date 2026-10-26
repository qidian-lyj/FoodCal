"""检查 YOLO 数据集是否完整：data.yaml、图片与标注数量是否匹配。

用法：
    python scripts/check_dataset.py data/datasets/food
"""
import argparse
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def inspect_split(root: Path, split: str):
    """返回 (图片数, 标注数, 缺标注的图片名列表)；目录不存在则返回 None。"""
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.exists():
        return None
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labels = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
    missing = [p.name for p in imgs if p.stem not in labels]
    return len(imgs), len(labels), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 YOLO 数据集完整性")
    parser.add_argument("root", help="数据集根目录（内含 data.yaml）")
    args = parser.parse_args()

    root = Path(args.root)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"找不到 {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    print(f"类别数: {len(names)} → {names}\n")

    ok = True
    for split in ("train", "val", "test"):
        result = inspect_split(root, split)
        if result is None:
            print(f"[跳过] {split}: 目录不存在")
            continue
        n_img, n_lbl, missing = result
        print(f"[{'OK' if not missing else '警告'}] {split}: 图片 {n_img} 张, 标注 {n_lbl} 个")
        if missing:
            ok = False
            for name in missing[:5]:
                print(f"    - 缺少标注: {name}")
            if len(missing) > 5:
                print(f"    - ... 共 {len(missing)} 张图片缺标注")

    status = "检查通过 ✅" if ok else "存在问题 ⚠️，请补齐缺失的标注文件"
    print(f"\n数据集{status}")


if __name__ == "__main__":
    main()
