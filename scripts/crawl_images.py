"""从百度/必应图片搜索爬取 20 种食堂菜的训练图片。

爬到的图片按 菜名/photos/菜名/ 结构存放，可直接接 build_dataset.py --auto-label
做自动标注。GLM 找不到目标菜的图片会被自动跳过，相当于多了一道过滤。

用法：
    python scripts/crawl_images.py --per-dish 60          # 每道菜爬 60 张（建议量，标注后会过滤一部分）
    python scripts/crawl_images.py --per-dish 20 --dishes 番茄炒蛋 饺子   # 只爬指定菜

注意：
    - 网图仅用于本地训练研究，不要把爬到的图片提交到 GitHub（photos/ 已在 .gitignore）
    - 有条件后仍建议补实拍：网图与手机实拍存在域差距，实拍越多线上效果越好
"""
import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image  # noqa: E402

from nutrition import NutritionDB  # noqa: E402

# 部分菜名直接搜效果一般，用更具体的搜索词
QUERY_OVERRIDES = {
    "清炒时蔬": "清炒时蔬 炒青菜",
    "蒸蛋羹": "蒸蛋羹 鸡蛋羹",
    "米饭": "米饭 一碗米饭",
    "包子": "包子 肉包 菜包",
    "炒面": "炒面 家常炒面",
}

MIN_SIZE = 200  # 短边小于该值的图片直接丢弃


def crawl_one(dish: str, query: str, dest: Path, per_dish: int) -> int:
    """爬一道菜，返回最终保留的图片数。"""
    from icrawler.builtin import BaiduImageCrawler, BingImageCrawler

    dest.mkdir(parents=True, exist_ok=True)
    for crawler_cls, name in ((BaiduImageCrawler, "baidu"), (BingImageCrawler, "bing")):
        try:
            crawler = crawler_cls(
                storage={"root_dir": str(dest)},
                downloader_threads=8,
                log_level=40,  # ERROR only
            )
            crawler.crawl(keyword=query, max_num=per_dish, min_size=(MIN_SIZE, MIN_SIZE))
        except Exception as exc:  # 单引擎失败不影响另一个
            print(f"  [{name}] 爬取异常（继续）: {exc}")

    # 去重 + 过滤 + 统一为 jpg
    seen, kept, removed = set(), 0, 0
    for p in sorted(dest.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            p.unlink(missing_ok=True)
            continue
        try:
            data = p.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            if md5 in seen:
                p.unlink()
                removed += 1
                continue
            with Image.open(p) as im:
                im = im.convert("RGB")
                if min(im.size) < MIN_SIZE:
                    p.unlink()
                    removed += 1
                    continue
                seen.add(md5)
                new_path = dest / f"{kept + 1:03d}.jpg"
                if p != new_path:
                    im.save(dest / f"__tmp_{kept:03d}.jpg", quality=90)
                    p.unlink()
                    (dest / f"__tmp_{kept:03d}.jpg").rename(new_path)
                kept += 1
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取食堂菜训练图片")
    parser.add_argument("--per-dish", type=int, default=60, help="每道菜目标张数（去重前）")
    parser.add_argument("--src", default="photos", help="输出根目录")
    parser.add_argument("--dishes", nargs="*", default=None, help="只爬指定菜（默认全部 20 种）")
    args = parser.parse_args()

    db = NutritionDB.load()
    dishes = args.dishes or db.zh_names()
    src = Path(args.src)

    print(f"共 {len(dishes)} 道菜，每道目标 {args.per_dish} 张\n")
    stats = []
    for i, dish in enumerate(dishes, 1):
        query = QUERY_OVERRIDES.get(dish, dish)
        print(f"[{i}/{len(dishes)}] {dish} ← 搜索词「{query}」")
        kept = crawl_one(dish, query, src / dish, args.per_dish)
        stats.append((dish, kept))
        print(f"  保留 {kept} 张")

    print("\n===== 汇总 =====")
    for dish, kept in stats:
        flag = "" if kept >= 20 else "  ← 偏少，建议补爬或换词"
        print(f"  {dish}: {kept} 张{flag}")
    total = sum(k for _, k in stats)
    print(f"\n总计 {total} 张，已保存到 {src.resolve()}")
    print("下一步：python scripts/build_dataset.py --src photos --out data/datasets/food --auto-label")


if __name__ == "__main__":
    main()
