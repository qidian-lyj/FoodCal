"""FoodCal 评测脚本：为技术报告提供量化指标。

子命令：
    detect     检测模型评测：mAP@50、mAP@50-95、各类别 AP、混淆矩阵
               传多个 --weights 可做消融对比（如 纯自动标注 vs 人工修正后）
    calories   热量估算评测：MAE / MAPE / ±20% 容差准确率 / 菜名匹配率
    engines    三引擎对比：识别准确率、单张延迟、API 调用次数

评测集格式（CSV，UTF-8，示例见 data/eval/*.example.csv）：
    calories_gt.csv  列: image,dish_zh,true_grams[,true_kcal]
    engines_gt.csv   列: image,dish_zh

用法示例：
    python scripts/evaluate.py detect --weights models/food.pt --data data/datasets/food/data.yaml
    python scripts/evaluate.py detect --weights runs/a/best.pt runs/b/best.pt --data ... --compare
    python scripts/evaluate.py calories --set data/eval/calories_gt.csv --engine hybrid
    python scripts/evaluate.py engines --set data/eval/engines_gt.csv --engines yolo,glm,hybrid
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from detector import FoodDetector  # noqa: E402
from nutrition import NutritionDB, estimate_grams, kcal_of  # noqa: E402
from vision import VisionRecognizer  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "evaluation"


def load_db() -> NutritionDB:
    return NutritionDB.load()


def resolve_zh(db: NutritionDB, raw_name: str) -> str:
    info = db.lookup(raw_name)
    return raw_name if info.name_en == "unknown" else info.name_zh


# ---------------- detect：检测指标 ----------------
def eval_one_detector(weights: str, args) -> tuple[dict, pd.DataFrame, Path | None]:
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, verbose=False)
    box = metrics.box
    names = model.names

    rows = []
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(int(ci))
        rows.append({
            "class_id": int(ci),
            "class": names[int(ci)],
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "AP@50": round(float(ap50), 4),
            "AP@50-95": round(float(ap), 4),
        })
    per_class = pd.DataFrame(rows).sort_values("AP@50", ascending=False)
    summary = {
        "weights": weights,
        "mAP@50": round(float(box.map50), 4),
        "mAP@50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }
    save_dir = Path(getattr(metrics, "save_dir", "."))
    cm = save_dir / "confusion_matrix.png"
    return summary, per_class, cm if cm.exists() else None


def cmd_detect(args) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, per_class_files = [], []
    for i, weights in enumerate(args.weights):
        tag = Path(weights).parent.parent.name if len(args.weights) > 1 else f"w{i}"
        print(f"\n===== 评测 {weights} =====")
        summary, per_class, cm_path = eval_one_detector(weights, args)
        summaries.append(summary)
        pc_file = out_dir / f"detect_{tag}_per_class.csv"
        per_class.to_csv(pc_file, index=False, encoding="utf-8-sig")
        per_class_files.append(pc_file)
        if cm_path:
            dst = out_dir / f"confusion_matrix_{tag}.png"
            shutil.copy2(cm_path, dst)
            print(f"混淆矩阵已保存: {dst}")
        print(per_class.to_string(index=False))

    cmp_df = pd.DataFrame(summaries)
    print("\n===== 汇总 =====")
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(out_dir / "detect_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "detect_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存到 {out_dir}（可直接引用进报告）")


# ---------------- 公共：单张图的 top-1 预测 ----------------
def predict_top1(engine: str, image, db: NutritionDB, detector, vision):
    """返回 (菜名列表 zh, top1_est_grams 或 None, api_calls)。"""
    if engine == "yolo":
        dets = detector.detect(image)
        names = [resolve_zh(db, d.name) for d in dets]
        grams = None
        if dets:
            h, w = image.shape[:2]
            d = dets[0]
            info = db.lookup(d.name)
            area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / float(w * h)
            grams = int(estimate_grams(area, info.typical_serving_g))
        return names, grams, 0

    if engine == "glm":
        items = vision.recognize(image, db.zh_names())
        names = [it["name_zh"] for it in items]
        grams = items[0]["grams"] if items else None
        return names, grams, 1

    # hybrid：YOLO 定位 + 每框 GLM 认菜；YOLO 没检出时退化为整图 GLM
    dets = detector.detect(image)
    calls = 0
    names, grams = [], None
    if dets:
        h, w = image.shape[:2]
        d = dets[0]
        try:
            glm = vision.name_dish(image[max(0, d.box[1]):d.box[3], max(0, d.box[0]):d.box[2]], db.zh_names())
            calls += 1
        except RuntimeError:
            glm = None
        names = [glm["name_zh"] if glm else d.name]
        info = db.lookup(names[0])
        area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / float(w * h)
        grams = glm["grams"] if glm else int(estimate_grams(area, info.typical_serving_g))
    else:
        items = vision.recognize(image, db.zh_names())
        calls += 1
        names = [it["name_zh"] for it in items]
        grams = items[0]["grams"] if items else None
    return names, grams, calls


def build_components(engine: str, weights: str):
    db = load_db()
    detector = FoodDetector(weights=weights, conf=0.25, food_only=True) if engine in ("yolo", "hybrid") else None
    vision = None
    if engine in ("glm", "hybrid"):
        import os

        api_key = os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            raise SystemExit(f"{engine} 引擎需要 API Key：请在 .env 配置 ZHIPU_API_KEY（https://open.bigmodel.cn 免费申请）")
        vision = VisionRecognizer(api_key=api_key)
    return db, detector, vision


# ---------------- calories：热量估算误差 ----------------
def cmd_calories(args) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    db, detector, vision = build_components(args.engine, args.weights)
    gt = pd.read_csv(args.set)

    rows = []
    for _, r in gt.iterrows():
        image = cv2.imread(str(r["image"]))
        if image is None:
            print(f"[跳过] 读不到图片: {r['image']}")
            continue
        names, est_grams, _ = predict_top1(args.engine, image, db, detector, vision)
        true_info = db.lookup(str(r["dish_zh"]))
        true_kcal = float(r["true_kcal"]) if "true_kcal" in gt.columns and pd.notna(r.get("true_kcal")) \
            else kcal_of(float(r["true_grams"]), true_info.kcal_per_100g)

        top1 = names[0] if names else ""
        info = db.lookup(top1) if top1 else true_info
        if est_grams is None:
            est_grams = info.typical_serving_g
        est_kcal = kcal_of(est_grams, info.kcal_per_100g)
        err = est_kcal - true_kcal
        rows.append({
            "image": r["image"],
            "真实": str(r["dish_zh"]),
            "预测": top1 or "(未检出)",
            "名字命中": top1 == str(r["dish_zh"]),
            "真实千卡": round(true_kcal),
            "估算千卡": round(est_kcal),
            "绝对误差": round(abs(err)),
            "相对误差%": round(abs(err) / true_kcal * 100, 1) if true_kcal > 0 else None,
        })

    df = pd.DataFrame(rows)
    mape = df["相对误差%"].dropna().mean() if len(df) else float("nan")
    summary = {
        "engine": args.engine,
        "样本数": len(df),
        "菜名匹配率": round(df["名字命中"].mean(), 4) if len(df) else None,
        "MAE(千卡)": round(df["绝对误差"].mean(), 1) if len(df) else None,
        "MAPE(%)": round(mape, 1) if len(df) else None,
        "±20%容差准确率": round((df["相对误差%"] <= 20).mean(), 4) if len(df) else None,
    }
    print(df.to_string(index=False))
    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"calories_{args.engine}.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"calories_{args.engine}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存到 {out_dir}")


# ---------------- engines：三引擎对比 ----------------
def cmd_engines(args) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    engines = [e.strip() for e in args.engines.split(",")]
    gt = pd.read_csv(args.set)

    rows = []
    for engine in engines:
        db, detector, vision = build_components(engine, args.weights)
        latencies, top1_hits, any_hits, api_calls = [], [], [], 0
        for _, r in gt.iterrows():
            image = cv2.imread(str(r["image"]))
            if image is None:
                continue
            t0 = time.perf_counter()
            names, _, calls = predict_top1(engine, image, db, detector, vision)
            latencies.append(time.perf_counter() - t0)
            api_calls += calls
            truth = str(r["dish_zh"])
            top1_hits.append(bool(names) and names[0] == truth)
            any_hits.append(truth in names)

        n = max(1, len(latencies))
        rows.append({
            "engine": engine,
            "top1准确率": round(sum(top1_hits) / n, 4),
            "命中准确率(含非首位)": round(sum(any_hits) / n, 4),
            "平均延迟(秒)": round(sum(latencies) / n, 2),
            "API调用总数": api_calls,
            "每张图API调用": round(api_calls / n, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "engines_comparison.csv", index=False, encoding="utf-8-sig")

    # 报告用图：准确率与延迟对比（标签用英文避免 matplotlib 中文字体问题）
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(df["engine"], df["top1准确率"], color="#4a90d9")
        ax1.set_title("Top-1 Accuracy")
        ax1.set_ylim(0, 1)
        ax2.bar(df["engine"], df["平均延迟(秒)"], color="#e07b39")
        ax2.set_title("Avg Latency (s/image)")
        fig.tight_layout()
        fig.savefig(out_dir / "engines_comparison.png", dpi=150)
        print(f"对比图已保存: {out_dir / 'engines_comparison.png'}")
    except Exception as exc:  # 图表失败不影响评测结果
        print(f"(图表生成失败，不影响数据: {exc})")
    print(f"结果已保存到 {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FoodCal 评测", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("detect", help="检测模型评测（mAP / 混淆矩阵 / 多权重消融）")
    p1.add_argument("--weights", nargs="+", required=True, help="一个或多个权重（多个 = 消融对比）")
    p1.add_argument("--data", required=True, help="data.yaml 路径")
    p1.add_argument("--split", default="val")
    p1.add_argument("--imgsz", type=int, default=640)
    p1.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p1.set_defaults(func=cmd_detect)

    p2 = sub.add_parser("calories", help="热量估算误差（MAE / MAPE）")
    p2.add_argument("--set", required=True, help="评测集 CSV（列: image,dish_zh,true_grams[,true_kcal]）")
    p2.add_argument("--engine", default="yolo", choices=["yolo", "glm", "hybrid"])
    p2.add_argument("--weights", default="yolov8n.pt")
    p2.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p2.set_defaults(func=cmd_calories)

    p3 = sub.add_parser("engines", help="三引擎对比（准确率 / 延迟 / API 调用）")
    p3.add_argument("--set", required=True, help="评测集 CSV（列: image,dish_zh）")
    p3.add_argument("--engines", default="yolo,glm,hybrid")
    p3.add_argument("--weights", default="yolov8n.pt")
    p3.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p3.set_defaults(func=cmd_engines)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
