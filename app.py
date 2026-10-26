"""FoodCal · 拍照识别食物并估算卡路里（Streamlit 网页界面）。

识别引擎（侧边栏可切换）：
    🔍 本地 YOLO —— 离线免费，类别取决于加载的权重
    ☁️ GLM 大模型 —— 视觉大模型整图识别，几乎能认任何菜（需免费 API Key）
    🤝 混合 —— YOLO 定位 + 大模型认菜名，推荐

运行方式：
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from detector import FoodDetector  # noqa: E402
from nutrition import NutritionDB, estimate_grams, kcal_of  # noqa: E402
from vision import VisionRecognizer  # noqa: E402
from visualize import draw_detections  # noqa: E402

ENGINE_YOLO = "🔍 本地 YOLO（离线）"
ENGINE_GLM = "☁️ GLM 大模型（云端）"
ENGINE_HYBRID = "🤝 混合（YOLO 定位 + GLM 认菜）"

st.set_page_config(page_title="FoodCal · 拍照算卡路里", page_icon="🍕", layout="wide")

st.title("🍕 FoodCal · 拍照识别食物并估算卡路里")
st.caption("YOLO 目标检测 + 视觉大模型 + 营养数据库；份量为粗略估计，可手动微调。结果仅供健康参考。")

# ---------------- 侧边栏：识别设置 ----------------
with st.sidebar:
    st.header("⚙️ 识别设置")
    engine = st.radio(
        "识别引擎",
        [ENGINE_YOLO, ENGINE_GLM, ENGINE_HYBRID],
        index=0,
        captions=["离线免费，只能识别训练过的类别", "几乎能认任何菜，需免费 API Key", "YOLO 找位置、大模型认菜名，推荐"],
        label_visibility="collapsed",
    )
    use_glm = engine != ENGINE_YOLO
    use_yolo = engine != ENGINE_GLM

    if use_yolo:
        st.divider()
        st.markdown("**YOLO 设置**")
        local_models = sorted((PROJECT_ROOT / "models").glob("*.pt"))
        model_options = ["yolov8n.pt（COCO 预训练）"] + [f"models/{p.name}" for p in local_models]
        model_choice = st.selectbox("模型权重", model_options)
        weights = "yolov8n.pt" if model_choice.startswith("yolov8n") else str(PROJECT_ROOT / model_choice)
        conf = st.slider("置信度阈值", 0.05, 0.90, 0.25, 0.05)
        food_only = st.checkbox("只保留食物类别", value=True, help="COCO 预训练模型会过滤非食物目标；自定义模型不受影响")

    api_key, vision_model = "", "glm-4.6v-flash"
    if use_glm:
        st.divider()
        st.markdown("**云端大模型设置**")
        api_key = st.text_input(
            "智谱 API Key",
            value=os.getenv("ZHIPU_API_KEY", ""),
            type="password",
            help="免费申请：https://open.bigmodel.cn ，GLM-4.6V-Flash 模型完全免费",
        )
        vision_model = st.text_input("视觉模型", value=os.getenv("ZHIPU_VISION_MODEL", "glm-4.6v-flash"))


@st.cache_resource(show_spinner="加载 YOLO 模型…")
def get_detector(weights: str, conf: float, food_only: bool) -> FoodDetector:
    return FoodDetector(weights=weights, conf=conf, food_only=food_only)


@st.cache_resource(show_spinner="加载营养数据库…")
def get_nutrition() -> NutritionDB:
    return NutritionDB.load()


detector = get_detector(weights, conf, food_only) if use_yolo else None
nutrition = get_nutrition()

if use_glm and not api_key:
    st.warning(
        "☁️ 你选择了使用 GLM 大模型，但还没有填 API Key。"
        "请在侧边栏输入，或到 [open.bigmodel.cn](https://open.bigmodel.cn) 免费申请"
        "（GLM-4.6V-Flash 模型完全免费），也可以先切换回本地 YOLO 引擎。"
    )
    st.stop()


def _crop(image, box):
    x1, y1, x2, y2 = box
    return image[max(0, y1):y2, max(0, x1):x2]


# ---------------- 图片输入 ----------------
tab_photo, tab_upload = st.tabs(["📷 拍照", "🖼️ 上传图片"])
with tab_photo:
    shot = st.camera_input("对准食物拍一张", key="camera")
with tab_upload:
    uploaded = st.file_uploader("或选择一张食物照片", type=["jpg", "jpeg", "png", "webp"])

img_file = shot or uploaded
if img_file is None:
    st.info("👆 拍照或上传一张食物图片开始识别")
    st.stop()

raw = img_file.getvalue()
photo_key = hashlib.md5(raw).hexdigest()[:8]
image_bgr = cv2.cvtColor(np.array(Image.open(BytesIO(raw)).convert("RGB")), cv2.COLOR_RGB2BGR)
img_area = float(image_bgr.shape[0] * image_bgr.shape[1])

# ---------------- 第 1 步：YOLO 检测（结果按照片缓存，避免滑动条时重复计算） ----------------
dets = []
if use_yolo:
    det_key = f"dets_{photo_key}_{Path(weights).name}"
    if det_key not in st.session_state:
        with st.spinner("YOLO 检测中…"):
            st.session_state[det_key] = detector.detect(image_bgr)
    dets = st.session_state[det_key]

# ---------------- 第 2 步：GLM 识别（调用结果按照片缓存，避免重复扣费） ----------------
def run_glm_recognition():
    return vision.recognize(image_bgr, nutrition.zh_names())


def run_hybrid_recognition():
    """对每个 YOLO 框裁剪后让 GLM 认菜；YOLO 没检出时退化为整图识别。"""
    results = []
    for det in dets:
        glm = None
        try:
            glm = vision.name_dish(_crop(image_bgr, det.box), nutrition.zh_names())
        except RuntimeError:
            pass  # 单框调用失败不影响其他框，退化为纯 YOLO 结果
        results.append({
            "name_zh": glm["name_zh"] if glm else det.name,
            "grams": glm["grams"] if glm else None,
            "conf": det.conf,
            "box": det.box,
            "source": "YOLO+GLM" if glm else "YOLO",
        })
    return results


recogs: list[dict] = []  # 统一格式：{name_zh, grams, conf, box, source}
if engine == ENGINE_YOLO:
    for det in dets:
        info = nutrition.lookup(det.name)
        zh = det.name if info.name_en == "unknown" else info.name_zh
        recogs.append({"name_zh": zh, "grams": None, "conf": det.conf, "box": det.box, "source": "YOLO"})
elif engine == ENGINE_GLM:
    recog_key = f"recog_{photo_key}_{vision_model}_{hashlib.md5(api_key.encode()).hexdigest()[:6]}"
    if recog_key not in st.session_state:
        with st.spinner("GLM 大模型识别中（约几秒）…"):
            st.session_state[recog_key] = run_glm_recognition()
    for r in st.session_state[recog_key]:
        recogs.append({"name_zh": r["name_zh"], "grams": r["grams"], "conf": r["conf"], "box": None, "source": "GLM"})
else:  # 混合
    recog_key = f"recog_{photo_key}_hybrid_{Path(weights).name}_{vision_model}_{hashlib.md5(api_key.encode()).hexdigest()[:6]}"
    if recog_key not in st.session_state:
        with st.spinner("YOLO 定位 + GLM 认菜中…"):
            st.session_state[recog_key] = run_hybrid_recognition()
    recogs = st.session_state[recog_key]
    if not dets and not recogs:
        # YOLO 未检出，退化为整图 GLM 识别
        if recog_key + "_full" not in st.session_state:
            with st.spinner("YOLO 未检出食物，改用 GLM 整图识别…"):
                st.session_state[recog_key + "_full"] = run_glm_recognition()
        for r in st.session_state[recog_key + "_full"]:
            recogs.append({"name_zh": r["name_zh"], "grams": r["grams"], "conf": r["conf"], "box": None, "source": "GLM"})

if not recogs:
    st.warning(
        "未检测到食物 🤔 试试：光线充足、正对食物拍摄，调低置信度阈值，"
        "或切换到 ☁️ GLM 大模型引擎。"
    )
    st.stop()

# ---------------- 第 3 步：份量滑块 + 热量计算 ----------------
st.subheader(f"1️⃣ 识别出 {len(recogs)} 项食物 · 请确认份量")
slider_cols = st.columns(min(3, len(recogs)))

items: list[dict] = []
boxes: list[tuple[int, int, int, int]] = []
labels: list[str] = []

for i, r in enumerate(recogs):
    info = nutrition.lookup(r["name_zh"])
    zh = r["name_zh"] if info.name_en == "unknown" else info.name_zh
    known = info.name_en != "unknown"

    if r["grams"] is not None:
        default_g = max(10, min(int(r["grams"]), 1500))
    else:
        area_ratio = (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) / img_area if r["box"] else 0.25
        default_g = max(10, min(int(round(estimate_grams(area_ratio, info.typical_serving_g))), 1500))

    grams = slider_cols[i % 3].slider(
        f"{zh}（{info.kcal_per_100g:.0f} 千卡/100克）",
        min_value=10,
        max_value=1500,
        value=default_g,
        step=10,
        key=f"grams_{photo_key}_{engine[:3]}_{i}",
    )
    kcal = round(kcal_of(grams, info.kcal_per_100g))
    if r["box"]:
        boxes.append(r["box"])
        labels.append(f"{zh} {r['conf']:.0%} · 约{kcal}千卡")
    items.append({
        "食物": zh,
        "置信度": f"{r['conf']:.0%}",
        "份量(克)": grams,
        "热量(千卡)": kcal,
        "营养库收录": "是" if known else "否（按150计）",
        "来源": r["source"],
    })

# ---------------- 第 4 步：标注图 + 明细 ----------------
st.subheader("2️⃣ 识别结果")
annotated = draw_detections(image_bgr, boxes, labels) if boxes else image_bgr

col_img, col_table = st.columns([3, 2])
with col_img:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.download_button(
        "💾 下载标注图",
        data=cv2.imencode(".jpg", annotated)[1].tobytes(),
        file_name=f"foodcal_{photo_key}.jpg",
        mime="image/jpeg",
    )
with col_table:
    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

total = sum(int(it["热量(千卡)"]) for it in items)
st.metric(label="🍱 本餐总计", value=f"{total} 千卡")

# ---------------- 第 5 步：记录今日饮食 ----------------
st.subheader("3️⃣ 记录")
if st.button("➕ 加入今日记录", type="primary"):
    log = st.session_state.setdefault("log", [])
    content = "、".join(f"{it['食物']}{it['份量(克)']}g" for it in items)
    log.append({"时间": datetime.now().strftime("%H:%M"), "内容": content, "热量(千卡)": total})

log = st.session_state.get("log", [])
if log:
    col_list, col_sum = st.columns([3, 1])
    with col_list:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
    with col_sum:
        st.metric("今日合计", f"{sum(int(r['热量(千卡)']) for r in log)} 千卡")
        if st.button("🗑️ 清空记录"):
            st.session_state["log"] = []
            st.rerun()
