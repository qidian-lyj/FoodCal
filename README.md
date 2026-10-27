# 🍕 FoodCal — Photo-Based Food Calorie Estimation

[中文文档](README.zh-CN.md) | English

YOLOv8 object detection + a GLM-4.6V vision-language model + a nutrition database, wrapped in a Streamlit web app. **Snap a photo, get an instant calorie estimate.**

![YOLO demo](docs/demo_yolo_pizza.jpg)
*YOLO detection with calorie labels — 3 pizza slices, ≈1,231 kcal*

![Custom model demo](docs/demo_custom_model.jpg)
*Self-trained `food_v1` on an unseen validation image: Mapo Tofu, 96% confidence, ≈439 kcal*

## 📊 Results (v1 · web-crawled data · 20 canteen dishes)

Training setup: YOLOv8n @ 640 px, 60 epochs, **2.74 h on CPU** (i7-14650HX), 824 images auto-labeled by a vision-language model — **zero human annotation, zero GPU, zero cost**.

| Metric | Value |
|---|---|
| **mAP@50** | **0.863** |
| mAP@50-95 | 0.532 |
| Precision / Recall | 0.896 / 0.755 |
| Latency (CPU, yolov8n) | 35.6 ms/image |

9 of 20 classes reach AP@50 ≈ 0.99. Weakest: baozi (0.478), steamed rice (recall 0.25), dumplings & mantou (validation sets of 2). Full analysis and error attribution in [REPORT.md](REPORT.md).

## ✨ Features

- **Three recognition engines**, switchable in the sidebar:
  - 🔍 **Local YOLO** — offline and free; limited to trained classes
  - ☁️ **GLM-4.6V (cloud)** — recognizes virtually any dish via a vision LLM (free API key)
  - 🤝 **Hybrid** — YOLO locates dishes, GLM names them; best accuracy/cost trade-off
- **87-item nutrition database** — canteen dishes, desserts, fruits, foreign food; unknown classes fall back to 150 kcal/100 g
- **Portion estimation** — from detection-box area or the VLM's own estimate, refined with sliders
- **Daily tracking** — per-meal totals, day summary, annotated-image download
- **Semi-automatic dataset pipeline** — web crawl → VLM auto-annotation (with quality filter) → YOLO training, fully resumable

## 🚀 Quick Start

```powershell
git clone https://github.com/qidian-lyj/FoodCal.git
cd FoodCal
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`yolov8n.pt` (~6 MB) downloads automatically on first run. Open http://localhost:8501.

**Enable the GLM engine (optional, free):** create an API key at [open.bigmodel.cn](https://open.bigmodel.cn), copy `.env.example` to `.env`, and fill in `ZHIPU_API_KEY` — or paste the key in the sidebar. `glm-4.6v-flash` is completely free.

## 💻 CLI

```powershell
python scripts/predict.py image.jpg
python scripts/predict.py image.jpg --weights models/food.pt --conf 0.3
```

## 🏋️ Train Your Own Detector

```powershell
# 1. Collect photos (one sub-folder per dish) or crawl web images
python scripts/crawl_images.py --per-dish 60

# 2. VLM auto-annotation + train/val split + data.yaml (resumable)
python scripts/build_dataset.py --src photos --out data/datasets/food --auto-label

# 3. Train
python scripts/train.py --data data/datasets/food/data.yaml --epochs 60 --name food_v1

# 4. Deploy + evaluate
Copy-Item runs\detect\food_v1*\weights\best.pt models\food.pt
python scripts/evaluate.py detect --weights models\food.pt --data data\datasets\food\data.yaml
```

Details in [docs/workflow-guide.md](docs/workflow-guide.md).

## 📈 Evaluation

`scripts/evaluate.py` covers three experiment groups, writing CSV/JSON/figures to `results/evaluation/`:

```powershell
python scripts/evaluate.py detect --weights models\food.pt --data data\datasets\food\data.yaml   # mAP + confusion matrix
python scripts/evaluate.py calories --set data/eval/calories_gt.csv --engine hybrid               # MAE / MAPE (needs real eval set)
python scripts/evaluate.py engines --set data/eval/engines_gt.csv --engines yolo,glm,hybrid       # accuracy vs latency vs cost
```

## 🧮 How Calorie Estimation Works

```
kcal = portion(g) × kcal-per-100g / 100
```

Portion source priority: **VLM estimate > box-area heuristic > default 150 g**, all adjustable via sliders. The heuristic assumes a box covering 25% of the frame equals one standard serving, scaled by `sqrt(area_ratio / 25%)`, clamped to 0.5–2.5×.

## 🌐 Deploy to Hugging Face Spaces

```powershell
python scripts/prepare_hf_space.py    # assembles an upload-ready hf-space/ folder
```

Then follow the 3-step upload in [hf-space/DEPLOY.md](hf-space/DEPLOY.md) — no token needed (web drag-and-drop).

## 📁 Project Structure

```
FoodCal/
├── app.py                    # Streamlit app (three-engine architecture)
├── src/
│   ├── detector.py           # YOLO wrapper
│   ├── vision.py             # GLM-4.6V vision LLM (recognize / name / auto-label)
│   ├── imgio.py              # Unicode-safe image IO (Windows CJK paths)
│   ├── nutrition.py          # nutrition DB + portion/kcal estimation
│   └── visualize.py          # CJK-aware annotation drawing (PIL)
├── scripts/
│   ├── crawl_images.py       # web-image crawler (per-dish queries, dedup)
│   ├── build_dataset.py      # photos → VLM auto-labels → YOLO dataset
│   ├── train.py              # fine-tune YOLOv8
│   ├── evaluate.py           # mAP / MAPE / engine benchmark
│   ├── check_dataset.py      # dataset sanity check
│   ├── predict.py            # CLI inference
│   └── prepare_hf_space.py   # assemble HF Spaces upload folder
├── data/
│   ├── calories.csv          # 87-item nutrition database (editable)
│   └── eval/                 # evaluation sets
├── docs/                     # demo images + workflow guide
└── REPORT.md                 # technical report (English)
```

## ⚠️ Limitations

- v1 is trained on **web images only**; expect a domain gap on in-situ cafeteria photos (v2 fine-tuning planned)
- Monocular photos cannot measure volume — portions are heuristic estimates, refined manually
- Nutrition values are per-dish averages; actual calories vary by recipe

## 📄 License

MIT
