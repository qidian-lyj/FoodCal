# FoodCal: Photo-Based Food Calorie Estimation with Detection and Vision-Language Models

**Technical Report · v1.0 · October 2026**
Repo: https://github.com/qidian-lyj/FoodCal

---

## 1. Motivation & Problem Statement

Estimating dietary intake typically requires manual logging, which is tedious and inaccurate. We ask a narrow, well-defined question:

> Given a single photo of a meal, can a system automatically identify the dishes present and estimate the total caloric content — cheaply enough for a student project, and accurately enough to be useful?

We scope the problem to **20 common university-cafeteria dishes** (Chinese home-style cooking), with a web interface that lets users photograph food and adjust estimated portions.

## 2. System Design

FoodCal is a three-engine pipeline, switchable at runtime:

| Engine | Role | Cost | Scope |
|---|---|---|---|
| 🔍 Local YOLO (YOLOv8n) | Offline detection + classification | Free | Trained classes only |
| ☁️ GLM-4.6V-Flash (cloud VLM) | Whole-image recognition + portion estimate | Free tier | Any dish |
| 🤝 Hybrid | YOLO localizes → GLM names each crop | Free tier | Best of both |

Calories are computed as `portion(g) × kcal/100g ÷ 100` against an 87-item nutrition database. Portion size comes from the VLM when available, otherwise a box-area heuristic (a detection box covering 25% of the frame ≙ one standard serving, scaled by `sqrt(area/25%)`, clamped to 0.5–2.5×), and is always user-adjustable.

### 2.1 The Data Flywheel

The central engineering idea is **using a large VLM to bootstrap a small detector**:

```
web-crawl 1,113 images (search-engine queries per dish)
  → GLM-4.6V auto-annotates bounding boxes per image
  → GLM simultaneously acts as a quality filter, rejecting images
    where the target dish is not actually visible (26% rejection rate)
  → 824 verified images train YOLOv8n
  → (planned) human spot-checks + in-situ photos for v2
```

Human effort is reduced to spot-checking; the VLM does the per-image labeling.

## 3. Dataset Construction

| | |
|---|---|
| Source | Baidu/Bing image search, per-dish Chinese queries |
| Images crawled | 1,113 (20 dishes, 41–66 per dish) |
| After VLM quality filter | **824** (700 train / 124 val, 85/15 split) |
| Annotation | VLM-predicted bounding boxes, YOLO format |
| Human annotation effort | ~0 (spot-check only) |

Engineering notes that mattered:

- **Cross-class filename collisions**: every dish folder numbered files `001.jpg…`; naive copying into a shared directory silently overwrote samples across classes. Fixed by class-ID filename prefixes — caught by monitoring an images-vs-labels count mismatch.
- **Rate-limit resilience**: the free VLM endpoint throttles aggressively (HTTP 429). Exponential backoff (2→40 s, 6 attempts) plus a "not-found" cache turned a crash-prone job into a fully resumable one.
- **CJK path handling**: `cv2.imread` fails silently on non-ASCII paths under the Windows GBK code page; replaced with an `imdecode(np.fromfile(...))` wrapper.

## 4. Experiments & Results

### 4.1 Detection (YOLOv8n, 60 epochs, CPU, 2.74 h)

Trained on 700 images; validated on 124 unseen images.

| Metric | Value |
|---|---|
| **mAP@50** | **0.863** |
| mAP@50-95 | 0.532 |
| Precision | 0.896 |
| Recall | 0.755 |
| Inference latency (CPU) | 35.6 ms/image |

Per-class AP@50 (best and worst):

| Class | AP@50 | | Class | AP@50 |
|---|---|---|---|---|
| Cola chicken wings | 0.995 | | Steamed rice | 0.588 |
| Kung pao chicken | 0.995 | | Braised pork | 0.797 |
| Tomato scrambled eggs | 0.995 | | White cut chicken | 0.853 |
| Stir-fried noodles | 0.995 | | Baozi | 0.478 |
| Fish-flavored pork | 0.995 | | Dumplings | 0.165 |

### 4.2 Error Analysis

- **Strong classes (AP@50 ≈ 0.99)** share distinctive color/texture signatures (sauces, stir-fry gloss) and consistent plating in web images.
- **Baozi (0.478)**: web images mix baozi, xiaolongbao, shengjianbao and raw frozen buns — high annotation ambiguity, confirmed as the top confusion pair in the confusion matrix.
- **Steamed rice (recall 0.25)**: white rice in white bowls has low contrast; the detector fires only on close-ups.
- **Dumplings / mantou**: only 2 validation images survived the quality filter, so their metrics are statistically unreliable — a sampling problem, not necessarily a model problem.
- **Portion-to-frame correlation**: since labels came from whole-dish photos, the detector's box prior is biased toward large, centered dishes.

### 4.3 Calorie Estimation Error — *pending real-world evaluation set*

The evaluation protocol is implemented (`scripts/evaluate.py calories`): ≥30 real trays with ground-truth portions → MAE / MAPE / ±20 % tolerance accuracy. **Awaiting in-situ data collection.**

### 4.4 Engine Comparison — *pending real-world evaluation set*

`scripts/evaluate.py engines` measures top-1 accuracy, per-image latency, and API-call count across engines. Informal observations: local YOLO adds ~0.6 ms preprocess + 36 ms inference; GLM round-trip is 3–8 s. The hybrid engine trades one VLM call per detected region for fine-grained dish naming.

## 5. Limitations

1. **Domain gap**: v1 is trained purely on web images (studio lighting, styled plating). In-situ cafeteria photos differ in lighting, clutter, and angles; real-world recall is expected to be lower. (See Future Work.)
2. **Volume is not observable** from a single monocular photo. Portions are heuristic estimates that the user corrects via sliders; the UI states this explicitly.
3. **Nutrition values are averages** per dish; actual calories vary by recipe and cooking oil.
4. **Class imbalance** after quality filtering (24–66 images per class in training; 2–9 in validation) depresses minority-class metrics.
5. **Crawled images are for local research use only** and are excluded from the repository.

## 6. Future Work

- **v2**: fine-tune on in-situ photos (targeting weak classes first); report web-only vs fine-tuned ablation.
- **Annotation-quality ablation**: compare VLM-only labels vs human-corrected labels on the same split.
- **Calorie error study** and **engine benchmark** on the real evaluation set (protocol implemented).
- **Portion regression**: replace the area heuristic with a reference-object (bowl/plate) scale estimator.
- **HF Spaces demo**: public click-through deployment (build in progress).

## 7. Reproducibility

All code, the 87-item nutrition database, dataset-construction scripts, and evaluation harness are open-sourced under MIT. Fixed random seeds (shuffle seed 42), pinned dependencies in `requirements.txt`, and resumable pipeline stages allow end-to-end reproduction on CPU-only hardware at zero API cost (free-tier `glm-4.6v-flash`).

---

*Results are estimates for research demonstration and do not constitute nutritional advice.*
