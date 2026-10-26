# 从拍照到出数字：完整作战手册

> 核心原则：**训练集**（单菜特写）和**评测集**（真实餐盘）是两批完全独立的照片，不要混用。

## 两组照片

| | 训练集 photos/ | 评测集 data/eval/ |
|---|---|---|
| 拍什么 | 单菜特写，一道菜占画面主体 | 真实餐盘，打完饭随手拍 |
| 目的 | 教模型认识每道菜 | 测整条管线的误差 |
| 数量 | 每菜 20~50 张 | 30~50 张 |
| 记录 | 无需记录 | 备忘录记：菜名 + 大概克数 |

## 第 ① 步：拍训练集（1~2 天）

- 每顿饭打 3~4 个菜，每道菜拍 20 张（约 2 分钟）
- 45° 俯拍为主，换 2~3 个光线窗口，远近各几张
- 微信文件传输助手传电脑，按菜名放：`photos/番茄炒蛋/`、`photos/宫保鸡丁/` ...
- 文件名随意，脚本不依赖

## 第 ② 步：自动标注 + 构建数据集

```powershell
python scripts/build_dataset.py --src photos --out data/datasets/food --auto-label
```

- 前提：`.env` 里已配置 `ZHIPU_API_KEY`
- 输出：`data/datasets/food/{train,val}/{images,labels}` + `data.yaml`
- 看跳过数量：跳过多 = 照片不清，删掉重拍

## 第 ③ 步：训练 v1

```powershell
python scripts/train.py --data data/datasets/food/data.yaml --epochs 80 --name food_v1
```

- CPU 半天，GPU 约 1 小时；ultralytics 自动选择设备
- 产物：`runs/detect/food_v1/weights/best.pt` + 训练曲线 `results.png`

## 第 ④ 步：检测评测

```powershell
python scripts/evaluate.py detect --weights runs/detect/food_v1/weights/best.pt --data data/datasets/food/data.yaml
```

- 重点看**混淆矩阵**：哪几道菜互相认错 → 报告的 Error Analysis 素材
- 某类 AP 很低 → 该类照片太少，补拍

## 第 ⑤ 步：热量与三引擎评测

拍评测集时备忘录记录真实值（米饭一碗约 200g、炒菜一勺 100~150g、荤菜一勺 80~120g），填入 `data/eval/calories_gt.csv`：

```csv
image,dish_zh,true_grams
data/eval/photos/tray001.jpg,番茄炒蛋,150
```

```powershell
python scripts/evaluate.py calories --set data/eval/calories_gt.csv --engine yolo
python scripts/evaluate.py calories --set data/eval/calories_gt.csv --engine hybrid
python scripts/evaluate.py engines --set data/eval/engines_gt.csv --engines yolo,glm,hybrid
```

## 第 ⑥ 步：部署 + 提交 GitHub

```powershell
Copy-Item runs\detect\food_v1\weights\best.pt models\food.pt
```

网页侧边栏选 `models/food.pt` + 混合引擎。把关键数字写进 README 后分批提交：

```powershell
git add README.md docs/
git commit -m "Train on canteen dataset: mAP@50 0.XX, hybrid accuracy XX%"
```

## 进阶：消融实验（报告最大亮点）

- v1 = 纯自动标注训练的模型
- v2 = 人工修正标注（X-AnyLabeling 抽查 30% 图片，修正后的 txt 放回 photos/ 旁，重跑 build_dataset，再训练）
- 对比：`python scripts/evaluate.py detect --weights runs/.../v1/best.pt runs/.../v2/best.pt --data ...`
- 结论一句话：**人工审核 X% 的标注带来 mAP +Y** —— 量化"数据飞轮"的价值

## 常见坑

1. 评测图混入训练图 → 数字虚高，报告失去可信度
2. 类别不平衡：某菜 <10 张 → 该类 AP 差，补拍
3. 训练集里出现"一盘多菜"的照片 → 标注混乱，训练集保持单菜一图
4. 自动标注 found:false 多 → 光线/角度问题，重拍那批
