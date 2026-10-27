"""组装 Hugging Face Spaces 上传目录（hf-space/）。

运行后 hf-space/ 里就是上传到 HF Space 需要的全部文件（含 DEPLOY.md 部署说明）。
模型权重存在时一并复制。上传步骤见 hf-space/DEPLOY.md：

    python scripts/prepare_hf_space.py
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
STAGE = PROJECT_ROOT / "hf-space"

SRC_FILES = ["detector.py", "vision.py", "imgio.py", "nutrition.py", "visualize.py", "__init__.py"]

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
ultralytics>=8.2.0
streamlit>=1.35.0
opencv-python-headless>=4.9.0
pillow>=10.0.0
pandas>=2.0.0
numpy>=1.26.0
requests>=2.31.0
python-dotenv>=1.0
pyyaml>=6.0
"""

PACKAGES = "fonts-noto-cjk\n"  # apt 装中文字体，保证标注图中文不糊

DEPLOY_MD = """# 🚀 部署到 Hugging Face Spaces（3 步，约 5 分钟）

本目录是给 HF Space 用的完整文件包。模型权重 `models/food.pt`（6.5 MB）
也在里面——**别把它提交到 GitHub**（已在 .gitignore），但可以传到 HF。

## 第 1 步：建 Space（2 分钟）

1. 注册/登录 https://huggingface.co
2. 右上角头像 → **New Space**
3. 填写：Space name 填 `FoodCal`；SDK 选 **Streamlit**；License 选 MIT；可见性 **Public**
4. 点 **Create Space**

## 第 2 步：上传文件（2 分钟）

进入刚建的 Space → **Files** 标签页 → **Add file → Upload files**，
把 `hf-space/` 里的**全部内容**拖进去（保持目录结构）：

```
app.py
src/            ← 整个文件夹（含 __init__.py）
data/calories.csv
models/food.pt
requirements.txt
packages.txt
```

> HF 网页支持拖拽整个文件夹，会保留目录结构。
> Space 自带的 README.md **不用动**（里面的 frontmatter 是 HF 自动生成的配置）。

## 第 3 步：等构建（3~5 分钟）

- HF 自动 pip install（首次约 5 分钟，PyTorch 较大）
- 状态 **Building → Running** 即部署完成
- 地址：`https://<你的用户名>-foodcal.hf.space`

## 说明

- Space 里 **GLM 引擎默认不可用**（不能内置 API Key）——访客可在侧边栏填自己的 Key 解锁；本地 YOLO 引擎开箱即用
- `packages.txt` 让 HF 自动安装中文字体（fonts-noto-cjk），保证标注图中文正常
- 换模型：本地替换 `models/food.pt` 后重新上传该文件
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "src").mkdir(parents=True)
    (STAGE / "data").mkdir()
    (STAGE / "models").mkdir()

    shutil.copy2(PROJECT_ROOT / "app.py", STAGE / "app.py")
    for f in SRC_FILES:
        shutil.copy2(SRC / f, STAGE / "src" / f)
    shutil.copy2(PROJECT_ROOT / "data" / "calories.csv", STAGE / "data" / "calories.csv")

    weights = PROJECT_ROOT / "models" / "food.pt"
    if weights.exists():
        shutil.copy2(weights, STAGE / "models" / "food.pt")
        print("已复制模型权重 models/food.pt（6.5 MB）")
    else:
        print("⚠ 未找到 models/food.pt，Space 将只能用 COCO 预训练模型")

    (STAGE / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (STAGE / "packages.txt").write_text(PACKAGES, encoding="utf-8")
    (STAGE / "DEPLOY.md").write_text(DEPLOY_MD, encoding="utf-8")

    print(f"\n组装完成: {STAGE.resolve()}")
    print("上传步骤见 hf-space/DEPLOY.md")


if __name__ == "__main__":
    main()
