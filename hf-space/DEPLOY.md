# 🚀 部署到 Hugging Face Spaces（3 步，约 5 分钟）

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
