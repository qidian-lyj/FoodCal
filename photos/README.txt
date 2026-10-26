把拍照收集的原始照片放在这里，每个子文件夹是一道菜，文件夹名就是类别名（中文即可）：

photos/
├── 番茄炒蛋/
│   ├── 001.jpg
│   └── ...
├── 宫保鸡丁/
├── 米饭/
└── ...

每道菜建议 20~50 张：正对拍、45° 拍、食堂不同光线/餐盘都拍一些。
放好后运行：

    python scripts/build_dataset.py --src photos --out data/datasets/food --auto-label

GLM 会自动给每张照片打标签，之后用 X-AnyLabeling 抽查修正即可。
