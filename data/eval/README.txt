评测集说明（申研报告的实验数据就在这里攒）

calories_gt.csv —— 热量估算评测集
    列: image(图片路径), dish_zh(真实菜名), true_grams(真实份量克), true_kcal(可选，不填则按数据库×真实份量计算)
    建议：拍 30~50 份真实餐盘，记录每个窗口 actual 菜品和打饭重量（或用标准营养表估算）

engines_gt.csv —— 三引擎对比评测集
    列: image(图片路径), dish_zh(真实菜名)
    与 calories 共用同一批图即可

运行：
    python scripts/evaluate.py calories --set data/eval/calories_gt.csv --engine hybrid
    python scripts/evaluate.py engines --set data/eval/engines_gt.csv

结果输出到 results/evaluation/（CSV + JSON + 对比图），可直接引用进报告。
