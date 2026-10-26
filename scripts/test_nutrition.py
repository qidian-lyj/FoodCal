"""营养库扩充后的验证测试。"""
import sys

sys.path.insert(0, "src")
from nutrition import NutritionDB

db = NutritionDB.load()
print("DB entries:", len(db))
assert len(db) == 87, f"expected 87, got {len(db)}"

# 新增条目精确查表
for zh in ["草莓", "提拉米苏", "意大利面", "榴莲", "蛋挞", "韩式拌饭", "宫保鸡丁", "芝士"]:
    info = db.lookup(zh)
    assert info.name_zh == zh, f"{zh} -> {info.name_zh} (lookup failed)"
print("exact lookups OK (8/8)")

# 分类抽查
cats = {}
for name in db.zh_names():
    cats[db.lookup(name).category] = cats.get(db.lookup(name).category, 0) + 1
print("categories:", dict(sorted(cats.items(), key=lambda x: -x[1])))
assert cats.get("甜点", 0) >= 15 and cats.get("水果", 0) >= 17 and cats.get("外国菜", 0) >= 14

# 模糊匹配防误伤：「凤梨」不应误命中单字「梨」（宁可落到 UNKNOWN，也不错查）
assert db.lookup("凤梨").name_en != "pear"
assert db.lookup("梨").name_zh == "梨"  # 精确匹配不受影响
assert db.lookup("咖喱").name_zh == "咖喱饭"  # 正向模糊仍可用
assert db.lookup("慕斯蛋糕").name_zh == "蛋糕"  # 反向模糊仍可用
assert db.lookup("宫保鸡丁").name_zh == "宫保鸡丁"
print("fuzzy-match guard OK")

# 提示列表规模（GLM prompt 里用）
hints = db.zh_names()
print(f"hints: {len(hints)} names, {len('、'.join(hints))} chars")
print("ALL NUTRITION TESTS PASSED")
