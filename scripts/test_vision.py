"""vision 模块离线冒烟测试（不调 API）。"""
import sys

import numpy as np

sys.path.insert(0, "src")
import vision
from vision import VisionRecognizer, _extract_json

# 1) JSON 解析容错
assert _extract_json('{"items": [{"名称": "披萨", "份量克": 200}]}') == {
    "items": [{"名称": "披萨", "份量克": 200}]
}
fenced = '好的，识别结果如下：\n```json\n{"名称": "番茄炒蛋", "份量克": 250, "置信度": 0.9}\n```\n以上。'
assert _extract_json(fenced)["名称"] == "番茄炒蛋"
assert _extract_json('思考中... 最终答案 {"found": true, "bbox": [10, 20, 30, 40]}')["found"] is True
assert _extract_json("完全不是 json") is None
print("json parse tests OK")

# 2) 无 Key 时给出清晰报错
try:
    VisionRecognizer(api_key="")
    raise AssertionError("should raise")
except ValueError as e:
    print("empty key raise OK:", str(e)[:40], "...")

# 3) bbox 缩放还原（模拟 2560 宽原图 -> 1280）
img = np.full((1920, 2560, 3), 100, dtype=np.uint8)


class FakeResp:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [
                {"message": {"content": '{"found": true, "bbox": [100, 100, 640, 640], "confidence": 0.9}'}}
            ]
        }


orig_post = vision.requests.post
vision.requests.post = lambda *a, **k: FakeResp()
try:
    vr = VisionRecognizer(api_key="test-key")
    box, conf = vr.find_bbox(img, "番茄炒蛋")
finally:
    vision.requests.post = orig_post

assert box == (200, 200, 1480, 1480), f"bbox restore wrong: {box}"  # 2x 缩放还原
print("bbox scale restore OK:", box, "conf:", round(conf, 2))
print("ALL VISION TESTS PASSED")
