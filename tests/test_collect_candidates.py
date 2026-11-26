"""候选池采集脚本的纯函数测试（不联网、不写文件）。"""
from scripts.collect_candidates import (
    build_candidates, extract_rumors, guess_topic, looks_like_popsci, preselect,
)


def test_extract_multiple_rumors_from_roundup():
    text = ("一、同时吃多种感冒药好得快？\n流言：可以同时吃多种药，提高感冒康复速度。\n"
            "真相：这是一种错误的做法。\n二、\n谣言：\n长期吃降压药会伤肾。\n辟谣：不会。")
    rs = extract_rumors(text)
    assert [r["rumor"] for r in rs] == ["可以同时吃多种药，提高感冒康复速度。", "长期吃降压药会伤肾。"]
    assert rs[0]["truth"].startswith("这是一种错误的做法")


def test_extract_skips_placeholder_and_too_short():
    assert extract_rumors("谣言：XX和XX一起吃会中毒。\n真相：……") == []
    assert extract_rumors("谣言：太短\n") == []


def test_topic_prefers_title_and_specific_topics():
    assert guess_topic("吃阿司匹林能防癌？") == "癌症"          # 癌症优先于用药/饮食
    assert guess_topic("颁发仪式") == "其他"                    # 「仪式」不能算养生器械
    assert guess_topic("一篇文章", "高血压患者注意") == "慢病"   # 标题猜不出才看正文


def test_popsci_filter_rejects_nav_and_debunk_pages():
    body = "高血压患者需要长期服药。" * 50
    assert looks_like_popsci("高血压用药常识", body)
    assert not looks_like_popsci("中国疾病预防控制中心-搜索", body)
    assert not looks_like_popsci("高血压用药常识", "谣言：吃药伤肾\n" + body)


def test_build_and_preselect_balances_labels():
    rumors = ["喝醋能软化血管，比吃药还管用。", "隔夜水喝了会致癌，千万别喝。", "吃保健品就能停掉降糖药。"]
    docs = [
        {"title": "汇总", "text": f"谣言：{r}\n真相：不能。",
         "source_name": "辟谣平台", "url": f"u{i}", "in_kb": True} for i, r in enumerate(rumors)
    ] + [
        {"title": t, "text": b * 60, "source_name": "卫健委", "url": f"v{i}", "in_kb": False}
        for i, (t, b) in enumerate([
            ("高血压患者如何监测血压", "家庭自测血压每天早晚各一次。"),
            ("糖尿病饮食怎么安排", "主食定量，粗细搭配，少吃甜食。"),
            ("癌症筛查有哪些手段", "低剂量螺旋CT可用于肺癌筛查。"),
        ])
    ]
    cands = build_candidates(docs)
    assert {c["label"] for c in cands} == {"假", "真"}
    preselect(cands, target=4)
    sel = [c for c in cands if c["selected"]]
    assert sum(c["label"] == "假" for c in sel) == 2
    assert sum(c["label"] == "真" for c in sel) == 2
