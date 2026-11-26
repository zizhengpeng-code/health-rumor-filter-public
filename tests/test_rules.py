"""行为面规则与风险矩阵的测试（纯逻辑，不联网、不调模型）。"""
from __future__ import annotations

import pytest

from src.behavior.rules import (
    behavior_level_from_types, effective_hits, risk_level_from_hits, screen_result,
)
from src.models import BehaviorHit
from src.risk_matrix import DEFAULT_MATRIX, decide, worst_label

# 与 config.yaml 的 behavior.risk_rules 保持一致
RULES = [
    {"level": "高", "any_of": ["医疗行为建议", "金钱相关"]},
    {"level": "中", "min_hits": 2},
    {"level": "低", "min_hits": 1},
    {"level": "低", "min_hits": 0},
]


@pytest.mark.parametrize("types,expected", [
    ([], "低"),
    (["转发诱导"], "低"),
    (["恐惧诉求"], "低"),
    (["医疗行为建议"], "高"),
    (["金钱相关"], "高"),
    (["转发诱导", "恐惧诉求"], "中"),
    (["消费导流", "恐惧诉求"], "中"),
    (["转发诱导", "恐惧诉求", "消费导流"], "中"),
    # 医疗行为建议 优先级高于"两类及以上"——顺序敏感
    (["医疗行为建议", "转发诱导"], "高"),
    (["金钱相关", "恐惧诉求", "消费导流"], "高"),
])
def test_risk_level_from_hits(types, expected):
    assert risk_level_from_hits(types, RULES) == expected


def test_duplicate_types_count_once():
    # 同一类型命中两次，只算一类 -> 应为低而不是中
    assert risk_level_from_hits(["转发诱导", "转发诱导"], RULES) == "低"


def test_confidence_threshold_filters_weak_hits():
    hits = [
        BehaviorHit(type="医疗行为建议", evidence="停药", confidence=0.9),
        BehaviorHit(type="恐惧诉求", evidence="很危险", confidence=0.3),
    ]
    kept = effective_hits(hits, 0.5)
    assert [h.type for h in kept] == ["医疗行为建议"]


def test_screen_result_applies_threshold_then_rules():
    hits = [
        BehaviorHit(type="恐惧诉求", evidence="再不看就晚了", confidence=0.8),
        BehaviorHit(type="转发诱导", evidence="转给家人", confidence=0.7),
        BehaviorHit(type="医疗行为建议", evidence="停药", confidence=0.2),  # 被阈值滤掉
    ]
    res = screen_result(hits, {"confidence_threshold": 0.5, "risk_rules": RULES})
    assert res.risk_level == "中"
    assert res.hit_types == ["转发诱导", "恐惧诉求"]


def test_screen_result_empty_is_low():
    res = screen_result([], {"confidence_threshold": 0.5, "risk_rules": RULES})
    assert res.risk_level == "低" and res.behaviors == []


# ---------------------------------------------------------------------------
# 风险矩阵
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cog,beh,expected", [
    ("假", "高", "尽快劝阻"),
    ("假", "中", "尽快劝阻"),
    ("假", "低", "告知误区"),
    ("部分属实", "高", "提醒留意"),
    ("部分属实", "中", "提醒留意"),
    ("部分属实", "低", "无需处理"),
    ("无法核查", "高", "提醒留意"),
    ("真", "高", "提醒留意"),
    ("真", "中", "无需处理"),
    ("真", "低", "无需处理"),
])
def test_risk_matrix_matches_spec(cog, beh, expected):
    assert decide(cog, beh, {"matrix": DEFAULT_MATRIX}).priority == expected


def test_decide_falls_back_when_config_missing():
    # 配置缺失不该崩，应回退到默认矩阵
    assert decide("假", "高", {}).priority == "尽快劝阻"


def test_worst_label_picks_most_severe():
    assert worst_label(["真", "假", "部分属实"]) == "假"
    assert worst_label(["真", "部分属实"]) == "部分属实"
    assert worst_label([]) == "无法核查"   # 保守兜底
