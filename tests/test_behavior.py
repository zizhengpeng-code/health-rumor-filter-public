"""M3 行为面滤网的离线测试（不联网、不花钱）。

用一个假 client 替换真实模型，重点验证三条**设计不变量**：
1. 行为面与真假无关——提示词里不许出现真假判定要求，行为命中不许改变认知面结论
2. 模型只报命中，风险等级一律由规则算——模型输出契约里没有 risk_level 字段
3. 低置信度命中不抬高风险等级
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.behavior.rules import behavior_level_from_types, screen_result
from src.behavior.screen import BehaviorScreen, screen
from src.config import load_config
from src.llm import LLMError
from src.models import BehaviorHit, BehaviorResult, BehaviorScreenOut, Document
from src.prompts import load_prompt

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


class FakeClient:
    """假模型：按预设返回命中列表，并记录被调用时的提示词。"""

    def __init__(self, behaviors=None, exc=None, cost_usd=0.0, tokens=0):
        self.behaviors = behaviors if behaviors is not None else []
        self.exc = exc
        self.calls: list[dict] = []
        # 真实 client 的契约里带 history（成本记账要用）。这里也给一份，
        # 让测试走的是和生产同一条路径，而不是靠"没有 history 也能跑"。
        self.history: list[Any] = []
        self._cost_usd, self._tokens = cost_usd, tokens

    def complete(self, prompt, system, schema, *, stage="llm", prompt_version=""):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema,
                           "stage": stage, "prompt_version": prompt_version})
        if self.exc:
            raise self.exc
        self.history.append(SimpleNamespace(
            cost_usd=self._cost_usd, prompt_tokens=self._tokens, completion_tokens=0))
        return schema.model_validate({"behaviors": self.behaviors}).model_dump(mode="json")


def hit(t, conf=0.9, ev="原文"):
    return {"type": t, "evidence": ev, "confidence": conf}


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def test_prompt_exists_and_is_v2(cfg):
    system, user, version = load_prompt("behavior_screen", cfg)
    assert version == "v2"          # D50：与标注规则对齐
    assert "{title}" in user and "{text}" in user
    for t in ("转发诱导", "医疗行为建议", "消费导流", "金钱相关", "恐惧诉求"):
        assert t in system, f"提示词里缺少行为类型定义：{t}"


def test_prompt_declares_independence_from_factuality(cfg):
    """核心不变量：行为面不看真假。提示词必须显式声明这一点。"""
    system, user, _ = load_prompt("behavior_screen", cfg)
    blob = system + user
    assert "不判断内容真假" in blob or "不判断真假" in blob or "与真假无关" in blob
    # 不能要求模型判定事实性——那是认知面的活
    for banned in ("判断真假", "是否谣言", "事实性判定", "辟谣"):
        assert banned not in blob, f"行为面提示词里不该出现「{banned}」"


def test_prompt_tells_model_not_to_output_risk_level(cfg):
    system, _, _ = load_prompt("behavior_screen", cfg)
    assert "不报等级" in system or "风险等级由代码" in system


# ---------------------------------------------------------------------------
# 模型输出契约：只有命中，没有等级
# ---------------------------------------------------------------------------
def test_screen_out_has_no_risk_level_field():
    """一旦模型也能给等级，就有了两个真值来源，不一致时无从裁决。"""
    assert "risk_level" not in BehaviorScreenOut.model_fields
    assert set(BehaviorScreenOut.model_fields) == {"behaviors"}


def test_screen_out_rejects_unknown_behavior_type():
    with pytest.raises(Exception):
        BehaviorScreenOut.model_validate({"behaviors": [hit("包治百病")]})


def test_screen_out_defaults_to_empty():
    assert BehaviorScreenOut.model_validate({}).behaviors == []


# ---------------------------------------------------------------------------
# 规则：模型报什么 -> 算成多严重
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("types,expected", [
    (["医疗行为建议"], "高"),
    (["金钱相关"], "高"),
    (["医疗行为建议", "转发诱导"], "高"),      # 高优先于"两类及以上"
    (["转发诱导", "消费导流"], "中"),          # 两类其他类型
    (["恐惧诉求", "消费导流"], "中"),
    (["转发诱导"], "低"),                      # 一类
    (["恐惧诉求"], "低"),
    ([], "低"),                                # 未命中
])
def test_risk_level_rules(cfg, types, expected):
    bcfg = cfg.get("behavior", {}) or {}
    assert behavior_level_from_types(types, bcfg) == expected


def test_rule_order_matters(cfg):
    """医疗行为建议单独命中就是"高"，不需要凑够两类——顺序敏感是刻意的。"""
    bcfg = cfg.get("behavior", {}) or {}
    assert behavior_level_from_types(["医疗行为建议"], bcfg) == "高"
    assert behavior_level_from_types(["医疗行为建议", "消费导流"], bcfg) == "高"


def test_screen_result_filters_by_confidence(cfg):
    """0.3 的弱命中不该把等级从"低"抬到"高"。"""
    bcfg = cfg.get("behavior", {}) or {}
    hits = [BehaviorHit(type="医疗行为建议", evidence="x", confidence=0.3)]
    res = screen_result(hits, bcfg)
    assert res.risk_level == "低"
    assert res.behaviors == []


def test_screen_result_keeps_confident_hits(cfg):
    bcfg = cfg.get("behavior", {}) or {}
    hits = [BehaviorHit(type="医疗行为建议", evidence="吃了就不用吃降压药", confidence=0.9)]
    res = screen_result(hits, bcfg)
    assert res.risk_level == "高"
    assert res.to_eval() == ["医疗行为建议"]


# ---------------------------------------------------------------------------
# 滤网
# ---------------------------------------------------------------------------
def test_screen_parses_and_applies_rules(cfg):
    c = FakeClient([hit("医疗行为建议", 0.95, "吃了这个就不用吃降压药了")])
    res = BehaviorScreen.build(cfg, client=c).screen("正文", "标题")
    assert isinstance(res, BehaviorResult)
    assert res.risk_level == "高"
    assert res.behaviors[0].evidence == "吃了这个就不用吃降压药了"


def test_screen_returns_low_when_nothing_hits(cfg):
    """大多数正常科普一类都不命中——空数组是正常结果，不是失败。"""
    res = BehaviorScreen.build(cfg, client=FakeClient([])).screen("正文")
    assert res.risk_level == "低" and res.behaviors == []


def test_hit_types_follow_declared_order(cfg):
    """按 BEHAVIOR_TYPES 声明顺序，不用 sorted()（中文按码点排不符合直觉）。"""
    c = FakeClient([hit("恐惧诉求"), hit("转发诱导")])
    res = BehaviorScreen.build(cfg, client=c).screen("x")
    assert res.to_eval() == ["转发诱导", "恐惧诉求"]


def test_screen_passes_title_and_text(cfg):
    c = FakeClient([])
    BehaviorScreen.build(cfg, client=c).screen("隔夜水致癌", "标题党")
    p = c.calls[0]["prompt"]
    assert "隔夜水致癌" in p and "标题党" in p


def test_screen_truncates_long_text(cfg):
    c = FakeClient([])
    s = BehaviorScreen.build(cfg, client=c)
    s.max_chars = 40
    s.screen("长" * 500, "标题")
    assert "长" * 40 in c.calls[0]["prompt"]
    assert "长" * 41 not in c.calls[0]["prompt"]


def test_screen_records_stage_and_prompt_version(cfg):
    c = FakeClient([])
    BehaviorScreen.build(cfg, client=c).screen("x")
    assert c.calls[0]["stage"] == "behavior_screen"
    assert c.calls[0]["prompt_version"] == "v2"
    assert c.calls[0]["schema"] is BehaviorScreenOut


def test_screen_propagates_llm_error(cfg):
    """调用失败必须抛出去，不能悄悄返回"未命中"——那会把失败伪装成正常结论。"""
    c = FakeClient(exc=LLMError("限流"))
    with pytest.raises(LLMError):
        BehaviorScreen.build(cfg, client=c).screen("x")


def test_screen_document(cfg):
    c = FakeClient([hit("消费导流", 0.9, "点击下方链接购买")])
    res = BehaviorScreen.build(cfg, client=c).screen_document(
        Document(id="d1", title="T", text="正文内容"))
    assert res.risk_level == "低"
    assert "正文内容" in c.calls[0]["prompt"]


def test_convenience_screen(cfg):
    res = screen("x", "y", cfg=cfg, client=FakeClient([hit("金钱相关")]))
    assert res.risk_level == "高"


def test_prompt_version_property(cfg):
    assert BehaviorScreen.build(cfg, client=FakeClient()).prompt_version == "v2"


def test_max_chars_inherits_ingest(cfg):
    """与 baseline 同一条规则：默认继承 ingest.max_chars，避免"被喂得少"导致的假增益。"""
    assert BehaviorScreen.build(cfg, client=FakeClient()).max_chars == cfg.get_path("ingest.max_chars")


# ---------------------------------------------------------------------------
# 集成：行为面是独立旁路，不许改变认知面结论
# ---------------------------------------------------------------------------
class FakeJudge:
    """假认知面：固定返回一个标签，并像真实调用一样往 client.history 记一笔。"""

    def __init__(self, label="真", exc=None, cost_usd=0.0, tokens=0):
        self.label, self.exc = label, exc
        self.client = FakeClient([], cost_usd=cost_usd, tokens=tokens)

    def judge(self, text, title=""):
        from src.models import BaselineVerdict
        if self.exc:
            raise self.exc
        self.client.history.append(SimpleNamespace(
            cost_usd=self.client._cost_usd,
            prompt_tokens=self.client._tokens, completion_tokens=0))
        return BaselineVerdict(label=self.label, confidence=0.9, reasoning="r")


def _sample():
    from src.models import LabeledSample
    return LabeledSample(id="x", title="标题", text="正文内容", topic="其他",
                         cognitive_label="真", behaviors=[], notes="")


def _rows(cfg):
    from eval.eval import EvalRow
    return EvalRow(id="x", topic="其他", cognitive_label="真", behaviors=[],
                   behavior_level="低", priority="无需处理")


def _predictor(cfg, label="真", behaviors=None):
    from eval.eval import RealPredictor
    cfg["_overrides"] = {"retrieve": False, "source_score": False, "debate": False,
                         "review": False, "behavior": True}
    return RealPredictor(cfg, judge=FakeJudge(label),
                         behavior=BehaviorScreen.build(cfg, client=FakeClient(behaviors or [])))


def test_behavior_does_not_change_cognitive_label(cfg):
    """核心不变量：一篇文章真假归真假，行为归行为。行为命中绝不能改写认知面结论。"""
    p = _predictor(cfg, label="真", behaviors=[hit("医疗行为建议")]).predict(_sample(), _rows(cfg))
    assert p.cognitive_label == "真"
    assert p.behavior_level == "高"


def test_true_article_can_still_be_high_risk(cfg):
    """完全属实的科普照样可以含"转给家里人"或购买引导——这正是双滤网要抓的。"""
    p = _predictor(cfg, label="真",
                   behaviors=[hit("消费导流"), hit("转发诱导")]).predict(_sample(), _rows(cfg))
    assert p.cognitive_label == "真"
    assert p.behavior_level == "中"
    assert set(p.behaviors) == {"转发诱导", "消费导流"}


def test_false_article_without_behavior_stays_low(cfg):
    """反过来也成立：谣言不等于有行为问题，不能因为内容是假的就硬凑命中。"""
    p = _predictor(cfg, label="假", behaviors=[]).predict(_sample(), _rows(cfg))
    assert p.cognitive_label == "假"
    assert p.behavior_level == "低"
    assert p.behaviors == []


def test_behavior_off_leaves_level_low(cfg):
    """消融开关关掉时，行为等级一律"低"——这张差值就是行为面的贡献。"""
    from eval.eval import RealPredictor
    cfg["_overrides"] = {"retrieve": False, "source_score": False, "debate": False,
                         "review": False, "behavior": False}
    p = RealPredictor(cfg, judge=FakeJudge("假")).predict(_sample(), _rows(cfg))
    assert p.behavior_level == "低" and p.behaviors == []


def test_behavior_failure_does_not_break_cognitive_verdict(cfg):
    """行为面调用失败：认知面结论必须保住，同时留下痕迹（不能伪装成"没命中"）。"""
    from eval.eval import RealPredictor
    cfg["_overrides"] = {"retrieve": False, "source_score": False, "debate": False,
                         "review": False, "behavior": True}
    p = RealPredictor(cfg, judge=FakeJudge("假"),
                      behavior=BehaviorScreen.build(cfg, client=FakeClient(exc=LLMError("限流")))
                      ).predict(_sample(), _rows(cfg))
    assert p.cognitive_label == "假"          # 认知面不受影响
    assert "行为面调用失败" in p.note          # 但失败被记录下来


def test_priority_uses_both_filters(cfg):
    """处置优先级是两张滤网的交汇点：认知面严重度 x 行为面风险等级。"""
    p = _predictor(cfg, label="假",
                   behaviors=[hit("医疗行为建议")]).predict(_sample(), _rows(cfg))
    assert p.cognitive_label == "假" and p.behavior_level == "高"
    assert p.priority == "尽快劝阻"


def test_cost_accumulates_across_both_filters(cfg):
    """两张滤网各自产生调用，成本必须合并统计，不能只算认知面。"""
    from eval.eval import RealPredictor
    cfg["_overrides"] = {"retrieve": False, "source_score": False, "debate": False,
                         "review": False, "behavior": True}
    judge = FakeJudge("真", cost_usd=0.0002, tokens=100)
    beh = BehaviorScreen.build(cfg, client=FakeClient([hit("金钱相关")],
                                                      cost_usd=0.0003, tokens=50))
    p = RealPredictor(cfg, judge=judge, behavior=beh).predict(_sample(), _rows(cfg))
    assert p.cost_usd == pytest.approx(0.0005)
    assert p.tokens == 150
