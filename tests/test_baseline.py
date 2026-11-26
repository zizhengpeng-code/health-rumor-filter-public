"""M2 baseline 的离线测试（不联网、不花钱）。

用一个假 client 替换真实模型，验证：提示词装载、输出解析、错误处理、成本透传。
真实模型的分数由 `python eval/eval.py --config baseline --split dev --model real` 产出。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cognitive.baseline import BaselineJudge, judge
from src.config import load_config
from src.llm import LLMError
from src.models import BaselineVerdict, Document
from src.prompts import load_prompt

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


class FakeClient:
    """假模型：按预设返回，并记录被调用时的提示词，便于断言提示词内容。"""

    def __init__(self, result=None, exc=None):
        self.result = result or {"label": "假", "confidence": 0.9, "reasoning": "与共识相反"}
        self.exc = exc
        self.calls: list[dict] = []

    def complete(self, prompt, system, schema, *, stage="llm", prompt_version=""):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema,
                           "stage": stage, "prompt_version": prompt_version})
        if self.exc:
            raise self.exc
        return schema.model_validate(self.result).model_dump(mode="json")


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def test_baseline_prompt_exists_and_is_v1(cfg):
    system, user, version = load_prompt("baseline_judge", cfg)
    assert version == "v1"
    assert "{title}" in user and "{text}" in user
    assert "无法核查" in system


def test_baseline_prompt_does_not_leak_retrieval(cfg):
    """baseline 是裸模型，提示词里绝不能出现检索/证据相关要求，否则就不是 baseline 了。"""
    system, user, _ = load_prompt("baseline_judge", cfg)
    blob = system + user
    for banned in ["检索", "证据编号", "信源评分", "多智能体", "辩论"]:
        assert banned not in blob, f"baseline 提示词里不该出现「{banned}」"


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def test_judge_parses_verdict(cfg):
    j = BaselineJudge.build(cfg, client=FakeClient())
    v = j.judge("喝醋能软化血管，每天一杯远离心脑血管病", "标题")
    assert isinstance(v, BaselineVerdict)
    assert v.label == "假" and v.confidence == 0.9


def test_judge_passes_title_and_text_into_prompt(cfg):
    c = FakeClient()
    BaselineJudge.build(cfg, client=c).judge("隔夜水致癌", "隔夜水不能喝")
    p = c.calls[0]["prompt"]
    assert "隔夜水致癌" in p and "隔夜水不能喝" in p


def test_judge_truncates_long_text(cfg):
    c = FakeClient()
    j = BaselineJudge.build(cfg, client=c)
    j.max_chars = 50
    j.judge("长" * 5000, "标题")
    assert "长" * 50 in c.calls[0]["prompt"]
    assert "长" * 51 not in c.calls[0]["prompt"]


def test_judge_records_stage_and_prompt_version(cfg):
    c = FakeClient()
    BaselineJudge.build(cfg, client=c).judge("x", "y")
    assert c.calls[0]["stage"] == "baseline"
    assert c.calls[0]["prompt_version"] == "v1"


def test_judge_uses_baseline_verdict_schema(cfg):
    c = FakeClient()
    BaselineJudge.build(cfg, client=c).judge("x")
    assert c.calls[0]["schema"] is BaselineVerdict


@pytest.mark.parametrize("label", ["真", "假", "部分属实", "无法核查"])
def test_judge_accepts_all_four_labels(cfg, label):
    c = FakeClient({"label": label, "confidence": 0.7, "reasoning": "r"})
    assert BaselineJudge.build(cfg, client=c).judge("x").label == label


def test_judge_rejects_invalid_label(cfg):
    c = FakeClient({"label": "半真半假", "confidence": 0.7})
    with pytest.raises(Exception):
        BaselineJudge.build(cfg, client=c).judge("x")


def test_judge_propagates_llm_error(cfg):
    """调用失败必须抛出去，由上层决定怎么记——不能悄悄返回一个默认标签。"""
    c = FakeClient(exc=LLMError("限流"))
    with pytest.raises(LLMError):
        BaselineJudge.build(cfg, client=c).judge("x")


def test_judge_document(cfg):
    c = FakeClient()
    j = BaselineJudge.build(cfg, client=c)
    v = j.judge_document(Document(id="d1", title="T", text="正文内容"))
    assert v.label == "假"
    assert "正文内容" in c.calls[0]["prompt"]


def test_convenience_judge(cfg):
    v = judge("隔夜水致癌", "标题", cfg=cfg, client=FakeClient())
    assert v.label == "假"


def test_prompt_version_property(cfg):
    assert BaselineJudge.build(cfg, client=FakeClient()).prompt_version == "v1"


# ---------------------------------------------------------------------------
# 正文截断上限的解析
# ---------------------------------------------------------------------------
def test_max_chars_inherits_ingest_when_zero(cfg):
    """config 里 max_chars: 0 表示"继承 ingest.max_chars"——两者必须一致，
    否则 baseline 会因为被喂得少而被人为压低，消融出来的增益就掺了水分。"""
    from src.cognitive.baseline import resolve_max_chars

    ingest = cfg.get_path("ingest.max_chars")
    assert resolve_max_chars(cfg) == ingest


def test_max_chars_explicit_override_wins():
    from src.cognitive.baseline import resolve_max_chars

    cfg = {"ingest": {"max_chars": 20000}, "cognitive": {"baseline": {"max_chars": 1234}}}
    assert resolve_max_chars(cfg) == 1234


def test_max_chars_falls_back_when_config_empty():
    from src.cognitive.baseline import resolve_max_chars

    assert resolve_max_chars({}) == 6000


def test_built_judge_uses_resolved_max_chars(cfg):
    j = BaselineJudge.build(cfg, client=FakeClient())
    assert j.max_chars == cfg.get_path("ingest.max_chars")
