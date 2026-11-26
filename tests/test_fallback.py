"""兜底机制（D41）的最小测试。全部用假判断器，不联网、不花钱。"""
from __future__ import annotations

from src.cognitive.fallback import FALLBACK_PREFIX, FallbackJudge, evidence_unusable
from src.models import BaselineVerdict, Evidence, EvidenceVerdict


def ev():
    return Evidence(id=0, claim_id=0, title="无关文章", url="https://x.gov.cn/a", source_name="某卫健委",
                    snippet="讲的是喝水", channel="kb")


class FakeInner:
    mode = "hybrid-rerank"
    client = object()
    last_trace = {"n_bm25": 3}
    last_debate_trace = {"debated": True}

    def __init__(self, verdict):
        self.verdict = verdict

    def judge(self, text, title=""):
        return self.verdict, [ev()]


class FakeBaseline:
    client = object()

    def __init__(self):
        self.calls = 0

    def judge(self, text, title=""):
        self.calls += 1
        return BaselineVerdict(label="真", confidence=0.9, reasoning="与常识一致")


def test_unusable_signal():
    assert evidence_unusable(EvidenceVerdict(label="无法核查", used_evidence=[1]))
    assert evidence_unusable(EvidenceVerdict(label="真", used_evidence=[]))
    assert not evidence_unusable(EvidenceVerdict(label="假", used_evidence=[2]))


def test_falls_back_to_baseline_when_evidence_unusable():
    base = FakeBaseline()
    j = FallbackJudge(inner=FakeInner(EvidenceVerdict(label="无法核查", confidence=0.9)), baseline=base)
    v, evs = j.judge("无糖饮料可能伤身体", "标题")
    assert base.calls == 1
    assert v.label == "真"
    assert v.confidence == 0.6                         # 无证据支撑，置信度封顶
    assert v.reasoning.startswith(FALLBACK_PREFIX)     # 不把模型记忆冒充成有据可查
    assert evs == [] and v.used_evidence == []         # 无关证据不进证据链
    assert j.last_trace["fallback_used"] is True and j.last_trace["n_bm25"] == 3
    assert j.last_debate_trace["fallback"]["evidence_verdict"]["label"] == "无法核查"


def test_keeps_evidence_verdict_when_usable():
    base = FakeBaseline()
    good = EvidenceVerdict(label="假", confidence=0.9, reasoning="证据[1]相反", used_evidence=[1])
    j = FallbackJudge(inner=FakeInner(good), baseline=base)
    v, evs = j.judge("正文", "标题")
    assert base.calls == 0                              # 证据可用就不多花一次调用
    assert v == good and len(evs) == 1
    assert j.last_trace["fallback_used"] is False


def test_cost_accounting_includes_baseline_client():
    base = FakeBaseline()
    inner = FakeInner(EvidenceVerdict(label="假", used_evidence=[1]))
    j = FallbackJudge(inner=inner, baseline=base)
    assert j.clients == [inner.client, base.client]
