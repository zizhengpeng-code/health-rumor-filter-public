"""辟谣生成、一键核查主流程、内容接入的最小测试。全部离线，不联网、不花钱。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import load_config
from src.ingest.fetch import IngestError, fetch_document, parse_wechat, strip_noise
from src.llm import LLMError
from src.models import (
    REBUTTAL_ENDING, BehaviorHit, BehaviorResult, Document, Evidence, EvidenceVerdict,
    RebuttalOut,
)
from src.pipeline import Checker
from src.report.rebuttal import RebuttalWriter, finalize, needs_rebuttal
from src.report.render import to_markdown

ROOT = Path(__file__).resolve().parent.parent


def cfg():
    return load_config(ROOT / "config.yaml")


def ev(name="国家卫生健康委员会"):
    return Evidence(id=0, claim_id=0, title="高血压用药误区", url="https://nhc.gov.cn/a.html",
                    source_name=name, domain="nhc.gov.cn", snippet="降压药不会成瘾，需遵医嘱长期服用。",
                    tier=1, channel="kb")


DOC = Document(id="d1", title="降压药吃上就停不下来？",
               text="降压药吃上就停不下来！喝三七粉三个月就能停药，赶紧转给家里老人！")


# ---------------------------------------------------------------------------
# 辟谣稿：长度、来源、结尾由代码兜底
# ---------------------------------------------------------------------------
def test_rebuttal_schema_rejects_overlong_text():
    with pytest.raises(ValidationError):
        RebuttalOut(text="字" * 291)


def test_finalize_appends_ending_and_drops_unknown_sources():
    out = RebuttalOut(text="我查了一下，国家卫健委说降压药不会成瘾，咱们别自己停药",
                      sources=["国家卫生健康委员会", "某养生公众号"])
    r = finalize(out, [ev()])
    assert r.text.endswith(REBUTTAL_ENDING)
    assert "成瘾。具体用药" in r.text or "停药。具体用药" in r.text   # 补了句号再接结尾
    assert r.sources == ["国家卫生健康委员会"]                      # 编造的来源被剔除
    assert len(r.text) <= 300


def test_finalize_does_not_duplicate_ending():
    r = finalize(RebuttalOut(text="内容大体没问题，咱们放心。" + REBUTTAL_ENDING), [])
    assert r.text.count(REBUTTAL_ENDING) == 1


def test_needs_rebuttal_only_skips_true_and_low_risk():
    assert not needs_rebuttal("真", "低")
    assert needs_rebuttal("真", "高")          # 内容属实但劝人停药，也要提醒
    assert needs_rebuttal("假", "低")


class FakeLLM:
    model = "fake"

    def __init__(self, payloads):
        self.payloads, self.prompts, self.history = list(payloads), [], []

    def complete(self, prompt, system, schema, *, stage="llm", prompt_version=""):
        self.prompts.append(prompt)
        return schema.model_validate(self.payloads.pop(0)).model_dump(mode="json")


def test_rebuttal_writer_passes_evidence_and_behaviors():
    llm = FakeLLM([{"text": "我查了一下，国家卫健委说降压药不会成瘾。", "sources": ["国家卫生健康委员会"]}])
    w = RebuttalWriter(cfg(), llm)
    r = w.write(title=DOC.title, text=DOC.text, label="假", evidence=[ev()],
                behaviors=[BehaviorHit(type="转发诱导", evidence="赶紧转给家里老人", confidence=0.9)])
    assert "降压药不会成瘾，需遵医嘱" in llm.prompts[0]          # 证据进了提示词
    assert "转发诱导：「赶紧转给家里老人」" in llm.prompts[0]       # 话术进了提示词
    assert r.text.endswith(REBUTTAL_ENDING)


def test_unverifiable_rebuttal_gets_no_evidence_and_cites_nothing():
    """v1 实测问题：无法核查时模型拿无关证据凑出处。现在一条证据都不给，来源也会被清空。"""
    llm = FakeLLM([{"text": "我查了一下，国家卫健委说这个没问题，咱们放心喝。", "sources": ["国家卫生健康委员会"]}])
    r = RebuttalWriter(cfg(), llm).write(title="无糖饮料", text="无糖饮料可能伤身体", label="无法核查",
                                         evidence=[ev()])
    assert "未检索到相关的权威证据" in llm.prompts[0]
    assert "降压药不会成瘾" not in llm.prompts[0]
    assert r.sources == []


def test_rebuttal_only_gets_evidence_used_in_judgement():
    from src.report.rebuttal import evidence_for_rebuttal
    evs = [ev("科普中国"), ev("国家卫生健康委员会"), ev("中国疾病预防控制中心")]
    assert [e.source_name for e in evidence_for_rebuttal("假", evs, used=[2])] == ["国家卫生健康委员会"]
    assert len(evidence_for_rebuttal("假", evs, used=[])) == 3       # 判定没标编号才退回前几条


# ---------------------------------------------------------------------------
# 主流程：两道滤网并行、单点失败不中断
# ---------------------------------------------------------------------------
class SlowBehavior:
    def screen(self, text, title=""):
        time.sleep(0.3)
        return BehaviorResult(risk_level="高", behaviors=[
            BehaviorHit(type="医疗行为建议", evidence="喝三七粉三个月就能停药", confidence=0.9)])


class SlowJudge:
    last_debate_trace = {"debated": False}

    def judge(self, text, title=""):
        time.sleep(0.3)
        return EvidenceVerdict(label="假", confidence=0.9, reasoning="与卫健委说法相反",
                               used_evidence=[1]), [ev()]


class FakeWriter:
    def __init__(self):
        self.called = 0

    def write(self, **kw):
        self.called += 1
        self.kw = kw
        return finalize(RebuttalOut(text="我查了一下，国家卫健委说降压药不会成瘾，可不能自己停药。", sources=["国家卫生健康委员会"]),
                        kw["evidence"])


def test_checker_runs_both_filters_in_parallel_and_decides():
    w = FakeWriter()
    t0 = time.perf_counter()
    r = Checker(cfg(), behavior=SlowBehavior(), judge=SlowJudge(), writer=w).check(DOC)
    assert time.perf_counter() - t0 < 0.55                    # 两路各 0.3 秒，并行才可能 < 0.6
    assert r.decision.priority == "尽快劝阻"                    # 假 × 高
    assert r.verdicts[0].label == "假" and r.behavior.risk_level == "高"
    assert w.called == 1 and r.rebuttal.endswith(REBUTTAL_ENDING)
    assert w.kw["priority"] == "尽快劝阻"                       # 档位传给文案，决定语气轻重（D51）
    md = to_markdown(r)
    for part in ("建议：尽快劝阻", "## 风险话术", "喝三七粉三个月就能停药", "## 证据链",
                 "国家卫生健康委员会", "## 发给爸妈的话", "不构成医疗建议"):
        assert part in md


class BrokenBehavior:
    def screen(self, text, title=""):
        raise LLMError("超时")


def test_behavior_failure_does_not_block_and_is_recorded():
    r = Checker(cfg(), behavior=BrokenBehavior(), judge=SlowJudge(), writer=FakeWriter()).check(DOC)
    assert r.verdicts[0].label == "假"                        # 认知面照常
    assert r.behavior.risk_level == "低"
    assert any("话术识别失败" in n for n in r.notes)            # 留痕，不装作真的没命中


class TrueJudge(SlowJudge):
    def judge(self, text, title=""):
        return EvidenceVerdict(label="真", confidence=0.95, reasoning="一致"), [ev()]


class CleanBehavior:
    def screen(self, text, title=""):
        return BehaviorResult(risk_level="低", behaviors=[])


def test_true_and_low_risk_gives_fixed_message_without_llm():
    """D51：子女每次都要拿到一段能发给爸妈的话；「真 × 低」不调模型，给固定的一句。"""
    from src.report.rebuttal import SAFE_MESSAGE
    w = FakeWriter()
    r = Checker(cfg(), behavior=CleanBehavior(), judge=TrueJudge(), writer=w).check(DOC)
    assert w.called == 0 and r.rebuttal == SAFE_MESSAGE and r.decision.priority == "无需处理"
    assert r.rebuttal.endswith(REBUTTAL_ENDING)
    assert "## 发给爸妈的话" in to_markdown(r)


# ---------------------------------------------------------------------------
# 内容接入
# ---------------------------------------------------------------------------
WECHAT_PAGE = """<html><head><meta property="og:title" content="降压药吃上就停不下来？老中医教你一招" />
<script>var nickname = "健康养生堂"; var msg_title = '降压药吃上就停不下来？';</script></head>
<body><div class="rich_media_content" id="js_content">
<p>很多人一查出高血压就开始吃降压药。</p><p>其实喝三七粉三个月就能停药。</p>
<p>警惕虚假广告，这句是正文不能删。</p>
<p>赶紧转给家里老人！</p><p>推荐阅读</p><p>震惊！这种水果千万别吃</p><p>长按识别二维码关注</p>
</div><script>var x=1;</script></body></html>"""


def test_parse_wechat_extracts_title_account_and_body():
    got = parse_wechat(WECHAT_PAGE)
    assert got["title"] == "降压药吃上就停不下来？老中医教你一招"
    assert got["account"] == "健康养生堂"
    assert "喝三七粉三个月就能停药" in got["text"]


def test_strip_noise_cuts_tail_but_keeps_body_mentions():
    text = parse_wechat(WECHAT_PAGE)["text"]
    clean = strip_noise(text, ["广告", "推荐阅读", "长按识别"])
    assert "推荐阅读" not in clean and "这种水果千万别吃" not in clean   # 文末推广区块被截掉
    assert "警惕虚假广告" in clean                                    # 长句里的「广告」不误删
    assert "赶紧转给家里老人" in clean                                 # 话术是证据，不能当噪声删


def test_fetch_document_wechat_and_blocked_page():
    c = cfg()
    doc = fetch_document("https://mp.weixin.qq.com/s/abc", c, fetcher=lambda u: WECHAT_PAGE)
    assert doc.account == "健康养生堂" and doc.url.endswith("/abc")
    blocked = "<html><body>环境异常 完成验证后即可继续访问</body></html>"
    with pytest.raises(IngestError, match="粘贴正文"):
        fetch_document("https://mp.weixin.qq.com/s/xyz", c, fetcher=lambda u: blocked)


def test_fetch_document_network_error_suggests_paste():
    def boom(u):
        raise ConnectionError("down")
    with pytest.raises(IngestError, match="粘贴正文"):
        fetch_document("https://example.com/a", cfg(), fetcher=boom)


def test_report_lists_only_used_evidence_with_original_numbers():
    """D52：只列判定用到的证据，保留原编号（对得上理由里的「证据[2]」），其余只给条数。"""
    from src.models import CheckReport, CognitiveVerdict
    evs = [ev(), ev(), ev()]
    evs[1].title, evs[0].title, evs[2].title = "用到的那条", "无关一", "无关二"
    cv = CognitiveVerdict(claim_id=0, claim_text="x", label="假", evidence=evs, used_evidence=[2])
    md = to_markdown(CheckReport(doc_id="d", title="t", verdicts=[cv]))
    assert "2. **" in md and "用到的那条" in md
    assert "无关一" not in md and "无关二" not in md
    assert "另检索到 2 条参考资料" in md


def test_report_falls_back_to_top_evidence_when_none_marked():
    from src.models import CheckReport, CognitiveVerdict
    cv = CognitiveVerdict(claim_id=0, claim_text="x", label="假", evidence=[ev()] * 5, used_evidence=[])
    md = to_markdown(CheckReport(doc_id="d", title="t", verdicts=[cv]))
    assert "没有标注具体用了哪几条" in md and "另检索到 2 条" in md
