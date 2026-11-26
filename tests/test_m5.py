"""M5：信源评分 + 分工式辩论（v2，D36）的最小测试。全部用假模型，不联网、不花钱。"""
from __future__ import annotations

from pathlib import Path

from src.cognitive.debate import DebateJudge, check_citations, split_piles, vote
from src.cognitive.source_score import SourceScorer, SourceScoredJudge, format_scored_evidence
from src.config import load_config
from src.models import AdvocateOut, Evidence, EvidenceVerdict, SourceCheckOut

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """按顺序吐出预设回复，并记下每次收到的提示词，便于断言「谁看到了什么」。"""

    model = "fake"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.history = []
        self.prompts = []
        self.stages = []

    def complete(self, prompt, system, schema, *, stage="llm", prompt_version=""):
        self.prompts.append(prompt)
        self.stages.append(stage)
        data = self.payloads.pop(0)
        return schema.model_validate(data).model_dump(mode="json")


class FakeRetriever:
    mode = "bm25"
    last_trace = {"n_bm25": 1, "n_vector": 0, "n_fused": 1}

    def __init__(self, evs, client=None):
        self.evs = evs
        self.client = client

    def retrieve(self, claim):
        return [e.model_copy() for e in self.evs]


def cfg():
    c = load_config(ROOT / "config.yaml")
    c["_overrides"] = {"retrieve": True, "source_score": True, "debate": True, "review": True}
    return c


def ev(i, snippet, domain="nhc.gov.cn", tier=1):
    return Evidence(id=i, claim_id=0, title="权威科普", url=f"https://{domain}/{i}.html",
                    source_name="国家卫生健康委员会", domain=domain, snippet=snippet, tier=tier,
                    channel="kb", matched_by="bm25", chunk_id=f"{i}-c1")


def quick(label, conf):
    return {"label": label, "confidence": conf, "reasoning": "快判", "used_evidence": [1]}


def adv(points=(), rebuttal="", concede=False):
    return {"points": [{"point": p, "cited_evidence": list(c)} for p, c in points],
            "rebuttal": rebuttal, "concede": concede}


def judge_out(label, conf=0.8, cont=False):
    return {"label": label, "confidence": conf, "reasoning": "裁决", "decisive_evidence": [1],
            "continue_debate": cont}


# ---------------------------------------------------------------------------
# 信源评分（M4/M5 共用）
# ---------------------------------------------------------------------------
def test_source_scorer_uses_tier_and_domain_rules():
    assert SourceScorer(cfg()).score(ev(0, "x", tier=4)) >= 0.9


def test_source_scored_judge_adds_scores_before_judging():
    c = cfg()
    inner = FakeRetriever([ev(0, "权威证据片段")])
    inner.client = FakeLLM([quick("真", 0.9)])
    judge = SourceScoredJudge(c, inner=inner, scorer=SourceScorer(c))
    verdict, evs = judge.judge("正文", "标题")
    assert isinstance(verdict, EvidenceVerdict) and verdict.label == "真"
    assert evs[0].score > 0 and "avg_source_score" in judge.last_trace
    assert "source_score=" in format_scored_evidence(evs)


# ---------------------------------------------------------------------------
# 分工靠 schema 与代码强制
# ---------------------------------------------------------------------------
def test_source_checker_cannot_return_a_verdict():
    """信源核查专家多写了结论也会被丢掉——它的输出契约里根本没有这个字段。"""
    out = SourceCheckOut.model_validate(
        {"assessments": [{"id": 1, "relevant": True, "stance": "支持"}], "label": "假"})
    assert "label" not in out.model_dump()


def test_split_piles_gives_sides_different_evidence():
    evs = [ev(i, f"s{i}") for i in range(4)]
    a = {1: {"relevant": True, "stance": "支持"}, 2: {"relevant": True, "stance": "反驳"},
         3: {"relevant": False, "stance": "支持"}}             # 第 4 条核查漏判 -> 相关·中立
    pro, con, relevant = split_piles(evs, a)
    assert pro == [1, 4] and con == [2, 4] and relevant == [1, 2, 4]


def test_check_citations_drops_out_of_range_ids():
    out = AdvocateOut.model_validate(adv([("论点A", [1, 2]), ("论点B", [9])]))
    out, dropped = check_citations(out, allowed=[1])
    assert out.points[0].cited_evidence == [1]
    assert out.points[1].cited_evidence == []                 # 渲染时标为「推测」
    assert sorted(dropped) == [2, 9]


def test_vote_tie_takes_more_severe_label():
    assert vote(["真", "假"]) == ("假", 0.5)
    assert vote(["真", "真", "部分属实"])[0] == "真"


# ---------------------------------------------------------------------------
# 两阶段流程
# ---------------------------------------------------------------------------
def test_confident_quick_verdict_skips_debate():
    c = cfg()
    debate = FakeLLM([])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("真", 0.95)])),
                    debate_client=debate, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    assert verdict.label == "真"
    assert debate.prompts == []                               # 一次辩论调用都没有
    assert j.last_debate_trace["debated"] is False and j.last_trace["debated"] is False


def test_uncertain_label_triggers_debate_even_if_confident():
    c = cfg()
    debate = FakeLLM([{"assessments": []}, adv(concede=True), adv([("p", [1])])])
    judge = FakeLLM([judge_out("真")])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("部分属实", 0.95)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    j.judge("正文", "标题")
    assert j.last_debate_trace["debated"] is True


def test_full_debate_each_side_only_sees_its_own_evidence():
    c = cfg()
    evs = [ev(0, "支持片段AAA"), ev(1, "反驳片段BBB"), ev(2, "无关片段CCC")]
    debate = FakeLLM([
        {"assessments": [{"id": 1, "relevant": True, "stance": "支持"},
                         {"id": 2, "relevant": True, "stance": "反驳"},
                         {"id": 3, "relevant": False, "stance": "中立"}]},
        adv([("证据1支持", [1, 2])]),                        # 正方越界引用了 2
        adv([("证据2反驳", [2])]),
        adv([("坚持", [1])], rebuttal="反方以偏概全"),
        adv([("证据2说明缺关键条件", [2])], rebuttal="正方误读"),
    ])
    # 第 1 轮后裁判要求继续，第 2 轮（最后一轮）结案
    judge = FakeLLM([judge_out("真", 0.6, cont=True), judge_out("部分属实", 0.7)])
    j = DebateJudge(cfg=c, retriever=FakeRetriever(evs, FakeLLM([quick("真", 0.6)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")

    assert debate.stages == ["debate_source_check", "debate_pro", "debate_con",
                             "debate_pro", "debate_con"]
    pro_r1, con_r1 = debate.prompts[1], debate.prompts[2]
    assert "支持片段AAA" in pro_r1 and "反驳片段BBB" not in pro_r1
    assert "反驳片段BBB" in con_r1 and "支持片段AAA" not in con_r1
    assert all("无关片段CCC" not in p for p in debate.prompts[1:])   # 无关证据谁都拿不到
    assert "无关片段CCC" not in judge.prompts[0]
    t = j.last_debate_trace
    assert t["dropped_citations"]["pro"] == [2]
    assert t["terminated_by"] == "round_limit" and len(t["rounds"]) == 2
    assert t["judge_steps"] == [{"round": 1, "continue": True, "label": "真"}]
    assert "第 1 轮" in judge.prompts[0] and "最后一轮" in judge.prompts[1]
    assert verdict.label == "部分属实" and t["final_label"] == "部分属实"


def test_judge_breaks_after_first_round_when_clear():
    """D45：结论已清楚时裁判第 1 轮就结案，不再辩第 2 轮（Liang 等：强行多辩反而变差）。"""
    c = cfg()
    debate = FakeLLM([{"assessments": [{"id": 1, "relevant": True, "stance": "支持"},
                                       {"id": 2, "relevant": True, "stance": "反驳"}]},
                      adv([("证据1支持", [1])]), adv([("细节问题", [2])])])
    judge = FakeLLM([judge_out("真", 0.9)])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "a"), ev(1, "b")], FakeLLM([quick("真", 0.6)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    t = j.last_debate_trace
    assert debate.stages == ["debate_source_check", "debate_pro", "debate_con"]   # 没有第 2 轮
    assert len(judge.prompts) == 1                                                 # 裁判只调用一次
    assert t["terminated_by"] == "judge_break" and verdict.label == "真"


def test_last_round_is_forced_to_close_even_if_judge_wants_more():
    c = cfg()
    debate = FakeLLM([{"assessments": []}, adv([("p", [1])]), adv([("q", [1])]),
                      adv([("p", [1])]), adv([("q", [1])])])
    judge = FakeLLM([judge_out("真", cont=True), judge_out("假", cont=True)])   # 第 2 次也想继续
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("真", 0.5)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    assert verdict.label == "假" and j.last_debate_trace["terminated_by"] == "round_limit"


def test_concede_ends_debate_early():
    c = cfg()
    debate = FakeLLM([{"assessments": [{"id": 1, "relevant": True, "stance": "反驳"}]},
                      adv(concede=True), adv([("证据1反驳", [1])])])
    judge = FakeLLM([judge_out("假", 0.9)])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("真", 0.5)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    assert len(j.last_debate_trace["rounds"]) == 1
    assert j.last_debate_trace["terminated_by"] == "consensus"
    assert verdict.label == "假"


def test_judge_majority_vote_sets_confidence():
    c = cfg()
    c["cognitive"]["debate"]["judge_runs"] = 3
    debate = FakeLLM([{"assessments": []}, adv(concede=True), adv([("p", [1])])])
    judge = FakeLLM([judge_out("假"), judge_out("真"), judge_out("真")])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("真", 0.5)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    assert verdict.label == "真" and verdict.confidence == round(2 / 3, 4)
    assert j.last_debate_trace["judge_votes"] == ["假", "真", "真"]


def test_debate_never_forces_unverifiable():
    """v1 的坑：审核卡住就改判「无法核查」，按风险矩阵会被放行。v2 结论只来自裁判。"""
    c = cfg()
    debate = FakeLLM([{"assessments": []}, adv([("p", [1])]), adv([("q", [1])]),
                      adv([("p", [1])]), adv([("q", [1])])])
    judge = FakeLLM([judge_out("假", 0.6)])
    j = DebateJudge(cfg=c, retriever=FakeRetriever([ev(0, "x")], FakeLLM([quick("无法核查", 0.4)])),
                    debate_client=debate, judge_client=judge, scorer=SourceScorer(c))
    verdict, _ = j.judge("正文", "标题")
    assert verdict.label == "假"
