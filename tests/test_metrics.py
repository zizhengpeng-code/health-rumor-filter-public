"""指标实现的正确性测试。

关键点：这些断言都是手工算出来的，不是跑一遍程序把结果抄下来。
只有这样，指标实现有 bug 时测试才会失败。
"""
from __future__ import annotations

from eval import metrics

LABELS4 = ["真", "假", "部分属实", "无法核查"]


def test_accuracy_hand_computed():
    yt = ["假", "真", "假", "部分属实"]
    yp = ["假", "假", "真", "部分属实"]
    assert metrics.accuracy(yt, yp) == 0.5


def test_accuracy_empty_is_zero():
    assert metrics.accuracy([], []) == 0.0


def test_per_class_prf_hand_computed():
    yt = ["假", "真", "假", "部分属实"]
    yp = ["假", "假", "真", "部分属实"]
    prf = metrics.per_class_prf(yt, yp, LABELS4)

    # 假：tp=1(第0条) fp=1(第1条) fn=1(第2条) -> P=R=F1=0.5
    assert prf["假"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    # 真：tp=0 fp=1(第2条) fn=1(第1条) -> 全 0
    assert prf["真"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    # 部分属实：tp=1 fp=0 fn=0 -> 全 1
    assert prf["部分属实"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
    # 无法核查：从未出现 -> 全 0
    assert prf["无法核查"]["f1"] == 0.0


def test_macro_f1_hand_computed():
    # 金标准里出现的类别：假、真、部分属实（无法核查 0 条，不计入）
    # F1: 假=0.5, 真=0.0, 部分属实=1.0 -> (0.5+0.0+1.0)/3 = 0.5
    yt = ["假", "真", "假", "部分属实"]
    yp = ["假", "假", "真", "部分属实"]
    assert metrics.macro_f1(yt, yp, LABELS4) == 0.5


def test_macro_f1_skip_absent_false_includes_zero_support():
    # 不跳过零支持类别时，"无法核查"的 F1=0 会把宏 F1 拉低到 0.375
    yt = ["假", "真", "假", "部分属实"]
    yp = ["假", "假", "真", "部分属实"]
    assert metrics.macro_f1(yt, yp, LABELS4, skip_absent=False) == 0.375


def test_absent_labels_reported_not_hidden():
    yt = ["假", "真"]
    assert metrics.absent_labels(yt, LABELS4) == ["部分属实", "无法核查"]
    assert metrics.present_labels(yt, LABELS4) == ["真", "假"]


def test_macro_f1_perfect_is_one():
    yt = ["假", "真", "部分属实", "无法核查"]
    assert metrics.macro_f1(yt, yt, LABELS4) == 1.0


def test_confusion_matrix_shape_and_sum():
    yt = ["假", "真", "假", "部分属实"]
    yp = ["假", "假", "真", "部分属实"]
    cm = metrics.confusion_matrix(yt, yp, LABELS4)
    assert set(cm) == set(LABELS4)
    assert cm["假"]["假"] == 1 and cm["假"]["真"] == 1
    assert cm["真"]["假"] == 1
    assert cm["部分属实"]["部分属实"] == 1
    total = sum(v for row in cm.values() for v in row.values())
    assert total == len(yt)


def test_recall_for_only_counts_positive_class():
    yt = ["尽快劝阻", "尽快劝阻", "无需处理", "无需处理"]
    yp = ["尽快劝阻", "无需处理", "尽快劝阻", "无需处理"]
    # 尽快劝阻：tp=1 fn=1 -> 0.5；无需处理被误判为尽快劝阻不该影响正类召回
    assert metrics.recall_for(yt, yp, "尽快劝阻") == 0.5


def test_multilabel_prf_hand_computed():
    yt = [{"医疗行为建议"}, {"转发诱导", "恐惧诉求"}, set()]
    yp = [{"医疗行为建议"}, {"转发诱导"}, {"消费导流"}]
    prf = metrics.multilabel_prf(yt, yp, ["医疗行为建议", "转发诱导", "恐惧诉求", "消费导流", "金钱相关"])
    assert prf["医疗行为建议"]["f1"] == 1.0
    # 转发诱导：tp=1 fn=0 fp=0 -> 1.0
    assert prf["转发诱导"]["f1"] == 1.0
    # 恐惧诉求：tp=0 fn=1 fp=0 -> 召回 0
    assert prf["恐惧诉求"]["recall"] == 0.0
    # 消费导流：tp=0 fn=0 fp=1 -> 精确率 0
    assert prf["消费导流"]["precision"] == 0.0


def test_group_accuracy_by_topic():
    topics = ["慢病", "慢病", "饮食", "饮食"]
    yt = ["假", "真", "假", "真"]
    yp = ["假", "假", "假", "真"]
    g = metrics.group_accuracy(topics, yt, yp)
    assert g["慢病"] == {"n": 2, "accuracy": 0.5}
    assert g["饮食"] == {"n": 2, "accuracy": 1.0}
    assert "用药" not in g  # 没样本的 topic 不出现


def test_cost_metrics_average():
    class S:
        cost_usd = 0.002
        latency_s = 2.0
        tokens = 100

    rows = [S(), S(), S()]
    c = metrics.cost_metrics(rows)
    assert c["avg_latency_s"] == 2.0
    assert c["avg_tokens"] == 100.0
    assert c["cost_per_1000_usd"] == 2.0  # 0.002 * 1000


# ---------------------------------------------------------------------------
# 可复现性指标
# ---------------------------------------------------------------------------
def _ev(channel):
    class E:
        pass
    e = E()
    e.channel = channel
    return e


def test_channel_distribution_counts_and_ratio():
    lists = [
        [_ev("kb"), _ev("kb")],                 # 全本地库 -> 可复现
        [_ev("kb"), _ev("site")],               # 混合
        [_ev("websearch")],                     # 全靠兜底
        [],                                     # 无证据
    ]
    d = metrics.channel_distribution(lists)
    assert d["total_evidence"] == 5          # 2 + 2 + 1 + 0
    assert d["by_channel"] == {"kb": 3, "site": 1, "websearch": 1}
    assert d["reproducible_evidence"] == 3
    assert d["reproducible_ratio"] == 0.6    # 3/5
    assert d["n_claims"] == 4
    assert d["claims_with_evidence"] == 3
    assert d["claims_kb_only"] == 1
    assert d["claims_kb_only_ratio"] == 0.25


def test_channel_distribution_empty_input():
    d = metrics.channel_distribution([])
    assert d["total_evidence"] == 0
    assert d["reproducible_ratio"] == 0.0
    assert d["claims_kb_only_ratio"] == 0.0


def test_channel_distribution_manual_counts_as_reproducible():
    d = metrics.channel_distribution([[_ev("manual")]])
    assert d["reproducible_ratio"] == 1.0


def test_channel_distribution_unknown_channel_not_reproducible():
    d = metrics.channel_distribution([[_ev(""), _ev("somethingelse")]])
    assert d["reproducible_ratio"] == 0.0
