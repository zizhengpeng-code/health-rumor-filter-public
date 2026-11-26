"""评测指标。全部手写实现，不依赖 sklearn，便于单测与复现。

指标口径（对应 CLAUDE.md「数据与评测」一节）：
- 认知面：准确率、宏 F1、混淆矩阵、各 topic 分数
- 行为面：各行为类型的精确率/召回率/F1，风险等级准确率
- 风险矩阵：对"尽快劝阻"类别的召回率（漏放比误伤更严重）
- 成本：每篇平均耗时、平均 token 数
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from src.models import BEHAVIOR_TYPES, COGNITIVE_LABELS, RISK_LEVELS, TOPICS


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """行=真实，列=预测。"""
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def per_class_prf(y_true: Sequence[str], y_pred: Sequence[str],
                  labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """每个类别的精确率/召回率/F1/支持数。分母为 0 时该项记 0。"""
    out: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": tp + fn,
        }
    return out


def present_labels(y_true: Sequence[str], labels: Sequence[str]) -> list[str]:
    """金标准中实际出现过的类别。"""
    seen = set(y_true)
    return [lab for lab in labels if lab in seen]


def absent_labels(y_true: Sequence[str], labels: Sequence[str]) -> list[str]:
    """金标准中一条都没有的类别。要单独报出来——否则评测集的覆盖缺口会被埋掉。"""
    seen = set(y_true)
    return [lab for lab in labels if lab not in seen]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str],
             skip_absent: bool = True) -> float:
    """宏 F1：各类别 F1 的算术平均。

    skip_absent=True（默认）时只对金标准中实际出现过的类别求平均。
    理由：一个在评测集里一条样本都没有的类别，F1 恒为 0，把它算进去
    会让所有配置的分数一起被稀释，消融对比失去分辨力；覆盖缺口由
    absent_labels 单独报告，不靠分数表达。
    """
    prf = per_class_prf(y_true, y_pred, labels)
    used = present_labels(y_true, labels) if skip_absent else list(labels)
    if not used:
        return 0.0
    return round(sum(prf[lab]["f1"] for lab in used) / len(used), 4)


def recall_for(y_true: Sequence[str], y_pred: Sequence[str], positive: str) -> float:
    """指定正类的召回率。"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    return round(tp / (tp + fn), 4) if (tp + fn) else 0.0


# ---------------------------------------------------------------------------
# 多标签（行为面）
# ---------------------------------------------------------------------------
def multilabel_prf(y_true: Sequence[Iterable[str]], y_pred: Sequence[Iterable[str]],
                   labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """行为面各类型的精确率/召回率/F1。样本为标签集合。"""
    ts = [set(x) for x in y_true]
    ps = [set(x) for x in y_pred]
    out: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(ts, ps) if lab in t and lab in p)
        fp = sum(1 for t, p in zip(ts, ps) if lab not in t and lab in p)
        fn = sum(1 for t, p in zip(ts, ps) if lab in t and lab not in p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": tp + fn,
        }
    return out


def multilabel_macro_f1(y_true: Sequence[Iterable[str]], y_pred: Sequence[Iterable[str]],
                        labels: Sequence[str], skip_absent: bool = True) -> float:
    prf = multilabel_prf(y_true, y_pred, labels)
    flat = [lab for s in y_true for lab in set(s)]
    used = [lab for lab in labels if lab in set(flat)] if skip_absent else list(labels)
    if not used:
        return 0.0
    return round(sum(prf[lab]["f1"] for lab in used) / len(used), 4)


# ---------------------------------------------------------------------------
# 分组（按 topic）
# ---------------------------------------------------------------------------
def group_accuracy(topics: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, Any]]:
    """各 topic 的准确率与样本数。样本太少的 topic 仍列出，便于发现问题。"""
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tp, t, p in zip(topics, y_true, y_pred):
        buckets[tp].append((t, p))
    out: dict[str, dict[str, Any]] = {}
    for tp in TOPICS:
        items = buckets.get(tp)
        if not items:
            continue
        out[tp] = {
            "n": len(items),
            "accuracy": round(sum(a == b for a, b in items) / len(items), 4),
        }
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def cognitive_metrics(samples: list[Any]) -> dict[str, Any]:
    """samples: 需含 .topic / .cognitive_label(金) / .pred_cognitive(预测)。"""
    y_true = [s.cognitive_label for s in samples]
    y_pred = [s.pred_cognitive for s in samples]
    topics = [s.topic for s in samples]
    return {
        "n": len(samples),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": macro_f1(y_true, y_pred, COGNITIVE_LABELS),
        "per_class": per_class_prf(y_true, y_pred, COGNITIVE_LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, COGNITIVE_LABELS),
        "by_topic": group_accuracy(topics, y_true, y_pred),
        # 评测集覆盖缺口：这些类别一条样本都没有，宏 F1 未计入
        "labels_absent": absent_labels(y_true, COGNITIVE_LABELS),
    }


def behavior_metrics(samples: list[Any]) -> dict[str, Any]:
    """行为面：各类型 PRF + 风险等级准确率。"""
    yt = [s.behaviors for s in samples]
    yp = [s.pred_behaviors for s in samples]
    lv_true = [s.behavior_level for s in samples]
    lv_pred = [s.pred_behavior_level for s in samples]
    return {
        "n": len(samples),
        "macro_f1": multilabel_macro_f1(yt, yp, BEHAVIOR_TYPES),
        "per_type": multilabel_prf(yt, yp, BEHAVIOR_TYPES),
        "risk_level_accuracy": round(accuracy(lv_true, lv_pred), 4),
        "risk_level_per_class": per_class_prf(lv_true, lv_pred, RISK_LEVELS),
        "types_absent": [t for t in BEHAVIOR_TYPES if not any(t in set(s) for s in yt)],
    }


def risk_matrix_metrics(samples: list[Any]) -> dict[str, Any]:
    """风险矩阵：重点是"尽快劝阻"的召回率——漏掉该尽快劝阻的最危险。"""
    y_true = [s.priority for s in samples]
    y_pred = [s.pred_priority for s in samples]
    return {
        "n": len(samples),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": macro_f1(y_true, y_pred, ["尽快劝阻", "提醒留意", "告知误区", "无需处理"]),
        "recall_尽快劝阻": recall_for(y_true, y_pred, "尽快劝阻"),
        "recall_提醒留意": recall_for(y_true, y_pred, "提醒留意"),
        "distribution_true": dict(Counter(y_true)),
        "distribution_pred": dict(Counter(y_pred)),
    }


def cost_metrics(samples: list[Any]) -> dict[str, Any]:
    n = len(samples) or 1
    cost = sum(getattr(s, "cost_usd", 0.0) for s in samples)
    lat = sum(getattr(s, "latency_s", 0.0) for s in samples)
    toks = sum(getattr(s, "tokens", 0) for s in samples)
    return {
        "n": len(samples),
        "total_cost_usd": round(cost, 6),
        "avg_cost_usd": round(cost / n, 6),
        "avg_latency_s": round(lat / n, 3),
        "avg_tokens": round(toks / n, 1),
        "cost_per_1000_usd": round(cost / n * 1000, 4),
    }


def summarize(samples: list[Any], with_behavior: bool = True,
              with_risk: bool = True, with_retrieval: bool = False) -> dict[str, Any]:
    """汇总全部指标。"""
    out: dict[str, Any] = {
        "cognitive": cognitive_metrics(samples),
        "cost": cost_metrics(samples),
    }
    if with_behavior:
        out["behavior"] = behavior_metrics(samples)
    if with_retrieval:
        out["retrieval"] = retrieval_metrics(samples)
    if with_risk:
        out["risk_matrix"] = risk_matrix_metrics(samples)
    return out


def retrieval_metrics(samples: list[Any]) -> dict[str, Any]:
    """检索层指标。

    这一层不评"检索得对不对"（没有人工标注的黄金证据集），只评**可观测的形态**：
    每条主张拿到几条证据、来自哪些通道、两路各召回多少、耗时多少。
    四行 retrieve-* 子消融靠这些数解释"分数为什么变"——只报最终准确率的话，
    涨了跌了都说不清是哪一步造成的。
    """
    n_ev = [int(getattr(s, "n_evidence", 0) or 0) for s in samples]
    chans: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for s in samples:
        chans.update(getattr(s, "evidence_channels", []) or [])
        matched.update(m for m in (getattr(s, "matched_by", []) or []) if m)

    total_ev = sum(n_ev)
    modes = Counter(getattr(s, "retrieve_mode", "") or "" for s in samples)
    traces = [getattr(s, "retrieval_trace", {}) or {} for s in samples]
    degraded = sum(1 for t in traces if t.get("rerank_degraded"))
    # 各阶段命中数（bm25 / vector / 融合后）——用于看两路是否真的都在出力
    def _avg(key: str) -> float:
        vals = [int(t.get(key) or 0) for t in traces]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "n": len(samples),
        "mode": modes.most_common(1)[0][0] if modes else "",
        "avg_evidence": round(total_ev / len(samples), 2) if samples else 0.0,
        "claims_with_evidence": sum(1 for x in n_ev if x > 0),
        "claims_without_evidence": sum(1 for x in n_ev if x == 0),
        "total_evidence": total_ev,
        "channel_distribution": dict(chans),
        # 可复现比例：kb / manual 是冻结快照，site / websearch 是活检索
        "reproducible_ratio": round(
            sum(v for k, v in chans.items() if k in REPRODUCIBLE_CHANNELS) / total_ev, 4
        ) if total_ev else 0.0,
        "matched_by": dict(matched),
        "avg_hits_bm25": _avg("n_bm25"),
        "avg_hits_vector": _avg("n_vector"),
        "avg_fused": _avg("n_fused"),
        "rerank_degraded_samples": degraded,
        # 重排只改顺序、不改候选集大小。没有这两个数，hybrid 与 hybrid-rerank
        # 在表上会长得一模一样，读表的人无法判断重排是否真的生效了。
        "rerank_moved_top1_samples": sum(
            1 for t in traces if t.get("rerank_moved_top1")),
        "avg_rerank_displaced": _avg("rerank_displaced"),
    }


# ---------------------------------------------------------------------------
# 可复现性：证据来自哪一层检索通道
# ---------------------------------------------------------------------------
REPRODUCIBLE_CHANNELS = {"kb", "manual"}


def channel_distribution(evidence_lists: Sequence[Iterable[Any]]) -> dict[str, Any]:
    """统计每条主张的证据分别来自哪个通道。

    kb / manual 是可复现的（本地冻结快照 / 人工录入）；
    site / websearch 是活检索，今天和下月返回不同。
    这个比例是"结果可复现"论证的关键数据——必须报出来，不能藏。
    """
    counts: Counter[str] = Counter()
    per_claim: list[dict[str, int]] = []
    for evs in evidence_lists:
        local: Counter[str] = Counter()
        for e in evs:
            ch = getattr(e, "channel", "") or "unknown"
            counts[ch] += 1
            local[ch] += 1
        per_claim.append(dict(local))

    total = sum(counts.values())
    repro = sum(n for ch, n in counts.items() if ch in REPRODUCIBLE_CHANNELS)
    claims_with_evidence = sum(1 for c in per_claim if c)
    claims_kb_only = sum(1 for c in per_claim if c and set(c) <= REPRODUCIBLE_CHANNELS)
    return {
        "total_evidence": total,
        "by_channel": dict(counts),
        "reproducible_evidence": repro,
        "reproducible_ratio": round(repro / total, 4) if total else 0.0,
        "n_claims": len(per_claim),
        "claims_with_evidence": claims_with_evidence,
        "claims_kb_only": claims_kb_only,
        "claims_kb_only_ratio": (round(claims_kb_only / len(per_claim), 4)
                                 if per_claim else 0.0),
        "per_claim": per_claim,
    }
