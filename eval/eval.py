"""评测入口。

用法：
    python eval/eval.py --config baseline --split dev
    python eval/eval.py --config all --split dev          # 跑全部消融配置并出对比表
    python eval/eval.py --config full --split dev --model stub:oracle   # 自检指标实现

M1 验收标准：用一个"固定答案的假模型"跑出全部指标，不消耗任何 token。

假模型（--model）：
    stub:fixed    永远返回同一个标签（默认"假"）——验证指标能反映真实分布
    stub:oracle   返回金标准——指标应全为 1.0，用于自检指标实现有无 bug
    stub:random   固定种子乱猜——指标应接近随机基线
    real          走真实 LLM 全流程（M2 起可用）

注意：--split test 只用于最终评测，禁止用于调提示词/阈值（config.eval.forbid_test_tuning）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 注意：不要把 eval/ 目录加进 sys.path——否则 eval/eval.py 会遮蔽 eval 包本身，
# 导致 `import eval.eval` 报 "'eval' is not a package"。
# 以脚本方式运行时（python eval/eval.py），Python 会自动把脚本目录放进 sys.path。

try:                      # 作为包导入时（pytest / 其他模块调用）
    from . import metrics  # type: ignore[import-not-found]
except ImportError:       # 作为脚本直接运行时（python eval/eval.py）
    import metrics  # type: ignore[no-redef]

from src.behavior.rules import behavior_level_from_types  # noqa: E402
from src.config import Config, load_config, run_stamp  # noqa: E402
from src.llm import StubClient  # noqa: E402
from src.models import (  # noqa: E402
    BEHAVIOR_TYPES, COGNITIVE_LABELS, LabeledSample, Prediction,
)
from src.risk_matrix import decide  # noqa: E402
from src.store import Store  # noqa: E402


# ---------------------------------------------------------------------------
# 样本行（喂给 metrics 的统一结构）
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    id: str
    topic: str
    # 金标准
    cognitive_label: str
    behaviors: list[str]
    behavior_level: str
    priority: str
    # 预测
    pred_cognitive: str = "无法核查"
    pred_behaviors: list[str] = field(default_factory=list)
    pred_behavior_level: str = "低"
    pred_priority: str = "无需处理"
    pred_confidence: float = 0.0
    pred_reasoning: str = ""
    # 成本
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens: int = 0
    note: str = ""              # 失败原因，不参与打分
    # 检索（M4）
    n_evidence: int = 0
    evidence_channels: list[str] = field(default_factory=list)
    matched_by: list[str] = field(default_factory=list)
    retrieve_mode: str = ""
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    # 多智能体讨论（M5）
    debate_trace: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_split(cfg: Config, split: str, limit: int | None = None) -> list[LabeledSample]:
    path = Path(cfg.get_path("paths.splits_dir")) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到评测集：{path}（先跑 scripts 或手工准备 {split}.jsonl）")
    out: list[LabeledSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(LabeledSample.model_validate_json(line))
            if limit and len(out) >= limit:
                break
    return out


def gold_row(s: LabeledSample, cfg: Config) -> EvalRow:
    """把标注样本转成评测行，补齐行为等级与处置优先级。"""
    bcfg = cfg.get("behavior", {}) or {}
    level = s.behavior_level or behavior_level_from_types(s.behaviors, bcfg)
    prio = decide(s.cognitive_label, level, cfg.get("risk_matrix", {})).priority
    return EvalRow(
        id=s.id, topic=s.topic, cognitive_label=s.cognitive_label,
        behaviors=list(s.behaviors), behavior_level=level, priority=prio,
    )


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
class StubPredictor:
    """假模型预测器：零成本、确定性。"""

    def __init__(self, mode: str, cfg: Config):
        self.cfg = cfg
        self.stub = StubClient(mode=mode, seed=int(cfg.get_path("eval.seed", 20261002)))
        self.mode = mode

    def predict(self, s: LabeledSample, row: EvalRow) -> Prediction:
        raw = self.stub.predict(s.id, row.cognitive_label, row.behaviors)
        # 与 RealPredictor 同一口径：行为面关闭时一律视为未命中。否则 oracle 自检在 baseline 上
        # 会显示「尽快劝阻召回 1.0」，而真实 baseline 结构上只能是 0。
        if not (self.cfg.get("_overrides") or {}).get("behavior"):
            raw["behaviors"] = []
        level = behavior_level_from_types(raw["behaviors"], self.cfg.get("behavior", {}) or {})
        prio = decide(raw["cognitive_label"], level, self.cfg.get("risk_matrix", {})).priority
        return Prediction(
            id=s.id, cognitive_label=raw["cognitive_label"], behaviors=raw["behaviors"],
            behavior_level=level, priority=prio,
        )


class RealPredictor:
    """真实模型预测器。

    已实现：baseline（裸模型）、retrieve（检索+证据判断，含 4 种检索模式）、
    source_score（规则化信源评分）、debate/review（多智能体讨论与审核闭环）、
    behavior_screen（行为面滤网）。

    这里保留 UNIMPLEMENTED 机制：后续如果新增消融开关但尚未接入，仍可在开跑前显式拦截，
    避免把静默降级的结果误当成完整消融。
    """

    # 尚未实现的模块开关：一旦配置里打开就拒绝运行
    UNIMPLEMENTED: tuple[str, ...] = ()

    def __init__(self, cfg: Config, store: Any = None, run_id: str = "adhoc", *,
                 judge: Any = None, behavior: Any = None):
        """judge / behavior 可注入，供离线测试替换成假模型（不联网、不花钱）。"""
        from src.cognitive.baseline import BaselineJudge

        overrides = cfg.get("_overrides") or {}
        blockers = [k for k in self.UNIMPLEMENTED if overrides.get(k)]
        if blockers:
            raise NotImplementedError(
                f"配置里开启了尚未实现的模块 {blockers}。real 模式当前不会静默降级。"
            )
        self.cfg = cfg
        self.store = store
        self.run_id = run_id
        self.retrieval_used = bool(overrides.get("retrieve"))

        # retrieve 开 -> 用证据链路；source_score/debate/review 是在证据链路上继续叠加。
        if judge is not None:
            self.judge = judge
        elif self.retrieval_used and (overrides.get("debate") or overrides.get("review")):
            from src.cognitive.debate import DebateJudge
            mode = overrides.get("retrieve_mode")
            self.judge = DebateJudge.build(cfg, store=store, run_id=run_id, mode=mode)
        elif self.retrieval_used and overrides.get("source_score"):
            from src.cognitive.source_score import SourceScoredJudge
            mode = overrides.get("retrieve_mode")
            self.judge = SourceScoredJudge.build(cfg, store=store, run_id=run_id, mode=mode)
        elif self.retrieval_used:
            from src.cognitive.evidence_judge import EvidenceJudge
            mode = overrides.get("retrieve_mode")
            self.judge = EvidenceJudge.build(cfg, store=store, run_id=run_id, mode=mode)
        else:
            self.judge = BaselineJudge.build(cfg, store=store, run_id=run_id)
        # 兜底（D41）：证据不可用时退回裸模型判断。包在任何证据判断器外面，是独立的消融开关
        if judge is None and self.retrieval_used and overrides.get("fallback"):
            from src.cognitive.fallback import FallbackJudge
            self.judge = FallbackJudge.build(cfg, self.judge, store=store, run_id=run_id)

        # 行为面是独立开关：关掉时行为等级一律按"未命中"推导（消融表要靠这个差值说话）
        if behavior is not None:
            self.behavior = behavior
        elif overrides.get("behavior"):
            from src.behavior.screen import BehaviorScreen
            self.behavior = BehaviorScreen.build(cfg, store=store, run_id=run_id)
        else:
            self.behavior = None

    @property
    def retrieve_mode(self) -> str:
        return getattr(self.judge, "mode", "") if self.retrieval_used else ""

    @property
    def _clients(self) -> list[Any]:
        """所有会产生成本的 client，用于统计 token 与费用。"""
        clients = getattr(self.judge, "clients", None)
        out = list(clients) if clients is not None else [self.judge.client]
        if self.behavior is not None:
            out.append(self.behavior.client)
        return out

    def _drain(self, before: dict[int, int]) -> tuple[float, int]:
        """取出本轮新增的调用记录，累加费用与 token。

        成本记账是**附属信息**，不能因为它把整轮评测搞崩：拿不到 history 就记 0，
        而不是抛异常。真正的分数比成本重要得多。
        """
        cost = 0.0
        tokens = 0
        for c in self._clients:
            hist = getattr(c, "history", None)
            if not isinstance(hist, list):
                continue
            for r in hist[before.get(id(c), 0):]:
                cost += float(getattr(r, "cost_usd", 0.0) or 0.0)
                tokens += int(getattr(r, "prompt_tokens", 0) or 0) + \
                    int(getattr(r, "completion_tokens", 0) or 0)
        return cost, tokens

    def predict(self, s: LabeledSample, row: EvalRow) -> Prediction:
        import time

        from src.llm import LLMError

        before = {}
        for c in self._clients:
            hist = getattr(c, "history", None)
            before[id(c)] = len(hist) if isinstance(hist, list) else 0
        t0 = time.perf_counter()
        note = ""

        # --- 认知面 ---
        evs: list[Any] = []
        try:
            if self.retrieval_used:
                verdict, evs = self.judge.judge(s.text, s.title)
            else:
                verdict = self.judge.judge(s.text, s.title)
            label, conf = verdict.label, verdict.confidence
            reasoning = getattr(verdict, "reasoning", "") or ""
        except LLMError as exc:
            # 单条失败不该让整轮评测崩掉——记为"无法核查"并留下原因
            label, conf, note = "无法核查", 0.0, f"认知面调用失败：{exc}"
            reasoning = ""

        row.n_evidence = len(evs)
        row.evidence_channels = [e.channel for e in evs]
        row.matched_by = [e.matched_by for e in evs]
        row.retrieve_mode = self.retrieve_mode
        row.retrieval_trace = dict(getattr(self.judge, "last_trace", {}) or {})
        row.debate_trace = dict(getattr(self.judge, "last_debate_trace", {}) or {})

        # --- 行为面（独立开关）---
        hit_types: list[str] = []
        if self.behavior is not None:
            try:
                bres = self.behavior.screen(s.text, s.title)
                hit_types = bres.to_eval()
            except LLMError as exc:
                # 行为面失败不改变认知面结论，但要留下痕迹——否则会被当成"真的没命中"
                note = (note + "；" if note else "") + f"行为面调用失败：{exc}"

        latency = time.perf_counter() - t0
        cost, tokens = self._drain(before)

        level = behavior_level_from_types(hit_types, self.cfg.get("behavior", {}) or {})
        prio = decide(label, level, self.cfg.get("risk_matrix", {})).priority
        return Prediction(
            id=s.id, cognitive_label=label, behaviors=hit_types, behavior_level=level,
            priority=prio, confidence=float(conf or 0.0), reasoning=reasoning,
            cost_usd=cost, latency_s=latency, tokens=tokens, note=note,
        )


# ---------------------------------------------------------------------------
# 单次评测
# ---------------------------------------------------------------------------
def run_one(cfg: Config, config_name: str, split: str, model_spec: str,
            limit: int | None = None, verbose: bool = True,
            store: Any = None, run_id: str = "adhoc") -> dict[str, Any]:
    overrides = (cfg.get_path("eval.configs", {}) or {}).get(config_name)
    if overrides is None:
        raise KeyError(f"config.yaml 里没有消融配置 `{config_name}`；"
                       f"可选：{list((cfg.get_path('eval.configs', {}) or {}).keys())}")
    cfg["_overrides"] = dict(overrides)

    samples = load_split(cfg, split, limit=limit)
    rows = [gold_row(s, cfg) for s in samples]

    if model_spec.startswith("stub:"):
        predictor: Any = StubPredictor(model_spec.split(":", 1)[1], cfg)
    elif model_spec == "real":
        predictor = RealPredictor(cfg, store=store, run_id=run_id)
    else:
        raise ValueError(f"未知 --model：{model_spec}")

    for s, row in zip(samples, rows):
        p = predictor.predict(s, row)
        row.pred_cognitive = p.cognitive_label
        row.pred_behaviors = list(p.behaviors)
        row.pred_behavior_level = p.behavior_level
        row.pred_priority = p.priority
        row.pred_confidence = p.confidence
        row.pred_reasoning = p.reasoning
        row.cost_usd = p.cost_usd
        row.latency_s = p.latency_s
        row.tokens = p.tokens
        row.note = p.note

    with_behavior = bool(overrides.get("behavior", False))
    with_retrieval = bool(overrides.get("retrieve", False))
    result = metrics.summarize(rows, with_behavior=with_behavior, with_risk=True,
                               with_retrieval=with_retrieval)
    failed = [r.id for r in rows if getattr(r, "note", "")]
    result["meta"] = {
        "config_name": config_name, "split": split, "model": model_spec,
        "n_samples": len(rows), "overrides": overrides,
        "failed_samples": failed,
        "prompt_versions": _prompt_versions(cfg, model_spec),
        "retrieval_setup": _retrieval_setup(cfg) if with_retrieval else {},
    }
    # 逐样本落盘：消融表只能看总分，而"哪个模块修好了哪条错"要靠逐样本对照。
    # 没有这一层，报告里就只能写"涨了 3 个点"，说不出涨在哪。
    result["rows"] = [
        {
            "id": r.id, "topic": r.topic,
            "gold": r.cognitive_label, "pred": r.pred_cognitive,
            "ok": r.cognitive_label == r.pred_cognitive,
            "gold_behaviors": list(r.behaviors), "pred_behaviors": list(r.pred_behaviors),
            "gold_level": r.behavior_level, "pred_level": r.pred_behavior_level,
            "gold_priority": r.priority, "pred_priority": r.pred_priority,
            "confidence": r.pred_confidence, "reasoning": r.pred_reasoning,
            "cost_usd": round(r.cost_usd, 8), "latency_s": round(r.latency_s, 3),
            "tokens": r.tokens, "note": r.note,
            "n_evidence": r.n_evidence, "evidence_channels": list(r.evidence_channels),
            "matched_by": list(r.matched_by), "retrieve_mode": r.retrieve_mode,
            # 逐样本的检索轨迹也要落盘。只存汇总的话，事后想问
            # "这条主张当时召回了几条、两路各出了多少力、重排有没有动它"
            # 就只能重跑一遍（还要再花一次钱）。审计线索必须和结果一起存下来。
            "retrieval_trace": dict(r.retrieval_trace or {}),
            "debate_trace": dict(r.debate_trace or {}),
        }
        for r in rows
    ]
    if verbose:
        print_report(result)
    return result


def _real_blockers(cfg: Config, names: list[str]) -> dict[str, list[str]]:
    """哪些配置在 real 模式下跑不了，以及卡在哪个模块。"""
    all_cfgs = cfg.get_path("eval.configs", {}) or {}
    out: dict[str, list[str]] = {}
    for n in names:
        ov = all_cfgs.get(n) or {}
        bad = [k for k in RealPredictor.UNIMPLEMENTED if ov.get(k)]
        if bad:
            out[n] = bad
    return out


def _retrieval_setup(cfg: Config) -> dict[str, Any]:
    """记下本次用的知识库快照与检索模型（D44：同一配置会有本地/API、新库/旧库多个版本的结果）。"""
    kb = cfg.get("kb", {}) or {}
    out: dict[str, Any] = {
        "kb_path": Path(cfg.get_path("paths.kb_path") or "").name,
        "embed": f"{(kb.get('embed') or {}).get('backend')}:{(kb.get('embed') or {}).get('model')}",
        "rerank": f"{(kb.get('rerank') or {}).get('backend')}:{(kb.get('rerank') or {}).get('model')}",
    }
    try:
        from src.kb.store import KnowledgeBase
        with KnowledgeBase(cfg.get_path("paths.kb_path")) as k:
            snap = k.snapshot() or {}
        out.update(kb_version=snap.get("kb_version"), kb_hash=snap.get("content_hash"),
                   kb_docs=snap.get("doc_count"))
    except Exception:                                           # noqa: BLE001
        pass
    return out


def _prompt_versions(cfg: Config, model_spec: str) -> dict[str, str]:
    """记录本次运行用到的提示词版本，保证结果可回溯。假模型不涉及提示词。

    只记**实际会跑到**的模块——把没开启的模块的版本也写进去，会让人误以为它跑过了。
    """
    if model_spec != "real":
        return {}
    from src.prompts import load_prompt

    names = ["baseline_judge"]
    ov = cfg.get("_overrides") or {}
    if ov.get("retrieve"):
        names = ["evidence_judge"]        # retrieve 开时用证据判断，baseline 提示词不参与
    if ov.get("debate") or ov.get("review"):
        from src.cognitive.debate import PROMPTS
        names = list(PROMPTS)             # M5 v2：快判 + 信源核查 + 正反方 + 裁判
    if ov.get("fallback") and ov.get("retrieve"):
        names.append("baseline_judge")    # 兜底时会调用裸模型
    if ov.get("behavior"):
        names.append("behavior_screen")
    out: dict[str, str] = {}
    if ov.get("source_score"):
        out["source_score"] = "rules"
    for n in names:
        try:
            _, _, v = load_prompt(n, cfg)
            out[n] = v
        except Exception:
            out[n] = "unknown"
    return out


def print_report(result: dict[str, Any]) -> None:
    m = result["meta"]
    c, r = result["cognitive"], result["risk_matrix"]
    print("=" * 68)
    print(f"配置 {m['config_name']}  |  split={m['split']}  |  模型={m['model']}  |  n={m['n_samples']}")
    print("-" * 68)
    print(f"认知面   准确率 {c['accuracy']:.4f}   宏F1 {c['macro_f1']:.4f}")
    for lab in COGNITIVE_LABELS:
        v = c["per_class"][lab]
        print(f"         {lab:<5s} P {v['precision']:.3f}  R {v['recall']:.3f}  "
              f"F1 {v['f1']:.3f}  n={v['support']}")
    print(f"风险矩阵 准确率 {r['accuracy']:.4f}  宏F1 {r['macro_f1']:.4f}  "
          f"尽快劝阻召回 {r['recall_尽快劝阻']:.4f}")
    if "behavior" in result:
        b = result["behavior"]
        print(f"行为面   宏F1 {b['macro_f1']:.4f}  风险等级准确率 {b['risk_level_accuracy']:.4f}")
        for t in BEHAVIOR_TYPES:
            v = b["per_type"][t]
            print(f"         {t:<6s} P {v['precision']:.3f}  R {v['recall']:.3f}  "
                  f"F1 {v['f1']:.3f}  n={v['support']}")
    if "retrieval" in result:
        rt = result["retrieval"]
        print(f"检索     模式 {rt['mode']:<16s} 平均 {rt['avg_evidence']} 条证据  "
              f"有证据 {rt['claims_with_evidence']}/{rt['n']}  "
              f"可复现占比 {rt['reproducible_ratio']:.4f}")
        if rt["matched_by"]:
            print(f"         命中来源 {rt['matched_by']}")
        if rt["rerank_degraded_samples"]:
            print(f"⚠️  {rt['rerank_degraded_samples']} 条样本重排降级（已回退到初检顺序）")
        # D41 教训：阿里云欠费时向量与重排静默降级，流程不报错，但结果已与其他运行不可比
        mode = rt.get("mode") or ""
        vec_dead = ("vector" in mode or "hybrid" in mode) and float(rt.get("avg_hits_vector") or 0) < 1
        rr_dead = rt["rerank_degraded_samples"] > rt["n"] / 2
        if vec_dead or rr_dead:
            print("❗❗ 向量检索或重排大面积失效（常见原因：API 欠费/额度用尽）。"
                  "本次结果只相当于关键词检索，不可与其他运行直接比较！")
    traces = [r.get("debate_trace") for r in (result.get("rows") or []) if r.get("debate_trace")]
    if traces:
        n_deb = sum(1 for t in traces if t.get("debated"))
        tail = "（其余快判把握足够，直接结束）" if n_deb < len(traces) else ""
        print(f"辩论     触发 {n_deb}/{len(traces)} 条{tail}")
    cost = result["cost"]
    print(f"成本     平均耗时 {cost['avg_latency_s']:.3f}s  平均 {cost['avg_tokens']:.0f} token  "
          f"千篇 ${cost['cost_per_1000_usd']:.4f}")
    absent = c.get("labels_absent") or []
    if absent:
        print(f"⚠️  评测集未覆盖的认知类别（未计入宏F1）：{'、'.join(absent)}")
    if "behavior" in result:
        tabs = result["behavior"].get("types_absent") or []
        if tabs:
            print(f"⚠️  评测集未覆盖的行为类型（未计入宏F1）：{'、'.join(tabs)}")
    failed = m.get("failed_samples") or []
    if failed:
        print(f"⚠️  {len(failed)} 条样本调用失败（已记为「无法核查」）："
              f"{'、'.join(failed[:8])}{' …' if len(failed) > 8 else ''}")
    _print_errors(result)
    pv = m.get("prompt_versions") or {}
    if pv:
        print(f"提示词版本：{'  '.join(f'{k}={v}' for k, v in pv.items())}")
    print("=" * 68)


def _print_errors(result: dict[str, Any], max_show: int = 10) -> None:
    """逐条列出判错的样本。总分之外，报告里真正有说服力的是错误清单。"""
    rows = result.get("rows") or []
    bad = [r for r in rows if not r.get("ok")]
    if not bad:
        print("✅ 无认知面误判")
        return
    print(f"❌ 认知面误判 {len(bad)}/{len(rows)} 条：")
    for r in bad[:max_show]:
        extra = f"  （{r['note']}）" if r.get("note") else ""
        print(f"   {r['id']}  gold={r['gold']}  pred={r['pred']}  {r['topic']}{extra}")
    if len(bad) > max_show:
        print(f"   … 另有 {len(bad) - max_show} 条")


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save_result(cfg: Config, result: dict[str, Any], split: str) -> Path:
    out_dir = Path(cfg.get_path("paths.results_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{result['meta']['config_name'].replace('+', 'plus')}_{split}_{run_stamp(cfg)}.json"
    path = out_dir / name
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_discussions(result: dict[str, Any], result_path: Path, max_samples: int = 3) -> Path | None:
    """M5 验收产物：导出若干样本的完整讨论记录。

    优先挑**真正辩论过**的样本，其中判错的排前面——一条快判就结束的记录说明不了多智能体在做什么，
    判错的样本最能看出辩论哪一步出了问题。
    """
    from src.cognitive.debate import trace_to_markdown

    rows = [r for r in (result.get("rows") or []) if r.get("debate_trace")]
    if not rows:
        return None
    rows.sort(key=lambda r: (not (r.get("debate_trace") or {}).get("debated"), r.get("ok", True)))
    n_debated = sum(1 for r in rows if (r.get("debate_trace") or {}).get("debated"))
    out = result_path.with_name(f"discussions_{result_path.stem}.md")
    lines = [
        f"# M5 多智能体讨论记录（{result['meta']['config_name']} / {result['meta']['split']}）",
        "",
        f"来源结果文件：`{result_path.name}`；触发辩论 {n_debated}/{len(rows)} 条",
        "",
    ]
    for r in rows[:max_samples]:
        trace = r.get("debate_trace") or {}
        lines += [
            f"## {r['id']}（gold={r['gold']}，pred={r['pred']}，topic={r['topic']}）",
            "",
            "### 证据",
            "",
        ]
        for i, e in enumerate(trace.get("evidence") or [], 1):
            lines.append(f"- [{i}] {e.get('source_name') or e.get('domain')} "
                         f"score={float(e.get('score') or 0):.2f} `{e.get('chunk_id') or ''}`")
        lines += [""] + trace_to_markdown(trace) + [""]
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def write_comparison(cfg: Config, results: list[dict[str, Any]], split: str,
                     skipped: dict[str, list[str]] | None = None) -> Path:
    """消融对比表——这是项目的核心产出之一。

    `skipped` 列出本次**没跑**的配置（模块尚未实现）。必须写进表里：
    一张只列了 2 行的消融表，如果不说明另外 4 行为什么不在，看的人会以为那就是全部结果。
    """
    out_dir = Path(cfg.get_path("paths.results_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 消融对比（split={split}，生成于 {run_stamp(cfg)}）", "",
        "| 配置 | 认知准确率 | 认知宏F1 | 尽快劝阻召回 | 行为面宏F1 | 风险等级准确率 | 千篇成本(USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, c, rm = r["meta"], r["cognitive"], r["risk_matrix"]
        b = r.get("behavior")

        def fmt(v: float | None) -> str:
            return "—" if v is None else f"{v:.4f}"

        lines.append(
            f"| {m['config_name']} | {c['accuracy']:.4f} | {c['macro_f1']:.4f} | "
            f"{rm['recall_尽快劝阻']:.4f} | "
            f"{fmt(b['macro_f1'] if b else None)} | "
            f"{fmt(b['risk_level_accuracy'] if b else None)} | "
            f"{r['cost']['cost_per_1000_usd']:.4f} |"
        )

    if skipped:
        lines += ["", "## 未覆盖的配置（本次跳过）", "",
                  "以下配置开启了尚未实现的模块，**不在上表中**，不要把它当成完整消融结果：", ""]
        for name, mods in skipped.items():
            lines.append(f"- `{name}` ← 缺少 {('、'.join(mods))}")

    # --- 检索层子消融（M4）---
    # 只在本次跑到了检索时才出这张表。四行 retrieve-* 的最终准确率可能完全相同——
    # 那说明"检索得好不好"没传导到结论上，是**结论**不是噪声，
    # 但必须靠检索层的形态指标（召回了几条、两路各出多少力）才说得清为什么。
    rt_rows = [r for r in results if r.get("retrieval")]
    if rt_rows:
        lines += ["", "## 检索层子消融（M4）", "",
                  "| 配置 | 检索方式 | 平均证据数 | 有证据主张 | BM25 平均命中 | 向量平均命中 | 融合后平均 | Top-1 被重排改写 | 平均位移 | 重排降级 | 认知准确率 | 认知宏F1 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rt_rows:
            rt, c = r["retrieval"], r["cognitive"]
            lines.append(
                f"| {r['meta']['config_name']} | `{rt.get('mode') or '—'}` | "
                f"{rt['avg_evidence']} | {rt['claims_with_evidence']}/{rt['n']} | "
                f"{rt['avg_hits_bm25']} | {rt['avg_hits_vector']} | {rt['avg_fused']} | "
                f"{rt.get('rerank_moved_top1_samples', 0)}/{rt['n']} | "
                f"{rt.get('avg_rerank_displaced', 0)} | "
                f"{rt['rerank_degraded_samples']} | "
                f"{c['accuracy']:.4f} | {c['macro_f1']:.4f} |"
            )
        lines += ["",
                  "> 「BM25/向量平均命中」是各路的**初检**命中数（进融合前），"
                  "「融合后平均」是 RRF 去重后的候选数。",
                  "> ⚠️ 某一路命中数**打满上限**（改写查询数 × `candidates`）时，"
                  "它只说明该路总能填满候选池，**不说明匹配质量**——别拿它比「哪路更好」。",
                  "> 若某路平均命中恒为 0，说明该路在本次数据上没出力——"
                  "那么它的行与相邻行分数相同，是**正常**的，不能当成「两路等价」。",
                  "> **重排只改顺序、不改候选集大小**，所以 hybrid 与 hybrid-rerank 的"
                  "命中数必然相同；这两行要靠「Top-1 被重排改写」和「平均位移」才分得开。",
                  "> 「重排降级」> 0 表示重排调用失败、退回了初检顺序（见 `retrieval_trace`）。"]

    lines += ["", "## 口径说明", "",
              "- 宏F1 只对**评测集中实际出现**的类别求平均；零支持的类别单独列出（`labels_absent`）。",
              "- 「尽快劝阻召回」是主指标：漏掉该尽快劝阻的样本，代价远大于误报。",
              "- 行为面关闭时，行为等级一律推为「低」——这张差值就是行为面模块的贡献。"]

    path = out_dir / f"comparison_{split}_{run_stamp(cfg)}.md"
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    # 再写一份固定路径的副本：结果目录按时间戳堆文件，跑多了就找不到"当前那张表"。
    # 带时间戳的那份是历史记录（不覆盖），latest 这份是"现在该看哪张"。
    latest = out_dir / f"ablation_{split}_latest.md"
    latest.write_text(text, encoding="utf-8")
    print(f"\n对比表已写入：{path}")
    print(f"当前有效副本：{latest}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="健康谣言双滤网核查 —— 评测")
    ap.add_argument("--config", default="baseline",
                    help="消融配置名（config.yaml eval.configs 里的键），或 all")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--model", default="stub:fixed",
                    help="stub:fixed | stub:oracle | stub:random | real")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条，便于快速冒烟")
    ap.add_argument("--config-file", default=None, help="指定 config.yaml 路径")
    ap.add_argument("--no-store", action="store_true", help="不写 SQLite（评测框架自检时用）")
    args = ap.parse_args(argv)
    try:                      # Windows 默认 GBK 终端打印 ⚠️ ✅ 会抛 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:         # noqa: BLE001
        pass

    cfg = load_config(args.config_file)

    if args.split == "test" and cfg.get_path("eval.forbid_test_tuning", True):
        print("⚠️  警告：正在评测 test 集。test 只用于最终评测，禁止用于调提示词/阈值。\n")

    names = list((cfg.get_path("eval.configs", {}) or {}).keys()) if args.config == "all" else [args.config]
    results: list[dict[str, Any]] = []
    stamp = run_stamp(cfg)

    # real 模式下先看有哪些模块还没实现。
    # 区别对待：
    #   --config <具体名>  明确点了名 -> 直接报错退出，不要给一个"看起来跑过了"的假结果
    #   --config all       意思是"把能跑的都跑一遍" -> 跳过未实现的，但在输出与对比表里
    #                      显著标注跳过了哪些。跳过必须是**可见**的，否则半张表会被当成全表。
    blocked: dict[str, list[str]] = {}
    if args.model == "real":
        blocked = _real_blockers(cfg, names)
        if blocked:
            print("⚠️  以下配置开启了尚未实现的模块：")
            for name, mods in blocked.items():
                print(f"     {name}  ← {mods}")
            if args.config != "all":
                # 列出**全部**可跑的配置，而不只是本次点名的那些——
                # 用户点了个跑不了的，最想知道的是"那我该跑什么"。
                all_names = list((cfg.get_path("eval.configs", {}) or {}).keys())
                runnable = [n for n in all_names if n not in _real_blockers(cfg, all_names)]
                print(f"\n当前 real 模式可跑的配置：{runnable or '（无）'}")
                print("等 M4 的信源评分（source_score）/ M5（辩论+审核）完成后再跑这些。")
                print("想看框架自检，用 --model stub:oracle（零成本，指标应全为 1.0）。")
                return 2
            runnable = [n for n in names if n not in blocked]
            if not runnable:
                print("\n❌ 没有任何可跑的配置。")
                return 2
            print(f"    本次继续跑：{runnable}\n")
            names = runnable

    store = None if args.no_store else Store(cfg.get_path("paths.db_path"))
    try:
        for name in names:
            run_id = f"{name}-{args.split}-{stamp}"
            res = run_one(cfg, name, args.split, args.model, limit=args.limit,
                          store=store, run_id=run_id)
            path = save_result(cfg, res, args.split)
            disc_path = save_discussions(res, path)
            if disc_path is not None:
                print(f"讨论记录已写入：{disc_path}")
            results.append(res)
            if store is not None:
                store.log_eval(
                    run_id=run_id, config_name=name, split=args.split,
                    n_samples=res["meta"]["n_samples"], model=args.model,
                    metrics=res, config_snapshot={"overrides": res["meta"]["overrides"]},
                    prompt_versions=res["meta"].get("prompt_versions") or {},
                    results_path=str(path),
                )
                # 真实模型模式下，顺带报一下 token 与成本记账
                if args.model == "real":
                    print(f"成本记账：{store.cost_summary(run_id)}")
            print(f"结果已写入：{path}\n")
    finally:
        if store is not None:
            store.close()

    if len(results) > 1:
        write_comparison(cfg, results, args.split, skipped=blocked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
