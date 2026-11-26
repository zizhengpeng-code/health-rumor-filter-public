"""评测框架端到端冒烟测试 —— 这就是 M1 的验收标准。

用假模型跑通全部指标，不消耗任何 token：
- oracle 假模型返回金标准 -> 所有指标必须为 1.0（否则说明指标实现有 bug）
- fixed  假模型永远返回"假" -> 认知准确率必须等于 dev 集中"假"的占比
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval as _eval_pkg  # noqa: F401  （确保 eval 目录可导入）
from eval.eval import load_split, main, run_one
from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


def test_split_files_exist_and_parse(cfg):
    for split in ("dev", "test"):
        rows = load_split(cfg, split)
        assert len(rows) > 0, f"{split}.jsonl 为空"
        for r in rows:
            assert r.cognitive_label in ("真", "假", "部分属实", "无法核查")
            assert r.text.strip()


def test_oracle_scores_perfect(cfg):
    """自检：金标准喂回去，指标必须全 1.0。"""
    res = run_one(cfg, "full", "dev", "stub:oracle", verbose=False)
    c = res["cognitive"]
    assert c["accuracy"] == 1.0
    assert c["macro_f1"] == 1.0
    assert res["risk_matrix"]["accuracy"] == 1.0
    assert res["risk_matrix"]["recall_尽快劝阻"] == 1.0
    assert res["behavior"]["macro_f1"] == 1.0
    assert res["behavior"]["risk_level_accuracy"] == 1.0
    # 假模型不该产生成本
    assert res["cost"]["total_cost_usd"] == 0.0


def test_oracle_confusion_matrix_is_diagonal(cfg):
    res = run_one(cfg, "full", "dev", "stub:oracle", verbose=False)
    cm = res["cognitive"]["confusion_matrix"]
    for true_label, row in cm.items():
        for pred_label, n in row.items():
            if true_label != pred_label:
                assert n == 0, f"oracle 在 {true_label}->{pred_label} 上出现了非对角元素"


def test_fixed_model_accuracy_equals_class_prior(cfg):
    """fixed 永远答"假"，准确率必须等于 dev 里"假"的占比——手工可验。"""
    rows = load_split(cfg, "dev")
    n_fake = sum(1 for r in rows if r.cognitive_label == "假")
    expected = round(n_fake / len(rows), 4)

    res = run_one(cfg, "baseline", "dev", "stub:fixed", verbose=False)
    assert res["cognitive"]["accuracy"] == expected
    # 而且它绝不该猜中"真"
    assert res["cognitive"]["per_class"]["真"]["recall"] == 0.0


def test_random_model_accuracy_is_near_chance(cfg):
    """random 乱猜，认知准确率应明显低于 oracle，且不至于为 0。"""
    res = run_one(cfg, "baseline", "dev", "stub:random", verbose=False)
    acc = res["cognitive"]["accuracy"]
    assert 0.0 <= acc < 1.0


def test_random_is_deterministic(cfg):
    a = run_one(cfg, "baseline", "dev", "stub:random", verbose=False)
    b = run_one(cfg, "baseline", "dev", "stub:random", verbose=False)
    assert a["cognitive"]["accuracy"] == b["cognitive"]["accuracy"]


def test_ablation_switches_behavior_module(cfg):
    """behavior 开关关掉时不该输出行为面指标。"""
    off = run_one(cfg, "baseline", "dev", "stub:oracle", verbose=False)
    on = run_one(cfg, "full", "dev", "stub:oracle", verbose=False)
    assert "behavior" not in off
    assert "behavior" in on


def test_limit_argument(cfg):
    res = run_one(cfg, "baseline", "dev", "stub:oracle", limit=3, verbose=False)
    assert res["meta"]["n_samples"] == 3


def test_unknown_config_raises(cfg):
    with pytest.raises(KeyError):
        run_one(cfg, "不存在的配置", "dev", "stub:oracle", verbose=False)


def test_cli_writes_result_file(cfg, tmp_path, capsys):
    """CLI 端到端：跑通并落盘，且不写 SQLite。"""
    code = main(["--config", "full", "--split", "dev", "--model", "stub:oracle", "--no-store"])
    assert code == 0
    out = capsys.readouterr().out
    assert "认知面" in out and "风险矩阵" in out

    results_dir = Path(cfg.get_path("paths.results_dir"))
    files = sorted(results_dir.glob("full_dev_*.json"))
    assert files, "结果文件未写入 results/"
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    assert data["cognitive"]["accuracy"] == 1.0


def test_cli_config_all_writes_comparison(cfg):
    code = main(["--config", "all", "--split", "dev", "--model", "stub:oracle", "--no-store"])
    assert code == 0
    results_dir = Path(cfg.get_path("paths.results_dir"))
    assert list(results_dir.glob("comparison_dev_*.md")), "消融对比表未生成"


def test_full_real_config_no_longer_blocked_after_m5(cfg):
    """M5 后 full 已接入真实链路；单测只查预检，不实际调 API。"""
    from eval.eval import _real_blockers

    assert _real_blockers(cfg, ["full"]) == {}


# ---------------------------------------------------------------------------
# 逐样本记录（消融表只能看总分，错误分析要靠它）
# ---------------------------------------------------------------------------
def test_result_contains_per_sample_rows(cfg):
    res = run_one(cfg, "baseline", "dev", "stub:oracle", verbose=False)
    rows = res["rows"]
    assert len(rows) == res["meta"]["n_samples"]
    for r in rows:
        assert {"id", "gold", "pred", "ok", "note"} <= set(r)
        # oracle 喂回金标准，每条都该判对
        assert r["ok"] is True


def test_per_sample_rows_flag_the_errors(cfg):
    """fixed 永远答"假"：被判错的条数必须等于"非假"样本数。"""
    res = run_one(cfg, "baseline", "dev", "stub:fixed", verbose=False)
    n_wrong = sum(1 for r in res["rows"] if not r["ok"])
    assert n_wrong == res["meta"]["n_samples"] - res["cognitive"]["per_class"]["假"]["support"]


def test_error_list_is_printed(cfg, capsys):
    run_one(cfg, "baseline", "dev", "stub:fixed", verbose=True)
    out = capsys.readouterr().out
    assert "误判" in out


# ---------------------------------------------------------------------------
# real 模式的预检：没实现的模块要开跑前就拦住
# ---------------------------------------------------------------------------
def test_all_configs_are_runnable_in_real_precheck_after_m5(cfg):
    """M5 后 +source / +debate / full 都不应再被预检拦截。"""
    from eval.eval import _real_blockers

    cfgs = list((cfg.get_path("eval.configs") or {}).keys())
    assert _real_blockers(cfg, cfgs) == {}


def test_behavior_config_is_runnable_in_real_mode(cfg):
    """M3 之后 +behavior 不再被拦——它是行为面自己的消融行。"""
    from eval.eval import _real_blockers

    assert _real_blockers(cfg, ["baseline", "+behavior"]) == {}


def test_real_blockers_reflect_implemented_modules(cfg):
    from eval.eval import _real_blockers

    assert _real_blockers(cfg, ["baseline"]) == {}
    # retrieve / source_score / debate / review / behavior 均已实现，full 不应再被拦。
    assert _real_blockers(cfg, ["full", "+source", "+debate"]) == {}
    # 四行检索子消融都只开 retrieve，应当全部可跑
    sub = ["retrieve-bm25", "retrieve-vector", "retrieve-hybrid", "retrieve-hybrid-rerank"]
    assert _real_blockers(cfg, sub) == {}


def test_comparison_table_lists_skipped_configs(cfg):
    """只列了 2 行的消融表必须说明另外几行为什么不在，否则会被当成完整结果。"""
    from eval.eval import run_one, write_comparison

    res = [run_one(cfg, n, "dev", "stub:oracle", verbose=False)
           for n in ("baseline", "+behavior")]
    path = write_comparison(cfg, res, "dev",
                            skipped={"+retrieve": ["retrieve"], "full": ["retrieve", "behavior"]})
    text = path.read_text(encoding="utf-8")
    assert "未覆盖的配置" in text
    assert "`+retrieve`" in text and "`full`" in text
    # 已经跑过的配置出现在表里
    assert "| baseline |" in text and "| +behavior |" in text
