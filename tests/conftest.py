"""pytest 公共配置：把项目根加入 sys.path，让 tests 能 import src 与 eval。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 只加项目根。不要把 eval/ 也加进来——eval/eval.py 会遮蔽 eval 包。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path, monkeypatch):
    """所有测试的评测输出写到临时目录。

    以前冒烟测试直接写 results/，每跑一次 pytest 就用假模型结果覆盖 ablation_dev_latest.md。
    autouse fixture 先于其他 fixture 执行，所以测试里 load_config() 拿到的已是临时目录。
    """
    monkeypatch.setenv("HRF_RESULTS_DIR", str(tmp_path / "results"))
