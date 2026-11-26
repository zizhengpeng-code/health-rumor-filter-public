"""真机联调：验证权威来源检索链路端到端可用。

这个脚本**会联网**（调 DeepSeek 做查询改写 + 开浏览器打权威站），
所以不放进 pytest，手工跑：

    python scripts/check_retrieve.py
    python scripts/check_retrieve.py --backend mock      # 只验证不联网的部分
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cognitive.retrieve import (  # noqa: E402
    SITE_ADAPTERS, authority_search, rewrite_queries,
)
from src.config import load_config  # noqa: E402

CASES = [
    "隔夜水致癌，千万不能喝",
    "喝醋能软化血管，每天一杯远离心脑血管病",
    "吃碱性食物能抗癌？调节体质酸碱平衡",
    "化疗会让人死得更快，不如回家吃中药",
    "维生素C能预防感冒吗",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="覆盖 config 里的 search.backend")
    ap.add_argument("--rewrite", default=None, help="覆盖 query_rewrite：llm | rule | none")
    ap.add_argument("--cases", type=int, default=len(CASES))
    args = ap.parse_args()

    cfg = load_config()
    if args.backend:
        cfg["cognitive"]["retrieve"]["search"]["backend"] = args.backend
    if args.rewrite:
        cfg["cognitive"]["retrieve"]["query_rewrite"] = args.rewrite

    backend = cfg["cognitive"]["retrieve"]["search"]["backend"]
    print(f"检索后端 = {backend}    查询改写 = {cfg['cognitive']['retrieve']['query_rewrite']}")
    print(f"已注册站点适配器：{list(SITE_ADAPTERS) or '（无）'}")
    print("=" * 78)

    ok = 0
    t_all = time.time()
    for text in CASES[: args.cases]:
        print(f"\n主张：{text}")
        t0 = time.time()
        try:
            kws = rewrite_queries(text, cfg)
        except Exception as exc:
            print(f"  ✗ 查询改写失败：{exc}")
            continue
        print(f"  改写：{kws}")

        hits, tried = authority_search(text, cfg)
        dt = time.time() - t0
        if hits:
            ok += 1
            print(f"  ✓ 命中 {len(hits)} 条（{dt:.1f}s）  实际尝试：{tried}")
            for h in hits[:3]:
                print(f"      [{h.tier}] {h.source_name} | {h.title[:50]}")
                print(f"          {h.url[:88]}")
        else:
            print(f"  ✗ 0 条（{dt:.1f}s）  实际尝试：{tried}")

    print("\n" + "=" * 78)
    print(f"命中 {ok}/{min(args.cases, len(CASES))} 条主张，总耗时 {time.time() - t_all:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
