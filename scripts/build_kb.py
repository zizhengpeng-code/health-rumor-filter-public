"""建本地权威知识库。

    python scripts/build_kb.py                          # 全量种子词
    python scripts/build_kb.py --keywords 10 --per-source 5   # 小样本试跑
    python scripts/build_kb.py --snapshot-only          # 只打快照，不抓

耗时参考：URL 发现是慢环节（浏览器 / 联网搜索），正文抓取是并发的纯 HTTP。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.kb.crawl import SEED_KEYWORDS, crawl  # noqa: E402
from src.kb.store import KnowledgeBase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", type=int, default=0, help="只用前 N 个种子词（0=全部）")
    ap.add_argument("--per-source", type=int, default=None, help="每（关键词×域）最多取几篇")
    ap.add_argument("--workers", type=int, default=None, help="正文抓取并发数")
    ap.add_argument("--snapshot-only", action="store_true", help="只打印当前库快照")
    args = ap.parse_args()

    cfg = load_config()
    db = cfg["paths"]["kb_path"]

    if args.snapshot_only:
        with KnowledgeBase(db) as kb:
            print(f"库路径 {db}")
            print(f"文档数 {kb.doc_count()}")
            print(f"来源分布 {kb.stats_by_source()}")
            print(f"快照 {kb.snapshot() or '（尚未冻结）'}")
        return 0

    kws = SEED_KEYWORDS[: args.keywords] if args.keywords else SEED_KEYWORDS
    kb_cfg = cfg.get("kb", {}) or {}
    t0 = time.time()
    res = crawl(
        cfg, keywords=kws,
        per_source=args.per_source or int(kb_cfg.get("per_source", 8)),
        workers=args.workers or int(kb_cfg.get("workers", 8)),
        db_path=db,
    )
    print(f"\n总耗时 {time.time() - t0:.1f}s")
    return 0 if res["total_docs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
