"""知识库覆盖度探针：评测集里有多少主张**只靠本地冻结库**就能拿到证据。

这个脚本回答一个很具体的问题：**建库这 36 分钟到底买到了什么？**
它同时产出 M4 最需要的那个数——`claims_kb_only_ratio`。

    python scripts/check_kb_coverage.py                # dev 集
    python scripts/check_kb_coverage.py --split test
    python scripts/check_kb_coverage.py --no-llm       # 用规则改写，不花钱

会调 LLM 做查询改写（每篇 1 次），因此不放进 pytest。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cognitive.retrieve import kb_search, rewrite_queries  # noqa: E402
from src.config import load_config  # noqa: E402
from src.kb.store import KnowledgeBase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="用规则改写，不调模型")
    args = ap.parse_args()

    cfg = load_config()
    if args.no_llm:
        cfg["cognitive"]["retrieve"]["query_rewrite"] = "rule"

    db = cfg["paths"]["kb_path"]
    with KnowledgeBase(db) as kb:
        snap = kb.snapshot() or {}
        n_docs = kb.doc_count()
    print(f"知识库 {db}")
    print(f"  文档数 {n_docs}   快照 {snap.get('kb_version', '（未冻结）')}   "
          f"指纹 {snap.get('content_hash', '—')}")
    print("=" * 78)

    split_path = Path(cfg["paths"]["splits_dir"]) / f"{args.split}.jsonl"
    samples = [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    covered = 0
    rows: list[dict] = []
    t0 = time.time()
    for s in samples:
        # 用标题当"核心主张"——评测集里正文是全文，真实管线会先做 claim_extract。
        # 这里探的是**库的覆盖度**，不是抽取质量，所以取标题更直接。
        claim = s.get("title") or s["text"][:60]
        queries = rewrite_queries(claim, cfg)
        hits = kb_search(claim, cfg, top_k=args.top_k)
        got = len(hits)
        covered += 1 if got else 0
        rows.append({"id": s["id"], "claim": claim, "queries": queries, "hits": got})
        mark = "✓" if got else "✗"
        print(f"{mark} {s['id']}  命中 {got}  查询 {queries}")
        if not got:
            print(f"     主张：{claim}")

    dt = time.time() - t0
    n = len(samples)
    print("=" * 78)
    print(f"L0 覆盖：{covered}/{n} = {covered / n:.3f}" if n else "无样本")
    print(f"claims_kb_only_ratio = {covered / n:.4f}" if n else "")
    print(f"总耗时 {dt:.1f}s（{dt / n:.1f}s/篇）" if n else "")

    missed = [r for r in rows if not r["hits"]]
    if missed:
        print(f"\n未覆盖 {len(missed)} 条——这些主张在本地库里找不到证据：")
        for r in missed:
            print(f"  {r['id']}  {r['claim']}")
        print("\n若某类主题集中缺失，说明种子词需要补；若只是同义表述差异，"
              "说明需要查询扩展（见 docs/decisions.md 待决项）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
