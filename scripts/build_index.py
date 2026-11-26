"""建索引：切块 + 向量化 + 冻结快照。

    python scripts/build_index.py                 # 切块 + 向量化 + 冻结
    python scripts/build_index.py --chunks-only   # 只切块，不算向量（不花钱）
    python scripts/build_index.py --snapshot-only # 只打快照

**不重爬**：现有 kb_docs 已经抓好了，切分与向量化都是纯本地/编码计算。
向量会落库缓存，重复跑只会补算缺失的部分——不会为同一块重复花钱。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.kb.embed import Embedder, EmbedError  # noqa: E402
from src.kb.search import retrieval_config  # noqa: E402
from src.kb.store import KnowledgeBase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-only", action="store_true", help="只切块，不做向量化")
    ap.add_argument("--snapshot-only", action="store_true", help="只打快照")
    ap.add_argument("--no-drop", action="store_true", help="保留已有切块（增量）")
    ap.add_argument("--limit", type=int, default=0, help="只向量化前 N 块（试跑用）")
    args = ap.parse_args()

    cfg = load_config()
    db = cfg["paths"]["kb_path"]
    emb = Embedder.build(cfg)

    if args.snapshot_only:
        with KnowledgeBase(db) as kb:
            meta = kb.freeze(embed_model=emb.model_tag if emb.enabled else "",
                             retrieval=retrieval_config(cfg))
            _print_snapshot(meta)
        return 0

    t0 = time.time()
    with KnowledgeBase(db) as kb:
        print(f"库 {db}")
        print(f"现有文档 {kb.doc_count()}，切块 {kb.chunk_count()}")

        print("\n[1/3] 切块 …")
        st = kb.rebuild_chunks(cfg, drop=not args.no_drop)
        print(f"  文档 {st['docs_total']} -> 可用 {st['docs_used']}，剔除 {st['docs_skipped']}")
        print(f"  切出 {st['chunks']} 块（平均 {st['chunks'] / max(1, st['docs_used']):.1f} 块/篇）")
        print(f"  切分参数指纹 {st['chunk_params_fingerprint']}")
        if st["skipped"]:
            print(f"  被剔除的原因分布：")
            from collections import Counter
            for why, n in Counter(s["reason"].split("（")[0] for s in st["skipped"]).items():
                print(f"     {why}: {n}")

        if args.chunks_only:
            print("\n--chunks-only：跳过向量化")
            return 0

        print(f"\n[2/3] 向量化（{emb.backend}:{emb.model}）…")
        if not emb.enabled:
            print("  embed.backend = none，跳过")
        else:
            todo = kb.chunks_for_embedding(emb.model_tag, limit=args.limit or None)
            done = kb.vector_count(emb.model_tag)
            print(f"  已有向量 {done}，待算 {len(todo)}")
            if todo:
                tb = time.time()
                batch: list[tuple] = []
                for i, ch in enumerate(todo, 1):
                    batch.append(ch)
                    if len(batch) >= 64 or i == len(todo):
                        try:
                            vecs = emb.encode([c["text"] for c in batch])
                        except EmbedError as exc:
                            print(f"  ✗ 编码失败，中断：{exc}")
                            break
                        kb.put_vectors(emb.model_tag, [(c["chunk_id"], v) for c, v in zip(batch, vecs)])
                        batch = []
                        if i % 320 == 0 or i == len(todo):
                            el = time.time() - tb
                            print(f"  [{i}/{len(todo)}] {el:.0f}s  "
                                  f"({i / max(el, 1e-6):.1f} 块/s)")
                print(f"  向量化完成，库内向量 {kb.vector_count(emb.model_tag)}")
            print(f"  编码统计 {emb.stats()}")

        print("\n[3/3] 冻结快照 …")
        meta = kb.freeze(embed_model=emb.model_tag if emb.enabled else "",
                         retrieval=retrieval_config(cfg))
        _print_snapshot(meta)

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    return 0


def _print_snapshot(meta: dict) -> None:
    print(f"  版本 {meta['kb_version']}   指纹 {meta['content_hash']}")
    print(f"  文档 {meta['doc_count']}（可用 {meta.get('docs_used')}，剔除 {meta.get('docs_skipped')}）")
    print(f"  切块 {meta['chunk_count']}   切分指纹 {meta.get('chunk_params_fingerprint')}")
    print(f"  向量 {meta.get('vector_count')}  ({meta.get('embed_model')})")
    print(f"  检索模式 {meta.get('retrieval')}")


if __name__ == "__main__":
    raise SystemExit(main())
