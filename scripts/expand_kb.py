"""扩充权威证据库（D41）：在副本 data/kb_v2.db 上扩，旧库 data/kb.db 不动。

    python scripts/expand_kb.py                 # 复制旧库 → 并入已抓文章 → 按新关键词补抓 → 剔除样本原文
    python scripts/expand_kb.py --skip-crawl    # 只做前两步（不联网发现，不花钱）
    # 然后建索引（切块 + 向量化 + 冻结快照）：
    HRF_KB_PATH=data/kb_v2.db python scripts/build_index.py

为什么要扩：D38 实测 205 条开发样本里 136 条在库里找不到相关证据，检索在库外反而掉分。

防泄漏（必须）：评测样本自己的原文绝不能进库——否则检索会直接搜到样本原文，分数是假的。
只对**新加入**的文档执行剔除；旧库里原有的 963 篇保持不变（样本的「在知识库」标记仍然有效，
分组报分照旧可比）。

关键词表按主题系统整理，不依据评测样本的标题。作者看过开发集，因此刻意没有加入记得的
几个开发集判错样本的话题（肺炎支原体、寄生虫、三伏贴/冬病夏治、无糖饮料/代糖、12320 热线）——
否则就是「看着考题备课」，扩库带来的提升会虚高。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

OLD, NEW = ROOT / "data" / "kb.db", ROOT / "data" / "kb_v2.db"

KEYWORDS: dict[str, list[str]] = {
    "慢病": ["高血压", "糖尿病", "高血脂", "冠心病", "心力衰竭", "房颤", "脑卒中", "慢阻肺", "哮喘",
             "慢性肾病", "脂肪肝", "痛风", "骨质疏松", "关节炎", "甲状腺结节", "甲亢", "贫血", "肥胖"],
    "用药": ["抗生素滥用", "降压药", "降糖药", "胰岛素", "他汀", "阿司匹林", "布洛芬", "对乙酰氨基酚",
             "感冒药", "安眠药", "抗抑郁药", "激素药", "维生素补充剂", "钙片", "保健品与药品",
             "中成药", "中药注射剂", "药物相互作用", "过期药", "儿童用药", "孕期用药", "静脉输液"],
    "饮食": ["食盐", "添加糖", "食用油", "反式脂肪", "隔夜菜", "剩饭", "腌制食品", "加工肉", "红肉",
             "鸡蛋", "牛奶", "豆制品", "转基因", "食品添加剂", "防腐剂", "咖啡", "喝茶", "饮酒",
             "酸奶", "蜂蜜", "红糖", "醋", "大蒜", "枸杞", "燕窝", "海参", "膳食纤维", "饮水",
             "果汁", "碳酸饮料", "生酮饮食", "轻断食", "素食", "减肥"],
    "癌症": ["肺癌筛查", "胃癌", "肝癌", "结直肠癌", "乳腺癌", "宫颈癌", "HPV疫苗", "甲状腺癌",
             "鼻咽癌", "食管癌", "致癌物", "化疗", "放疗", "靶向药", "免疫治疗", "肿瘤标志物",
             "癌症早筛", "防癌体检", "抗癌食物"],
    "养生器械": ["理疗仪", "保健品骗局", "量子产品", "磁疗", "负离子", "足浴", "艾灸", "拔罐", "刮痧",
                 "针灸", "按摩器", "远红外", "石墨烯", "富氢水", "净水器"],
    "其他": ["流感", "新冠", "疫苗接种", "诺如病毒", "手足口病", "狂犬病", "艾滋病", "结核病", "乙肝",
             "幽门螺杆菌", "食物中毒", "中暑", "睡眠", "熬夜", "运动健身", "久坐", "近视", "口腔健康",
             "皮肤护理", "脱发", "便秘", "体检报告", "心理健康", "老年痴呆", "老人跌倒", "更年期",
             "儿童生长发育", "孕期保健", "母乳喂养", "秋冬进补", "排毒"],
}

# 新增来源：中国健康教育网（D27「待决」里记过可直接抓）
EXTRA_DOMAINS = [{"domain": "cche.org.cn", "name": "中国健康教育网", "tier": 1}]


def url_key(u: str) -> str:
    return (u or "").split("#", 1)[0].rstrip("/").replace("https://", "http://").lower()


def sample_urls() -> set[str]:
    meta = json.loads((ROOT / "data" / "labeled" / "round1_meta.json").read_text(encoding="utf-8"))
    return {url_key(m["url"]) for m in meta.values() if m.get("url")}


def remove_samples(db: Path, samples: set[str], min_id: int) -> int:
    """只删 id > min_id（本次新加入）的样本原文。触发器会同步删全文索引。"""
    con = sqlite3.connect(db)
    rows = con.execute("SELECT id, url FROM kb_docs WHERE id > ?", (min_id,)).fetchall()
    bad = [i for i, u in rows if url_key(u) in samples]
    con.executemany("DELETE FROM kb_docs WHERE id = ?", [(i,) for i in bad])
    con.commit()
    con.close()
    return len(bad)


CACHE = ROOT / "data" / "raw" / "expand_cache.json"


def crawl_resumable(cfg: dict, keywords: list[str], domains: dict, samples: set[str],
                    per_source: int, workers: int) -> bool:
    """可断点续传的补抓。返回 False 表示因 API 不可用中途停止（进度已保存）。

    D43 教训：原 crawl() 把发现结果放在内存里、全部抓完才入库，任务超时一次就全丢；
    联网搜索余额不足时又静默返回空，把失败的关键词当成已完成。这里改为：
      - 每个关键词发现完就写缓存；API 不可用（402 等）立即停，不标记为已完成
      - 正文每抓 50 篇入库一次，抓取结果也写缓存
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import urlparse

    from src.cognitive.retrieve import SITE_ADAPTERS, BrowserSession
    from src.cognitive.websearch import WebSearchUnavailable
    from src.kb.crawl import discover, fetch_article
    from src.kb.store import KnowledgeBase

    cache = {"done_keywords": [], "pending": {}, "fetched": {}}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    def save() -> None:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    with KnowledgeBase(str(NEW)) as kb:
        in_kb = {url_key(r[0]) for r in kb.conn.execute("SELECT url FROM kb_docs")}

    todo = [k for k in keywords if k not in cache["done_keywords"]]
    print(f"② 发现：剩 {len(todo)}/{len(keywords)} 个关键词 × {len(domains)} 个来源", flush=True)
    session = BrowserSession(cfg) if any(d in SITE_ADAPTERS for d in domains) else None
    aborted = False
    try:
        # 联网搜索的来源并发查（各自独立的 HTTP 请求）；浏览器适配器的来源串行（浏览器会话不能多线程共用）
        web_doms = [d for d in domains if d not in SITE_ADAPTERS]
        site_doms = [d for d in domains if d in SITE_ADAPTERS]
        pool = ThreadPoolExecutor(max_workers=max(1, len(web_doms)))
        for i, kw in enumerate(todo, 1):
            found: dict[str, str] = {}
            try:
                futs = [pool.submit(discover, kw, d, cfg, per_source, None) for d in web_doms]
                for d in site_doms:
                    for h in discover(kw, d, cfg, per_source, session):
                        found.setdefault(h.url, kw)
                for f in futs:
                    for h in f.result():
                        found.setdefault(h.url, kw)
            except WebSearchUnavailable as exc:
                print(f"   ✗ {kw}：{exc}", flush=True)
                aborted = True
                break
            for u, k in found.items():
                if u.startswith("http") and url_key(u) not in in_kb and url_key(u) not in samples:
                    cache["pending"].setdefault(u, k)
            cache["done_keywords"].append(kw)
            save()
            if i % 5 == 0 or i == len(todo):
                print(f"   [{i}/{len(todo)}] {kw}：待抓网址累计 {len(cache['pending'])}", flush=True)
    finally:
        if session is not None:
            session.close()
        if "pool" in locals():
            pool.shutdown(wait=False, cancel_futures=True)

    urls = [u for u in cache["pending"] if u not in cache["fetched"]]
    print(f"   抓正文：{len(urls)} 条（{workers} 并发）", flush=True)
    batch: list[dict] = []
    added = 0

    def flush() -> None:
        nonlocal added, batch
        if batch:
            with KnowledgeBase(str(NEW)) as kb:
                added += kb.add_many(batch)
            batch = []
        save()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_article, u, cfg): u for u in urls}
        for n, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                got = fut.result()
            except Exception:                                  # noqa: BLE001
                got = None
            cache["fetched"][u] = bool(got)
            if got:
                host = urlparse(u).netloc.lower()
                src = next((d for d in domains if d in host), host)
                meta = domains.get(src, {"name": src, "tier": 3})
                batch.append({"source": src, "source_name": meta["name"],
                              "tier": int(meta.get("tier", 1)), "title": got[0] or u, "url": u,
                              "text": got[1], "seed_keyword": cache["pending"].get(u, "")})
            if len(batch) >= 50:
                flush()
            if n % 100 == 0:
                print(f"   [{n}/{len(urls)}] 已入库 {added}", flush=True)
    flush()
    print(f"   本轮新入库 {added} 篇", flush=True)
    return not aborted


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crawl", action="store_true")
    ap.add_argument("--per-source", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    if not NEW.exists():
        shutil.copy(OLD, NEW)
        print(f"已复制旧库 → {NEW}")
    cfg = load_config()
    cfg["paths"]["kb_path"] = str(NEW)
    rcfg = cfg["cognitive"]["retrieve"]
    known = {d["domain"] for d in rcfg["authority_domains"]}
    rcfg["authority_domains"] += [d for d in EXTRA_DOMAINS if d["domain"] not in known]
    domains = {d["domain"]: d for d in rcfg["authority_domains"]}

    from src.kb.store import KnowledgeBase
    with KnowledgeBase(str(NEW)) as kb:
        base_count = kb.doc_count()
    con = sqlite3.connect(NEW)
    max_old_id = con.execute("SELECT MAX(id) FROM kb_docs").fetchone()[0] or 0
    con.close()
    samples = sample_urls()
    print(f"起始文档 {base_count}；评测样本原文 {len(samples)} 个网址（新加入时剔除）")

    # ① 并入候选池采集时已经抓好的文章（不花钱）
    cache_path = ROOT / "data" / "raw" / "discover_cache.json"
    if cache_path.exists():
        from urllib.parse import urlparse
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        docs = []
        for u, got in cache.get("docs", {}).items():
            if not got or url_key(u) in samples:
                continue
            host = urlparse(u).netloc.lower()
            src = next((d for d in domains if d in host), None)
            if src is None:
                continue
            docs.append({"source": src, "source_name": domains[src]["name"],
                         "tier": int(domains[src].get("tier", 1)), "title": got[0] or u,
                         "url": u, "text": got[1][:int(cfg["kb"].get("max_chars", 8000))],
                         "seed_keyword": cache.get("pending", {}).get(u, "")})
        with KnowledgeBase(str(NEW)) as kb:
            added = kb.add_many(docs)
        print(f"① 并入已抓文章：候选 {len(docs)}，新增 {added}")

    # ② 按新关键词补抓（断点续传，见 D43）
    if not args.skip_crawl:
        kws = [k for ks in KEYWORDS.values() for k in ks]
        t0 = time.time()
        ok = crawl_resumable(cfg, kws, domains, samples, args.per_source, args.workers)
        print(f"   补抓耗时 {time.time() - t0:.0f}s")
        if not ok:
            print("！发现阶段因 API 不可用中止，进度已保存；充值后重新运行本脚本会从断点继续。")

    # ③ 剔除新加入文档里的样本原文
    n_bad = remove_samples(NEW, samples, max_old_id)
    with KnowledgeBase(str(NEW)) as kb:
        print(f"③ 剔除样本原文 {n_bad} 篇；当前文档 {kb.doc_count()}，来源分布 {kb.stats_by_source()}")
    print("下一步：HRF_KB_PATH=data/kb_v2.db python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
