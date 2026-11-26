"""本地知识库的离线测试（不联网）。

重点验证：中文检索能用、快照指纹稳定、URL 去重、短查询降级。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.kb.crawl import SEED_KEYWORDS, extract_article
from src.kb.store import KnowledgeBase

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def kb(tmp_path):
    with KnowledgeBase(tmp_path / "kb.db") as k:
        k.add_many([
            {"source": "kepuchina.cn", "source_name": "科普中国", "tier": 1,
             "title": "隔夜水真的不能喝吗？", "url": "https://www.kepuchina.cn/a/1",
             "text": "隔夜水中的亚硝酸盐含量极低，远低于国家安全标准，可以正常饮用。"},
            {"source": "piyao.org.cn", "source_name": "中国互联网联合辟谣平台", "tier": 1,
             "title": "被传了很多年！隔夜水不能喝？会致癌？",
             "url": "https://www.piyao.org.cn/a/2",
             "text": "实验数据显示，隔夜水反复烧开后亚硝酸盐含量仍远低于限值，致癌说法没有依据。"},
            {"source": "chinacdc.cn", "source_name": "中国疾病预防控制中心", "tier": 1,
             "title": "高血压患者如何自我管理", "url": "https://www.chinacdc.cn/a/3",
             "text": "原发性高血压需要长期规范用药，擅自停药会导致血压反弹。"},
        ])
        yield k


def test_doc_count_and_stats(kb):
    assert kb.doc_count() == 3
    assert kb.stats_by_source() == {"chinacdc.cn": 1, "kepuchina.cn": 1, "piyao.org.cn": 1}


def test_trigram_search_finds_chinese(kb):
    hits = kb.search("隔夜水", top_k=5)
    assert len(hits) == 2
    assert all("隔夜水" in (h["title"] + h["text"]) for h in hits)


def test_search_by_longer_term(kb):
    hits = kb.search("亚硝酸盐", top_k=5)
    assert len(hits) == 2


def test_search_specific_term_hits_only_relevant(kb):
    hits = kb.search("高血压", top_k=5)
    assert len(hits) == 1
    assert "高血压" in hits[0]["title"]


def test_short_query_falls_back_to_like(kb):
    # trigram 要求 >=3 字符，2 字查询必须走 LIKE 降级而不是返回空
    hits = kb.search("血压", top_k=5)
    assert len(hits) == 1
    assert "高血压" in hits[0]["title"]


def test_search_source_filter(kb):
    hits = kb.search("隔夜水", top_k=5, sources=["piyao.org.cn"])
    assert len(hits) == 1
    assert hits[0]["source"] == "piyao.org.cn"


def test_search_empty_query_returns_nothing(kb):
    assert kb.search("", top_k=5) == []
    assert kb.search("   ", top_k=5) == []


def test_search_special_chars_do_not_crash(kb):
    # FTS5 语法字符必须被转义，否则会抛 OperationalError
    for q in ['隔夜水"', "隔夜水*", "隔夜水 AND", "(隔夜水)", "隔夜水:水"]:
        kb.search(q, top_k=5)   # 不抛异常即通过


def test_url_dedupe(kb):
    added = kb.add_document(source="kepuchina.cn", title="重复标题",
                            url="https://www.kepuchina.cn/a/1", text="重复内容" * 20)
    assert added is False
    assert kb.doc_count() == 3


def test_rejects_incomplete_documents(kb):
    assert kb.add_document(source="x", title="", url="https://x/1", text="abc") is False
    assert kb.add_document(source="x", title="t", url="", text="abc") is False
    assert kb.add_document(source="x", title="t", url="https://x/2", text="") is False
    assert kb.doc_count() == 3


def test_content_hash_is_stable_and_changes_on_insert(kb):
    h1 = kb.content_hash()
    assert h1 == kb.content_hash(), "同样的库必须给出同样的指纹"
    kb.add_document(source="x", source_name="X", tier=3, title="新文章",
                    url="https://x/new", text="内容" * 50)
    assert kb.content_hash() != h1, "内容变了指纹必须变"


def test_freeze_writes_snapshot(kb):
    snap = kb.freeze("test-v1")
    assert snap["kb_version"] == "test-v1"
    assert snap["doc_count"] == 3
    assert snap["by_source"]["kepuchina.cn"] == 1
    assert kb.snapshot()["content_hash"] == snap["content_hash"]


def test_snapshot_empty_before_freeze(tmp_path):
    with KnowledgeBase(tmp_path / "empty.db") as k:
        assert k.snapshot() == {}
        assert k.doc_count() == 0
        assert k.content_hash() == k.content_hash()


# ---------------------------------------------------------------------------
# 正文抽取
# ---------------------------------------------------------------------------
def test_extract_article_prefers_trafilatura():
    html = """<html><head><title>隔夜水真的不能喝吗</title></head><body>
      <nav>导航 首页 关于</nav>
      <article><p>隔夜水中的亚硝酸盐含量极低，远低于国家安全标准。</p>
      <p>实验表明反复烧开也不会显著增加亚硝酸盐。</p></article>
      <footer>版权所有</footer></body></html>"""
    title, text = extract_article(html)
    assert "隔夜水" in title or "隔夜水" in text
    assert "亚硝酸盐" in text
    assert "版权所有" not in text, "页脚噪声应被去掉"


def test_extract_article_falls_back_on_plain_html():
    html = "<html><head><title>标题</title></head><body><p>正文内容</p></body></html>"
    title, text = extract_article(html)
    assert title == "标题"
    assert "正文内容" in text


def test_extract_article_handles_garbage():
    title, text = extract_article("")
    assert title == "" and text == ""


# ---------------------------------------------------------------------------
# 种子词
# ---------------------------------------------------------------------------
def test_seed_keywords_cover_all_topics():
    """种子词必须覆盖 CLAUDE.md 里的五个 topic 分类，否则建库会有覆盖缺口。"""
    joined = " ".join(SEED_KEYWORDS)
    for topic_hint in ["高血压", "抗生素", "隔夜水", "癌症", "理疗仪"]:
        assert topic_hint in joined, f"种子词缺少 {topic_hint} 方向的覆盖"
    assert len(SEED_KEYWORDS) >= 30


def test_kb_config_present():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg["paths"]["kb_path"]
    assert cfg["cognitive"]["retrieve"]["layers"] == ["kb", "site", "websearch"]
