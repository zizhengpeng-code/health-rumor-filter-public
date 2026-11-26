"""M4 切块层的离线测试（不联网、不花钱）。

重点验证几条**硬约束**——它们在实测里都被违反过，不能只靠"看起来对"：
1. 块长严格不超过 max_chars（重叠占用本块预算）
2. 相邻块确实重叠
3. 二进制/PDF 内容被拦在切分之外
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.kb.chunk import (
    chunk_document, chunk_params, chunk_text, is_binary_text, is_usable_article,
    params_fingerprint, split_paragraphs,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# 分段
# ---------------------------------------------------------------------------
def test_split_paragraphs_by_blank_line():
    assert split_paragraphs("第一段\n\n第二段") == ["第一段", "第二段"]


def test_split_paragraphs_single_newline_also_splits():
    """权威站正文常一行一句，单换行也得当分段。"""
    assert split_paragraphs("第一句\n第二句") == ["第一句", "第二句"]


def test_split_paragraphs_drops_blank():
    assert split_paragraphs("\n\n  \n第一段\n\n\n") == ["第一段"]


def test_split_paragraphs_normalizes_whitespace():
    assert split_paragraphs("第一段\u3000\u3000有全角空格") == ["第一段 有全角空格"]


# ---------------------------------------------------------------------------
# 块长硬约束
# ---------------------------------------------------------------------------
def test_chunk_respects_max_chars():
    """配置说 500 就得是 500。早先的实现把重叠算在预算之外，切出过 796 字的块。"""
    text = "。".join(f"这是第{i}个句子内容" for i in range(200))
    for c in chunk_text(text, min_chars=300, max_chars=500, overlap=50):
        assert len(c) <= 500, f"块长 {len(c)} 超过上限 500"


def test_chunk_respects_max_chars_with_long_unbroken_paragraph():
    """一整段没有句末标点（如转写稿），也必须守住上限。"""
    for c in chunk_text("长" * 5000, min_chars=300, max_chars=500, overlap=50):
        assert len(c) <= 500


def test_chunk_respects_max_chars_with_many_short_paragraphs():
    text = "\n".join("短句" for _ in range(500))
    for c in chunk_text(text, min_chars=300, max_chars=500, overlap=50):
        assert len(c) <= 500


def test_chunk_overlap_is_real(cfg):
    text = "。".join(f"这是第{i}个句子的内容，用来凑长度" for i in range(60))
    cs = chunk_text(text, min_chars=300, max_chars=500, overlap=50)
    assert len(cs) > 1
    for prev, cur in zip(cs, cs[1:]):
        # 第 i 块尾部 50 字必须出现在第 i+1 块开头
        assert cur.startswith(prev[-50:])


def test_zero_overlap_means_no_overlap():
    text = "。".join(f"句子{i}" * 10 for i in range(60))
    cs = chunk_text(text, min_chars=300, max_chars=500, overlap=0)
    assert len(cs) > 1
    for prev, cur in zip(cs, cs[1:]):
        assert not cur.startswith(prev[-20:])


def test_overlap_larger_than_max_is_clamped():
    """overlap 配成比 max_chars 还大时不能死循环或产出空块。"""
    cs = chunk_text("。".join(f"句子{i}" for i in range(300)), min_chars=50,
                    max_chars=100, overlap=999)
    assert cs
    for c in cs:
        assert len(c) <= 100


def test_empty_and_whitespace_text_yield_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []
    assert chunk_text("x", min_chars=300, max_chars=0) == []


def test_short_text_yields_single_chunk():
    cs = chunk_text("很短的一句话。", min_chars=300, max_chars=500, overlap=50)
    assert len(cs) == 1 and cs[0] == "很短的一句话。"


# ---------------------------------------------------------------------------
# 二进制 / 质检
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "%PDF-1.7\n%����\n1 0 obj\n<> /Outlines 5 0 R",
    "PK\x03\x04\x14\x00\x00\x00",
    "GIF89a\x01\x00\x01\x00",
    "\x89PNG\r\n\x1a\n",
])
def test_binary_magic_detected(text):
    assert is_binary_text(text) is True


def test_normal_chinese_text_is_not_binary():
    assert is_binary_text("高血压患者需要长期规范用药，不能随意停药。") is False


def test_control_char_heavy_text_is_binary():
    assert is_binary_text("\x01\x02\x03\x04\x05\x06\x07\x08" * 50) is True


def test_empty_text_is_binary():
    assert is_binary_text("") is True


def test_is_usable_article_reasons():
    ok, why = is_usable_article("正常的一段健康科普正文。" * 20)
    assert ok and why == ""
    ok, why = is_usable_article("%PDF-1.7\n二进制")
    assert not ok and "二进制" in why
    ok, why = is_usable_article("太短")
    assert not ok and "过短" in why
    ok, why = is_usable_article("")
    assert not ok


def test_pdf_documents_are_excluded_from_chunking(cfg):
    """实测库里 60 篇是 PDF 二进制，绝不能进检索库。"""
    doc = {"id": 1, "title": "T", "text": "%PDF-1.7\n%����\n1 0 obj", "source": "x"}
    from src.kb.chunk import is_usable_article as usable
    assert usable(doc["text"])[0] is False


# ---------------------------------------------------------------------------
# 元数据与参数指纹
# ---------------------------------------------------------------------------
def test_chunk_document_carries_metadata(cfg):
    doc = {"id": 42, "title": "隔夜水致癌吗", "source": "kepuchina.cn",
           "source_name": "科普中国", "tier": 1, "url": "https://x/1",
           "published": "2024-10-09",
           "text": "。".join(f"第{i}句内容，凑够长度" for i in range(80))}
    cs = chunk_document(doc, cfg)
    assert cs
    for i, c in enumerate(cs):
        assert c.chunk_id == f"42-c{i}"
        assert c.doc_id == 42 and c.idx == i
        assert c.title == "隔夜水致癌吗"
        assert c.source_name == "科普中国"
        assert c.url == "https://x/1"
        assert c.published == "2024-10-09"
        assert c.tier == 1


def test_chunk_id_is_stable(cfg):
    doc = {"id": 7, "title": "T", "text": "。".join(f"句{i}" for i in range(100))}
    a = [c.chunk_id for c in chunk_document(doc, cfg)]
    b = [c.chunk_id for c in chunk_document(doc, cfg)]
    assert a == b


def test_chunk_params_read_from_config(cfg):
    p = chunk_params(cfg)
    assert p == {"min_chars": 300, "max_chars": 500, "overlap": 50}


def test_params_fingerprint_changes_with_params():
    a = {"kb": {"chunk": {"min_chars": 300, "max_chars": 500, "overlap": 50}}}
    b = {"kb": {"chunk": {"min_chars": 300, "max_chars": 600, "overlap": 50}}}
    assert params_fingerprint(a) != params_fingerprint(b)
    assert params_fingerprint(a) == params_fingerprint(
        {"kb": {"chunk": {"min_chars": 300, "max_chars": 500, "overlap": 50}}})
