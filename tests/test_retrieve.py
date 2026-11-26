"""检索模块的离线测试。

全部不联网：解析用固定 HTML 片段，改写走规则通道，检索走 mock 后端。
真机联调（浏览器 + 权威站）在 scripts/check_retrieve.py 里手工跑，
不放进单元测试，避免 CI 依赖外网。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.cognitive.retrieve import (
    SITE_ADAPTERS, _filter_site_links, authority_search, engine_search,
    is_antispider, parse_bing, parse_sogou, rewrite_query_rule, rewrite_queries,
)
from src.models import QueryRewrite
from src.prompts import load_prompt, render

ROOT = Path(__file__).resolve().parent.parent

# 真实 Sogou 结果页的结构（截取自实测抓到的页面）
SOGOU_HTML = """
<div class="vrwrap"><h3 class="vr-title"><a href="/link?url=xxx">高血压和高血压病是一回事吗?</a></h3></div>
<div class="r-sech" id="sogou_vr_30010469_0"
     data-url="https://www.piyao.org.cn/20250206/939ccecd879a4903bd42004df6cbd142/c.html"
     data-title="%E9%AB%98%E8%A1%80%E5%8E%8B%E5%92%8C%E9%AB%98%E8%A1%80%E5%8E%8B%E7%97%85%E6%98%AF%E4%B8%80%E5%9B%9E%E4%BA%8B%E5%90%97%3F"
     data-rank="0"></div>
<div class="r-sech" id="sogou_vr_30010469_1"
     data-url="https://www.piyao.org.cn/20240521/9265f8fd0cdd4e40b6661100b35f7bc0/c.html"
     data-title="%E9%AB%98%E8%A1%80%E5%8E%8B%E8%AF%AF%E5%8C%BA"
     data-rank="1"></div>
"""

ANTISPIDER_HTML = '<html><head><title>请输入验证码</title></head><body>antispider</body></html>'

BING_HTML = """
<li class="b_algo"><h2><a href="https://www.who.int/zh/news-room/fact-sheets/detail/hypertension">
高血压 - World Health Organization</a></h2></li>
<li class="b_algo"><h2><a href="https://baike.baidu.com/item/高血压">高血压（疾病名称）_百度百科</a></h2></li>
"""


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def test_parse_sogou_extracts_real_urls_from_data_url():
    hits = parse_sogou(SOGOU_HTML)
    assert len(hits) == 2
    assert hits[0].url == "https://www.piyao.org.cn/20250206/939ccecd879a4903bd42004df6cbd142/c.html"
    assert hits[0].domain == "www.piyao.org.cn"
    # data-title 是 URL 编码，要解码成中文
    assert "高血压" in hits[0].title
    assert hits[0].title.endswith("一回事吗?")


def test_parse_sogou_ignores_non_http_and_dedupes():
    html = SOGOU_HTML + '<div data-url="javascript:void(0)"></div>' + SOGOU_HTML
    hits = parse_sogou(html)
    assert len(hits) == 2, "重复 URL 与非法 URL 都应被过滤"


def test_parse_sogou_empty_html():
    assert parse_sogou("") == []


def test_parse_bing():
    hits = parse_bing(BING_HTML)
    assert len(hits) == 2
    assert hits[0].domain == "www.who.int"
    assert "高血压" in hits[0].title


def test_is_antispider_detects_block_page():
    assert is_antispider(ANTISPIDER_HTML) is True
    assert is_antispider("https://www.sogou.com/antispider/?m=1") is True
    assert is_antispider(SOGOU_HTML) is False


def test_filter_site_links_drops_footer_noise():
    pairs = [
        ["隔夜水真的不能喝吗？", "https://www.kepuchina.cn/article/articleinfo?ar_id=1"],
        ["京ICP备16016202号", "https://beian.miit.gov.cn/"],
        ["短", "https://www.kepuchina.cn/article/articleinfo?ar_id=2"],       # 标题太短
        ["外部站点标题很长很长", "https://www.other.com/a"],                    # 非本站
        ["隔夜水真的不能喝吗？", "https://www.kepuchina.cn/article/articleinfo?ar_id=1"],  # 重复
    ]
    out = _filter_site_links(pairs, "kepuchina.cn")
    assert len(out) == 1
    assert out[0].url.endswith("ar_id=1")


# ---------------------------------------------------------------------------
# 查询改写
# ---------------------------------------------------------------------------
def test_rewrite_query_rule_strips_stopwords_and_punctuation():
    q = rewrite_query_rule("隔夜水致癌，千万不能喝")
    assert "，" not in q and "千万" not in q
    assert "隔夜水" in q


def test_rewrite_query_rule_handles_empty():
    assert rewrite_query_rule("") == ""
    assert rewrite_query_rule("！！！") == "！！！"


def test_rewrite_queries_mode_none_returns_raw(cfg):
    cfg["cognitive"]["retrieve"]["query_rewrite"] = "none"
    assert rewrite_queries("隔夜水致癌", cfg) == ["隔夜水致癌"]


def test_rewrite_queries_mode_rule_does_not_touch_network(cfg):
    cfg["cognitive"]["retrieve"]["query_rewrite"] = "rule"
    out = rewrite_queries("隔夜水致癌，千万不能喝", cfg)
    assert isinstance(out, list) and len(out) == 1 and out[0]


def test_rewrite_queries_falls_back_when_llm_fails(cfg):
    """LLM 改写抛异常时必须退回规则法，不能把检索整条丢掉。"""
    cfg["cognitive"]["retrieve"]["query_rewrite"] = "llm"

    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("模拟接口挂了")

    out = rewrite_queries("隔夜水致癌，千万不能喝", cfg, client=Boom())
    assert len(out) == 1 and "隔夜水" in out[0]


def test_query_rewrite_schema_caps_at_three():
    r = QueryRewrite(keywords=["a", "b", "c", "d", "e"])
    assert r.keywords == ["a", "b", "c"]
    assert QueryRewrite(keywords=["  ", ""]).keywords == []


# ---------------------------------------------------------------------------
# 检索编排（mock 后端，不联网）
# ---------------------------------------------------------------------------
def test_engine_search_mock_returns_empty(cfg):
    assert engine_search("高血压", cfg, backend="mock") == []


def test_authority_search_mock_backend_returns_empty_and_reports_queries(cfg):
    cfg["cognitive"]["retrieve"]["search"]["backend"] = "mock"
    cfg["cognitive"]["retrieve"]["query_rewrite"] = "rule"
    hits, queries = authority_search("隔夜水致癌", cfg)
    assert hits == []
    assert queries, "即使没命中，也要如实报告尝试过哪些查询词"


def test_site_adapter_registry_has_kepuchina():
    assert "kepuchina.cn" in SITE_ADAPTERS


# ---------------------------------------------------------------------------
# 提示词加载
# ---------------------------------------------------------------------------
def test_load_query_rewrite_prompt(cfg):
    system, user, version = load_prompt("query_rewrite", cfg)
    assert version == "v1"
    assert "检索" in system
    assert "{claim}" in user


def test_render_keeps_unknown_placeholders():
    assert render("a={a} b={b}", a=1) == "a=1 b={b}"


def test_load_prompt_missing_raises(cfg):
    from src.prompts import PromptNotFound
    with pytest.raises(PromptNotFound):
        load_prompt("不存在的提示词", cfg)
