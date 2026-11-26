"""M4 向量化与重排的离线测试（不联网、不花钱）。

用假 httpx 替换真实请求，重点验证：
1. backend 开关真的能关掉（none 不偷偷发请求）
2. 向量表的 model_tag 把后端与模型名都编进去（换模型必须失效重算）
3. 重排失败**降级**而不是抛异常——初检候选不能因为精排挂了被一起丢掉
4. 降级必须是**可见**的（degraded 标记带出去），不能悄悄返回初检顺序冒充重排结果
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.kb.embed import Embedder, EmbedError, embed_config
from src.kb.rerank import Reranker, rerank_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
def test_embed_config_reads_yaml(cfg):
    c = embed_config(cfg)
    # 后端同理：开发用 local、上线切 api（D42），以 YAML 为准
    assert c["backend"] == cfg.get_path("kb.embed.backend")
    # 刻意**不**断言具体模型名：这个 key 上的可用模型换过三次
    # （bge-m3 不可用 -> v4 -> v3 -> v2，见 D29/D30/D31），
    # 把模型名写死在测试里，等于让"供应商额度"这种外部因素把测试搞红。
    # 要验的是"YAML 里的值真的被读进来了"，所以拿 YAML 自己当期望值。
    assert c["model"] == cfg.get_path("kb.embed.model")
    assert c["model"]
    assert c["timeout"] == 60


def test_rerank_config_reads_yaml(cfg):
    c = rerank_config(cfg)
    assert c["backend"] == cfg.get_path("kb.rerank.backend")
    assert c["model"] == cfg.get_path("kb.rerank.model")
    assert c["model"]
    assert c["max_doc_chars"] == 1200


def test_model_tag_encodes_backend_and_model():
    """同名模型不同后端维度可能不同，tag 必须把两者都编进去。"""
    a = Embedder(backend="api", model="m")
    b = Embedder(backend="local", model="m")
    assert a.model_tag == "api:m"
    assert b.model_tag == "local:m"
    assert a.model_tag != b.model_tag


@pytest.mark.parametrize("backend", ["none", "", "off"])
def test_disabled_backends(backend):
    assert Embedder(backend=backend, model="m").enabled is False
    assert Reranker(backend=backend, model="m").enabled is False


def test_none_backend_refuses_to_encode():
    """关掉了就不能偷偷发请求——否则 retrieve-bm25 消融行会偷偷花钱。"""
    with pytest.raises(EmbedError):
        Embedder(backend="none", model="m").encode(["x"])


def test_unknown_backend_raises():
    with pytest.raises(EmbedError):
        Embedder(backend="奇异后端", model="m").encode(["x"])


# ---------------------------------------------------------------------------
# 向量化：假 httpx
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, code, payload):
        self.status_code = code
        self._p = payload
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self):
        return self._p


def _fake_post_factory(calls, code=200, dim=8, fail_times=0):
    state = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "payload": json, "headers": headers})
        state["n"] += 1
        if state["n"] <= fail_times:
            return _Resp(500, {"error": "boom"})
        if code != 200:
            return _Resp(code, {"error": {"message": "nope"}})
        n = len(json["input"])
        return _Resp(200, {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(n)]})

    return fake_post


def test_api_encode_returns_vectors(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_post_factory(calls))
    e = Embedder(backend="api", model="m", base_url="https://x/v1", api_key="k")
    v = e.encode(["a", "b"])
    assert len(v) == 2 and len(v[0]) == 8
    assert calls[0]["url"] == "https://x/v1/embeddings"
    assert calls[0]["headers"]["Authorization"] == "Bearer k"


def test_api_encode_batches_by_batch_size(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_post_factory(calls))
    e = Embedder(backend="api", model="m", base_url="https://x", api_key="k", batch_size=3)
    e.encode([f"t{i}" for i in range(7)])
    assert [len(c["payload"]["input"]) for c in calls] == [3, 3, 1]


def test_api_sends_dimensions_only_when_configured(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_post_factory(calls))
    Embedder(backend="api", model="m", base_url="https://x", api_key="k").encode(["a"])
    assert "dimensions" not in calls[0]["payload"]
    calls.clear()
    Embedder(backend="api", model="m", base_url="https://x", api_key="k",
             dim=512).encode(["a"])
    assert calls[0]["payload"]["dimensions"] == 512


def test_api_retries_on_500_then_succeeds(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_post_factory(calls, fail_times=2))
    e = Embedder(backend="api", model="m", base_url="https://x", api_key="k", max_attempts=3)
    assert len(e.encode(["a"])) == 1
    assert len(calls) == 3


def test_api_does_not_retry_on_404(monkeypatch):
    """404 说明模型名不对，重试没有意义，白花钱。"""
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_post_factory(calls, code=404))
    e = Embedder(backend="api", model="bad", base_url="https://x", api_key="k", max_attempts=3)
    with pytest.raises(EmbedError):
        e.encode(["a"])
    assert len(calls) == 1


def test_api_missing_credentials_raises():
    with pytest.raises(EmbedError) as ei:
        Embedder(backend="api", model="m", base_url="", api_key="").encode(["a"])
    assert "EMBED_BASE_URL" in str(ei.value)


def test_encode_empty_returns_empty():
    assert Embedder(backend="api", model="m", base_url="x", api_key="k").encode([]) == []


def test_embedder_stats(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", _fake_post_factory([]))
    e = Embedder(backend="api", model="m", base_url="https://x", api_key="k")
    e.encode(["a", "b"])
    s = e.stats()
    assert s["calls"] == 1 and s["texts_encoded"] == 2 and s["model_tag"] == "api:m"


# ---------------------------------------------------------------------------
# 重排：降级行为
# ---------------------------------------------------------------------------
def test_none_backend_keeps_original_order():
    r = Reranker(backend="none", model="m")
    res = r.rerank("q", ["a", "b", "c"])
    assert res.order == [0, 1, 2] and res.degraded is False


def test_empty_docs_returns_empty():
    assert Reranker(backend="api", model="m").rerank("q", []).order == []


def _fake_rerank_post(calls, code=200, order=None):
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "payload": json})
        if code != 200:
            return _Resp(code, {"error": "nope"})
        idx = order or [1, 0]
        return _Resp(200, {"output": {"results": [
            {"index": i, "relevance_score": 0.9 - 0.1 * n} for n, i in enumerate(idx)]}})

    return fake_post


def test_api_rerank_orders_by_score(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_rerank_post(calls, order=[2, 0, 1]))
    r = Reranker(backend="api", model="m", base_url="https://x", api_key="k")
    res = r.rerank("q", ["a", "b", "c"])
    assert res.order == [2, 0, 1] and res.degraded is False
    assert calls[0]["url"].endswith("/api/v1/services/rerank/text-rerank/text-rerank")


def test_rerank_truncates_docs_before_sending(monkeypatch):
    import httpx
    calls: list = []
    monkeypatch.setattr(httpx, "post", _fake_rerank_post(calls))
    Reranker(backend="api", model="m", base_url="https://x", api_key="k",
             max_doc_chars=10).rerank("q", ["长" * 100])
    assert len(calls[0]["payload"]["input"]["documents"][0]) == 10


def test_rerank_degrades_on_http_error(monkeypatch):
    """精排挂了不能拖垮检索链——初检候选还在，原样返回，但必须标记降级。"""
    import httpx
    monkeypatch.setattr(httpx, "post", _fake_rerank_post([], code=403))
    r = Reranker(backend="api", model="m", base_url="https://x", api_key="k", max_attempts=1)
    res = r.rerank("q", ["a", "b"])
    assert res.order == [0, 1]          # 原样
    assert res.degraded is True         # 但可见
    assert "403" in res.reason
    assert r.stats()["degraded_calls"] == 1


def test_rerank_degrades_on_network_exception(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("断网")

    monkeypatch.setattr(httpx, "post", boom)
    res = Reranker(backend="api", model="m", base_url="https://x", api_key="k",
                   max_attempts=1).rerank("q", ["a", "b"])
    assert res.degraded is True and res.order == [0, 1]


def test_rerank_missing_credentials_degrades_not_raises():
    res = Reranker(backend="api", model="m", base_url="", api_key="",
                   max_attempts=1).rerank("q", ["a"])
    assert res.degraded is True


def test_rerank_top_n(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", _fake_rerank_post([], order=[2, 0, 1]))
    res = Reranker(backend="api", model="m", base_url="https://x", api_key="k"
                   ).rerank("q", ["a", "b", "c"], top_n=2)
    assert res.order == [2, 0]


def test_unknown_rerank_backend_degrades():
    res = Reranker(backend="奇异", model="m").rerank("q", ["a"])
    assert res.degraded is True and res.order == [0]
