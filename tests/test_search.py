"""M4 混合检索的离线测试（不联网、不花钱）。

用临时知识库 + 假编码器，重点验证：
1. RRF 按**名次**融合，不按分数（BM25 与余弦相似度量纲不可比）
2. 单路检索不做融合（融合会平白改变名次）
3. 四种检索模式真的走不同的路径（消融表不能是四行一样的数）
4. 重排降级会写进 trace（否则"为什么这行分数变了"解释不清）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.kb import search as S
from src.kb.store import KnowledgeBase

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture()
def tmp_kb(tmp_path, cfg):
    """建一个小库：3 篇文档 -> 若干块。

    正文必须长于 `kb.min_chars`(200)，否则会被质检当成导航页剔除、切不出块。
    这里刻意写到 400+ 字，贴近真实科普文章的体量。
    """
    db = tmp_path / "kb.db"
    with KnowledgeBase(db) as kb:
        kb.add_document(
            source="nhc.gov.cn", source_name="国家卫健委", tier=1,
            title="高血压用药常见误区",
            url="https://x/1",
            text="高血压需要长期规范用药，不能因为血压降下来了就自行停药。\n"
                 "随意停药会导致血压反弹，显著增加脑卒中与心肌梗死的发生风险。\n"
                 "降压药不会成瘾，也不存在所谓的停药窗口期，这些说法都是常见误解。\n"
                 "患者应当在医生指导下调整剂量，并坚持定期监测血压变化情况。\n"
                 "生活方式干预是治疗的基础，但不能替代药物治疗，两者需要配合进行。\n"
                 "减盐、控制体重、规律运动都有助于血压管理，但起效需要时间。\n"
                 "如果出现头晕、乏力等不适，应当及时就医而不是自行调整用药方案。")
        kb.add_document(
            source="piyao.org.cn", source_name="中国互联网联合辟谣平台", tier=1,
            title="隔夜水致癌吗",
            url="https://x/2",
            text="隔夜水中的亚硝酸盐含量远低于对人体产生危害的剂量。\n"
                 "实验数据显示，反复烧开的水同样不会致癌，相关传言缺乏科学依据。\n"
                 "饮用水安全的关键在于容器卫生与存放环境，而不是放置时间的长短。\n"
                 "只要使用干净的容器并在阴凉处存放，隔夜饮用并不会带来健康风险。\n"
                 "与其担心隔夜水，不如关注每日饮水量是否充足这个更重要的问题。\n"
                 "成年人每天建议饮水一千五百毫升至一千七百毫升，视气候与活动量调整。\n"
                 "白开水是最经济的补水方式，含糖饮料不宜用来替代日常饮水。")
        kb.add_document(
            source="kepuchina.cn", source_name="科普中国", tier=1,
            title="维生素C能预防感冒吗",
            url="https://x/3",
            text="维生素C不能预防普通感冒，这是多项临床研究得出的一致结论。\n"
                 "对于缩短感冒病程的作用也十分有限，通常只有轻微的缓解效果。\n"
                 "通过均衡饮食获取的维生素C比大剂量补充剂更为可靠和安全。\n"
                 "长期大剂量服用还可能带来胃肠不适等不良反应，不宜盲目补充。\n"
                 "预防感冒的关键是勤洗手、保持室内通风以及规律作息与充足睡眠。\n"
                 "在流感高发季节，接种疫苗是更为有效的防护手段，尤其对老年人。\n"
                 "如果症状持续加重或出现高热，应当及时就医而不是自行用药。")
        kb.rebuild_chunks(cfg)
    return str(db)


class FakeEmbedder:
    """确定性假编码器：按关键词命中数造一个稀疏向量。

    不追求语义真实，只保证"同一段文本两次编码结果相同"且"含关键词的向量更接近查询"，
    这样向量路的行为可预测、可断言。
    """

    backend, model = "fake", "fake-embed"

    def __init__(self, dim=16):
        self.dim = dim
        self.enabled = True

    @property
    def model_tag(self) -> str:
        return "fake:fake-embed"

    def _vec(self, text: str):
        import numpy as np

        v = np.zeros(self.dim, dtype="float32")
        for i, kw in enumerate(["高血压", "降压药", "隔夜水", "亚硝酸盐", "维生素C", "感冒"]):
            if kw in text:
                v[i] = 1.0
        if v.sum() == 0:
            v[-1] = 0.01
        return v

    def encode_one(self, text: str):
        return list(self._vec(text))

    def encode(self, texts):
        return [list(self._vec(t)) for t in texts]


@pytest.fixture()
def kb_with_vectors(tmp_kb, cfg):
    """给临时库灌入假向量。"""
    emb = FakeEmbedder()
    with KnowledgeBase(tmp_kb) as kb:
        rows = [dict(r) for r in kb.conn.execute("SELECT chunk_id, text FROM kb_chunks")]
        kb.put_vectors(emb.model_tag, [(r["chunk_id"], emb._vec(r["text"])) for r in rows])
    S.clear_matrix_cache()
    return tmp_kb


def _cfg_with(cfg, db, **hybrid):
    cfg["paths"]["kb_path"] = db
    cfg.setdefault("cognitive", {}).setdefault("retrieve", {}).setdefault("hybrid", {})
    cfg["cognitive"]["retrieve"]["hybrid"].update(hybrid)
    return cfg


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def test_rrf_favours_items_hit_by_both_paths():
    """两路都命中的条目应当排在最前——这正是融合的意义。"""
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    fused = S.fuse_rrf([a, b], k=60)
    assert fused[0]["chunk_id"] == "y"


def test_rrf_score_formula():
    fused = S.fuse_rrf([[{"chunk_id": "a"}], [{"chunk_id": "a"}]], k=60)
    # 两路都是第 1 名：2 * 1/(60+0+1)
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_dedupes_and_keeps_payload():
    fused = S.fuse_rrf([[{"chunk_id": "a", "text": "T"}], [{"chunk_id": "a", "text": "T"}]])
    assert len(fused) == 1 and fused[0]["text"] == "T"


def test_rrf_ignores_rows_without_key():
    fused = S.fuse_rrf([[{"chunk_id": "a"}, {"no_id": 1}]])
    assert [r["chunk_id"] for r in fused] == ["a"]


def test_rrf_rank_matters_more_than_count():
    """一路第 1 名 + 另一路没有，应当胜过两路都第 5 名。"""
    first = S.fuse_rrf([[{"chunk_id": "a"}], [{"chunk_id": "b"}]])
    assert first[0]["chunk_id"] == "a"


def test_rrf_empty():
    assert S.fuse_rrf([]) == []
    assert S.fuse_rrf([[], []]) == []


def test_rrf_merges_channel_marks_from_every_path():
    """两路都命中的块，融合后必须**同时**带 `bm25_rank` 与 `vec_rank`。

    这是 `matched_by` 能不能算出 "both" 的前提。早期版本用 `setdefault`
    只保留先见到的那一行，而 BM25 的列表总是排在向量前面，
    于是"两路都命中"的块一律被标成 `bm25`，`both` 这个分支永远走不到。
    丢掉的恰恰是融合里最可信的信号——hybrid 该赢过单路，靠的就是这批块。
    """
    bm25 = [{"chunk_id": "x", "bm25_rank": 0, "bm25_score": 1.0}]
    vec = [{"chunk_id": "x", "vec_rank": 3, "vec_score": 0.7}]
    fused = S.fuse_rrf([bm25, vec], k=60)
    assert len(fused) == 1
    row = fused[0]
    assert row["bm25_rank"] == 0 and row["vec_rank"] == 3
    assert row["bm25_score"] == 1.0 and row["vec_score"] == 0.7


def test_rrf_does_not_overwrite_existing_marks():
    """先到的名次不能被后到的覆盖——否则记录的就不是"最先召回它的那一路"了。"""
    a = [{"chunk_id": "x", "bm25_rank": 0}]
    b = [{"chunk_id": "x", "bm25_rank": 9}]
    assert S.fuse_rrf([a, b], k=60)[0]["bm25_rank"] == 0


def test_rrf_does_not_mutate_input_rows():
    """融合会往记录里并字段，不能把调用方的原始列表改掉（两路列表可能被复用）。"""
    bm25 = [{"chunk_id": "x", "bm25_rank": 0}]
    vec = [{"chunk_id": "x", "vec_rank": 1}]
    S.fuse_rrf([bm25, vec], k=60)
    assert "vec_rank" not in bm25[0]
    assert "bm25_rank" not in vec[0]


# ---------------------------------------------------------------------------
# 检索模式开关
# ---------------------------------------------------------------------------
def test_bm25_search_finds_exact_term(cfg, tmp_kb):
    cfg["paths"]["kb_path"] = tmp_kb
    rows = S.bm25_search(cfg, "亚硝酸盐", top_k=5)
    assert rows and "隔夜水" in rows[0]["title"]


def test_bm25_search_returns_rank_and_score(cfg, tmp_kb):
    cfg["paths"]["kb_path"] = tmp_kb
    rows = S.bm25_search(cfg, "高血压", top_k=5)
    assert rows
    assert rows[0]["bm25_rank"] == 0
    assert rows[0]["bm25_score"] >= rows[-1]["bm25_score"]


def test_vector_search_finds_related(cfg, kb_with_vectors):
    cfg = _cfg_with(cfg, kb_with_vectors)
    rows = S.vector_search(cfg, "高血压", top_k=3, embedder=FakeEmbedder())
    assert rows and "高血压" in rows[0]["text"]
    assert rows[0]["vec_rank"] == 0


def test_vector_search_empty_when_disabled(cfg, tmp_kb):
    cfg["paths"]["kb_path"] = tmp_kb
    from src.kb.embed import Embedder
    rows = S.vector_search(cfg, "高血压", top_k=3, embedder=Embedder(backend="none", model="m"))
    assert rows == []


def test_vector_search_handles_dim_mismatch(cfg, kb_with_vectors):
    """查询向量与库维度不一致时要安全返回空，而不是崩在矩阵乘法上。"""
    cfg = _cfg_with(cfg, kb_with_vectors)
    rows = S.vector_search(cfg, "高血压", top_k=3, embedder=FakeEmbedder(dim=99))
    assert rows == []


# ---------------------------------------------------------------------------
# 四种模式
# ---------------------------------------------------------------------------
def test_mode_bm25_only(cfg, tmp_kb):
    cfg = _cfg_with(cfg, tmp_kb, bm25=True, vector=False, rerank=False)
    out = S.hybrid_search(cfg, "亚硝酸盐")
    assert out["trace"]["bm25"] and not out["trace"]["vector"]
    assert out["results"]


def test_mode_vector_only(cfg, kb_with_vectors):
    cfg = _cfg_with(cfg, kb_with_vectors, bm25=False, vector=True, rerank=False)
    out = S.hybrid_search(cfg, "高血压", embedder=FakeEmbedder())
    assert out["trace"]["vector"] and not out["trace"]["bm25"]
    assert out["results"]


def test_mode_hybrid_fuses(cfg, kb_with_vectors):
    cfg = _cfg_with(cfg, kb_with_vectors, bm25=True, vector=True, rerank=False)
    out = S.hybrid_search(cfg, "高血压", embedder=FakeEmbedder())
    assert out["trace"]["bm25"] and out["trace"]["vector"]
    assert out["trace"]["n_fused"] >= 1


def test_single_path_does_not_fuse(cfg, tmp_kb):
    """单路时不该做融合——融合会平白改变名次，让消融结果失真。"""
    cfg = _cfg_with(cfg, tmp_kb, bm25=True, vector=False, rerank=False)
    out = S.hybrid_search(cfg, "高血压")
    assert all("rrf_score" not in r for r in out["results"])


def test_mode_hybrid_rerank_calls_reranker(cfg, kb_with_vectors):
    cfg = _cfg_with(cfg, kb_with_vectors, bm25=True, vector=True, rerank=True)

    class FakeReranker:
        backend, model = "fake", "fake-rerank"
        enabled = True

        def __init__(self):
            self.seen: list[str] = []

        def rerank(self, query, docs, top_n=None):
            from src.kb.rerank import RerankResult
            self.seen = list(docs)
            n = len(docs)
            return RerankResult(order=list(range(n)), scores=[0.5] * n)

    rr = FakeReranker()
    out = S.hybrid_search(cfg, "高血压", embedder=FakeEmbedder(), reranker=rr)
    assert out["trace"]["rerank"] is True
    assert rr.seen, "重排器应当收到候选文档"
    assert all("rerank_score" in r for r in out["results"])


def test_rerank_degradation_is_visible_in_trace(cfg, kb_with_vectors):
    """降级必须写进 trace——否则"为什么这行分数变了"解释不清。"""
    from src.kb.rerank import Reranker

    cfg = _cfg_with(cfg, kb_with_vectors, bm25=True, vector=True, rerank=True)
    broken = Reranker(backend="api", model="m", base_url="", api_key="", max_attempts=1)
    out = S.hybrid_search(cfg, "高血压", embedder=FakeEmbedder(), reranker=broken)
    assert out["trace"]["rerank_degraded"] is True
    assert out["trace"].get("rerank_reason")
    assert out["results"], "降级后仍要返回初检候选，不能返回空"


def test_no_paths_enabled_returns_empty(cfg, tmp_kb):
    cfg = _cfg_with(cfg, tmp_kb, bm25=False, vector=False, rerank=False)
    out = S.hybrid_search(cfg, "高血压")
    assert out["results"] == []


def test_top_k_limits_results(cfg, kb_with_vectors):
    cfg = _cfg_with(cfg, kb_with_vectors, bm25=True, vector=True, rerank=False)
    cfg["cognitive"]["retrieve"]["top_k"] = 1
    out = S.hybrid_search(cfg, "高血压", embedder=FakeEmbedder())
    assert len(out["results"]) <= 1


def test_retrieval_config_defaults(cfg):
    rc = S.retrieval_config(cfg)
    assert rc["bm25"] and rc["vector"] and rc["rerank"]
    assert rc["candidates"] == 20 and rc["top_k"] == 5


def test_missing_kb_path_returns_empty(cfg):
    cfg["paths"]["kb_path"] = "C:/不存在的路径/kb.db"
    assert S.bm25_search(cfg, "高血压", 5) == []
