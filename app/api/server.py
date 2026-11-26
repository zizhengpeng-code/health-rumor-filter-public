"""FastAPI 后端 —— 包装健康谣言双滤网核查 pipeline。

接口：
  POST /api/check    核查文章（URL 或正文）
  GET  /api/health   健康检查

启动：
  cd app/api && python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# 把项目根目录加到 sys.path，让 src 包可导入
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase, Field

from src.config import load_config
from src.models import CheckReport
from src.pipeline import Checker, check_text
from src.ingest.fetch import fetch_document, IngestError
from src.report.render import to_markdown, PRIORITY_HINT

# ---------------------------------------------------------------------------
app = FastAPI(title="健康谣言双滤网核查 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 开发期放通；上线收紧到小程序域名
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 全局 Checker 实例（加载本地知识库、嵌入模型等，初始化较慢，复用）
_checker: Checker | None = None


def get_checker() -> Checker:
    global _checker
    if _checker is None:
        _checker = Checker()
    return _checker


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class CheckRequest(PydanticBase):
    url: str = Field(default="", description="公众号文章链接")
    text: str = Field(default="", description="文章正文（粘贴模式）")
    title: str = Field(default="", description="文章标题（粘贴模式可选）")


class EvidenceItem(PydanticBase):
    id: int = 0
    source_name: str = ""
    domain: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    tier: int = 4
    score: float = 0.0


class BehaviorHitItem(PydanticBase):
    type: str
    evidence: str
    confidence: float = 0.0


class CheckResponse(PydanticBase):
    """返回给小程序的结构化结果。"""
    ok: bool = True
    error: str = ""

    title: str = ""
    source_url: str = ""
    latency_s: float = 0.0

    # 风险矩阵结论
    priority: str = ""                    # 尽快劝阻 / 提醒留意 / 告知误区 / 无需处理
    priority_hint: str = ""               # 给子女的一句话解释

    # 认知面
    cognitive_label: str = ""             # 真 / 假 / 部分属实 / 无法核查
    confidence: float = 0.0
    reasoning: str = ""

    # 行为面
    behavior_level: str = ""             # 高 / 中 / 低
    behaviors: list[BehaviorHitItem] = Field(default_factory=list)

    # 证据链
    evidence: list[EvidenceItem] = Field(default_factory=list)

    # 发给爸妈的话
    rebuttal: str = ""
    rebuttal_sources: list[str] = Field(default_factory=list)

    # 运行说明
    notes: list[str] = Field(default_factory=list)

    # Markdown 完整报告（供"查看完整报告"用）
    markdown_report: str = ""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def report_to_response(r: CheckReport, source_url: str = "") -> CheckResponse:
    """把 CheckReport 转成小程序友好的 JSON。"""
    v = r.verdicts[0] if r.verdicts else None
    d = r.decision
    b = r.behavior

    behaviors = []
    if b and b.behaviors:
        behaviors = [BehaviorHitItem(type=h.type, evidence=h.evidence, confidence=h.confidence)
                      for h in b.behaviors]

    evidence = []
    if v:
        from src.report.render import split_evidence
        shown, _ = split_evidence(v.evidence, v.used_evidence)
        evidence = [EvidenceItem(
            id=i,
            source_name=e.source_name or e.domain,
            domain=e.domain,
            title=e.title,
            url=e.url,
            snippet=e.snippet[:200] if e.snippet else "",
            tier=e.tier,
            score=e.score,
        ) for i, e in shown]

    return CheckResponse(
        title=r.title,
        source_url=source_url,
        latency_s=r.latency_s,
        priority=d.priority if d else "",
        priority_hint=PRIORITY_HINT.get(d.priority, "") if d else "",
        cognitive_label=v.label if v else "无法核查",
        confidence=v.confidence if v else 0.0,
        reasoning=v.reasoning if v else "",
        behavior_level=b.risk_level if b else "低",
        behaviors=behaviors,
        evidence=evidence,
        rebuttal=r.rebuttal,
        rebuttal_sources=r.rebuttal_sources,
        notes=r.notes,
        markdown_report=to_markdown(r),
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/check", response_model=CheckResponse)
def check_article(req: CheckRequest):
    """核查一篇文章。url 和 text 至少传一个。"""
    # 1. 输入校验
    if not req.url and not req.text:
        return CheckResponse(ok=False, error="请输入文章链接或粘贴正文")
    if req.text and len(req.text.strip()) < 50:
        return CheckResponse(ok=False, error="正文太短，请粘贴完整内容（至少 50 字）")

    cfg = load_config()

    try:
        if req.url:
            # 2a. 从 URL 抓取文章
            try:
                doc = fetch_document(req.url, cfg)
            except IngestError as exc:
                return CheckResponse(ok=False, error=str(exc))
            report = get_checker().check(doc)
            return report_to_response(report, source_url=req.url)
        else:
            # 2b. 直接核查粘贴的正文
            report = check_text(req.text, title=req.title, cfg=cfg)
            return report_to_response(report)
    except Exception as exc:
        traceback.print_exc()
        return CheckResponse(ok=False, error=f"核查流程出错：{exc}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
