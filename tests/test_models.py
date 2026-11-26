"""数据模型（schema 契约）测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    BehaviorHit, BehaviorResult, CheckReport, CognitiveVerdict, Document, Evidence,
    LabeledSample, SEVERITY,
)


def test_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        Document(id="d1", text="   ")


def test_document_char_count():
    d = Document(id="d1", text="高血压不能停药")
    assert d.char_count == 7


def test_behavior_hit_confidence_bounds():
    with pytest.raises(ValidationError):
        BehaviorHit(type="恐惧诉求", evidence="x", confidence=1.5)
    with pytest.raises(ValidationError):
        BehaviorHit(type="恐惧诉求", evidence="x", confidence=-0.1)


def test_behavior_hit_evidence_required():
    with pytest.raises(ValidationError):
        BehaviorHit(type="恐惧诉求", evidence="", confidence=0.9)


def test_behavior_result_hit_types_dedup_sorted():
    r = BehaviorResult(risk_level="中", behaviors=[
        BehaviorHit(type="恐惧诉求", evidence="a", confidence=0.9),
        BehaviorHit(type="恐惧诉求", evidence="b", confidence=0.8),
        BehaviorHit(type="转发诱导", evidence="c", confidence=0.7),
    ])
    assert r.hit_types == ["转发诱导", "恐惧诉求"]
    assert r.to_eval() == ["转发诱导", "恐惧诉求"]


def test_behavior_type_must_be_in_enum():
    with pytest.raises(ValidationError):
        BehaviorHit(type="随便编的类型", evidence="x", confidence=0.9)  # type: ignore[arg-type]


def test_evidence_defaults_tier_and_score():
    e = Evidence(id=0, claim_id=0, title="t", url="https://www.piyao.org.cn/a.html")
    assert e.tier == 4 and e.score == 0.0


def test_claim_importance_bounds():
    from src.models import Claim
    with pytest.raises(ValidationError):
        Claim(id=0, text="x", importance=9)


def test_labeled_sample_roundtrip():
    raw = ('{"id":"s1","text":"喝醋软化血管","topic":"慢病","cognitive_label":"假",'
           '"behaviors":["医疗行为建议"]}')
    s = LabeledSample.model_validate_json(raw)
    assert s.topic == "慢病" and s.behaviors == ["医疗行为建议"]
    assert LabeledSample.model_validate_json(s.model_dump_json()).id == "s1"


def test_labeled_sample_rejects_bad_label():
    with pytest.raises(ValidationError):
        LabeledSample.model_validate_json(
            '{"id":"s1","text":"x","cognitive_label":"半真半假"}')


def test_check_report_worst_label():
    r = CheckReport(doc_id="d1", verdicts=[
        CognitiveVerdict(claim_id=0, claim_text="a", label="真"),
        CognitiveVerdict(claim_id=1, claim_text="b", label="假"),
        CognitiveVerdict(claim_id=2, claim_text="c", label="部分属实"),
    ])
    assert r.worst_label == "假"


def test_check_report_empty_verdicts_is_uncertain():
    assert CheckReport(doc_id="d1").worst_label == "无法核查"


def test_severity_order_is_total():
    assert SEVERITY["假"] > SEVERITY["部分属实"] > SEVERITY["无法核查"] > SEVERITY["真"]
