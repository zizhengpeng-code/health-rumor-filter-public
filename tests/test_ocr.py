"""截图核查与不支持平台提示（D54）的最小测试。全部离线，不联网、不花钱。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import load_config
from src.ingest.fetch import IngestError, extract_url, fetch_document, unsupported_platform
from src.ingest.ocr import OCRError, ocr_images


class FakeOCR:
    """模拟 OpenAI 兼容接口：按顺序吐出每张图的识别结果，并记下收到的图片。"""

    def __init__(self, texts):
        self.texts, self.urls = list(texts), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages):
        self.urls.append(messages[0]["content"][0]["image_url"]["url"])
        msg = SimpleNamespace(content=self.texts.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20
JPG = b"\xff\xd8\xff" + b"0" * 20


def test_ocr_joins_screens_in_order_and_sends_correct_mime():
    fake = FakeOCR(["第一屏：喝莓茶能停药", "第二屏：加微信领取"])
    text = ocr_images([PNG, JPG], load_config(), client=fake)
    assert text == "第一屏：喝莓茶能停药\n\n第二屏：加微信领取"
    assert fake.urls[0].startswith("data:image/png;base64,")
    assert fake.urls[1].startswith("data:image/jpeg;base64,")


def test_ocr_rejects_too_many_images():
    with pytest.raises(OCRError, match="最多"):
        ocr_images([JPG] * 4, load_config(), client=FakeOCR(["x"] * 4))


def test_extract_url_from_app_share_text():
    raw = "【莓茶降糖神方 - 小红书】 😆 abc123 😆 http://xhslink.com/a/AbCd，复制本条信息，打开小红书App查看"
    assert extract_url(raw) == "http://xhslink.com/a/AbCd"
    assert extract_url("https://mp.weixin.qq.com/s/xyz") == "https://mp.weixin.qq.com/s/xyz"


@pytest.mark.parametrize("url,name", [
    ("http://xhslink.com/a/AbCd", "小红书"),
    ("https://www.xiaohongshu.com/explore/123", "小红书"),
    ("https://channels.weixin.qq.com/web/pages/feed?eid=1", "视频号"),
    ("https://v.douyin.com/abc/", "抖音"),
    ("https://mp.weixin.qq.com/s/xyz", ""),
])
def test_unsupported_platform_detection(url, name):
    assert unsupported_platform(url) == name


def test_unsupported_platform_link_gives_clear_hint_without_fetching():
    called = []
    with pytest.raises(IngestError, match="上传截图"):
        fetch_document("看看这个 http://xhslink.com/a/AbCd", load_config(), fetcher=lambda u: called.append(u))
    assert called == []                                          # 根本不去抓


def test_cloud_server_only_accepts_own_screenshot_files(monkeypatch):
    import importlib
    import sys
    from pathlib import Path
    monkeypatch.setenv("KB_FILEID", "cloud://prod-abc.1234-prod-abc-100/kb/kb_cloud.db.gz")
    monkeypatch.setenv("HRF_KB_PATH", str(Path(__file__).parent / "_no_such_kb.db"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "cloudrun"))
    monkeypatch.setattr("threading.Thread.start", lambda self: None)   # 不在测试里真去下载知识库
    server = importlib.import_module("server")
    ok = "cloud://prod-abc.1234-prod-abc-100/screenshots/1.jpg"
    assert server._check_image_ids([ok]) == ""
    assert server._check_image_ids(["cloud://prod-abc.1234-prod-abc-100/kb/kb_cloud.db.gz"])   # 不在截图目录
    assert server._check_image_ids(["cloud://prod-evil.999/screenshots/1.jpg"])               # 别的环境
    assert server._check_image_ids(["https://example.com/screenshots/1.jpg"])                 # 外部网址
    assert server._check_image_ids([ok] * 4)                                                  # 超过张数


def test_coordinate_output_is_retried_with_second_prompt():
    """实测 qwen-vl-ocr 在某些提示词下输出坐标；要换提示词重试，而不是拿坐标去核查。"""
    fake = FakeOCR(["252,128,95,423,90\n332,306,95,577,90", "每天喝一杯莓茶，30天停针又停药"])
    assert ocr_images([JPG], load_config(), client=fake) == "每天喝一杯莓茶，30天停针又停药"
    assert len(fake.urls) == 2


def test_coordinates_twice_raises_instead_of_checking_numbers():
    fake = FakeOCR(["252,128,95,423,90", "332,306,95,577,90"])
    with pytest.raises(OCRError, match="没能识别出文字"):
        ocr_images([JPG], load_config(), client=fake)
