"""从搜狗微信搜索收集真实公众号文章，作为行为面样本候选（D47）。

    python scripts/collect_wechat.py              # 按内置关键词收集（可断点续跑）
    python scripts/collect_wechat.py --export     # 只把已收集的导出成 Excel

为什么要这样收：行为面需要「原文里带话术」的文章，而辟谣平台摘录的谣言只剩一句话，
「转给家里老人」「加微信购买」「交费入会」这些话都被删了（D34/D38）；用户提供的辟谣链接里
也没有原文截图（D47）。直接搜还在发布中的公众号文章，拿到的才是原文。

**全部是公开发布的真实文章，不构造任何内容。** 文章来自普通公众号，不在知识库的权威来源里，无泄漏。

节流：搜狗反爬较严（D6：连续十几次即封 IP），每次请求间隔 3–6 秒；被拦截时保存进度并退出。
抓取复用 `src/ingest/fetch.py`（同时检验公众号抓取在真实链接上的表现）。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.ingest.fetch import IngestError, fetch_document  # noqa: E402

CACHE = ROOT / "data" / "raw" / "wechat_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9"}

# 按话术类型组织的检索词：用典型说法去搜，原文里带这些话的文章才会被搜到
QUERIES: dict[str, list[str]] = {
    "金钱相关": ["养生会员 入会 返利", "保健品 投资 分红", "健康讲座 会员费", "理疗 充值 套餐 老人",
                "养老项目 稳赚 不赔", "免费体验 理疗 押金", "交钱入会 养生"],
    "消费导流": ["加微信 领取 降糖", "点击下方 购买 降压", "原价 今天 特价 保健品", "扫码 领取 体验装 关节",
                "直播间 神药 下单"],
    "医疗行为建议": ["不用吃药 降压 偏方", "停药 糖尿病 食疗 根治", "代替胰岛素 食物", "偏方 根治 痛风",
                    "降压药 伤肝伤肾 别吃"],
    "转发诱导": ["转给家里老人 养生", "不转不是中国人 健康", "功德无量 转发 偏方", "快转给父母 一定要看"],
    "恐惧诉求": ["医生都不敢告诉你", "再不看就晚了 健康", "震惊 致癌 千万别吃", "紧急提醒 老人 千万不要"],
}

# 初步排序用：原文里出现这些话术迹象越多，越可能是有用的样本（只用于排序，不当作标签）
SIGNALS = {
    "金钱相关": r"入会|会员费|转账|汇款|投资|分红|返利|充值|押金|稳赚|年费",
    "消费导流": r"加微信|微信号|扫码|二维码|点击下方|下单|购买|领取|体验装|原价|特价|包邮|咨询电话",
    "医疗行为建议": r"不用吃药|停药|不吃药|代替.{0,4}药|根治|偏方|替代胰岛素|摆脱.{0,4}药",
    "转发诱导": r"转给|转发|分享给|不转不是|功德无量|告诉家人|让更多人",
    "恐惧诉求": r"再不看|就晚了|千万别|千万不要|震惊|紧急|医生都不敢|后果不堪",
}


class Blocked(RuntimeError):
    pass


DELAY = (3.0, 6.0)          # --slow 时改为 (12, 20)


def sleep() -> None:
    time.sleep(random.uniform(*DELAY))


def blocked_now() -> bool:
    import httpx
    r = httpx.get("https://weixin.sogou.com/weixin?type=2&query=" + urllib.parse.quote("健康"),
                  headers=UA, timeout=20, follow_redirects=True)
    return "antispider" in str(r.url)


def wait_until_unblocked(interval_min: int = 15, max_hours: float = 6) -> bool:
    """搜狗临时封锁会自动解除；只等待、不绕过验证码。"""
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        try:
            if not blocked_now():
                return True
        except Exception:                                      # noqa: BLE001
            pass
        print(f"  {time.strftime('%H:%M')} 仍被拦截，{interval_min} 分钟后再试", flush=True)
        time.sleep(interval_min * 60)
    return False


def search(client, query: str) -> list[tuple[str, str]]:
    r = client.get("https://weixin.sogou.com/weixin?type=2&query=" + urllib.parse.quote(query))
    if "antispider" in str(r.url) or "antispider" in r.text[:2000]:
        raise Blocked("搜狗反爬拦截")
    out = []
    for h, t in re.findall(r'<h3>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S):
        out.append((html.unescape(h), html.unescape(re.sub("<[^>]+>", "", t)).strip()))
    return out


def resolve(client, link: str, referer: str) -> str:
    r = client.get("https://weixin.sogou.com" + link, headers={**UA, "Referer": referer})
    if "antispider" in str(r.url):
        raise Blocked("搜狗反爬拦截（跳转链接）")
    return "".join(re.findall(r"url \+= '([^']*)'", r.text)).replace("@", "")


def signals(text: str) -> dict[str, int]:
    return {k: len(re.findall(p, text)) for k, p in SIGNALS.items()}


def collect(cfg) -> bool:
    """返回 True 表示全部关键词已搜完；False 表示中途被拦截。"""
    import httpx
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else \
        {"done_queries": [], "articles": {}}

    def save() -> None:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    client = httpx.Client(headers=UA, timeout=20, follow_redirects=True)
    seen_titles = {a["title"] for a in cache["articles"].values()}
    todo = [(k, q) for k, qs in QUERIES.items() for q in qs if q not in cache["done_queries"]]
    # 「金钱相关」搜出来几乎全是防骗新闻（这类话术多在私域群和线下，D47），排到最后
    todo.sort(key=lambda kq: kq[0] == "金钱相关")
    print(f"待搜索 {len(todo)} 组关键词；已收集 {len(cache['articles'])} 篇", flush=True)
    try:
        for kind, q in todo:
            hits = search(client, q)
            referer = "https://weixin.sogou.com/weixin?type=2&query=" + urllib.parse.quote(q)
            got = 0
            for link, title in hits:
                if title in seen_titles:
                    continue
                sleep()
                url = resolve(client, link, referer)
                if not url.startswith("http"):
                    continue
                try:
                    doc = fetch_document(url, cfg)
                except IngestError:
                    continue
                key = hashlib.md5((doc.title + doc.text[:200]).encode()).hexdigest()[:12]
                if key in cache["articles"]:
                    continue
                cache["articles"][key] = {
                    "id": "wx-" + key, "query": q, "query_kind": kind, "title": doc.title or title,
                    "account": doc.account, "text": doc.text, "url": url,
                    "signals": signals(doc.title + "\n" + doc.text),
                    "fetched": time.strftime("%Y-%m-%d %H:%M"),
                }
                seen_titles.add(title)
                got += 1
            cache["done_queries"].append(q)
            save()
            print(f"  [{kind}] {q}：新增 {got} 篇，累计 {len(cache['articles'])}", flush=True)
            sleep()
    except Blocked as exc:
        save()
        print(f"！{exc}，进度已保存。过一段时间再运行会从断点继续。", flush=True)
        return False
    return True


def export() -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    arts = sorted(cache["articles"].values(),
                  key=lambda a: (-sum(1 for v in a["signals"].values() if v), -sum(a["signals"].values())))
    wb = Workbook()
    ws = wb.active
    ws.title = "公众号文章"
    head = ["编号", "入选", "真假", "主题", "备注", "话术迹象（仅供参考，不是标签）", "搜索词", "标题",
            "公众号", "样本正文", "链接"]
    widths = [14, 7, 9, 9, 18, 26, 18, 30, 14, 70, 20]
    for i, (h, w) in enumerate(zip(head, widths), 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="3B5B7A")
        ws.column_dimensions[c.column_letter].width = w
    ws.freeze_panes = "B2"
    for r, a in enumerate(arts, 2):
        sig = "、".join(f"{k}×{v}" for k, v in a["signals"].items() if v) or "无"
        vals = [a["id"], "", "", "", "", sig, a["query"], a["title"], a["account"],
                a["text"][:32000], a["url"]]
        for c, v in enumerate(vals, 1):
            ws.cell(row=r, column=c, value=v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 60
    out = ROOT / "data" / "raw" / f"wechat_candidates_{time.strftime('%Y%m%d-%H%M')}.xlsx"
    wb.save(out)
    print(f"已导出 {len(arts)} 篇：{out}")
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--slow", action="store_true", help="请求间隔拉长到 12–20 秒")
    ap.add_argument("--wait", action="store_true", help="先等搜狗临时封锁自动解除（每 15 分钟检查）")
    ap.add_argument("--loop", action="store_true",
                    help="被拦截后等解封再继续，直到全部关键词搜完（最长 12 小时）")
    args = ap.parse_args(argv)
    global DELAY
    if args.slow:
        DELAY = (12.0, 20.0)
    if args.loop:
        cfg, deadline = load_config(), time.time() + 12 * 3600
        while time.time() < deadline:
            if collect(cfg):
                print("全部关键词已搜完。", flush=True)
                break
            time.sleep(10 * 60)                                # 刚被拦截，先歇一会再查
            if not wait_until_unblocked(max_hours=(deadline - time.time()) / 3600):
                print("等待超时，仍被拦截；稍后手动重跑。", flush=True)
                break
    elif args.wait and not wait_until_unblocked():
        print("等待超时，仍被拦截；稍后手动重跑。", flush=True)
    elif not args.export:
        collect(load_config())
    if CACHE.exists():
        export()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
