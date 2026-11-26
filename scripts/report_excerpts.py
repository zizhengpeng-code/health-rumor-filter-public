"""从防骗报道里摘出推销方原话，作为「金钱相关」的补充样本（D48）。

    python scripts/report_excerpts.py            # 校验并写出 data/labeled/report_excerpts.jsonl
    python scripts/report_excerpts.py --eval     # 另外用行为面滤网跑一遍，看能否识别（调用 DeepSeek）

为什么需要：公开公众号里搜不到「金钱相关」话术的原文，搜到的几乎都是事后防骗报道（D47）。
报道里用引号原样引用了推销方的说法，这是真实话术的唯一公开来源。

**摘录标准（A 级，进入评测）**：
1. 必须是报道里**加引号的直接引语**，且报道写明出自推销方（商家、工作人员、宣传材料）；
2. 记者转述（如「李明告诉老人们，每年交 570 元会费……」）不收——那是记者的话，不是原话；
3. 受害者、警方、律师的话不收；与健康养生无关的骗局（纪念币、字画）不收。

B 级只记录、不进评测：自媒体的亲历转述、防骗科普里作者编写的情景对话——来源无法核实是原话。

每条摘录都会程序校验「是报道正文的逐字子串」，保证没有改写或构造。
这些样本单独评测，不并入 dev/test，也不参与任何提示词或阈值调整。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "raw" / "wechat_cache.json"
OUT = ROOT / "data" / "labeled" / "report_excerpts.jsonl"

# (报道编号, 逐字摘录, 报道中写明的说话方, 话术标签, 级别)
EXCERPTS: list[tuple[str, str, str, list[str], str]] = [
    ("wx-0c83a771df56", "当天营业额30%的广告费作为返利回馈投资者",
     "商家宣传（老妈乐）", ["金钱相关"], "A"),
    ("wx-0c83a771df56", "只要1280元的入会费，便可以在10个月内返利回本，最高可以拿到2000元",
     "老妈乐会员制度宣称", ["金钱相关"], "A"),
    ("wx-0c83a771df56", "投资25600返利64320",
     "老妈乐会员档位（翡翠会员）", ["金钱相关"], "A"),
    ("wx-0c83a771df56", "投资7200元，返利22000元",
     "博邦商城宣称", ["金钱相关"], "A"),
    ("wx-fc32a60bf34a", "全球最简单、最强大的公益创业平台，秉承着众扶是美德，是爱的理念，"
                        "公司这个平台不收大家一分钱，只为人人可以参与创富。",
     "健康俱乐部工作人员", ["金钱相关"], "A"),
    # ---- B 级：只记录，不进评测 ----
    ("wx-e4a60cac20cc", "这个不是普通补品，是内部渠道、国家扶持项目，不光能调理身体，还能投资返利。"
                        "买得多赚得多，存着还能翻倍，专门带自家人发财的，外人我都不介绍！",
     "自媒体作者转述亲戚推销原话", ["金钱相关", "消费导流"], "B"),
    ("wx-bfdb87c32498", "您可以先充值，再介绍朋友进来，我们给您返利。错过首批入会，以后就收费了，"
                        "以后您要买我们产品，从充值金扣除就行了。",
     "防骗科普情景对话（作者编写）", ["金钱相关", "恐惧诉求"], "B"),
]


def build() -> list[dict]:
    arts = {a["id"]: a for a in json.loads(CACHE.read_text(encoding="utf-8"))["articles"].values()}
    rows, bad = [], []
    for n, (aid, quote, speaker, behaviors, tier) in enumerate(EXCERPTS, 1):
        a = arts.get(aid)
        if a is None or quote not in a["text"]:
            bad.append((aid, quote[:20]))
            continue
        rows.append({"id": f"rx-{n:03d}", "text": quote, "source_type": "报道摘录", "tier": tier,
                     "behaviors": behaviors, "cognitive_label": None, "speaker": speaker,
                     "report_title": a["title"], "report_account": a["account"], "url": a["url"]})
    if bad:
        raise SystemExit(f"以下摘录不是报道原文的逐字子串，拒绝写出：{bad}")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"已写出 {len(rows)} 条（A 级 {sum(r['tier'] == 'A' for r in rows)}）：{OUT}")
    return rows


def evaluate(rows: list[dict]) -> None:
    from src.behavior.screen import BehaviorScreen
    from src.config import load_config
    screen = BehaviorScreen.build(load_config())
    for tier in ("A", "B"):
        sub = [r for r in rows if r["tier"] == tier]
        hit = 0
        print(f"\n== {tier} 级（{len(sub)} 条）==")
        for r in sub:
            res = screen.screen(r["text"])
            types = [b.type for b in res.behaviors]
            ok = "金钱相关" in types
            hit += ok
            print(f"  {'✓' if ok else '✗'} {r['id']} 风险={res.risk_level} 命中={types or '无'}｜{r['text'][:30]}")
        print(f"  金钱相关召回：{hit}/{len(sub)}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    rows = build()
    if ap.parse_args().eval:
        evaluate(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
