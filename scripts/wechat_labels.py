"""公众号补充样本（D47）的筛选与话术标签：由 Claude 逐篇阅读后给出，每个标签附原文依据（D49）。

    python scripts/wechat_labels.py           # 校验并写出 data/labeled/wechat_supplement.jsonl + Excel
    python scripts/wechat_labels.py --eval    # 另外用行为面滤网跑一遍，与标签对比（调用 DeepSeek）

这是**大模型辅助标注**，判断规则与第一轮完全相同（behavior_labels_round1.RULES），
尤其是：被辟谣/防骗文章「引用」的推销原话不算文章自己的话术（那些原话另见 report_excerpts.py）。
不标真假：这批样本只用于行为面补充评测和报告案例，认知面成绩仍以 dev/test 为准。
单独统计，不并入 dev/test，不参与任何提示词或阈值调整。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.behavior_labels_round1 import RULES  # noqa: E402,F401  （同一套规则）

CACHE = ROOT / "data" / "raw" / "wechat_cache.json"
OUT = ROOT / "data" / "labeled" / "wechat_supplement.jsonl"
TYPES = ["转发诱导", "医疗行为建议", "消费导流", "金钱相关", "恐惧诉求"]

M, F, Z, X = "医疗行为建议", "恐惧诉求", "转发诱导", "消费导流"

# 不入选：编号 -> 原因
EXCLUDE = {
    "wx-704ff5c1c2f4": "与健康无关（化妆品特价）",
    "wx-d8478a84459d": "与健康无关（海淘特价）",
    "wx-941f34d0b7ca": "正文缺失（只抓到页面导航）",
    "wx-9034349f2493": "与 wx-9f04db06e3c4（南国都市报原文）重复，为转载",
}

# 入选样本的话术标签：编号 -> [(类型, 原文依据, 是否边界)]；入选但未列出 = 五类都不命中
LABELS: dict[str, list[tuple[str, str, bool]]] = {
    # ---- 家庭群式养生文 ----
    "wx-c76a98f2651a": [(M, "土豆中含有一种可治疗胃溃疡的特种抗菌分子", False),
                        (Z, "记得给身边的每一位朋友看看哦", False)],
    "wx-2a0768726a4b": [(M, "30天，停针又停药", False),
                        (M, "请立刻停止！！！", False),
                        (X, "二维码3秒→打开名片→ 添加金妹微信", False),
                        (F, "可是西药、胰岛素到底给患者带来了什么？", True)],
    "wx-3db5b86c0b26": [(M, "通过刺激可使迷走神经兴奋，促进胰岛素的分泌，从而有效降低血糖", True),
                        (Z, "一定要发给身边有需要的朋友！", False),
                        (X, "微信转账交费后预约成功", False)],
    "wx-3432b536e9e1": [(F, "未来世界范围内的死亡率或达到10%", False),
                        (Z, "将保存下来的图转发到自己的微信朋友圈", False),
                        (X, "感兴趣的童鞋可以报名庆余阁独家举办的运气培训班，详情可扫码加微信", False)],
    "wx-88232e35bc7b": [(M, "满足这3个条件，糖尿病人就有机会不用长期服药", True),
                        (X, "扫码下方二维码，或添加微信", True)],
    "wx-5b7fbdc9835a": [(M, "即便是你少用降糖药，甚至不用降糖药", True)],
    "wx-414ccbfde8df": [(M, "赶紧调整自己的食物量，调整自己的药物量", True)],
    "wx-f27287f5f28e": [(M, "对于肝火亢盛型高血压、顽固性高血压、头痛、眩晕等病均有良好的功效", False),
                        (X, "点击下方小程序即可购买！", False)],
    "wx-a986f710dfe9": [(Z, "更重要的是记得“分享”给身边人", False)],
    "wx-e4d6fdf005f9": [(Z, "并转发给更多人看", False)],
    "wx-fb2464c8cfb6": [(M, "就能轻松解决失眠问题", False),
                        (Z, "别忘了告诉朋友们", False)],
    "wx-aa2671dafd9f": [(X, "有需要的肾友可以点下面小程序购买！", False)],
    "wx-b081804ab5d9": [(M, "想要解决这些毛病，一定要多吃“它”", False)],
    # ---- 防骗文章自己的话术（引用的推销原话不算） ----
    "wx-f2c5d9d8fa70": [(X, "扫码咨询参会", True)],
    "wx-ebbef3fbb2c4": [(Z, "快告诉爸爸妈妈们，一定要警惕！", True)],
    "wx-5b7938a1eca5": [(Z, "转发这篇文章到家庭群", False)],
    "wx-621686949052": [(Z, "觉得有用，转给家人朋友", False)],
}


def build() -> list[dict]:
    arts = {a["id"]: a for a in json.loads(CACHE.read_text(encoding="utf-8"))["articles"].values()}
    unknown = (set(LABELS) | set(EXCLUDE)) - set(arts)
    if unknown:
        raise SystemExit(f"编号不存在：{sorted(unknown)}")
    bad, rows = [], []
    for aid, a in arts.items():
        if aid in EXCLUDE:
            continue
        labels = LABELS.get(aid, [])
        for t, ev, _ in labels:
            if ev not in a["title"] + "\n" + a["text"]:
                bad.append((aid, t, ev))
        rows.append({
            "id": aid, "source_type": "公众号原文", "title": a["title"], "account": a["account"],
            "text": a["text"], "url": a["url"], "query_kind": a["query_kind"],
            "behaviors": sorted({t for t, _, _ in labels}, key=TYPES.index),
            "behavior_evidence": [{"type": t, "evidence": ev, "borderline": b} for t, ev, b in labels],
            "cognitive_label": None, "label_by": "Claude 辅助标注（规则同第一轮）",
        })
    if bad:
        raise SystemExit("以下依据不是原文的逐字子串：\n" + "\n".join(map(str, bad)))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_pos = sum(bool(r["behaviors"]) for r in rows)
    print(f"入选 {len(rows)} 篇（有话术 {n_pos}，无话术 {len(rows) - n_pos}），剔除 {len(EXCLUDE)} 篇 → {OUT}")
    for t in TYPES:
        print(f"  {t}：{sum(t in r['behaviors'] for r in rows)}")
    return rows


def export(rows: list[dict], preds: dict[str, list[str]] | None = None) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "入选样本"
    head = ["编号", "公众号", "标题", "话术标签", "原文依据（★=边界）"] + (["系统识别"] if preds else []) + ["链接"]
    widths = [17, 14, 36, 22, 60] + ([22] if preds else []) + [20]
    for i, (h, w) in enumerate(zip(head, widths), 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="3B5B7A")
        ws.column_dimensions[c.column_letter].width = w
    ws.freeze_panes = "B2"
    rows = sorted(rows, key=lambda r: (not r["behaviors"], r["id"]))
    for n, r in enumerate(rows, 2):
        ev = "\n".join(f"{'★' if e['borderline'] else ''}【{e['type']}】{e['evidence']}"
                       for e in r["behavior_evidence"]) or "—"
        vals = [r["id"], r["account"], r["title"], "、".join(r["behaviors"]) or "无", ev]
        if preds:
            vals.append("、".join(preds.get(r["id"], [])) or "无")
        vals.append(r["url"])
        for c, v in enumerate(vals, 1):
            ws.cell(row=n, column=c, value=v).alignment = Alignment(wrap_text=True, vertical="top")
    ws2 = wb.create_sheet("剔除")
    ws2.append(["编号", "原因"])
    for k, v in EXCLUDE.items():
        ws2.append([k, v])
    ws3 = wb.create_sheet("判断规则")
    ws3["A1"] = RULES
    ws3["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws3.column_dimensions["A"].width = 110
    out = ROOT / "data" / "labeled" / f"公众号补充样本_话术标签_{time.strftime('%Y%m%d')}.xlsx"
    wb.save(out)
    print(f"Excel：{out}")
    return out


def evaluate(rows: list[dict]) -> dict[str, list[str]]:
    from src.behavior.screen import BehaviorScreen
    from src.config import load_config
    screen = BehaviorScreen.build(load_config())
    preds: dict[str, list[str]] = {}
    for r in rows:
        try:
            res = screen.screen(r["text"], r["title"])
            preds[r["id"]] = sorted({b.type for b in res.behaviors}, key=TYPES.index)
        except Exception as exc:                               # noqa: BLE001
            print(f"  ! {r['id']} 识别失败：{exc}")
            preds[r["id"]] = []
    print("\n类型        金标 预测 命中   P      R      F1")
    f1s = []
    for t in TYPES:
        g = {r["id"] for r in rows if t in r["behaviors"]}
        p = {k for k, v in preds.items() if t in v}
        tp = len(g & p)
        P = tp / len(p) if p else 0.0
        R = tp / len(g) if g else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        if g:
            f1s.append(F)
        print(f"  {t:<8} {len(g):>3} {len(p):>4} {tp:>4}  {P:.2f}  {R:.2f}  {F:.2f}")
    print(f"  宏 F1（有金标的类）：{sum(f1s) / len(f1s):.2f}")
    exact = sum(set(preds[r['id']]) == set(r["behaviors"]) for r in rows)
    print(f"  整篇完全一致：{exact}/{len(rows)}")
    print("\n不一致的样本：")
    for r in rows:
        if set(preds[r["id"]]) != set(r["behaviors"]):
            print(f"  {r['id']} {r['title'][:24]}｜标签={r['behaviors'] or '无'} 系统={preds[r['id']] or '无'}")
    out = ROOT / "results" / f"wechat_supplement_eval_{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    return preds


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    rows = build()
    preds = evaluate(rows) if args.eval else None
    export(rows, preds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
