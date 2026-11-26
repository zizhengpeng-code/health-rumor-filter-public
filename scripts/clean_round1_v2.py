"""第一轮评测集清洗（D55，2026-10-12 用户决定）。

    python scripts/clean_round1_v2.py

规则（全部按文本形态判定，不看任何系统预测结果）：

1. 问句形式的谣言句（「辟谣文中的谣言句」且以问号结尾）→ 机械改为陈述句：
   去掉句首编号（如「1.」）、句末「吗/呢」、把问号改成句号。原文的字不增不改，真伪标签不变。
   例：「尿酸高就会得痛风吗？」→「尿酸高就会得痛风。」
   原因：辟谣标题里的问句本身不断言谣言，模型可能把它当成在提问而判「真」（D38 发现 4）。

3. 重复文章（全文逐字 12 字片段重合 ≥0.8）：跨集合的去掉测试集副本（防泄漏），开发集内的去掉编号靠后者。
   不是可核查说法的样本（导航页、标题片段）一并剔除。

2. 标题抓成了机构 / 网站名的样本（D38 发现 4）：
   - 正文是正常的健康内容 → 清空标题，保留样本；
   - 正文是导航页、文章列表或与健康说法无关的工作汇报 → 剔除（REMOVE，逐条写明原因）。

原划分备份为 dev_v1.jsonl / test_v1.jsonl；清洗后的写回 dev.jsonl / test.jsonl，划分不变（不重新抽样）。
每条被改动的样本在 notes 里追加说明，清洗记录写入 data/labeled/clean_v2_log.json。
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
META = json.loads((ROOT / "data" / "labeled" / "round1_meta.json").read_text(encoding="utf-8"))

# 整个标题就是机构名（前面不能有「_ | ？ -」等分隔符——「单纯补钙不能治疗骨质疏松_云南省卫生健康委员会」是正常标题）
SITE_TITLE = re.compile(r"^[^_|？?\-—]{2,24}(委员会|控制中心|预防控制局|管理局|安全所|控制所|健康所|人民政府([_\-].{0,8})?)$")

# 正文不是可核查的健康说法 → 剔除
REMOVE = {
    "c0272": "正文是云南省疾控局首页的新闻标题列表，不是一篇健康科普",
    "c0264": "正文是页面导航（各省卫健委名称列表），实质内容不足",
    "c0253": "正文是多篇文章的摘要列表，不是单一健康说法",
    "c0162": "正文是 12320 热线的工作汇报，不含可核查的健康说法",
    "c0286": "正文是营养与健康所网站的新闻标题列表，不是健康说法",
    "c0024": "只有一个标题片段（「甲状腺”和“碘盐”的各种纠葛」），不是可核查的说法",
    # 重复文章（全文 12 字片段重合 ≥0.8）。跨集合的去掉测试集那份——开发集那份已在开发中被看过，
    # 留着测试集副本就等于测试集泄漏；开发集内部重复的去掉编号靠后的一份，避免同一篇文章重复计分。
    "c0164": "与开发集 c0163 为同一篇文章（重合 1.00），测试集泄漏",
    "c0183": "与开发集 c0403 为同一篇文章（重合 0.88），测试集泄漏",
    "c0204": "与开发集 c0205 为同一篇文章（重合 0.83），测试集泄漏",
    "c0225": "与开发集 c0228 为同一篇文章（重合 0.88），测试集泄漏",
    "c0165": "与 c0163 为同一篇文章（重合 0.95），开发集内重复",
    "c0174": "与 c0173 为同一篇文章（重合 0.96），开发集内重复",
    "c0248": "与 c0179 为同一篇文章（重合 0.99），开发集内重复",
    "c0402": "与 c0180 为同一篇文章（重合 0.97），开发集内重复",
}


def question_to_statement(text: str) -> str:
    t = re.sub(r"^\s*\d+[.、．]\s*", "", text.strip())
    t = re.sub(r"(吗|呢)?\s*[？?]", "。", t)
    return t


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    log = {"question_rewritten": [], "title_cleared": [], "removed": []}
    for split in ("dev", "test"):
        src = SPLITS / f"{split}.jsonl"
        bak = SPLITS / f"{split}_v1.jsonl"
        if not bak.exists():
            shutil.copy2(src, bak)                             # 只在第一次运行时备份原始划分
        rows = [json.loads(l) for l in bak.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for r in rows:
            rid, kind = r["id"], META[r["id"]]["kind"]
            if rid in REMOVE:
                log["removed"].append({"id": rid, "split": split, "reason": REMOVE[rid]})
                continue
            if kind == "辟谣文中的谣言句" and re.search(r"[？?]\s*$", r["text"].strip()):
                new = question_to_statement(r["text"])
                log["question_rewritten"].append({"id": rid, "split": split, "from": r["text"], "to": new})
                r["text"] = new
                r["notes"] = (r.get("notes") or "") + "；清洗v2=问句机械改为陈述句"
            if SITE_TITLE.match(r.get("title", "").strip()):
                log["title_cleared"].append({"id": rid, "split": split, "title": r["title"]})
                r["title"] = ""
                r["notes"] = (r.get("notes") or "") + "；清洗v2=标题为机构名，已清空"
            out.append(r)
        src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        print(f"{split}: {len(rows)} → {len(out)}")
    (ROOT / "data" / "labeled" / "clean_v2_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in log.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
