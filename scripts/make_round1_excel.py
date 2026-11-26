"""第一轮标注合并：真假/主题取标注同学的结果，话术取 Claude 辅助标注，并生成人工盲检表。

    python scripts/make_round1_excel.py

输入：data/labeled/round1_from_annotator.xlsx（标注同学交回的表）
输出：
    data/labeled/round1_labeled.xlsx         合并后的完整结果（含话术依据、边界标记）
    data/labeled/round1_spotcheck_blind.xlsx 给抽检人的盲检表：60 条，不显示任何已有话术标签
    data/labeled/round1_spotcheck_key.json   盲检表的答案（编号 -> Claude 标签），算 Kappa 用，不要发给抽检人
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from scripts.behavior_labels_round1 import BORDERLINE_NEGATIVE, LABELS, RULES  # noqa: E402
from src.models import BEHAVIOR_TYPES  # noqa: E402

LAB = ROOT / "data" / "labeled"
HEAD_FILL = PatternFill("solid", fgColor="3B5B7A")
HEAD_FONT = Font(bold=True, color="FFFFFF")
YELLOW = PatternFill("solid", fgColor="FFF7D6")
WRAP = Alignment(wrap_text=True, vertical="top")
N_POS, N_NEG, SEED = 20, 40, 20261003


def load_annotated() -> list[dict]:
    ws = load_workbook(LAB / "round1_from_annotator.xlsx")["候选文章"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    return [dict(zip(head, r)) for r in rows[1:] if r[head.index("入选")] == "是"]


def header(ws, cols: list[tuple[str, int]]) -> None:
    for i, (name, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "B2"


def rules_sheet(wb: Workbook, title: str = "判断规则") -> None:
    ws = wb.create_sheet(title)
    ws.column_dimensions["A"].width = 110
    for i, line in enumerate(RULES.strip().splitlines(), 1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = WRAP
    ws["A1"].font = Font(bold=True, size=13)


def main() -> int:
    rows = load_annotated()

    # ---------- 合并结果 ----------
    wb = Workbook()
    ws = wb.active
    ws.title = "标注结果"
    cols = [("编号", 9), ("真假", 9), ("主题", 9), *[(t, 9) for t in BEHAVIOR_TYPES],
            ("话术依据（原文）", 40), ("边界案例", 8), ("备注", 20),
            ("标注同学原先的话术（未采用）", 22), ("标题", 24), ("样本正文", 60)]
    header(ws, cols)
    n_pos = 0
    for r_i, r in enumerate(rows, 2):
        labs = LABELS.get(r["编号"], [])
        n_pos += bool(labs)
        types = {t for t, _, _ in labs}
        ev = "\n".join(f"【{t}】{e}" for t, e, _ in labs)
        border = "是" if any(b for _, _, b in labs) or r["编号"] in BORDERLINE_NEGATIVE else ""
        note = r.get("备注") or ""
        if r["编号"] in BORDERLINE_NEGATIVE:
            note = (note + "；" if note else "") + "未标原因：" + BORDERLINE_NEGATIVE[r["编号"]]
        old = "、".join(t for t in BEHAVIOR_TYPES if r.get(t) == "是")
        vals = [r["编号"], r["真假"], r["主题"], *["是" if t in types else "" for t in BEHAVIOR_TYPES],
                ev, border, note, old, r.get("标题") or "", (r.get("样本正文") or "")[:32000]]
        for c_i, v in enumerate(vals, 1):
            ws.cell(row=r_i, column=c_i, value=v).alignment = WRAP
        ws.row_dimensions[r_i].height = 45
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(rows) + 1}"
    rules_sheet(wb)
    wb.save(LAB / "round1_labeled.xlsx")

    # ---------- 盲检表 ----------
    rnd = random.Random(SEED)
    pos = [r for r in rows if r["编号"] in LABELS]
    neg = [r for r in rows if r["编号"] not in LABELS]
    pick = rnd.sample(pos, min(N_POS, len(pos))) + rnd.sample(neg, min(N_NEG, len(neg)))
    rnd.shuffle(pick)

    wb2 = Workbook()
    intro = wb2.active
    intro.title = "说明"
    intro.column_dimensions["A"].width = 100
    for i, line in enumerate([
        "抽检说明",
        "",
        "1. 一共 60 条。请先读完「判断规则」表，再逐条看「样本正文」。",
        "2. 正文里出现对应话术，就在那一列选「是」；没有就空着。可以同时选多类。",
        "3. 请独立判断，不要和别人商量，也不要看别的标注结果。拿不准的在备注里写一句。",
        "4. 只标话术，不用管真假。",
        "5. 做完把文件发回即可。",
    ], 1):
        c = intro.cell(row=i, column=1, value=line)
        c.alignment = WRAP
    intro["A1"].font = Font(bold=True, size=14)

    ws2 = wb2.create_sheet("抽检")
    cols2 = [("编号", 9), *[(t, 9) for t in BEHAVIOR_TYPES], ("备注", 20), ("标题", 26), ("样本正文", 70)]
    header(ws2, cols2)
    dv = DataValidation(type="list", formula1='"是"', allow_blank=True)
    ws2.add_data_validation(dv)
    dv.add(f"B2:{get_column_letter(1 + len(BEHAVIOR_TYPES))}{len(pick) + 1}")
    for r_i, r in enumerate(pick, 2):
        vals = [r["编号"], *[""] * len(BEHAVIOR_TYPES), "", r.get("标题") or "",
                (r.get("样本正文") or "")[:32000]]
        for c_i, v in enumerate(vals, 1):
            cell = ws2.cell(row=r_i, column=c_i, value=v)
            cell.alignment = WRAP
            if 2 <= c_i <= 2 + len(BEHAVIOR_TYPES):
                cell.fill = YELLOW
        ws2.row_dimensions[r_i].height = 60
    rules_sheet(wb2)
    wb2.save(LAB / "round1_spotcheck_blind.xlsx")

    key = {r["编号"]: sorted({t for t, _, _ in LABELS.get(r["编号"], [])}) for r in pick}
    (LAB / "round1_spotcheck_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    cnt = Counter(t for r in rows for t, _, _ in LABELS.get(r["编号"], []))
    print(f"入选 {len(rows)} 条，有话术 {n_pos} 条；各类：{dict(cnt)}")
    print(f"盲检表 {len(pick)} 条（有话术 {sum(1 for r in pick if r['编号'] in LABELS)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
