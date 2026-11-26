"""第二版话术盲检表 + Kappa 计算。

第一版盲检（round1_spotcheck_returned_v1.xlsx）Kappa≈0：抽检人给 60 行里的 59 行各挑了恰好一类，
把「出现了才勾」当成了「五选一」（D35）。第二版针对这个设计缺陷：
  - 加「无话术」列，每行必须二选一（要么无话术，要么勾具体类型），表里有「检查」列提示未填/冲突
  - 先做 10 道练习题（不在正式 60 条里），做完对答案再开始正式抽检

    python scripts/make_spotcheck_v2.py                    # 生成 data/labeled/round1_spotcheck_v2.xlsx
    python scripts/make_spotcheck_v2.py --score <交回的文件>  # 和 round1_spotcheck_key.json 比对算 Kappa
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.formatting.rule import FormulaRule  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from scripts.behavior_labels_round1 import RULES  # noqa: E402
from src.models import BEHAVIOR_TYPES  # noqa: E402

LAB = ROOT / "data" / "labeled"
HEAD_FILL = PatternFill("solid", fgColor="3B5B7A")
HEAD_FONT = Font(bold=True, color="FFFFFF")
YELLOW = PatternFill("solid", fgColor="FFF7D6")
RED = PatternFill("solid", fgColor="F8D0D0")
WRAP = Alignment(wrap_text=True, vertical="top")
NONE = "无话术"

# 练习题：编号 -> (答案, 理由)。挑的都是第一版里最容易误判的类型，且不在正式 60 条里。
PRACTICE: dict[str, tuple[list[str], str]] = {
    "c0098": ([], "只是一个说法（「撸串会致癌」），没有劝人做事，也没有吓唬人的话，所以无话术。说法错不错交给真假那一栏。"),
    "c0020": (["医疗行为建议"], "把一种食物当成治病手段，而且没加「有助于」之类的限定（「能降血糖」）。"),
    "c0101": ([], "错误结论本身不算恐惧诉求。要有「危险了！」「再不……就晚了」这类吓唬人的说法才算。"),
    "c0161": (["转发诱导"], "标题里有「请转给爸妈」。正文提醒「不要盲目下单」是在劝人别买，不算消费导流。"),
    "c0087": ([], "「味精会致癌」是一个说法，没有推人做事，无话术。"),
    "c0010": (["医疗行为建议"], "原文劝人「不需要去医院检查」，属于劝人不就医。"),
    "c0189": ([], "官方的使用注意事项。「请遵医嘱购买使用」是正常提醒，没有给购买渠道，不算消费导流。"),
    "c0259": (["恐惧诉求"], "标题「救命！……竟藏着致癌危机？」用夸张的话吓唬人。"),
    "c0219": ([], "「赶紧一起来了解一下吧」只是招呼读者，没有吓唬，不算恐惧诉求。"),
    "c0192": (["消费导流"], "文末推荐医院科室的理疗服务，并写了「欢迎您来……咨询」和地址。"),
}


def load_rows() -> dict[str, dict]:
    ws = load_workbook(LAB / "round1_from_annotator.xlsx")["候选文章"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    return {r[0]: dict(zip(head, r)) for r in rows[1:]}


def header(ws, cols: list[tuple[str, int]]) -> None:
    for i, (name, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "C2"


def fill_sheet(ws, ids: list[str], data: dict[str, dict]) -> None:
    """编号 | 检查 | 无话术 | 五类 | 备注 | 标题 | 样本正文。「检查」列用公式提示未填/冲突。"""
    cols = [("编号", 9), ("检查", 8), (NONE, 8), *[(t, 9) for t in BEHAVIOR_TYPES],
            ("备注", 18), ("标题", 26), ("样本正文", 70)]
    header(ws, cols)
    first, last = get_column_letter(3), get_column_letter(3 + len(BEHAVIOR_TYPES))
    types_first = get_column_letter(4)
    n = len(ids)
    dv = DataValidation(type="list", formula1='"是"', allow_blank=True,
                        showErrorMessage=True, error="请从下拉框选「是」，或留空")
    ws.add_data_validation(dv)
    dv.add(f"{first}2:{last}{n + 1}")
    for i, cid in enumerate(ids, 2):
        r = data[cid]
        ws.cell(row=i, column=1, value=cid)
        ws.cell(row=i, column=2, value=(
            f'=IF(AND({first}{i}="是",COUNTIF({types_first}{i}:{last}{i},"是")>0),"冲突",'
            f'IF(COUNTIF({first}{i}:{last}{i},"是")=0,"未填","OK"))'))
        ws.cell(row=i, column=4 + len(BEHAVIOR_TYPES) + 1, value="")
        ws.cell(row=i, column=4 + len(BEHAVIOR_TYPES) + 2, value=r.get("标题") or "")
        ws.cell(row=i, column=4 + len(BEHAVIOR_TYPES) + 3, value=(r.get("样本正文") or "")[:32000])
        for c in range(1, len(cols) + 1):
            ws.cell(row=i, column=c).alignment = WRAP
        for c in range(3, 4 + len(BEHAVIOR_TYPES)):
            ws.cell(row=i, column=c).fill = YELLOW
        ws.row_dimensions[i].height = 60
    ws.conditional_formatting.add(
        f"B2:B{n + 1}", FormulaRule(formula=[f'B2<>"OK"'], fill=RED))


def make() -> Path:
    data = load_rows()
    key = json.loads((LAB / "round1_spotcheck_key.json").read_text(encoding="utf-8"))
    assert not set(PRACTICE) & set(key), "练习题不能和正式抽检重复"
    ids = list(key)
    random.Random(7).shuffle(ids)          # 换个顺序，避免和第一版逐行对照

    wb = Workbook()
    intro = wb.active
    intro.title = "先看这里"
    intro.column_dimensions["A"].width = 100
    lines = [
        "话术抽检（第二版）",
        "",
        "我们要标的是：这段文字里，有没有「推着读者去做事」的说法。",
        "大部分一句话的谣言都没有话术——那就勾「无话术」。勾「无话术」是完全正常、而且最常见的答案。",
        "",
        "三个例子：",
        "  「烧烤类食品含有苯并芘，可致癌。」→ 无话术（只是一个说法，没推人做事）",
        "  「多吃黑木耳能降血压。」→ 医疗行为建议（把食物当成治病手段）",
        "  「赶紧转给家里老人！再不看就晚了！」→ 转发诱导 + 恐惧诉求",
        "",
        "怎么做：",
        "  1. 先做「练习」表的 10 题，做完再看「练习答案」表对一下。错了就看理由，弄明白再往下。",
        "  2. 再做「正式抽检」表的 60 条。每行要么勾「无话术」，要么勾一个或几个具体类型，不能都空着，也不能同时勾。",
        "     「检查」列会自动提示：显示「未填」或「冲突」的行要改，全部变成「OK」才算做完。",
        "  3. 长文章请读完再判断，话术常常在结尾（比如「欢迎来咨询」「转给爸妈」）。",
        "  4. 自己独立判断，不要和别人商量。拿不准的写在备注里。",
        "",
        "判断标准见「判断规则」表。做完把整个文件发回即可。",
    ]
    for i, line in enumerate(lines, 1):
        intro.cell(row=i, column=1, value=line).alignment = WRAP
    intro["A1"].font = Font(bold=True, size=14)
    for i in (3, 4):
        intro.cell(row=i, column=1).font = Font(bold=True)

    fill_sheet(wb.create_sheet("练习"), list(PRACTICE), data)

    ans = wb.create_sheet("练习答案")
    header(ans, [("编号", 9), ("答案", 16), ("理由", 70), ("样本正文（节选）", 50)])
    for i, (cid, (types, why)) in enumerate(PRACTICE.items(), 2):
        vals = [cid, "、".join(types) or NONE, why, (data[cid].get("样本正文") or "")[:150]]
        for c, v in enumerate(vals, 1):
            ans.cell(row=i, column=c, value=v).alignment = WRAP
        ans.row_dimensions[i].height = 45

    fill_sheet(wb.create_sheet("正式抽检"), ids, data)

    rules = wb.create_sheet("判断规则")
    rules.column_dimensions["A"].width = 110
    for i, line in enumerate(RULES.strip().splitlines(), 1):
        rules.cell(row=i, column=1, value=line).alignment = WRAP
    rules["A1"].font = Font(bold=True, size=13)

    out = LAB / "round1_spotcheck_v2.xlsx"
    wb.save(out)
    return out


def _kappa(a: list[bool], b: list[bool]) -> tuple[float, float | None]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (None if pe >= 1 else (po - pe) / (1 - pe))


def score(path: str) -> int:
    key = json.loads((LAB / "round1_spotcheck_key.json").read_text(encoding="utf-8"))
    wb = load_workbook(path)
    ws = wb["正式抽检"] if "正式抽检" in wb.sheetnames else wb["抽检"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    got = {r[0]: dict(zip(head, r)) for r in rows[1:] if r[0] in key}

    def on(v) -> bool:                      # 有人会填 1 而不是「是」
        return v not in (None, "", 0, "否")

    unfinished = [c for c, r in got.items()
                  if not on(r.get(NONE)) and not any(on(r.get(t)) for t in BEHAVIOR_TYPES)]
    conflict = [c for c, r in got.items()
                if on(r.get(NONE)) and any(on(r.get(t)) for t in BEHAVIOR_TYPES)]
    print(f"交回 {len(got)}/{len(key)} 条；未填 {len(unfinished)}，冲突 {len(conflict)}")
    ids = list(got)
    for t in [*BEHAVIOR_TYPES, "有任何话术"]:
        if t == "有任何话术":
            a = [bool(key[c]) for c in ids]
            b = [any(on(got[c].get(x)) for x in BEHAVIOR_TYPES) for c in ids]
        else:
            a = [t in key[c] for c in ids]
            b = [on(got[c].get(t)) for c in ids]
        po, k = _kappa(a, b)
        ks = "—（两边都没有）" if k is None else f"{k:.2f}"
        print(f"  {t}：Claude {sum(a)} / 抽检 {sum(b)}，一致率 {po:.2f}，Kappa {ks}")
    diff = [(c, key[c], [t for t in BEHAVIOR_TYPES if on(got[c].get(t))]) for c in ids
            if set(key[c]) != {t for t in BEHAVIOR_TYPES if on(got[c].get(t))}]
    print(f"不一致 {len(diff)} 条：")
    for c, a, b in diff:
        print(f"  {c}  Claude={a or NONE}  抽检={b or NONE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", help="交回的抽检文件路径；不给则生成第二版抽检表")
    args = ap.parse_args(argv)
    if args.score:
        return score(args.score)
    print(f"已生成：{make()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
