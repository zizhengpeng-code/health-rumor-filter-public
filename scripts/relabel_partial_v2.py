"""「省略限定条件算部分属实」口径下的真伪改标（D55，2026-10-12 用户决定「算」）。

    python scripts/clean_round1_v2.py      # 先清洗（会从 *_v1 备份重新生成 dev/test）
    python scripts/relabel_partial_v2.py   # 再改标，并导出给同学确认的 Excel

口径：文章或说法的主体方向有依据，但核心健康说法省略了权威来源要求的限定条件
（适用人群、剂量/频次、前提条件、「辅助/有助于」这类程度限定），以绝对化、普遍化或必然化的形式表述
→ 部分属实。只有标题夸大而正文明确补回限定的，仍为「真」；个别数字笔误不改。

做法（防止「照着系统答案改金标准」）：
- 把**全部** 147 条「真」样本（开发集 + 测试集，清洗后）逐条重读，判断时不看系统预测；
- 每条改标都摘出原文依据（程序校验为逐字子串）并写明省略了什么限定；
- CLEAR（证据明确）直接改标；BORDER（边界）只列出，等同学确认后再改；
- 原标签保存在 cognitive_label_v1，评测同时报告改标前后两套成绩。
- 说明：开发集上系统判「真→部分属实」的 14 条中，有 4 条（c0028、c0198、c0211、c0284）复核后仍维持「真」；
  另有 13 条改标/边界样本不在系统错例里，其中不少在测试集——改标不是照搬系统的判断。

这是大模型（Claude）辅助的复核，最终以同学确认为准（导出 Excel 的「同学确认」列）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"

# 编号 -> (原文依据, 省略了什么 / 为什么)
CLEAR: dict[str, tuple[str, str]] = {
    "c0158": ("紫锥菊(即松果菊)是抵抗感冒最有效的天然草药",
              "凭单项研究称「最有效」，省略了证据有限、研究结论不一致的限定"),
    "c0170": ("刮痧对高热、腹泻、感冒、中暑、颈椎病、腰腿痛、类风湿关节炎、慢性胃炎等慢性病有很好的疗效",
              "疗效说法省略「辅助、缓解症状」限定；同文「出痧越多越好？理论上是这样」与 c0015 等样本的权威结论相反"),
    "c0192": ("该仪器在治疗慢性疼痛和促进术后伤口愈合方面作用突出",
              "适用范围列出痛风、骨质疏松、前列腺增生等，未注明辅助性质与证据等级"),
    "c0207": ("脑卒中的发生与性别和年龄关系不大",
              "年龄是脑卒中最主要的危险因素之一；同文「尚未发现因使用他汀类药物发生身体损害的案例」把罕见不良反应说成不存在"),
    "c0220": ("大约在三到五年之后，就会有内分泌问题，出现肥胖或血糖代谢问题",
              "把观察性研究中的关联说成确定的因果和时间表，省略「可能、部分人群」限定"),
    "c0243": ("具有减肥瘦身、美容养颜的目的",
              "无限定的功效说法；同文「夏天拔罐可以最大程度去除体内过多的湿气」缺乏依据"),
    "c0246": ("还可以明显改善失眠、偏头疼、颈椎病以及其他颈肩肌肉酸疼等症状",
              "省略「辅助、改善舒适度」限定，与 c0171「保健枕不能治病」的权威说法相反"),
    "c0258": ("维生素C对消化道肿瘤的发生有强大的预防作用",
              "把观察性研究结论说成「强大的预防作用」；同文称β-胡萝卜素可降低多种癌症风险，而补充剂试验甚至增加吸烟者肺癌风险"),
    "c0265": ("通常是炎症恢复期",
              "持续存在的磨玻璃结节是早期肺腺癌的常见表现（c0266 明确提示要重视），省略了「短期复查后消失的才可能是炎症」的限定"),
    "c0292": ("然后充分炒熟后再加点醋，龙葵素遇醋易分解，这样就可以避免中毒了",
              "与 c0290「仅挖掉芽眼、削皮无法清除已渗透的毒素」相反，省略了「发芽多、变绿者不可食用」的限定"),
    "c0300": ("所以可预防和辅助治疗高血压和心肌梗死",
              "无依据的疗效说法；同文「和一些红糖大枣一起泡，可以帮助提神可以补血提气」与 c0288 的权威说法相反"),
}

BORDER: dict[str, tuple[str, str]] = {
    "c0106": ("也不会对人体造成任何伤害", "省略「完好合规的微波炉、正常使用」限定（同主题 c0104 写明了「完好合规」）"),
    "c0160": ("它不仅仅是加速细胞的衰老与死亡，而且还会抵制新细胞的生长，或者使新细胞畸形",
              "把高剂量电离辐射的危害套用到日常电子产品，省略剂量与辐射类型限定（正文后面的辟谣部分正确）"),
    "c0178": ("优于单纯药物治疗组", "来源不明的数据与疗效比较，省略证据等级；文中多处精确数字无法核实"),
    "c0186": ("凡是以振动、抖动达到按摩或减肥效果的电器，都不太适合老人，长期使用还会对皮肤、皮下组织、内脏有损害",
              "把「部分有基础病的老人、不当使用」扩大到所有老人和所有产品"),
    "c0199": ("1 个月后心梗风险极高", "把皱眉、失眠、表情淡漠等非特异表现直接与一个月后的心梗挂钩"),
    "c0224": ("这样不仅影响疗效，还会加大药物的不良反应",
              "省略「缓控释、肠溶等特殊剂型」限定（c0226 写明普通片剂一般可以掰开）"),
    "c0244": ("提示体内水湿之气较重，需要连续的多次拔罐或配合中药熏蒸治疗",
              "以罐印判断体质并据此建议治疗，缺乏依据（c0169：罐印颜色不是衡量湿气的标准）"),
    "c0256": ("即便是较低的暴露水平也与某些类型的癌症和白血病有关",
              "省略「长期、较高浓度暴露」限定，与 c0255、c0257 的权威说法不一致"),
    "c0273": ("如果没有及时切除，在5年之内就会发生癌变", "把「癌变风险较高」说成必然"),
    "c0277": ("女性在绝经以后5～10年，男性在65～70岁一般都会出现骨质疏松",
              "普遍化，省略「风险显著增加」的限定；同文「豆腐是一种含有大量雌激素的食品」有误"),
    "c0295": ("刺激胰岛素过度分泌，导致胰岛素抵抗，最终引发代谢紊乱",
              "把证据有限的机制和观察性关联说成确定的因果"),
    "c0296": ("那它和含糖饮料在导致糖尿病发生方面并没有本质区别",
              "把证据有限、结论不一的研究说成定论（c0297 的权威表述为「需要辩证地看待」）"),
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    rows, where = {}, {}
    for split in ("dev", "test"):
        for line in (SPLITS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]], where[r["id"]] = r, split

    bad = []
    for d in (CLEAR, BORDER):
        for rid, (ev, _) in d.items():
            r = rows.get(rid)
            if r is None or ev not in (r.get("title", "") + "\n" + r["text"]):
                bad.append(rid)
    if bad:
        raise SystemExit(f"依据不是原文逐字子串或样本不存在：{bad}")

    changed = 0
    for rid, (ev, why) in CLEAR.items():
        r = rows[rid]
        if r["cognitive_label"] != "真":
            continue                                           # 已改过（重复运行）
        r["cognitive_label_v1"] = "真"
        r["cognitive_label"] = "部分属实"
        r["notes"] = (r.get("notes") or "") + f"；改标v2=真→部分属实（省略限定条件，Claude复核待同学确认）：{why}"
        changed += 1
    for split in ("dev", "test"):
        out = [r for rid, r in rows.items() if where[rid] == split]
        (SPLITS / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"改标 {changed} 条（明确）；边界 {len(BORDER)} 条待确认")
    for name, d in (("明确", CLEAR), ("边界", BORDER)):
        n_dev = sum(where[i] == "dev" for i in d)
        print(f"  {name}：开发集 {n_dev}，测试集 {len(d) - n_dev}")

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "改标确认"
    head = ["编号", "集合", "类别", "原标签", "建议标签", "原文依据", "省略了什么 / 理由", "同学确认（同意/不同意）", "备注"]
    widths = [9, 7, 8, 8, 9, 50, 50, 14, 20]
    for i, (h, w) in enumerate(zip(head, widths), 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="3B5B7A")
        ws.column_dimensions[c.column_letter].width = w
    n = 2
    for cat, d in (("明确（已改）", CLEAR), ("边界（未改）", BORDER)):
        for rid, (ev, why) in d.items():
            for col, v in enumerate([rid, where[rid], cat, "真", "部分属实", ev, why, "", ""], 1):
                ws.cell(row=n, column=col, value=v).alignment = Alignment(wrap_text=True, vertical="top")
            n += 1
    ws2 = wb.create_sheet("口径说明")
    ws2["A1"] = (__doc__ or "").strip()
    ws2["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws2.column_dimensions["A"].width = 120
    out = ROOT / "data" / "labeled" / f"真伪改标确认_省略限定条件_{time.strftime('%Y%m%d')}.xlsx"
    wb.save(out)
    print(f"Excel：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
