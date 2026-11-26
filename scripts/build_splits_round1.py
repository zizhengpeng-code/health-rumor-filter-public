"""第一轮真实标注 -> 正式评测集（dev 70% / test 30%，按 主题×真假 分层）。

    python scripts/build_splits_round1.py

标签来源（写进每条样本的 notes，报告里要如实交代，见 D35/D37）：
    cognitive_label / topic —— 标注同学（data/labeled/round1_from_annotator.xlsx）
    behaviors               —— Claude 辅助标注（scripts/behavior_labels_round1.py），**未经人工验证**

产出：
    data/labeled/round1.jsonl      全部 292 条
    data/labeled/round1_meta.json  编号 -> {kind, in_kb, url}，分组报分用
    data/splits/dev.jsonl / test.jsonl
旧的合成占位集备份为 data/splits/synthetic_dev.jsonl / synthetic_test.jsonl（只在首次运行时备份）。

test 集一经生成即冻结：不得用它调提示词、阈值或规则（CLAUDE.md 工作约定第 5 条）。
重跑本脚本会得到同样的划分（固定种子），但不要因为 test 分数不好就改种子重划。
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import load_workbook  # noqa: E402

from scripts.behavior_labels_round1 import LABELS  # noqa: E402
from src.models import LabeledSample  # noqa: E402

LAB = ROOT / "data" / "labeled"
SPLITS = ROOT / "data" / "splits"
CANDIDATES = ROOT / "data" / "raw" / "candidates_20261003-113433.jsonl"
SEED, DEV_RATIO = 20261003, 0.7


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    cand = {}
    with open(CANDIDATES, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cand[c["cid"]] = c

    ws = load_workbook(LAB / "round1_from_annotator.xlsx")["候选文章"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    sel = [dict(zip(head, r)) for r in rows[1:] if r[head.index("入选")] == "是"]

    samples: list[LabeledSample] = []
    meta: dict[str, dict] = {}
    for r in sel:
        cid = r["编号"]
        c = cand[cid]
        behaviors = sorted({t for t, _, _ in LABELS.get(cid, [])})
        note = "；".join(x for x in [
            f"类型={c['kind']}", f"在知识库={'是' if c['in_kb'] else '否'}",
            "话术=Claude辅助标注(未经人工验证)", f"标注备注={r['备注']}" if r.get("备注") else "",
        ] if x)
        samples.append(LabeledSample(
            id=cid, title=r.get("标题") or "", text=c["text"], topic=r["主题"],
            cognitive_label=r["真假"], behaviors=behaviors, source_url=c["url"], notes=note,
        ))
        meta[cid] = {"kind": c["kind"], "in_kb": c["in_kb"], "url": c["url"]}

    LAB.mkdir(parents=True, exist_ok=True)
    with open(LAB / "round1.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
    (LAB / "round1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    # 旧的合成集只备份一次，避免重跑时把真实划分当成"旧的"覆盖掉备份
    for name in ("dev", "test"):
        src, dst = SPLITS / f"{name}.jsonl", SPLITS / f"synthetic_{name}.jsonl"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    strata: dict[tuple[str, str], list[LabeledSample]] = defaultdict(list)
    for s in samples:
        strata[(s.topic, s.cognitive_label)].append(s)
    rnd = random.Random(SEED)
    dev, test = [], []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda s: s.id)
        rnd.shuffle(group)
        n_dev = round(len(group) * DEV_RATIO) if len(group) > 1 else 1
        dev += group[:n_dev]
        test += group[n_dev:]
    for name, part in (("dev", dev), ("test", test)):
        part.sort(key=lambda s: s.id)
        with open(SPLITS / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in part:
                f.write(s.model_dump_json() + "\n")

    for name, part in (("dev", dev), ("test", test)):
        print(f"{name}: {len(part)} 条  真假 {dict(Counter(s.cognitive_label for s in part))}")
        print(f"      主题 {dict(Counter(s.topic for s in part))}")
        print(f"      有话术 {sum(1 for s in part if s.behaviors)}  "
              f"不在知识库 {sum(1 for s in part if not meta[s.id]['in_kb'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
