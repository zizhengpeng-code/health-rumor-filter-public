"""第一轮真实数据的分组报分：按样本类型、是否在知识库拆开看认知准确率。

    python scripts/analyze_round1.py results/baseline_dev_X.json results/full_dev_X.json ...
    python scripts/analyze_round1.py --latest dev        # 自动取各配置最新一份 real 结果

为什么要拆（D34 的两个已知局限）：
  1. 格式混杂：假样本多是一句话，真样本多是整篇文章——模型可能靠长短/文风判真假
  2. 知识库重叠：库内文章抽出的样本，检索会直接命中原文，分数偏乐观
只看总分会把这两个偏差埋掉。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "labeled" / "round1_meta.json"
ORDER = ["baseline", "+behavior", "+retrieve", "+source", "+debate", "+fallback", "full"]


def latest(split: str) -> list[Path]:
    out = []
    for name in ORDER:
        stem = name.replace("+", "plus")
        files = sorted((ROOT / "results").glob(f"{stem}_{split}_*.json"))
        for f in reversed(files):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d["meta"]["model"] == "real" and d["meta"]["n_samples"] > 50:
                out.append(f)
                break
    return out


def acc(rows: list[dict]) -> str:
    if not rows:
        return "—"
    return f"{sum(r['ok'] for r in rows) / len(rows):.3f} (n={len(rows)})"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--latest", metavar="SPLIT")
    args = ap.parse_args(argv)
    files = [Path(f) for f in args.files] or (latest(args.latest) if args.latest else [])
    if not files:
        print("没有可分析的结果文件")
        return 1
    meta = json.loads(META.read_text(encoding="utf-8"))

    groups = {
        "一句话谣言（假为主）": lambda m: m["kind"] == "辟谣文中的谣言句",
        "科普原文（真为主）": lambda m: m["kind"] == "权威科普原文",
        "辟谣文章全文": lambda m: m["kind"] == "辟谣文章全文",
        "在知识库": lambda m: m["in_kb"],
        "不在知识库": lambda m: not m["in_kb"],
    }
    lines = ["| 配置 | 总体 | " + " | ".join(groups) + " | 部分属实召回 |",
             "|---|---|" + "---|" * len(groups) + "---|"]
    confusion_note = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = [r for r in d["rows"] if r["id"] in meta]
        cells = [acc([r for r in rows if fn(meta[r["id"]])]) for fn in groups.values()]
        partial = [r for r in rows if r["gold"] == "部分属实"]
        pr = f"{sum(r['pred'] == '部分属实' for r in partial)}/{len(partial)}"
        lines.append(f"| {d['meta']['config_name']} | {acc(rows)} | " + " | ".join(cells) + f" | {pr} |")
        # 错在哪：按 gold→pred 计数
        cnt: dict[str, int] = defaultdict(int)
        for r in rows:
            if not r["ok"]:
                cnt[f"{r['gold']}→{r['pred']}"] += 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
        confusion_note.append(f"- {d['meta']['config_name']}：" +
                              "，".join(f"{k} {v} 条" for k, v in top))
    print("\n".join(lines))
    print("\n主要错误类型（金标准→预测）：")
    print("\n".join(confusion_note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
