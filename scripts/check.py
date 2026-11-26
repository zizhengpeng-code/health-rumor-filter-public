"""一键核查一篇文章（命令行版）。

    python scripts/check.py --url https://mp.weixin.qq.com/s/xxxx
    python scripts/check.py --file article.txt --title "降压药吃上就停不下来？"
    python scripts/check.py --text "多吃苦瓜能降血糖，赶紧转给家里老人！"

打印 Markdown 报告，并把 JSON + Markdown 存到 results/checks/。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, run_stamp  # noqa: E402
from src.models import Document  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="家庭群健康谣言双滤网核查")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--file")
    src.add_argument("--text")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.url:
        from src.ingest.fetch import IngestError, fetch_document
        try:
            doc = fetch_document(args.url, cfg)
        except IngestError as exc:
            print(f"抓取失败：{exc}")
            return 2
    else:
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        doc = Document(id="paste-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10],
                       source_type="paste", title=args.title, text=text)

    from src.pipeline import Checker
    from src.report.render import to_markdown

    report = Checker(cfg).check(doc, progress=lambda m: print(m, flush=True))
    md = to_markdown(report)
    print("\n" + md)

    out = Path(cfg.get_path("paths.results_dir")) / "checks"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{run_stamp(cfg)}_{doc.id}"
    (out / f"{stem}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(f"已保存：{out / stem}.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
