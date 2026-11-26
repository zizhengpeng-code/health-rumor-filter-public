"""生成微信云托管部署包（D53）。

    python app/cloudrun/build.py          # 生成 app/cloudrun/dist/（推到 GitHub 公开仓库，云托管绑定它发布）
    python app/cloudrun/build.py --kb     # 另外生成云端精简知识库 data/kb_cloud.db.gz（上传到云托管对象存储）

部署包里**只有**代码、提示词、云端配置：没有 data/（标注数据、测试集、知识库），没有 .env（密钥）。
密钥在云托管「服务设置 - 环境变量」里填；知识库放对象存储，容器启动时下载。

云端配置与本机的差别（只改这几处，其余与本机完全一致）：
  - 检索层 [kb, site, websearch] → [kb, websearch]：site 层要浏览器，云上没有
  - search.backend browser → mock：同上
  - 向量 / 重排走阿里云接口：由环境变量 HRF_RETRIEVAL=api 切换（server.py 里设了默认值）
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIST = HERE / "dist"

CONFIG_EDITS = [
    ("layers: [kb, site, websearch]", "layers: [kb, websearch]          # 云端：site 层需要浏览器，去掉"),
    ("      backend: browser\n", "      backend: mock               # 云端：没有浏览器\n"),
]


def build_dist() -> None:
    DIST.mkdir(exist_ok=True)
    for p in DIST.iterdir():                                    # 保留 .git，方便重复生成后直接推送
        if p.name == ".git":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", DIST / "src", ignore=ignore)
    shutil.copytree(ROOT / "prompts", DIST / "prompts", ignore=ignore)
    for name in ("server.py", "Dockerfile", "requirements.txt"):
        shutil.copy2(HERE / name, DIST / name)
    shutil.copy2(HERE / "dot_gitignore", DIST / ".gitignore")
    shutil.copy2(HERE / "dot_dockerignore", DIST / ".dockerignore")

    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    for old, new in CONFIG_EDITS:
        if old not in cfg:
            raise SystemExit(f"config.yaml 里找不到要改的配置：{old.strip()}")
        cfg = cfg.replace(old, new)
    (DIST / "config.yaml").write_text(cfg, encoding="utf-8")
    (DIST / "README.md").write_text(
        "# 健康谣言双滤网核查 · 云托管后端\n\n"
        "由主项目 `app/cloudrun/build.py` 自动生成，请勿手改。\n"
        "只含代码与提示词；知识库与密钥不在仓库里（见 server.py 顶部说明）。\n", encoding="utf-8")

    # 安全检查：绝不能带上密钥或数据
    leaked = [p for p in DIST.rglob("*") if ".git" not in p.parts and (
        p.name in (".env", "data") or p.suffix in (".db", ".gz", ".jsonl", ".xlsx", ".json"))]
    if leaked:
        raise SystemExit(f"部署包里出现了不该有的文件：{leaked}")
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"部署包：{DIST}（{size / 1024:.0f} KB）")

    # 云托管「手动上传代码包」用的 zip（≤ 2MB）：只收部署文件，排除 .git 和运行时产生的 data/
    import zipfile
    zpath = HERE / "hrf-api.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DIST.rglob("*")):
            rel = p.relative_to(DIST)
            if p.is_dir() or rel.parts[0] in (".git", "data"):
                continue
            z.write(p, rel.as_posix())
    print(f"上传包：{zpath}（{zpath.stat().st_size / 1024:.0f} KB）")


def build_kb() -> None:
    src = ROOT / "data" / "kb_v2.db"
    tmp = ROOT / "data" / "kb_cloud.db"
    out = ROOT / "data" / "kb_cloud.db.gz"
    shutil.copy2(src, tmp)
    con = sqlite3.connect(tmp)
    n = con.execute("DELETE FROM kb_vectors WHERE model LIKE 'local:%'").rowcount   # 云上只用 API 向量
    con.commit()
    con.execute("VACUUM")
    left = con.execute("SELECT model, COUNT(*) FROM kb_vectors GROUP BY model").fetchall()
    con.close()
    with open(tmp, "rb") as f, gzip.open(out, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g, 1 << 20)
    print(f"删去本地向量 {n} 条，保留 {left}")
    print(f"精简库 {tmp.stat().st_size / 2**20:.0f} MB → 压缩后 {out.stat().st_size / 2**20:.0f} MB：{out}")
    tmp.unlink()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", action="store_true", help="同时生成云端精简知识库")
    args = ap.parse_args()
    build_dist()
    if args.kb:
        build_kb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
