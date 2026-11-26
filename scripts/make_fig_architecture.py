"""报告图 4-1：系统总体技术架构（v2，D53/D54/D51 之后）。

    python scripts/make_fig_architecture.py   → docs/图4-1_系统总体技术架构_v2.png / .svg / .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "docs" / "图4-1_系统总体技术架构_v2.png"

INK, MUTED = "#1F2A37", "#5B6B7C"
C_IN, C_BEH, C_COG, C_RISK, C_GEN, C_DEP = "#E8F1FB", "#FDF0E6", "#E9F6EE", "#F3ECFA", "#FDECEC", "#F2F4F7"
E_IN, E_BEH, E_COG, E_RISK, E_GEN, E_DEP = "#3B7DD8", "#D9822B", "#2E9E5B", "#7B4FC4", "#D64545", "#8A97A6"

fig, ax = plt.subplots(figsize=(13, 8.6), dpi=200)
ax.set_xlim(0, 130)
ax.set_ylim(0, 86)
ax.axis("off")


def box(x, y, w, h, fill, edge, title, lines=(), title_size=11.5, line_size=9.2, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.6 - i * 3.3, ln, ha="center", va="top", fontsize=line_size, color=MUTED)


def arrow(x1, y1, x2, y2, color=MUTED, text="", tx=0, ty=0, size=8.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.3, color=color, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", va="center",
                fontsize=size, color=color)


# 标题
ax.text(65, 84, "家庭群健康谣言双滤网核查系统 · 总体技术架构", ha="center", va="center",
        fontsize=15, fontweight="bold", color=INK)

# ① 接入层
box(3, 50, 30, 28, C_IN, E_IN, "① 接入层", [
    "公众号链接：页面解析 + 去噪",
    "粘贴正文",
    "上传截图：qwen-vl-ocr 识别文字",
    "小红书 / 视频号 / 抖音链接：",
    "提示改用截图或粘贴正文",
])

# ② 双滤网（并行）
box(41, 64, 40, 14, C_BEH, E_BEH, "② 行为面滤网（话术）", [
    "五类话术 + 逐字原文依据（DeepSeek）",
    "风险等级由规则计算：医疗 / 金钱 → 高",
])
box(41, 28, 40, 32, C_COG, E_COG, "② 认知面滤网（真伪）", [
    "查询改写 → BM25 + 向量 → RRF → 重排",
    "冻结证据库：3100 篇 / 13028 段（6 类权威来源）",
    "信源评分（政府与疾控 0.95 …）",
    "阶段一：快速判断（置信度 ≥0.85 直接采用）",
    "阶段二：分工辩论",
    "信源核查 → 正方 / 反方（按证据立场分材料）→ 裁判",
    "程序校验引用编号；裁判逐轮决定是否结案",
    "兜底：无可用证据 → 模型自身知识（≤0.6，注明）",
], line_size=8.8)

# ③ 研判层
box(89, 50, 38, 28, C_RISK, E_RISK, "③ 研判层：风险矩阵", [
    "真伪 × 行为风险 → 建议档位",
    "",
    "尽快劝阻：假 × 中/高",
    "提醒留意：存疑 × 中/高；真 × 高",
    "告知误区：假 × 低",
    "无需处理：其余",
], line_size=9.4)

# ④ 生成层
box(89, 20, 38, 24, C_GEN, E_GEN, "④ 生成层", [
    "发给爸妈的话（子女口吻，≤300 字）",
    "按档位定轻重，按话术给出路",
    "代码约束：来源核对 / 字数 / 结尾",
    "核查报告：只列实际用到的证据",
], line_size=9.2)

# 部署
box(3, 3, 124, 11, C_DEP, E_DEP, "部署：微信小程序（子女端）  调用  微信云托管（FastAPI：任务提交 / 轮询，按用户限次）", [
    "大模型：DeepSeek（全部判断角色）    检索与识别：阿里云百炼（text-embedding-v4、gte-rerank-v2、qwen-vl-ocr）    证据库：对象存储，启动时下载",
], title_size=10.5, line_size=8.6)

# 箭头
arrow(33.6, 71, 41, 71, E_IN, "统一为文档", -0.4, 2.0, size=7.6)
arrow(33, 60, 41, 46, E_IN)
ax.text(35.6, 47.4, "两路并行", fontsize=8.6, color=E_IN, ha="center", va="center")
arrow(81, 71, 89, 66, E_BEH, "行为风险", 0, 2.2)
arrow(81, 44, 89, 58, E_COG, "真伪结论", 1.5, -1.6)
arrow(108, 50, 108, 44, E_RISK, "档位", 3.2, 0)
arrow(81, 31, 89, 28, E_COG)
ax.text(85, 25.2, "实际用到的证据", fontsize=8.6, color=E_COG, ha="center")
arrow(108, 20, 108, 14.5, E_GEN)
ax.text(111.5, 17.3, "返回小程序", fontsize=8.6, color=E_GEN, ha="left")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
# 矢量版：文字转成路径，换电脑没有微软雅黑也不会变样（代价是文字不能在 Word 里直接改，改字请改本脚本）
plt.rcParams["svg.fonttype"] = "path"
for ext in ("svg", "pdf"):
    fig.savefig(OUT.with_suffix("." + ext), bbox_inches="tight", facecolor="white")
print(OUT, "(+ .svg / .pdf)")
