"""采集 300 条真实样本的候选池，导出成给标注同学用的 Excel。

    python scripts/collect_candidates.py --skip-discover     # 只用本地知识库，秒出，不花钱
    python scripts/collect_candidates.py                     # 再联网发现新文章（约 30 分钟，几毛钱）
    python scripts/collect_candidates.py --discover-keywords 5   # 小批量试跑联网发现

产出（data/raw/）：
    candidates_<时间戳>.xlsx    给标注同学：只需在 Excel 里确认/修改，不用碰代码
    candidates_<时间戳>.jsonl   全部候选的完整正文，回收标注时按编号对回去

两类候选：
    假 —— 从辟谣文章里把「谣言：/流言：……」那一句拆出来（一篇汇总文常含多条）
    真 —— 权威站的科普原文（排除发布会实录、通知公告等非科普页）

**真假只按来源结构推断，不调用模型**：辟谣文里的谣言句 = 假，科普原文 = 真。
这样预填不会把模型的判断带进金标准；行为面五列一律留空，必须人工判断（见 D34）。

「知识库里已有」= 该文章就在 data/kb.db 里。这类样本评测时检索会直接搜到原文，
偏乐观，选样时优先用不在库里的。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, run_stamp  # noqa: E402
from src.models import BEHAVIOR_TYPES, TOPICS  # noqa: E402

# 联网发现用的关键词：刻意**不和建库种子词重复**，这样新发现的文章大概率不在库里，
# 也顺带检验知识库对没见过的主题的覆盖度。
DISCOVER_KEYWORDS: list[str] = [
    # 慢病
    "血糖 偏方", "降血脂 食疗", "心梗 前兆", "中风 谣言", "骨质疏松 补钙", "甲状腺 谣言",
    # 用药
    "胶囊 掰开", "输液 预防", "头孢 喝酒", "安眠药 依赖", "布洛芬 谣言", "中成药 副作用",
    # 饮食
    "隔夜茶", "空腹 牛奶", "鸡蛋 胆固醇", "红糖 补血", "发芽土豆", "无糖饮料",
    "酸奶 谣言", "柠檬水", "喝汤 营养",
    # 癌症
    "甲醛 致癌", "烧烤 致癌", "乳腺结节", "肺结节", "幽门螺杆菌 胃癌", "HPV 疫苗 谣言", "抗癌 偏方",
    # 养生器械
    "负离子", "艾灸", "拔罐", "刮痧", "保健枕", "足贴 排毒", "远红外 理疗",
    # 其他
    "熬夜 补觉", "冬病夏治", "排毒 谣言", "老人 疫苗 谣言", "体检 谣言",
]

# 谣言句的起止标记
# 不用「说法：」——实测会抓到新闻稿里的普通句子
_RUMOR_START = re.compile(r"【?(流言|谣言|传言|网传|误区)\s*[】：:]\s*")
_RUMOR_END = re.compile(
    r"\n|【?(真相|辟谣|真相解读|实际上|点评|解读|专家|科学解读|事实)\s*[】：:]")
_TRUTH_START = re.compile(r"【?(真相|辟谣|真相解读|点评|解读|科学解读|事实)\s*[】：:]\s*")

# 主题判定：按顺序匹配，先命中先返回（癌症、用药、器械比"吃喝"更具体，放前面）
TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("癌症", ("癌", "肿瘤", "结节", "HPV", "化疗", "放疗", "白血病", "致癌")),
    ("用药", ("药", "输液", "抗生素", "头孢", "胶囊", "布洛芬", "阿司匹林", "他汀", "胰岛素",
              "疫苗", "注射", "剂量")),
    ("养生器械", ("仪器", "治疗仪", "理疗仪", "器械", "磁疗", "艾灸", "拔罐", "刮痧", "足贴", "负离子", "理疗",
                 "按摩椅", "足浴", "保健枕", "量子", "远红外", "床垫")),
    ("慢病", ("高血压", "血压", "糖尿病", "血糖", "血脂", "胆固醇", "心梗", "中风", "脑卒中",
              "痛风", "冠心病", "骨质疏松", "甲状腺", "心脏", "血管", "慢性")),
    ("饮食", ("吃", "喝", "食", "饮", "菜", "水", "奶", "蛋", "茶", "汤", "酒", "醋", "肉",
              "果", "粮", "油", "盐", "糖", "营养", "维生素", "补钙")),
]
_HEALTH_WORDS = ("病", "健康", "医", "身体", "营养", "细菌", "病毒", "感染", "症", "养生",
                 "保健", "体质", "免疫", "器官", "肝", "肾", "胃", "肠", "睡眠", "减肥")
_NOT_POPSCI = ("新闻发布会", "文字实录", "通知", "公告", "会议", "招标", "采购", "名单",
               "工作动态", "领导", "简报", "政策解读", "答记者问", "目录", "检索", "搜索",
               "品牌网站", "资源服务", "文献服务", "网站地图", "首页")
# 正文开头出现这些，说明抓到的是搜索页/导航页而不是文章
_JUNK_TEXT = ("为您找到相关结果", "网站地图", "学术热点追踪", "版权所有")
_DEBUNK_WORDS = ("辟谣", "谣言", "流言", "真相", "误区", "传言", "网传")


# ---------------------------------------------------------------------------
# 抽取与归类（纯函数，便于单测）
# ---------------------------------------------------------------------------
def guess_topic(text: str, fallback: str = "") -> str:
    """先看 text（通常是标题），猜不出再看 fallback（正文开头）。"""
    for src in (text, fallback):
        t = _topic_of(src)
        if t != "其他":
            return t
    return "其他"


def _topic_of(text: str) -> str:
    for topic, keys in TOPIC_RULES:
        if any(k in text for k in keys):
            return topic
    return "其他"


def is_health(text: str) -> bool:
    return any(k in text for _, keys in TOPIC_RULES for k in keys) or \
        any(k in text for k in _HEALTH_WORDS)


def extract_rumors(text: str, max_len: int = 300) -> list[dict[str, str]]:
    """从辟谣文里拆出所有「谣言：……」句，连同紧随其后的辟谣节选。"""
    out: list[dict[str, str]] = []
    for m in _RUMOR_START.finditer(text):
        rest = text[m.end():]
        # 谣言句可能另起一行（"谣言：\n听说……"），先跳过开头的空白
        lead = len(rest) - len(rest.lstrip())
        body = rest[lead:]
        end = _RUMOR_END.search(body)
        rumor = body[: end.start()] if end else body[:max_len]
        rumor = rumor.strip(" 　\t\"“”")
        if not (6 <= len(rumor) <= max_len) or "XX" in rumor.upper():
            continue
        tm = _TRUTH_START.search(body)
        truth = body[tm.end(): tm.end() + 300].strip() if tm else ""
        out.append({"rumor": rumor, "truth": truth, "marker": m.group(1)})
    return out


def looks_like_popsci(title: str, text: str) -> bool:
    """科普原文：不是辟谣格式、不是发布会/通知、长度像一篇文章。"""
    if _RUMOR_START.search(text):
        return False
    if any(k in title for k in _NOT_POPSCI) or any(k in text[:400] for k in _JUNK_TEXT):
        return False
    # 标题本身要有健康主题词：站点名、栏目名这类标题（"大'申'说辟谣"）一律不要
    return 400 <= len(text) and is_health(title)


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s or "").lower()


def _grams(s: str, n: int = 3) -> set[str]:
    s = _norm(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def near_dup(a: set[str], b: set[str], thr: float = 0.6) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= thr


def _url_key(u: str) -> str:
    return (u or "").split("#", 1)[0].rstrip("/").replace("https://", "http://")


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
def load_kb_docs(kb_path: str) -> list[dict[str, Any]]:
    if not Path(kb_path).exists():
        return []
    con = sqlite3.connect(kb_path)
    rows = con.execute("select source, source_name, title, url, text from kb_docs").fetchall()
    con.close()
    return [{"source": r[0], "source_name": r[1], "title": r[2], "url": r[3], "text": r[4],
             "in_kb": True} for r in rows]


def discover_new_docs(cfg: dict[str, Any], keywords: list[str], kb_urls: set[str],
                      per_source: int, workers: int, cache_path: Path) -> list[dict[str, Any]]:
    """联网发现 + 抓正文。结果缓存到磁盘，中断后重跑不会重复花钱。"""
    from src.cognitive.retrieve import SITE_ADAPTERS, BrowserSession, _domain
    from src.kb.crawl import discover, fetch_article

    cache: dict[str, Any] = {"done_keywords": [], "pending": {}, "docs": {}}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def save() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    auth = rcfg.get("authority_domains") or []
    domains = [d["domain"] for d in auth if d.get("domain")]
    name_of = {d["domain"]: d.get("name", "") for d in auth}

    todo = [k for k in keywords if k not in cache["done_keywords"]]
    print(f"联网发现：{len(todo)} 个关键词 × {len(domains)} 个权威域"
          f"（缓存里已完成 {len(cache['done_keywords'])} 个）", flush=True)
    session = BrowserSession(cfg) if any(d in SITE_ADAPTERS for d in domains) else None
    try:
        for i, kw in enumerate(todo, 1):
            for dom in domains:
                try:
                    hits = discover(kw, dom, cfg, per_source, session)
                except Exception as exc:                       # noqa: BLE001
                    print(f"  ! {kw} @ {dom} 发现失败：{exc}", flush=True)
                    hits = []
                for h in hits:
                    if h.url.startswith("http") and _url_key(h.url) not in kb_urls:
                        cache["pending"].setdefault(h.url, kw)
            cache["done_keywords"].append(kw)
            save()
            print(f"  [{i}/{len(todo)}] {kw}：累计新 URL {len(cache['pending'])} 条", flush=True)
    finally:
        if session is not None:
            session.close()

    urls = [u for u in cache["pending"] if u not in cache["docs"]]
    print(f"抓正文：{len(urls)} 条（{workers} 并发）", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_article, u, cfg): u for u in urls}
        for n, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                got = fut.result()
            except Exception:                                  # noqa: BLE001
                got = None
            cache["docs"][u] = list(got) if got else None
            if n % 25 == 0:
                save()
                print(f"  [{n}/{len(urls)}]", flush=True)
    save()

    out = []
    for u, got in cache["docs"].items():
        if not got:
            continue
        dom = _domain(u)
        src = next((d for d in domains if d in dom), dom)
        out.append({"source": src, "source_name": name_of.get(src, src), "title": got[0] or u,
                    "url": u, "text": got[1], "in_kb": False,
                    "keyword": cache["pending"].get(u, "")})
    return out


def build_candidates(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把文章拆成候选样本并去重。不在库里的排前面，去重时优先保留它们。"""
    docs = sorted(docs, key=lambda d: d["in_kb"])
    cands: list[dict[str, Any]] = []
    seen_grams: list[set[str]] = []
    seen_head: set[str] = set()

    def keep(sample_text: str) -> bool:
        g = _grams(sample_text[:200])
        if any(near_dup(g, s) for s in seen_grams):
            return False
        seen_grams.append(g)
        return True

    for d in docs:
        title, text = d["title"] or "", d["text"] or ""
        rumors = extract_rumors(text)
        if rumors:
            for r in rumors:
                if not is_health(r["rumor"]) or not keep(r["rumor"]):
                    continue
                cands.append({
                    "label": "假", "kind": "辟谣文中的谣言句", "title": "", "text": r["rumor"],
                    "truth": r["truth"], "topic": guess_topic(r["rumor"], title),
                    "source_name": d["source_name"], "url": d["url"], "doc_title": title,
                    "in_kb": d["in_kb"], "marker": r["marker"],
                })
        elif looks_like_popsci(title, text):
            head = hashlib.md5(_norm(text[:300]).encode()).hexdigest()
            if head in seen_head or not keep(title + text[:150]):
                continue
            seen_head.add(head)
            debunk = any(k in title + text[:200] for k in _DEBUNK_WORDS)
            cands.append({
                "label": "真", "kind": "辟谣文章全文" if debunk else "权威科普原文",
                "title": title, "text": text,
                "truth": "", "topic": guess_topic(title, text[:300]),
                "source_name": d["source_name"], "url": d["url"], "doc_title": title,
                "in_kb": d["in_kb"],
            })
    return cands


def preselect(cands: list[dict[str, Any]], target: int) -> None:
    """预选约 target 条：真假各半，各 topic 轮流取，不在库里的优先。原地写 selected。"""
    per_label = target // 2
    for c in cands:
        c["selected"] = False
    for label in ("假", "真"):
        pools = {t: [c for c in cands if c["label"] == label and c["topic"] == t]
                 for t in TOPICS}
        for t in pools:
            # 不在库里的优先；「真」里科普原文优先于整篇辟谣文；谣言句短的优先（长的多是抽错的）
            # 「误区：」后面有时跟的是纠正后的正确说法（"刮痧并非越紫越好"），也往后排
            pools[t].sort(key=lambda c: (c["in_kb"], c["kind"] == "辟谣文章全文",
                                         c.get("marker") == "误区",
                                         len(c["text"]) > 120 if c["label"] == "假" else 0))
        picked = 0
        while picked < per_label and any(pools.values()):
            for t in TOPICS:
                if pools[t] and picked < per_label:
                    pools[t].pop(0)["selected"] = True
                    picked += 1


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
INSTRUCTIONS = """怎么用这张表（不用装任何软件，Excel / WPS 打开即可）

一、你要做的只有三件事
1. 看「候选文章」表的黄色列。电脑已经预填了「入选」「真假」「主题」，你只需要把填错的改掉。
   · 入选：电脑已挑好约 300 条（是），不想要的改成「否」，想换进来的改成「是」。
   · 真假：电脑是按来源推断的——辟谣文章里的谣言句填了「假」，官方科普原文填了「真」。
     你看一眼，觉得不对就改；拿不准、或者"大方向对但有夸大/漏了条件"的，选「部分属实」并在备注里写一句为什么。
   · 主题：电脑按关键词猜的，分错了就改。
2. 标「话术」：五个话术列默认是空的，**必须你自己判断**，电脑没有填。
   读样本正文，出现对应的话术就在那一列选「是」，没有就空着。判断标准见下面第三节。
   一句话谣言通常没有话术，几秒就能过；主要花时间的是带推销、带"转发"的长文章。
3. 「家庭群收集」表：把家里长辈在家庭群转过的健康文章正文粘进去（每行一篇），
   同样填真假、主题、话术。名字、电话等隐私信息请删掉。

二、不用管的列
灰色列（来源、链接、知识库里已有、电脑建议……）是给我们核对用的，不用改。
「统计」表会根据你选的「入选」自动更新各主题、真假的数量，可以随时看是否均衡。

三、五类话术怎么判断（以 docs/annotation_guide 定稿版为准）
· 转发诱导：用道德绑架或利益诱导要求转发。例："转给家里老人""不转不是中国人"。
  "欢迎分享"这种没有绑架意味的不算。
· 医疗行为建议：建议停药、换药、改剂量，或用偏方/保健品/饮食替代正规治疗。
  例："吃了这个就不用吃降压药了"。"遵医嘱调整用药"不算。
· 消费导流：推荐买保健品、器械、理疗服务、课程，附链接、二维码或联系方式。
  例："点击下方链接购买""加微信领体验装"。
· 金钱相关：要钱本身——入会费、转账、投资理财。例："交199元入会""投资稳赚"。
  卖实物或服务的会员费算「消费导流」，不算这一类。
· 恐惧诉求：夸大危害、制造紧迫感来推动行动。例："再不看就晚了""医生都不敢告诉你"。
  客观说风险不算；只是说了一个错误结论（如"癌症患者都是酸性体质"）也不算。
一条可以同时选多类。原则：看说话方式，不看内容真假；标了就要能指出原文是哪句。

四、标完以后
把整个 Excel 发回给我们即可，不用改文件名，也不要删列。
"""

HEADERS = [
    # (列名, 宽度, 是否需要人工填)
    ("编号", 9, False), ("入选", 7, True), ("真假", 9, True), ("主题", 9, True),
    *[(t, 8 if len(t) <= 4 else 11, True) for t in BEHAVIOR_TYPES],
    ("备注", 20, True),
    ("标题", 24, False), ("样本正文", 60, False), ("辟谣原文节选（供参考）", 40, False),
    ("类型", 14, False), ("来源", 14, False), ("链接", 28, False),
    ("知识库里已有", 9, False), ("电脑建议真假", 9, False), ("电脑建议主题", 9, False),
]


def write_excel(path: Path, cands: list[dict[str, Any]], family_rows: int = 80) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    yellow = PatternFill("solid", fgColor="FFF7D6")
    grey = PatternFill("solid", fgColor="F0F0F0")
    head_fill = PatternFill("solid", fgColor="3B5B7A")
    head_font = Font(bold=True, color="FFFFFF")
    wrap = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    ws0 = wb.active
    ws0.title = "使用说明"
    ws0.column_dimensions["A"].width = 110
    for i, line in enumerate(INSTRUCTIONS.splitlines(), 1):
        c = ws0.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if line and line[0] in "一二三四" and "、" in line[:3]:
            c.font = Font(bold=True, size=12)
    ws0.cell(row=1, column=1).font = Font(bold=True, size=14)

    col = {name: i for i, (name, _, _) in enumerate(HEADERS, 1)}

    def setup_sheet(ws, n_rows: int) -> None:
        for name, (i, (_, w, manual)) in zip(col, enumerate(HEADERS, 1)):
            c = ws.cell(row=1, column=i, value=name)
            c.fill, c.font = head_fill, head_font
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "B2"
        ws.row_dimensions[1].height = 30
        last = n_rows + 1
        dvs = [
            (DataValidation(type="list", formula1='"是,否"', allow_blank=True), ["入选"]),
            (DataValidation(type="list", formula1='"真,假,部分属实,无法核查"', allow_blank=True),
             ["真假"]),
            (DataValidation(type="list", formula1='"' + ",".join(TOPICS) + '"', allow_blank=True),
             ["主题"]),
            (DataValidation(type="list", formula1='"是"', allow_blank=True), list(BEHAVIOR_TYPES)),
        ]
        for dv, names in dvs:
            ws.add_data_validation(dv)
            for nm in names:
                letter = get_column_letter(col[nm])
                dv.add(f"{letter}2:{letter}{last}")

    ws = wb.create_sheet("候选文章")
    setup_sheet(ws, len(cands))
    for r, c in enumerate(cands, 2):
        snippet = (c["text"] or "")[:32000]
        values = {
            "编号": c["cid"], "入选": "是" if c["selected"] else "否",
            "真假": c["label"], "主题": c["topic"], "备注": "",
            "标题": c["title"], "样本正文": snippet, "辟谣原文节选（供参考）": c["truth"],
            "类型": c["kind"], "来源": c["source_name"], "链接": c["url"],
            "知识库里已有": "是" if c["in_kb"] else "否",
            "电脑建议真假": c["label"], "电脑建议主题": c["topic"],
        }
        for name, (i, (_, _, manual)) in zip(col, enumerate(HEADERS, 1)):
            cell = ws.cell(row=r, column=i, value=values.get(name, None))
            cell.alignment = wrap
            cell.fill = yellow if manual else grey
        ws.row_dimensions[r].height = 60
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{len(cands) + 1}"

    wf = wb.create_sheet("家庭群收集")
    setup_sheet(wf, family_rows)
    for r in range(2, family_rows + 2):
        wf.cell(row=r, column=col["编号"], value=f"fam-{r - 1:03d}").fill = grey
        wf.cell(row=r, column=col["入选"], value="是")
        wf.cell(row=r, column=col["类型"], value="家庭群收集").fill = grey
        for name, (i, (_, _, manual)) in zip(col, enumerate(HEADERS, 1)):
            cell = wf.cell(row=r, column=i)
            cell.alignment = wrap
            if manual or name in ("标题", "样本正文", "来源", "链接"):
                cell.fill = yellow
        wf.row_dimensions[r].height = 45

    # 统计：用公式，她改了「入选」「真假」「主题」会自动更新
    st = wb.create_sheet("统计")
    st["A1"] = "已入选样本分布（候选文章 + 家庭群收集，随你的修改自动更新）"
    st["A1"].font = Font(bold=True, size=12)
    labels = ["真", "假", "部分属实", "无法核查"]
    st.cell(row=3, column=1, value="主题 \\ 真假").font = Font(bold=True)
    for j, lab in enumerate(labels + ["合计"], 2):
        st.cell(row=3, column=j, value=lab).font = Font(bold=True)
    L = {k: get_column_letter(col[k]) for k in ("入选", "真假", "主题")}

    def cnt(sheet: str, topic: str | None, lab: str | None) -> str:
        parts = [f"'{sheet}'!{L['入选']}:{L['入选']},\"是\""]
        if topic:
            parts.append(f"'{sheet}'!{L['主题']}:{L['主题']},\"{topic}\"")
        if lab:
            parts.append(f"'{sheet}'!{L['真假']}:{L['真假']},\"{lab}\"")
        return f"COUNTIFS({','.join(parts)})"

    for i, t in enumerate(list(TOPICS) + ["合计"], 4):
        st.cell(row=i, column=1, value=t).font = Font(bold=(t == "合计"))
        for j, lab in enumerate(labels + [None], 2):
            tp = None if t == "合计" else t
            st.cell(row=i, column=j,
                    value=f"={cnt('候选文章', tp, lab)}+{cnt('家庭群收集', tp, lab)}")
    st.cell(row=4 + len(TOPICS) + 2, column=1,
            value="目标：合计约 300 条，真假接近 1:1，每个主题都要有。").font = Font(italic=True)
    st.column_dimensions["A"].width = 14

    wb.active = 0                                      # 打开时先看到「使用说明」
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="采集真实样本候选池并导出 Excel")
    ap.add_argument("--skip-discover", action="store_true", help="只用本地知识库，不联网")
    ap.add_argument("--discover-keywords", type=int, default=0, help="只用前 N 个发现关键词（0=全部）")
    ap.add_argument("--per-source", type=int, default=6, help="每（关键词×域）最多取几篇")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=int, default=300, help="预选条数")
    args = ap.parse_args(argv)

    cfg = load_config()
    raw_dir = Path(cfg.get_path("paths.raw_dir"))
    t0 = time.time()

    docs = load_kb_docs(cfg.get_path("paths.kb_path"))
    print(f"本地知识库：{len(docs)} 篇", flush=True)
    if not args.skip_discover:
        kws = DISCOVER_KEYWORDS[: args.discover_keywords] if args.discover_keywords \
            else DISCOVER_KEYWORDS
        kb_urls = {_url_key(d["url"]) for d in docs}
        docs += discover_new_docs(cfg, kws, kb_urls, args.per_source, args.workers,
                                  raw_dir / "discover_cache.json")

    cands = build_candidates(docs)
    preselect(cands, args.target)
    # 编号：入选的排前面，便于她从上往下看
    cands.sort(key=lambda c: (not c["selected"], c["label"], c["topic"], c["in_kb"]))
    for i, c in enumerate(cands, 1):
        c["cid"] = f"c{i:04d}"

    stamp = run_stamp(cfg)
    xlsx = raw_dir / f"candidates_{stamp}.xlsx"
    jsonl = raw_dir / f"candidates_{stamp}.jsonl"
    write_excel(xlsx, cands)
    with open(jsonl, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 汇总
    from collections import Counter
    sel = [c for c in cands if c["selected"]]
    print(f"\n候选共 {len(cands)} 条（假 {sum(c['label'] == '假' for c in cands)} / "
          f"真 {sum(c['label'] == '真' for c in cands)}），预选 {len(sel)} 条")
    print(f"  预选里不在知识库的：{sum(not c['in_kb'] for c in sel)} 条")
    for lab in ("假", "真"):
        dist = Counter(c["topic"] for c in sel if c["label"] == lab)
        print(f"  预选 {lab}：{dict(dist)}")
    print(f"\nExcel：{xlsx}\n完整正文：{jsonl}\n耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
