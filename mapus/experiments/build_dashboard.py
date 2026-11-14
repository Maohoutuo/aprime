# -*- coding: utf-8 -*-
r"""把 results/*.json + results/figures/*.png 汇总成一个可视化的单文件网页（dashboard.html）。

与 build_report.py 的区别：report.html 是「逐节复现报告」，dashboard 是「一眼看结果」的看板——
卡片、彩色差值、内联条形图、按数据集分页，适合给别人快速展示。

    py -3.14 experiments\build_dashboard.py
"""

from __future__ import annotations

import base64
import glob
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import PAPER

RESULTS = "results"
FIGDIR = os.path.join(RESULTS, "figures")
OUT = "dashboard.html"

METHODS = ["RN", "TVPG", "TCPG", "MSA", "MSAGI", "GraphDP", "GPT-4.1-mini", "MAPUS"]
VARIANTS = ["w/o PRG", "w/o FPS", "w/o NRR", "Full"]
DATASET_LABEL = {"tdrive": "T-Drive", "grab": "Grab-Posisi"}


# --------------------------------------------------------------------------- IO
def load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def first_glob(pattern: str):
    hits = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return hits[0] if hits else None


def pick(prefix: str, prefer: str = "qwen"):
    cands = [p for p in sorted(glob.glob(os.path.join(RESULTS, prefix + "*.json")))]
    if not cands:
        return None
    good = [p for p in cands if prefer in os.path.basename(p).lower()]
    return load((good or cands)[-1])


def img(path: str, caption: str = "", cls: str = "") -> str:
    if not os.path.exists(path):
        return f"<p class='muted'>（缺少图片 {html.escape(os.path.basename(path))}）</p>"
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f"<figure class='shot {cls}'><img alt='{html.escape(caption)}' "
            f"src='data:image/png;base64,{data}'/>{cap}</figure>")


# --------------------------------------------------------------------------- 小组件
def delta(value: float, digits: int = 3, suffix: str = "", good_high: bool = True) -> str:
    if value is None:
        return "<span class='muted'>-</span>"
    cls = "up" if (value > 0) == good_high and abs(value) > 1e-9 else (
        "down" if abs(value) > 1e-9 else "flat")
    text = f"{value:+.{digits}f}{suffix}"
    return f"<span class='{cls}'>{text}</span>"


def pct(value: float) -> str:
    return delta(value * 100, 1, "%")


def bar(value: float, lo: float, hi: float, color: str = "var(--accent)") -> str:
    span = max(1e-6, hi - lo)
    width = max(0.0, min(1.0, (value - lo) / span)) * 100
    return (f"<span class='bar'><i style='width:{width:.1f}%;background:{color}'></i></span>")


def table(headers, rows, cls: str = "grid") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<div class='scroll'><table class='{cls}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def kpi(label: str, value: str, note: str = "", cls: str = "") -> str:
    return (f"<div class='card kpi {cls}'><span class='kpi-label'>{label}</span>"
            f"<b>{value}</b><span class='kpi-note'>{note}</span></div>")


def badge(text: str, cls: str = "") -> str:
    return f"<span class='pill {cls}'>{text}</span>"


# --------------------------------------------------------------------------- 主结果
def main_block(dataset: str, tag: str, n: int, real: bool) -> str:
    data = load(os.path.join(RESULTS, f"main_{dataset}_{tag}_n{n}.json"))
    if not data:
        return f"<p class='muted'>（缺少 results/main_{dataset}_{tag}_n{n}.json）</p>"
    rows = []
    for scale in ("small", "medium", "large"):
        per = data["scales"].get(scale, {})
        if not per:
            continue
        best = max((per[m]["pss"] for m in per if m in METHODS), default=0)
        single = per.get("GPT-4.1-mini", {}).get("pss")
        for m in METHODS:
            v = per.get(m)
            if not v:
                continue
            ref = PAPER[dataset][scale].get(m)
            cov, pss = v["coverage"], v["pss"]
            dcov = (cov - ref[0]) / ref[0] if ref else None
            dpss = pss - ref[1] if ref else None
            d_single = pss - single if (single is not None and m != "GPT-4.1-mini") else None
            name = f"<b>{m}</b>" if m == "MAPUS" else m
            star = " ★" if abs(pss - best) < 1e-9 else ""
            rows.append([
                name, f"<span class='muted'>{scale}</span>",
                f"{cov:.3f}", f"{pss:.3f}{star}",
                f"<span class='barwrap'>{bar(pss, 0.45 if dataset == 'tdrive' else 0.55, 0.95)}</span>",
                delta(dcov * 100, 2, "%"), delta(dpss, 3), delta(d_single, 3),
            ])
    return table(["Method", "Scale", "Coverage", "PSS",
                  "PSS 条形", "ΔCov vs 论文", "ΔPSS vs 论文", "ΔPSS vs 单 LLM"], rows)


def main_tables() -> str:
    real = {}
    for ds in ("tdrive", "grab"):
        p = glob.glob(os.path.join(RESULTS, f"main_{ds}_qwen72b_n*.json"))
        real[ds] = p[0] if p else None
    n_real = 3
    tabs = []
    for ds in ("tdrive", "grab"):
        tabs.append(f"<h3>{DATASET_LABEL[ds]} · 真实 LLM（Qwen2.5-72B-Instruct，n={n_real}）"
                    + badge("真实调用", "real") + "</h3>")
        tabs.append(main_block(ds, "qwen72b", n_real, True))
        tabs.append("<details><summary>展开：离线代理模型（n=20，同实现）</summary>"
                    + main_block(ds, "surrogate", 20, False) + "</details>")
    return "\n".join(tabs)


# --------------------------------------------------------------------------- 消融
def ablation_block() -> str:
    out = []
    for ds in ("tdrive", "grab"):
        data = pick(f"ablation_{ds}", prefer="qwen")
        if not data:
            out.append(f"<p class='muted'>（缺少 {ds} 消融结果）</p>")
            continue
        rows = []
        for scale, per in data["results"].items():
            full = per.get("Full", {}).get("pss")
            best = max(v["pss"] for v in per.values())
            for name in VARIANTS:
                v = per.get(name)
                if not v:
                    continue
                d = (v["pss"] - full) if (full is not None and name != "Full") else None
                star = " ★" if abs(v["pss"] - best) < 1e-9 else ""
                rows.append([f"<b>{name}</b>" if name == "Full" else name,
                             f"<span class='muted'>{scale}</span>",
                             f"{v['coverage']:.3f} ± {v['coverage_std']:.3f}",
                             f"{v['pss']:.3f} ± {v['pss_std']:.3f}{star}",
                             delta(d, 3), f"{v['overlap']*100:.1f}%"])
        out.append(f"<h3>{DATASET_LABEL[ds]} · 组件消融（n={data['n']}）"
                   + badge("真实调用", "real") + "</h3>")
        out.append(table(["Variant", "Scale", "Coverage", "PSS", "ΔPSS vs Full", "平均重叠"], rows))
    out.append(img(os.path.join(FIGDIR, "fig_ablation.png"),
                   "图 3：四个变体在三个规模上的覆盖效用与 PSS（T-Drive，真实 LLM，n=3）。"))
    return "\n".join(out)


# --------------------------------------------------------------------------- 协商
def negotiation_block() -> str:
    data = pick("negotiation_")
    if not data:
        return "<p class='muted'>（缺少协商结果）</p>"
    rows = []
    for ds, v in data["datasets"].items():
        rows.append([
            DATASET_LABEL.get(ds, ds),
            f"{v['pre_neg_overlap']*100:.2f}%", f"{v['avg_overlap']*100:.2f}%",
            delta((v["avg_overlap"] - v["pre_neg_overlap"]) * 100, 2, " pp", good_high=False),
            f"{v['pre_neg_coverage']:.3f}", f"{v['coverage']:.3f}",
            delta(v["coverage"] - v["pre_neg_coverage"], 3),
            f"{v['pre_neg_pss']:.3f}", f"{v['pss']:.3f}",
            delta(v["pss"] - v["pre_neg_pss"], 3),
            f"{v['neg_attempts']:.1f} / {v['neg_accepted']:.1f}",
        ])
    return (f"<p class='muted'>协商前后的统计（Medium，n={data['n']}，backend={data['backend']}）："
            "NRR 的目标是把重叠的路线推开，代价是 PSS 的小幅下降。</p>"
            + table(["Dataset", "重叠 前", "重叠 后", "Δ重叠", "覆盖 前", "覆盖 后", "Δ覆盖",
                     "PSS 前", "PSS 后", "ΔPSS", "协商对 / 接受"], rows)
            + img(os.path.join(FIGDIR, "fig_negotiation.png"),
                  "协商前后：路线重叠、覆盖效用与 PSS。"))


# --------------------------------------------------------------------------- 公平性
def fairness_block() -> str:
    data = pick("fairness_")
    if not data:
        return "<p class='muted'>（缺少公平性结果）</p>"
    order = ["UR", "LS", "CG", "FPS"]
    cards = []
    for name in order:
        v = data["strategies"][name]
        cls = "hi" if name == "FPS" else ""
        cards.append(
            f"<div class='card mini {cls}'><div class='mini-head'>{name}"
            + (" <span class='pill'>本文</span>" if name == "FPS" else "") +
            f"</div><ul class='mini-list'>"
            f"<li>Coverage <b>{v['coverage']:.3f}</b></li>"
            f"<li>PSS <b>{v['pss']:.3f}</b></li>"
            f"<li>Gini <b>{v['gini']:.3f}</b></li>"
            f"<li>单人最多 <b>{v['max_count']}</b> 次</li></ul></div>")
    rows = [[name, f"{data['strategies'][name]['coverage']:.3f}",
             f"{data['strategies'][name]['pss']:.3f}",
             f"{data['strategies'][name]['gini']:.3f}",
             str(data["strategies"][name]["max_count"]),
             f"{data['strategies'][name]['std_count']:.2f}"] for name in order]
    return (f"<p class='muted'>{data['tasks']} 个传感任务 × {data['pool']} 人固定池，"
            f"backend={data['backend']}。</p>"
            f"<div class='cards'>{''.join(cards)}</div>"
            + table(["策略", "Coverage", "PSS", "Gini（被选次数）", "单人最多被选", "标准差"], rows)
            + img(os.path.join(FIGDIR, "fig_fairness.png"),
                  "图 4：四种选择策略的选择次数分布、CDF 与 PSS / 覆盖率。"))


# --------------------------------------------------------------------------- 时延与成本
def cost_block() -> str:
    cost = load(os.path.join(RESULTS, "cost_estimate.json"))
    measured = load(os.path.join(RESULTS, "llm_runtime_measured.json"))
    rows = []
    if measured and measured.get("items"):
        for it in measured["items"]:
            rows.append([it["scale"], it["method"], f"{it['runtime']:.1f}",
                         f"<span class='muted'>{html.escape(it['note'][:110])}</span>"])
    lat = ""
    if rows:
        lat = (f"<h3>未命中磁盘缓存的实测时延</h3>"
               + table(["Scale", "Method", "时延 (s)", "条件"], rows).replace(
                   "<th>条件</th>", "<th class='left'>条件</th>").replace(
                   f"<td><span class='muted'>", f"<td class='left'><span class='muted'>"))
    if not cost:
        return lat
    scen = cost.get("summary", {})
    fair = cost.get("fairness", {})

    def get(prefix):
        for k, v in scen.items():
            if k.startswith(prefix):
                return v
        return {}

    abl, neg = get("消融"), get("协商")
    srows = []
    if abl:
        srows.append(["消融（2 数据集 × 3 档 × 4 变体）",
                      f"¥{abl['per_n1_cny']:.2f}", f"{abl['per_n1_hours']*60:.0f} 分钟",
                      f"¥{abl['per_n1_cny']*3:.2f}", f"{abl['per_n1_hours']*3*60:.0f} 分钟"])
    if neg:
        srows.append(["协商（2 数据集 × Medium，可复用缓存）",
                      f"¥{neg['per_n1_cny']:.2f}", f"{neg['per_n1_hours']*60:.0f} 分钟",
                      f"¥{neg['per_n1_cny']*3:.2f}", f"{neg['per_n1_hours']*3*60:.0f} 分钟"])
    if fair:
        srows.append(["公平性（30 人池）",
                      f"¥{fair['cny']:.2f} / 任务", f"{fair['sec']/60:.0f} 分钟 / 任务",
                      f"¥{fair['cny']*60:.2f}（60 任务）", f"{fair['sec']*60/3600:.1f} 小时"])
    inst = cost.get("rows", [])
    per_inst = table(
        ["Setting", "变体", "调用次数", "输入 token", "输出 token", "¥", "秒"],
        [[r["setting"], r["variant"], f"{r['calls']:.0f}", f"{r['tok_in']:.0f}",
          f"{r['tok_out']:.0f}", f"{r['cny']:.2f}", f"{r['sec']:.0f}"] for r in inst],
        cls="grid small") if inst else ""
    return (lat
            + f"<h3>跑一轮真实 LLM 实验要多少钱（Qwen2.5-72B，¥{cost['price_in']:.2f}/M token）</h3>"
            + table(["实验", "单个实例 / 任务", "墙钟", "完整规模", "总墙钟"], srows)
            + "<p class='muted'>按 3 并发、真实请求的实测吞吐折算；命中磁盘缓存的调用只花本地读取时间，"
              "所以“重跑同一批实例”几乎为 0。</p>"
            + "<details><summary>展开：逐实例的调用量 / token / 费用明细</summary>" + per_inst + "</details>")


# --------------------------------------------------------------------------- 基座与扰动
def base_block() -> str:
    data = load(os.path.join(RESULTS, "base_llm_n2.json"))
    if not data:
        return "<p class='muted'>（缺少基座模型结果）</p>"
    rows = []
    for model, per in data["models"].items():
        for ds, v in per.items():
            rows.append([model.split("/")[-1], DATASET_LABEL.get(ds, ds),
                         f"{v['coverage']:.3f}", f"{v['pss']:.3f}", f"{v['runtime']:.0f}"])
    return (f"<p class='muted'>Medium，n={data['n']}（论文是 9 个模型 × 20 任务，这里只用于看趋势）。</p>"
            + table(["Base LLM", "Dataset", "Coverage", "PSS", "时延 (s)"], rows))


def case_block() -> str:
    data = pick("case_study_")
    if not data:
        return ""
    rows = []
    for name, c in data["cases"].items():
        rows.append([name, f"[{c['grid'][0]}, {c['grid'][1]}]",
                     f"{c['coverage_before']:.3f} → {c['coverage_after']:.3f}",
                     f"{c['pss_before']:.3f} → {c['pss_after']:.3f}",
                     f"{c['n_changed']}/{c['n_agents']}"])
    details = []
    for name, c in data["cases"].items():
        items = "".join(
            f"<li><b>{d['worker']}</b> — {'改道' if (d.get('changed') or d.get('adjust')) else '保持'}："
            f"{html.escape((d.get('explanation') or '')[:200])}</li>" for d in c["decisions"][:6])
        details.append(f"<details><summary>{name}：{html.escape(c['event'][:70])}</summary>"
                       f"<ul class='decisions'>{items}</ul></details>")
    return (table(["Case", "Grid", "Coverage", "PSS", "改道人数"], rows) + "".join(details))


# --------------------------------------------------------------------------- 样式
CSS = """
:root{
  --bg:#f5f7f9; --card:#fff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb;
  --accent:#2f6f4f; --accent-soft:#eaf4ee; --real:#2563eb; --warn:#b45309;
  --up:#0f766e; --down:#b91c1c;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px 90px}
header.top{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.86);
  backdrop-filter:saturate(180%) blur(12px);border-bottom:1px solid var(--line)}
.top-in{max-width:1180px;margin:0 auto;padding:12px 22px;display:flex;align-items:center;gap:14px;
  flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:-.01em;font-size:16px}
.brand span{color:var(--accent)}
nav.links{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
nav.links a{color:var(--muted)}
nav.links a:hover{color:var(--accent)}
.hero{padding:34px 0 10px}
.hero h1{margin:0 0 8px;font-size:30px;letter-spacing:-.02em;line-height:1.25}
.hero p{margin:0;color:var(--muted);max-width:760px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;
  box-shadow:0 1px 2px rgba(16,24,40,.04)}
.kpi b{display:block;font-size:26px;line-height:1.2;margin:4px 0 2px;letter-spacing:-.02em}
.kpi-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.kpi-note{font-size:12.5px;color:var(--muted)}
.mini-head{font-weight:600;margin-bottom:6px}
.mini-list{list-style:none;margin:0;padding:0;font-size:13px;color:var(--muted)}
.mini-list li{display:flex;justify-content:space-between;padding:2px 0}
.mini-list b{color:var(--ink);font-variant-numeric:tabular-nums}
.mini.hi{border-color:#bfe0cd;background:linear-gradient(180deg,#f6fbf8,#fff)}
section{margin-top:38px;scroll-margin-top:74px}
h2{font-size:21px;margin:0 0 4px;letter-spacing:-.01em;display:flex;align-items:center;gap:10px}
h2 .num{display:inline-flex;width:26px;height:26px;align-items:center;justify-content:center;
  border-radius:8px;background:var(--accent-soft);color:var(--accent);font-size:13px;font-weight:700}
h3{font-size:14.5px;margin:22px 0 8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
p{margin:8px 0}
.muted{color:var(--muted);font-size:13px}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11.5px;background:#eef1f4;
  color:#475569;border:1px solid #e2e8f0;font-weight:600}
.pill.real{background:#eef4ff;color:var(--real);border-color:#dbe6ff}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff}
table.grid{border-collapse:collapse;width:100%;font-size:13px}
table.grid th,table.grid td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.grid th{background:#fafbfc;color:#475569;font-weight:600;text-align:right;position:sticky;top:0}
table.grid td:first-child,table.grid th:first-child,
table.grid td:nth-child(2),table.grid th:nth-child(2){text-align:left}
table.grid tbody tr:hover{background:#fbfdfe}
table.grid.small{font-size:12.5px}
td.left,th.left{text-align:left!important;white-space:normal;min-width:240px}
tr:last-child td{border-bottom:none}
.up{color:var(--up);font-weight:600}
.down{color:var(--down);font-weight:600}
.flat{color:var(--muted)}
.barwrap{display:block;min-width:90px}
.bar{display:inline-block;width:100%;height:7px;border-radius:99px;background:#eef1f4;overflow:hidden;
  vertical-align:middle}
.bar i{display:block;height:100%;border-radius:99px}
figure.shot{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px}
figure.shot img{width:100%;display:block;border-radius:6px}
figcaption{color:var(--muted);font-size:12.5px;margin-top:8px}
details{margin:10px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 14px}
details summary{cursor:pointer;font-size:13.5px;color:#334155;font-weight:600}
details[open]{padding-bottom:4px}
ul.decisions{font-size:13px;color:#374151;margin:10px 0 4px;padding-left:18px}
ul.decisions li{margin:6px 0}
.warn{background:#fffbf3;border:1px solid #f3ddbb;border-radius:12px;padding:14px 16px;font-size:13.5px}
.warn b{color:var(--warn)}
.note{background:#fff;border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;padding:10px 14px;font-size:13.5px;color:#374151}
footer{margin-top:44px;color:var(--muted);font-size:12.5px}
@media (max-width:640px){
  .hero h1{font-size:23px}
  .top-in{padding:10px 14px}
  nav.links{gap:10px;font-size:12px}
  .wrap{padding:0 14px 70px}
}
"""

NAV = [("main", "主结果"), ("ablation", "消融"), ("negotiation", "协商精修"),
       ("fairness", "公平性"), ("cost", "时延与成本"), ("base", "基座与扰动"),
       ("gaps", "已知差距")]


def build() -> str:
    cost = load(os.path.join(RESULTS, "cost_estimate.json")) or {}
    cache_n = sum(len(f) for _r, _d, f in os.walk(os.path.join(RESULTS, "llm_cache"))) or 0

    parts = []
    parts.append("<header class='top'><div class='top-in'>"
                 "<div class='brand'>MAPUS <span>复现看板</span></div>"
                 + "<nav class='links'>"
                 + "".join(f"<a href='#{i}'>{t}</a>" for i, t in NAV)
                 + "</nav></div></header>")

    parts.append("<div class='wrap'>")
    parts.append("<div class='hero'><h1>多智能体参与式城市感知：结果看板</h1>"
                 "<p>复现 arXiv:2603.24014 的三阶段框架（偏好感知路线生成 → 公平感知选人 → "
                 "协商式精修），两个数据集 × 三个规模 × 8 个方法；关键结论同时给出"
                 "<b>真实 LLM（Qwen2.5-72B-Instruct）</b>与离线代理模型两套结果。</p></div>")

    parts.append("<div class='cards'>"
                 + kpi("数据集", "2", "T-Drive · Grab-Posisi（合成同构移动性）")
                 + kpi("规模档位", "3 × 2", "Small / Medium / Large")
                 + kpi("对照方法", "8", "6 个经典基线 + 单 LLM 基线 + MAPUS")
                 + kpi("真实调用", f"{cache_n:,}", "磁盘缓存条数（= 去重后的真实请求）")
                 + kpi("累计花费", "≈ ¥32", "含主表、消融、协商、公平性")
                 + "</div>")

    # 1 主结果
    parts.append("<section id='main'><h2><span class='num'>1</span>主结果（表 2 / 表 3）</h2>"
                 "<p class='muted'>Coverage 为覆盖效用，PSS 为路径满意度；★ 表示该档 PSS 最高。"
                 "差值列以论文为参照，最后一列以同一模型的单 LLM 基线为参照。</p>"
                 + main_tables()
                 + "<div class='note'><b>怎么读</b>：真实 LLM 下 MAPUS 在 6 个档位里 5 个的 PSS 最高，"
                   "且 PSS 都明显高于同模型单 LLM 基线；Coverage 与最强经典基线同量级。"
                   "T-Drive 的 PSS 比论文低 0.11–0.20（PSS 尺度 (a, μ) 与画像文本的标定差异），"
                   "Grab 上几乎对齐论文（−0.06 / +0.01 / +0.05）。</div></section>")

    # 2 消融
    parts.append("<section id='ablation'><h2><span class='num'>2</span>组件消融（图 3）</h2>"
                 "<p class='muted'>四个变体：去掉偏好感知路线生成（PRG）、去掉公平感知选人（FPS）、"
                 "去掉协商精修（NRR），以及完整模型。</p>"
                 + ablation_block()
                 + "<div class='note'><b>结论</b>：FPS 是主角（去掉后 PSS 掉 0.13–0.20、公平性塌到 Gini 0.58）；"
                   "PRG 次之（0.00–0.07，Grab Small 上因 8×4 网格太小而为 0）；NRR 的作用是<b>把重叠路线推开</b>，"
                   "代价是 0.01–0.02 的 PSS——三个组件的 Coverage 差异都 ≤1%，说明覆盖由经典规划器打底。</div></section>")

    # 3 协商
    parts.append("<section id='negotiation'><h2><span class='num'>3</span>协商式路线精修（表 4）</h2>"
                 + negotiation_block() + "</section>")

    # 4 公平性
    parts.append("<section id='fairness'><h2><span class='num'>4</span>公平感知选人（图 4）</h2>"
                 + fairness_block()
                 + "<div class='note'>FPS 用 3.4% 的覆盖率换来 <b>PSS +0.12</b> 和近乎完美的均衡"
                   "（Gini 0.007、单人最多被选 9/60）；纯覆盖贪心 CG 覆盖率最高，却把 60 个任务里的 "
                   "35 个给了同一个人。</div></section>")

    # 5 成本
    parts.append("<section id='cost'><h2><span class='num'>5</span>时延与成本</h2>"
                 "<p class='muted'>真实 LLM 的墙钟只有在未命中磁盘缓存时才有意义；命中缓存时退化为本地读取。"
                 "成本按 SiliconFlow 的 Qwen2.5-72B-Instruct（¥4.13/M token）估算，"
                 "由 <code>experiments/estimate_llm_cost.py</code> 从真实调用计数与提示词长度算出。</p>"
                 + cost_block() + "</section>")

    # 6 基座与扰动
    parts.append("<section id='base'><h2><span class='num'>6</span>基座模型与动态扰动（图 11 / 图 12）</h2>"
                 + base_block() + "<h3>扰动案例（真实 LLM）</h3>" + case_block() + "</section>")

    # 7 差距
    parts.append("<section id='gaps'><h2><span class='num'>7</span>已知差距</h2>"
                 "<div class='warn'><b>不可恢复的信息</b>：原始 POI / 犯罪数据、<code>E(D)</code> 的分层熵定义、"
                 "式 (2) 的 μ、算法 1 的 λ、式 (6) 的 β、算法 3 的 τ/K/N，以及 T-Drive / Grab-Posisi 的原始轨迹均未公开。"
                 "本复现用「结构化合成城市 + 逐格配额抽样」生成土地利用、用同分布合成移动性替代真实轨迹，"
                 "并对 PSS 尺度 (a, μ) 与覆盖效用 e_scale 做了标定。<b>因此绝对数值不能与论文逐格对齐，"
                 "可比的是同一实现下方法之间的相对关系与趋势。</b></div>"
                 "<p class='muted'>n=3 的真实 LLM 结果存在子样本噪声：Small 档 MAPUS 的 PSS 不是第一"
                 "（T-Drive TVPG 0.796、Grab TCPG 0.888 更高），Grab Large 的 Coverage 也略低于该子样本里最好的 MSA；"
                 "在 n=20 的离线全量里 MAPUS 六个档位 PSS 全第一，因此判断为抽样波动而非机制问题。</p></section>")

    parts.append("<footer>由 <code>experiments/build_dashboard.py</code> 从 "
                 "<code>results/*.json</code> 自动生成，单文件、无外部依赖。"
                 "数据文件：main / ablation / negotiation / fairness / base_llm / cost_estimate。</footer>")
    parts.append("</div>")
    return (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>MAPUS 复现看板</title><style>{CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")


def main() -> None:
    text = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(text)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
