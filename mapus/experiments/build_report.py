# -*- coding: utf-8 -*-
r"""把 results/*.json + results/figures/*.png 汇总成单文件 HTML 报告。

    py -3.14 experiments\build_report.py
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


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _img(path: str, caption: str) -> str:
    if not os.path.exists(path):
        return f"<p class='muted'>（缺少图片 {html.escape(os.path.basename(path))}）</p>"
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return (f"<figure><img src='data:image/png;base64,{data}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def _table(headers, rows, cls="tbl") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for r in rows:
        cells = "".join(f"<td>{c}</td>" for c in r)
        body.append(f"<tr>{cells}</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _latest(cands):
    """同一实验有多个结果文件时，优先用真实 LLM（非 surrogate）的那份。"""
    real = [c for c in cands if "surrogate" not in os.path.basename(c).lower()]
    return (real or cands)[-1]


def _infer_n(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    for token in reversed(stem.split("_")):
        if token.startswith("n") and token[1:].isdigit():
            return int(token[1:])
    return 20


def main_table(dataset: str, tag: str, n: int, label: str) -> str:
    data = _load(os.path.join(RESULTS, f"main_{dataset}_{tag}_n{n}.json"))
    if not data:
        return f"<p class='muted'>（缺少 {dataset} 主结果：results/main_{dataset}_{tag}_n{n}.json）</p>"
    methods = ["RN", "TVPG", "TCPG", "MSA", "MSAGI", "GraphDP", "GPT-4.1-mini", "MAPUS"]
    rows = []
    for scale in ("small", "medium", "large"):
        if scale not in data["scales"]:
            continue
        per = data["scales"][scale]
        for m in methods:
            v = per.get(m)
            if not v:
                continue
            ref = PAPER[dataset][scale].get(m)
            cov, pss = v["coverage"], v["pss"]
            dcov = f"{(cov-ref[0])/ref[0]*100:+.1f}%" if ref else "-"
            dpss = f"{pss-ref[1]:+.3f}" if ref else "-"
            cls = " style='font-weight:600'" if m == "MAPUS" else ""
            rows.append([f"<span{cls}>{m}</span>", scale,
                         f"{cov:.3f} ± {v['coverage_std']:.3f}",
                         f"{pss:.3f} ± {v['pss_std']:.3f}",
                         f"{ref[0]:.3f}" if ref else "-",
                         f"{ref[1]:.3f}" if ref else "-", dcov, dpss])
    return (f"<h3>{label}（n={data['n']}，backend={data['backend']}"
            f"{'，model=' + str(data['model']) if data.get('model') and data['model'] != 'surrogate' else ''}）</h3>"
            + _table(["Method", "Scale", "Coverage（本文）", "PSS（本文）",
                      "Coverage（论文）", "PSS（论文）", "ΔCoverage", "ΔPSS"], rows))


def ablation_table() -> str:
    cands = sorted(glob.glob(os.path.join(RESULTS, "ablation_*.json")))
    if not cands:
        return "<p class='muted'>（缺少消融结果）</p>"
    real = [c for c in cands if "surrogate" not in os.path.basename(c).lower()]
    blocks = []
    for path in (real or [_latest(cands)])[-2:]:
        data = _load(path)
        if not data:
            continue
        rows = []
        for scale, per in data["results"].items():
            for name in ("w/o PRG", "w/o FPS", "w/o NRR", "Full"):
                v = per.get(name)
                if not v:
                    continue
                cls = " style='font-weight:600'" if name == "Full" else ""
                rows.append([f"<span{cls}>{name}</span>", scale,
                             f"{v['coverage']:.3f} ± {v['coverage_std']:.3f}",
                             f"{v['pss']:.3f} ± {v['pss_std']:.3f}", f"{v['overlap']:.3f}"])
        blocks.append(
            f"<h3>图 3 数据：消融实验（{data['dataset']}，n={data['n']}，"
            f"backend={data['backend']}）</h3>"
            + _table(["Variant", "Scale", "Coverage", "PSS", "Avg. overlap"], rows))
    return "".join(blocks) or "<p class='muted'>（缺少消融结果）</p>"


def negotiation_table() -> str:
    cands = sorted(glob.glob(os.path.join(RESULTS, "negotiation_*.json")))
    if not cands:
        return "<p class='muted'>（缺少协商实验结果）</p>"
    data = _load(_latest(cands))
    rows = []
    for d, v in data["datasets"].items():
        rows.append([d, f"{v['pre_neg_overlap']*100:.2f}", f"{v['avg_overlap']*100:.2f}",
                     f"{v['pre_neg_coverage']:.3f}", f"{v['coverage']:.3f}",
                     f"{v['pre_neg_entropy']:.3f}", f"{v['entropy']:.3f}",
                     f"{v['pre_neg_count']:.1f}", f"{v['count']:.1f}",
                     f"{v['pre_neg_pss']:.3f}", f"{v['pss']:.3f}"])
    return (f"<h3>表 4 数据：协商式路线精修（Medium，n={data['n']}，"
            f"backend={data['backend']}）</h3>"
            + _table(["Dataset", "Overlap 前(%)", "Overlap 后(%)", "Coverage 前", "Coverage 后",
                      "Entropy 前", "Entropy 后", "Count 前", "Count 后", "PSS 前", "PSS 后"], rows))


def fairness_table() -> str:
    cands = sorted(glob.glob(os.path.join(RESULTS, "fairness_*.json")))
    if not cands:
        return "<p class='muted'>（缺少公平性结果）</p>"
    data = _load(_latest(cands))
    rows = []
    for name, v in data["strategies"].items():
        rows.append([name, f"{v['coverage']:.3f}", f"{v['pss']:.3f}", f"{v['gini']:.3f}",
                     str(v["max_count"]), f"{v['std_count']:.2f}"])
    return (f"<h3>图 4 数据：公平感知选择（T-Drive Medium，{data['tasks']} 个任务，"
            f"池 {data['pool']} 人，backend={data['backend']}）</h3>"
            + _table(["Strategy", "Coverage", "PSS", "Gini of selection counts",
                      "Max #selections", "Std"], rows))


def runtime_table() -> str:
    cands = sorted(glob.glob(os.path.join(RESULTS, "runtime_*.json")))
    if not cands:
        return "<p class='muted'>（缺少运行时结果）</p>"
    data = _load(cands[-1])
    rows = []
    for scale, per in data["runtime"].items():
        for name, v in per.items():
            rows.append([scale, name, f"{v['mean']:.4g}"])
    note = ("<p class='muted'>离线代理模型没有 API 往返时延，所以 MAPUS / 单 LLM 两行"
            "比论文低两三个数量级；有意义的对照是<b>经典基线之间的相对耗时</b>，"
            "以及真实模型未命中磁盘缓存时的实测时延（见下表，数据来自 "
            "<code>results/llm_runtime_measured.json</code>）。"
            "<b>注意</b>：真实 LLM 的墙钟时间在命中磁盘缓存时会趋近 0，"
            "<code>main_tdrive_qwen72b</code> 里绝大多数行都是缓存回放，不能当作真实时延使用。</p>")
    measured = _load(os.path.join(RESULTS, "llm_runtime_measured.json"))
    if measured and measured.get("items"):
        extra = [[it["scale"], f"{it['method']}（真实 LLM）", f"{it['runtime']:.1f}", it["note"]]
                 for it in measured["items"]]
        note += _table(["Scale", "Method", "Runtime (s, Qwen2.5-72B)", "来源"], extra)
    return (f"<h3>表 5 数据：运行时间（{data['dataset']}，n={data['n']}，"
            f"backend={data['backend']}）</h3>" + _table(["Scale", "Method", "Runtime (s)"], rows)
            + note)


def base_llm_block() -> str:
    data = _load(os.path.join(RESULTS, "base_llm_n2.json"))
    if not data:
        return "<p class='muted'>（缺少基座模型结果 results/base_llm_n2.json）</p>"
    rows = []
    for model, per in data["models"].items():
        for dataset, v in per.items():
            src = v.get("runtime_source", "")
            rows.append([model.split("/")[-1], dataset,
                         f"{v['coverage']:.3f}", f"{v['pss']:.3f}",
                         f"{v['runtime']:.1f}",
                         "实测（未命中缓存）" if src.startswith("measured")
                         else "沿用主实验实测值（本次为缓存回放）"])
    return ("<h3>图 11 数据：不同基座 LLM（Medium，n=%d）</h3>" % data["n"]
            + _table(["Base LLM", "Dataset", "Coverage", "PSS", "Runtime (s)", "Runtime 来源"], rows)
            + "<p class='muted'>论文图 11 用 9 个基座模型 × 两个数据集 × 20 个任务；"
              "本复现受额度与限流约束，缩小为 2 个可用的硅基流动模型 × 两个数据集 × "
              "2 个任务，因此只用于复现<b>趋势</b>（轻量模型的 PSS 与重量级模型同量级，"
              "覆盖效用的差距大于 PSS 的差距），不能逐点对齐论文。</p>")


def case_study_block() -> str:
    cands = sorted(glob.glob(os.path.join(RESULTS, "case_study_*.json")))
    if not cands:
        return "<p class='muted'>（缺少案例研究结果）</p>"
    data = _load(_latest(cands))
    out = [f"<h3>图 12 数据：动态扰动（backend={data['backend']}）</h3>"]
    rows = []
    for name, c in data["cases"].items():
        rows.append([name, f"[{c['grid'][0]}, {c['grid'][1]}]",
                     f"{c['coverage_before']:.3f} → {c['coverage_after']:.3f}",
                     f"{c['pss_before']:.3f} → {c['pss_after']:.3f}",
                     f"{c['n_changed']}/{c['n_agents']}"])
    out.append(_table(["Case", "Grid", "Coverage", "PSS", "Changed agents"], rows))
    for name, c in data["cases"].items():
        out.append(f"<p class='muted'>{html.escape(c['event'])}</p>")
        items = "".join(
            f"<li><b>{d['worker']}</b>: {'调整' if d['changed'] else '保持'} — "
            f"{html.escape((d['explanation'] or '')[:220])}</li>"
            for d in c["decisions"][:8])
        out.append(f"<details><summary>各参与者的决定</summary><ul class='decisions'>{items}</ul></details>")
    return "\n".join(out)


def calibration_block() -> str:
    data = _load(os.path.join(RESULTS, "calibration.json"))
    if not data:
        return "<p class='muted'>（缺少标定结果）</p>"
    rows = []
    for k, v in (data.get("cosine") or {}).items():
        rows.append([k, f"{v:.4f}", f"{data['risk'][k]:.4f}",
                     f"{data.get('fitted_pss', {}).get(k, float('nan')):.4f}",
                     f"{data['anchors'][k]:.3f}"])
    return ("<h3>PSS 的尺度标定（式 2 的 a 与 μ）</h3>"
            f"<p>prior κ = {data['prior']}，scale a = {data.get('scale', float('nan')):.3f}，"
            f"μ = {data.get('mu', float('nan'))}，RMSE = {data.get('rmse', float('nan')):.4f}</p>"
            + _table(["Method", "余弦相似度", "risk", "PSS（拟合后）", "论文锚点"], rows))


CSS = """
:root{--fg:#1b1f24;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;--accent:#2f6f4f;}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1080px;padding:32px 22px 80px;font:15px/1.7 -apple-system,
 "Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;color:var(--fg);background:var(--bg)}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:#111}
p{margin:8px 0}
.muted{color:var(--muted);font-size:13px}
code{background:#f5f6f7;padding:1px 5px;border-radius:4px;font-size:13px}
pre{background:#f7f8f9;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px}
table.tbl{border-collapse:collapse;width:100%;margin:10px 0 4px;font-size:13px}
table.tbl th,table.tbl td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
table.tbl th:first-child,table.tbl td:first-child,table.tbl th:nth-child(2),table.tbl td:nth-child(2){text-align:left}
table.tbl thead th{background:#fafbfc;font-weight:600;color:#374151}
table.tbl tbody tr:hover{background:#fcfdfe}
figure{margin:16px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--muted);font-size:12.5px;margin-top:6px}
ul.decisions{font-size:13px;color:#374151}
details{margin:8px 0;font-size:13px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.kpi{flex:1 1 180px;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:19px}
.tag{display:inline-block;background:#eef6f1;color:var(--accent);border-radius:999px;
 padding:1px 9px;font-size:12px;margin-right:6px}
.warn{background:#fff8ed;border:1px solid #f3ddbb;border-radius:8px;padding:10px 14px;font-size:13.5px}
"""


def build() -> str:
    parts = []
    parts.append("<h1>MAPUS 复现报告</h1>")
    parts.append("<p><span class='tag'>arXiv:2603.24014v1</span>"
                 "Language-Grounded Multi-Agent Planning for Personalized and Fair "
                 "Participatory Urban Sensing</p>")
    parts.append(
        "<div class='kpis'>"
        "<div class='kpi'><b>3 阶段</b>任务派发 / 公平选择 / 协商精修</div>"
        "<div class='kpi'><b>2 数据集</b>T-Drive · Grab-Posisi（合成同构移动性）</div>"
        "<div class='kpi'><b>3×6 配置</b>Small / Medium / Large × 6 个基线</div>"
        "<div class='kpi'><b>真实 LLM</b>OpenAI 兼容网关，含磁盘缓存</div>"
        "</div>")

    parts.append("<h2>1. 复现了什么</h2>")
    parts.append("""
<ul>
<li><b>城市属性层</b>：6 类土地利用（vegetation / industrial / institutional / medical /
residential / commercial）与犯罪风险场，对应式 (3)(4) 与附录 B.1。</li>
<li><b>参与者画像</b>：5 个 archetype（生态爱好者 / 都市居民 / 产业工人 / 社区助手 / 探索者），
含人口统计、爱好、经济状况与土地利用偏好向量，对应附录 B.2 与图 10。</li>
<li><b>PSS 指标</b>：<code>S(r)=a·cos(p_u,h(r_u)) − μ·risk(r_u)</code>，对应式 (2)。</li>
<li><b>Participant Agent</b>：接单决策 + 算法 1（TSPTW 可行基线 → LLM 迭代精修 → 可行性校验/投影）。</li>
<li><b>Coordinator Agent</b>：算法 2（边际覆盖增益 + 公平分 + LLM 平局裁决）与算法 3（重叠配对协商）。</li>
<li><b>对照组</b>：RN / TVPG / TCPG / MSA / MSAGI / GraphDP 六个经典基线 + 单 LLM 一次性规划基线。</li>
<li><b>实验</b>：表 2/3 主结果、图 3 消融、表 4 协商统计、图 4 公平性、表 5 运行时、图 11 基座模型、图 12 扰动案例。</li>
</ul>""")

    parts.append("<h2>2. 主结果（表 2 / 表 3）</h2>")
    parts.append(main_table("tdrive", "surrogate", 20, "T-Drive（离线代理模型）"))
    parts.append(main_table("grab", "surrogate", 20, "Grab-Posisi（离线代理模型）"))
    if glob.glob(os.path.join(RESULTS, "main_tdrive_qwen72b_n*.json")):
        n = _infer_n(glob.glob(os.path.join(RESULTS, "main_tdrive_qwen72b_n*.json"))[0])
        parts.append(main_table("tdrive", "qwen72b", n,
                                "T-Drive（真实 LLM：Qwen2.5-72B-Instruct）"))
    if glob.glob(os.path.join(RESULTS, "main_grab_qwen72b_n*.json")):
        n = _infer_n(glob.glob(os.path.join(RESULTS, "main_grab_qwen72b_n*.json"))[0])
        parts.append(main_table("grab", "qwen72b", n,
                                "Grab-Posisi（真实 LLM：Qwen2.5-72B-Instruct）"))

    parts.append("<h2>3. 消融（图 3）</h2>")
    parts.append(ablation_table())
    parts.append(_img(os.path.join(FIGDIR, "fig_ablation.png"), "图 3：三个组件的消融。"))

    parts.append("<h2>4. 协商式精修（表 4）</h2>")
    parts.append(negotiation_table())
    parts.append(_img(os.path.join(FIGDIR, "fig_negotiation.png"), "协商前后：路线重叠与覆盖。"))

    parts.append("<h2>5. 公平性（图 4）</h2>")
    parts.append(fairness_table())
    parts.append(_img(os.path.join(FIGDIR, "fig_fairness.png"), "图 4：四种选择策略。"))

    parts.append("<h2>6. 运行时间（表 5）</h2>")
    parts.append(runtime_table())

    parts.append("<h2>7. 基座模型与扰动</h2>")
    parts.append(base_llm_block())
    parts.append(_img(os.path.join(FIGDIR, "fig_base_llm.png"), "图 11：不同基座 LLM 的性能—时延权衡。"))
    parts.append(case_study_block())

    parts.append("<h2>8. 城市属性与标定</h2>")
    parts.append(_img(os.path.join(FIGDIR, "fig_land_use.png"), "图 9：土地利用与犯罪分布（合成城市形态）。"))
    parts.append(calibration_block())

    parts.append("<h2>9. 与论文的已知差距</h2>")
    parts.append("""<div class='warn'>
<b>不可恢复的信息</b>：原始 POI / 犯罪数据、<code>E(D)</code> 的分层熵定义（引自 Ji et al. 2016）、
式 (2) 的 μ、算法 1 的 λ、式 (6) 的 β、算法 3 的 τ/K/N、以及 T-Drive / Grab-Posisi 的原始轨迹均未公开。
本工程用「结构化合成城市 + 逐格配额抽样」生成土地利用，用同分布的合成移动性替代真实轨迹，
并对 PSS 的尺度 (a, μ) 与覆盖效用的 e_scale 各做了单参数/双参数标定（见 §8）。
<b>因此绝对数值不可与论文逐格对齐，可比的是同一实现下的方法间相对关系与趋势。</b>
</div>""")
    return f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>" \
           f"<title>MAPUS 复现报告</title><style>{CSS}</style></head><body>" \
           + "\n".join(parts) + "</body></html>"


def main() -> None:
    html_text = build()
    path = "report.html"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    print(f"wrote {path} ({len(html_text)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
