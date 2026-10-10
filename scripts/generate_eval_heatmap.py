#!/usr/bin/env python3
"""Generate a standalone Base-vs-LoRA RoboTwin evaluation heatmap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_result_dirs(directories: list[str]) -> dict[str, float]:
    results: dict[str, float] = {}
    for directory in directories:
        root = Path(directory)
        for result_file in sorted(root.glob("*/_result.txt")):
            text = result_file.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"(?:^|\n)\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$", text)
            if not matches:
                raise ValueError(f"Cannot parse success rate from {result_file}")
            results[result_file.parent.name] = float(matches[-1])
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-clean-dir", action="append", required=True)
    parser.add_argument("--baseline-randomized-dir", action="append", required=True)
    parser.add_argument("--lora-clean-dir", action="append", required=True)
    parser.add_argument("--lora-randomized-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="LingBot-VLA LoRA Evaluation")
    parser.add_argument("--baseline-label", default="Base")
    parser.add_argument("--lora-label", default="LoRA-90k")
    parser.add_argument("--baseline-episodes", type=int, default=100)
    parser.add_argument("--lora-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "title": args.title,
        "baselineLabel": args.baseline_label,
        "loraLabel": args.lora_label,
        "baselineEpisodes": args.baseline_episodes,
        "loraEpisodes": args.lora_episodes,
        "clean": {
            "baseline": load_result_dirs(args.baseline_clean_dir),
            "lora": load_result_dirs(args.lora_clean_dir),
        },
        "randomized": {
            "baseline": load_result_dirs(args.baseline_randomized_dir),
            "lora": load_result_dirs(args.lora_randomized_dir),
        },
    }
    html = TEMPLATE.replace("__REPORT_DATA__", json.dumps(payload, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LoRA evaluation heatmap</title>
  <style>
    :root { --ink:#17242b; --muted:#65757d; --line:#dce6e9; --paper:#f4f8f8; --teal:#075e6b; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    .page { max-width:1480px; margin:auto; padding:38px 28px 60px; }
    header { display:flex; gap:24px; justify-content:space-between; align-items:flex-end; margin-bottom:20px; }
    h1 { margin:0 0 6px; font-size:clamp(26px,3vw,42px); letter-spacing:-.035em; }
    .subtitle { color:var(--muted); }
    .warning { margin:16px 0 22px; padding:13px 16px; border:1px solid #e5c77c; border-radius:12px; background:#fff9e8; color:#69501b; }
    .toolbar { display:flex; gap:10px; flex-wrap:wrap; }
    button { border:1px solid var(--line); background:white; padding:8px 12px; border-radius:9px; cursor:pointer; color:var(--ink); }
    button.active { color:white; background:var(--teal); border-color:var(--teal); }
    .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:22px; }
    .metric { background:white; border:1px solid var(--line); border-radius:14px; padding:15px 17px; box-shadow:0 6px 22px rgba(30,64,70,.05); }
    .metric .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .metric .value { margin-top:5px; font-size:25px; font-weight:750; }
    .metric .detail { color:var(--muted); font-size:12px; margin-top:2px; }
    .grids { display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }
    .panel { background:white; border:1px solid var(--line); border-radius:16px; overflow:hidden; box-shadow:0 8px 26px rgba(30,64,70,.06); }
    .panel-title { padding:17px 18px 14px; border-bottom:1px solid var(--line); }
    .panel-title h2 { margin:0; font-size:21px; }
    .panel-title p { margin:3px 0 0; color:var(--muted); font-size:12px; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; }
    th { position:sticky; top:0; z-index:2; background:#f8fbfb; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    th,td { padding:7px 10px; border-bottom:1px solid #edf2f3; text-align:center; }
    th:first-child,td:first-child { width:43%; text-align:left; padding-left:18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    tbody tr:hover td:first-child { background:#f7fbfb; }
    .heat { font-variant-numeric:tabular-nums; font-weight:650; }
    .delta { font-variant-numeric:tabular-nums; font-weight:700; }
    .delta.up { color:#08783f; } .delta.down { color:#b33d38; } .delta.flat { color:#738187; }
    .missing { color:#a4afb3; background:#f5f7f7 !important; }
    footer { color:var(--muted); margin-top:18px; font-size:12px; }
    @media (max-width:900px) { .grids{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr} header{align-items:flex-start;flex-direction:column} }
    @media print { body{background:white}.page{padding:10px}.toolbar{display:none}.panel,.metric{box-shadow:none}.warning{break-inside:avoid} }
  </style>
</head>
<body>
<main class="page">
  <header>
    <div><h1 id="title"></h1><div class="subtitle">RoboTwin · Base 与 LoRA 逐任务成功率对比</div></div>
    <div class="toolbar" aria-label="row filter">
      <button class="active" data-filter="all">全部任务</button>
      <button data-filter="up">仅进步</button>
      <button data-filter="down">仅退步</button>
      <button data-filter="delta">按变化排序</button>
    </div>
  </header>
  <div class="warning" id="warning"></div>
  <section class="summary" id="summary"></section>
  <section class="grids">
    <article class="panel"><div class="panel-title"><h2 id="clean-title"></h2><p>固定场景评测</p></div><div id="clean"></div></article>
    <article class="panel"><div class="panel-title"><h2 id="randomized-title"></h2><p>仿真器按随机种子现场生成场景</p></div><div id="randomized"></div></article>
  </section>
  <footer>成功率来自评测目录中的 <code>_result.txt</code>；该页面不依赖外部资源，可直接保存、截图或打印为 PDF。</footer>
</main>
<script>
const DATA = __REPORT_DATA__;
let filter = 'all';
const pct = x => x == null ? '—' : `${(x*100).toFixed(1)}%`;
const mean = obj => { const v=Object.values(obj); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; };
function color(v){ if(v==null)return ''; const a=.07+.82*v; return `rgba(1,103,120,${a})`; }
function textColor(v){ return v!=null && v>.52 ? '#fff' : '#17333a'; }
function rowsFor(setting){
  const d=DATA[setting], tasks=[...new Set([...Object.keys(d.baseline),...Object.keys(d.lora)])].sort();
  let rows=tasks.map(task=>({task,b:d.baseline[task]??null,l:d.lora[task]??null})).map(r=>({...r,delta:r.b==null||r.l==null?null:r.l-r.b}));
  if(filter==='up') rows=rows.filter(r=>r.delta>0);
  if(filter==='down') rows=rows.filter(r=>r.delta<0);
  if(filter==='delta') rows.sort((a,b)=>(b.delta??-9)-(a.delta??-9));
  return rows;
}
function renderTable(setting){
  const rows=rowsFor(setting);
  const head=`<table><thead><tr><th>Task</th><th>${DATA.baselineLabel}</th><th>${DATA.loraLabel}</th><th>Δ</th></tr></thead><tbody>`;
  const body=rows.map(r=>{
    const delta=r.delta==null?'—':`${r.delta>0?'+':''}${(r.delta*100).toFixed(1)}`;
    const cls=r.delta==null||r.delta===0?'flat':r.delta>0?'up':'down';
    const cell=(v,episodes)=>`<td class="heat ${v==null?'missing':''}" title="${episodes} episode(s)" style="background:${color(v)};color:${textColor(v)}">${pct(v)}</td>`;
    return `<tr><td title="${r.task}">${r.task}</td>${cell(r.b,DATA.baselineEpisodes)}${cell(r.l,DATA.loraEpisodes)}<td class="delta ${cls}">${delta}</td></tr>`;
  }).join('');
  document.getElementById(setting).innerHTML=head+body+'</tbody></table>';
}
function render(){
  document.getElementById('title').textContent=DATA.title;
  const cB=mean(DATA.clean.baseline),cL=mean(DATA.clean.lora),rB=mean(DATA.randomized.baseline),rL=mean(DATA.randomized.lora);
  document.getElementById('clean-title').textContent=`Clean · ${DATA.loraLabel} ${pct(cL)}`;
  document.getElementById('randomized-title').textContent=`Randomized · ${DATA.loraLabel} ${pct(rL)}`;
  const cards=[['Clean Base',cB,`${Object.keys(DATA.clean.baseline).length} tasks × ${DATA.baselineEpisodes} episodes`],['Clean LoRA',cL,`${Object.keys(DATA.clean.lora).length} tasks × ${DATA.loraEpisodes} episode(s)`],['Randomized Base',rB,`${Object.keys(DATA.randomized.baseline).length} tasks × ${DATA.baselineEpisodes} episodes`],['Randomized LoRA',rL,`${Object.keys(DATA.randomized.lora).length} tasks × ${DATA.loraEpisodes} episode(s)`]];
  document.getElementById('summary').innerHTML=cards.map(x=>`<div class="metric"><div class="label">${x[0]}</div><div class="value">${pct(x[1])}</div><div class="detail">${x[2]}</div></div>`).join('');
  document.getElementById('warning').innerHTML=`<strong>样本量提示：</strong>${DATA.baselineLabel} 每任务 ${DATA.baselineEpisodes} 局，${DATA.loraLabel} 当前每任务 ${DATA.loraEpisodes} 局。LoRA 的 0% / 100% 是小样本结果，只用于快速筛查；正式精度需等多 episode 评测完成后刷新。`;
  renderTable('clean'); renderTable('randomized');
}
document.querySelectorAll('button[data-filter]').forEach(btn=>btn.addEventListener('click',()=>{filter=btn.dataset.filter;document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));btn.classList.add('active');renderTable('clean');renderTable('randomized');}));
render();
</script>
</body></html>'''


if __name__ == "__main__":
    main()
