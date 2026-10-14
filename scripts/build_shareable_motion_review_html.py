#!/usr/bin/env python3
"""Build a single-file, offline RoboTwin trajectory review HTML."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path


DERIVATIVE_ZH = {"velocity": "速度", "acceleration": "加速度", "jerk": "Jerk"}


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def metric_base(metric: str) -> str:
    for suffix in ("_max_abs_raw", "_p99_abs_raw"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def metric_label(metric: str) -> str:
    base = metric_base(metric)
    signal, derivative = base.split("_", 1)
    statistic = "P99" if "_p99_abs_raw" in metric else "峰值"
    return f"{signal} · {DERIVATIVE_ZH[derivative]} · {statistic}"


def joint_label(index: int) -> str:
    if 0 <= index <= 5:
        return f"左臂 J{index}（dim {index}）"
    if 7 <= index <= 12:
        return f"右臂 J{index - 7}（dim {index}）"
    return f"dim {index}"


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(args.manifest.read_text())
    manifest_by_key = {(item["task"], int(item["episode"])): item for item in manifest}

    items = []
    needed_videos: set[str] = set()
    for row in rows:
        if not as_bool(row["smoothness_priority_review_flag"]):
            continue
        source = manifest_by_key[(row["task"], int(row["episode"]))]
        stage = "clean" if as_bool(row["remove_flag"]) else "manual"
        metric = row["worst_metric"]
        base = metric_base(metric)
        candidate_video = source["filename"]
        normal_video = source["normal_reference"]["filename"]
        needed_videos.update((candidate_video, normal_video))
        items.append(
            {
                "id": f"{row['task']}:ep{int(row['episode'])}",
                "stage": stage,
                "task": row["task"],
                "episode": int(row["episode"]),
                "metric": metric_label(metric),
                "z": float(row["worst_abs_robust_z"]),
                "step": max(float(row["state_max_joint_step"]), float(row["action_max_joint_step"])),
                "frame": int(row[f"{base}_frame"]),
                "joint": joint_label(int(row[f"{base}_joint_dim"])),
                "state_static": float(row["state_static_ratio_1e6"]),
                "action_static": float(row["action_static_ratio_1e6"]),
                "instruction_zh": source["instruction_zh"],
                "instruction": source["instruction"],
                "candidate_video": candidate_video,
                "candidate_time": float(source["critical_time"]),
                "normal_episode": int(source["normal_reference"]["episode"]),
                "normal_instruction_zh": source["normal_reference"]["instruction_zh"],
                "normal_video": normal_video,
                "normal_time": float(source["normal_reference"]["critical_time"]),
            }
        )
    items.sort(key=lambda item: (item["stage"] != "clean", -item["step"]))

    videos = {}
    for number, filename in enumerate(sorted(needed_videos), 1):
        path = args.clips_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"[{number:02d}/{len(needed_videos)}] embedding {filename}", flush=True)
        videos[filename] = data_uri(path)

    payload = json.dumps({"items": items, "videos": videos}, ensure_ascii=False).replace("</", "<\\/")
    args.output.write_text(TEMPLATE.replace("__PAYLOAD__", payload))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTwin 轨迹清洗人工复核</title>
<style>
:root{--bg:#f5f6f3;--panel:#fff;--text:#171918;--muted:#686d69;--line:#d8dcd8;--accent:#155f52;--danger:#9d2929;--warn:#975500}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.wrap{max-width:1500px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 7px}h2{font-size:19px}.lead{color:var(--muted);max-width:980px}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.stat b{display:block;font-size:22px}.stat span{color:var(--muted)}.guide{background:var(--panel);border-left:4px solid var(--accent);padding:12px 15px;margin:15px 0}.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:11px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.toolbar input,.toolbar select,.toolbar button,button{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:8px 11px}.toolbar button,button{cursor:pointer}.stats{margin-left:auto;color:var(--muted)}.card{display:grid;grid-template-columns:minmax(650px,2fr) minmax(350px,1fr);gap:18px;background:var(--panel);border:1px solid var(--line);border-radius:9px;margin:16px 0;padding:16px}.card.clean{border-left:5px solid var(--danger)}.card.manual{border-left:5px solid var(--warn)}.videos{display:grid;grid-template-columns:1fr 1fr;gap:9px}.video-title{font-weight:650;margin-bottom:5px}.candidate-title{color:var(--danger)}.normal-title{color:var(--accent)}video{width:100%;background:#000;border-radius:5px}.tools{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.tools button{font-size:12px;padding:5px 8px}.title{font-size:18px;font-weight:700}.pill{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:6px}.pill.clean{color:var(--danger);background:color-mix(in srgb,var(--danger) 10%,transparent)}.pill.manual{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent)}.instruction{margin:10px 0;padding:10px 12px;background:var(--bg);border-left:3px solid var(--accent)}.instruction .zh{font-weight:650}.instruction .en{color:var(--muted);font-size:12px;margin-top:4px}.meta{font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);margin:10px 0}.decision label{display:block;padding:6px 0}.decision textarea{width:100%;min-height:80px;border:1px solid var(--line);border-radius:5px;padding:8px;background:var(--bg);color:var(--text)}.hidden{display:none}.footer{color:var(--muted);font-size:12px;margin:22px 0}@media(max-width:1050px){.card{grid-template-columns:1fr}}@media(max-width:720px){.wrap{padding:14px}.summary{grid-template-columns:1fr 1fr}.videos{grid-template-columns:1fr}.stats{margin-left:0;width:100%}}@media(prefers-color-scheme:dark){:root{--bg:#171918;--panel:#222523;--text:#edf0ed;--muted:#a9b0ab;--line:#3b403d;--accent:#6cb5a7;--danger:#ef8484;--warn:#e6a866}}
</style></head><body><main class="wrap">
<h1>RoboTwin 轨迹清洗人工复核</h1>
<p class="lead">这是可离线转发的单文件版本，17 条候选和正常参考视频均已嵌入。请逐条选择“保留 / 删除 / 不确定”，填写备注后导出 CSV 或 JSON 发回。</p>
<div class="summary"><div class="stat"><b>17</b><span>待复核候选</span></div><div class="stat"><b>4</b><span>算法建议清洗</span></div><div class="stat"><b>13</b><span>算法建议人工判断</span></div><div class="stat"><b>24</b><span>内嵌三相机视频</span></div></div>
<div class="guide"><b>判断标准：</b>候选与右侧同任务正常参考对比。机械臂瞬移、画面与 action 不同步、碰撞后失控、任务演示失败可删除；正常快速接触、按压、抓取或放置应保留。算法建议只是排序依据，最终以视频为准。</div>
<div class="toolbar"><input id="reviewer" placeholder="复核人（选填）"><select id="filter"><option value="all">全部</option><option value="unreviewed">未复核</option><option value="delete">已判删除</option><option value="keep">已判保留</option><option value="uncertain">不确定</option></select><button id="exportCsv">导出 CSV</button><button id="exportJson">导出 JSON</button><span class="stats" id="stats"></span></div>
<section id="list"></section><div class="footer">数据来源：RoboTwin official clean demonstrations · 审计阈值：优先复核 robust Z ≥ 3.5 且最大关节步长 ≥ 0.25 rad/帧。</div>
</main><script>
const payload=__PAYLOAD__,items=payload.items,videos=payload.videos,key='robotwin-motion-review-v2';let saved=JSON.parse(localStorage.getItem(key)||'{}');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function persist(){localStorage.setItem(key,JSON.stringify(saved));renderStats()};function decision(id){return saved[id]?.decision||''};function pct(x){return(100*x).toFixed(1)+'%'};
function render(){const f=document.querySelector('#filter').value,root=document.querySelector('#list');root.innerHTML='';items.forEach((x,i)=>{const d=decision(x.id);if(f==='unreviewed'&&d||!['all','unreviewed'].includes(f)&&d!==f)return;const el=document.createElement('article');el.className='card '+x.stage;const recommendation=x.stage==='clean'?'建议清洗':'人工复核';el.innerHTML=`<div><div class="videos"><div><div class="video-title candidate-title">异常候选 · episode ${x.episode}</div><video class="candidate" controls preload="metadata"></video></div><div><div class="video-title normal-title">正常参考 · episode ${x.normal_episode}</div><video class="normal" controls preload="metadata"></video></div></div><div class="tools"><button data-candidate>跳到候选异常帧</button><button data-normal>跳到正常参考关键帧</button><button data-rate="0.25">两边 0.25×</button><button data-rate="0.5">两边 0.5×</button><button data-rate="1">两边 1×</button></div></div><div><div class="title">${i+1}. ${esc(x.task)} · episode ${x.episode}<span class="pill ${x.stage}">${recommendation}</span></div><div class="instruction"><div class="zh">${esc(x.instruction_zh)}</div><div class="en">${esc(x.instruction)}</div></div><div class="meta">主导维度：${esc(x.metric)}<br>轨迹级 robust Z：${x.z.toFixed(2)}<br>最大关节步长：${x.step.toFixed(6)} rad/帧<br>定位：frame ${x.frame} · ${esc(x.joint)}<br>state/action 静止比例：${pct(x.state_static)} / ${pct(x.action_static)}<br>正常参考：${esc(x.normal_instruction_zh)}</div><div class="decision">${[['keep','保留'],['delete','删除'],['uncertain','不确定']].map(([v,l])=>`<label><input type="radio" name="${x.id}" value="${v}" ${d===v?'checked':''}> ${l}</label>`).join('')}<label>备注<textarea placeholder="记录瞬移、碰撞、任务失败或保留理由">${esc(saved[x.id]?.note||'')}</textarea></label></div></div>`;const cv=el.querySelector('video.candidate'),nv=el.querySelector('video.normal');cv.src=videos[x.candidate_video];nv.src=videos[x.normal_video];el.querySelector('[data-candidate]').onclick=()=>{cv.pause();cv.currentTime=x.candidate_time};el.querySelector('[data-normal]').onclick=()=>{nv.pause();nv.currentTime=x.normal_time};el.querySelectorAll('[data-rate]').forEach(b=>b.onclick=()=>{const r=Number(b.dataset.rate);cv.playbackRate=r;nv.playbackRate=r;cv.play();nv.play()});el.querySelectorAll('input[type=radio]').forEach(r=>r.onchange=()=>{saved[x.id]={...(saved[x.id]||{}),decision:r.value};persist()});el.querySelector('textarea').oninput=e=>{saved[x.id]={...(saved[x.id]||{}),note:e.target.value};persist()};root.appendChild(el)});renderStats()}
function renderStats(){const ds=items.map(x=>decision(x.id));document.querySelector('#stats').textContent=`已复核 ${ds.filter(Boolean).length}/17 · 保留 ${ds.filter(x=>x==='keep').length} · 删除 ${ds.filter(x=>x==='delete').length} · 不确定 ${ds.filter(x=>x==='uncertain').length}`}
function result(){const reviewer=document.querySelector('#reviewer').value.trim();return items.map(x=>({reviewer,...x,video_data_omitted:true,decision:decision(x.id),note:saved[x.id]?.note||''}))};function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.querySelector('#filter').onchange=render;document.querySelector('#exportJson').onclick=()=>download('robotwin_motion_review.json',JSON.stringify(result(),null,2),'application/json');document.querySelector('#exportCsv').onclick=()=>{const cols=['reviewer','task','episode','stage','metric','z','step','frame','joint','decision','note'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';download('robotwin_motion_review.csv',[cols.join(','),...result().map(x=>cols.map(c=>q(x[c])).join(','))].join('\n'),'text/csv')};render();
</script></body></html>'''


if __name__ == "__main__":
    main()
