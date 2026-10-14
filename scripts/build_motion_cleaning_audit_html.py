#!/usr/bin/env python3
"""Build a self-contained HTML report from the RoboTwin motion audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SIGNALS = ("state", "action")
DERIVATIVES = ("velocity", "acceleration", "jerk")
DERIVATIVE_ZH = {"velocity": "速度", "acceleration": "加速度", "jerk": "Jerk"}
UNITS = {"velocity": "rad/s", "acceleration": "rad/s²", "jerk": "rad/s³"}


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def joint_label(index: int) -> str:
    if 0 <= index <= 5:
        return f"左臂 J{index}（dim {index}）"
    if 7 <= index <= 12:
        return f"右臂 J{index - 7}（dim {index}）"
    if index == 6:
        return "左夹爪（dim 6）"
    if index == 13:
        return "右夹爪（dim 13）"
    return f"dim {index}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(args.manifest.read_text())
    videos = {(item["task"], int(item["episode"])): item for item in manifest}

    candidates = []
    for row in rows:
        remove = as_bool(row["remove_flag"])
        priority = as_bool(row["smoothness_priority_review_flag"])
        broad = as_bool(row["smoothness_review_flag"])
        if not broad and not remove:
            continue
        stage = "clean" if remove else "manual" if priority else "observe"
        metric = row["worst_metric"]
        base = metric_base(metric)
        video = videos.get((row["task"], int(row["episode"])))
        dimension_z = {
            f"{signal}_{derivative}": max(
                float(row[f"episode_z_{signal}_{derivative}_max_abs_raw"]),
                float(row[f"episode_z_{signal}_{derivative}_p99_abs_raw"]),
            )
            for signal in SIGNALS
            for derivative in DERIVATIVES
        }
        candidates.append(
            {
                "stage": stage,
                "task": row["task"],
                "episode": int(row["episode"]),
                "length": int(row["length"]),
                "metric": metric,
                "metric_label": metric_label(metric),
                "signal": base.split("_", 1)[0],
                "derivative": base.split("_", 1)[1],
                "z": float(row["worst_abs_robust_z"]),
                "step": max(float(row["state_max_joint_step"]), float(row["action_max_joint_step"])),
                "frame": int(row[f"{base}_frame"]),
                "joint": int(row[f"{base}_joint_dim"]),
                "joint_label": joint_label(int(row[f"{base}_joint_dim"])),
                "raw_derivative": float(row[f"{base}_raw"]),
                "state_static": float(row["state_static_ratio_1e6"]),
                "action_static": float(row["action_static_ratio_1e6"]),
                "dimension_z": dimension_z,
                "video": video["filename"] if video else None,
                "normal_video": video["normal_reference"]["filename"] if video else None,
                "instruction_zh": video["instruction_zh"] if video else None,
                "remove_reasons": row["remove_reasons"],
            }
        )
    order = {"clean": 0, "manual": 1, "observe": 2}
    candidates.sort(key=lambda item: (order[item["stage"]], -item["step"], -item["z"]))

    matrix = []
    for signal in SIGNALS:
        for derivative in DERIVATIVES:
            key = f"{signal}_{derivative}"
            clean = sum(item["stage"] == "clean" and item["dimension_z"][key] >= 3.5 for item in candidates)
            manual = sum(item["stage"] == "manual" and item["dimension_z"][key] >= 3.5 for item in candidates)
            observe = sum(item["stage"] == "observe" and item["dimension_z"][key] >= 3.5 for item in candidates)
            matrix.append(
                {
                    "signal": signal,
                    "derivative": derivative,
                    "label": f"{signal} · {DERIVATIVE_ZH[derivative]}",
                    "unit": UNITS[derivative],
                    "clean": clean,
                    "manual": manual,
                    "observe": observe,
                }
            )

    focus = [item for item in candidates if item["stage"] in {"clean", "manual"}]
    joint_counts = Counter(item["joint"] for item in focus)
    joints = [
        {"joint": index, "label": joint_label(index), "count": count}
        for index, count in sorted(joint_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]

    static_watch = []
    for row in rows:
        state_static = float(row["state_static_ratio_1e6"])
        action_static = float(row["action_static_ratio_1e6"])
        if state_static > 0.5 and action_static > 0.5:
            static_watch.append(
                {
                    "task": row["task"],
                    "episode": int(row["episode"]),
                    "state_static": state_static,
                    "action_static": action_static,
                    "length": int(row["length"]),
                }
            )
    static_watch.sort(key=lambda item: min(item["state_static"], item["action_static"]), reverse=True)

    data = json.dumps(
        {"candidates": candidates, "matrix": matrix, "joints": joints, "static": static_watch},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = TEMPLATE.replace("__AUDIT_DATA__", data)
    args.output.write_text(html)
    print(args.output)


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTwin 机器人轨迹清洗审计</title>
<style>
:root{--bg:#f6f7f5;--panel:#fff;--text:#171918;--muted:#666c68;--line:#d8ddd9;--accent:#165f52;--clean:#9c2525;--manual:#9a5700;--observe:#486276;--ok:#287a4b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.wrap{max-width:1480px;margin:auto;padding:30px}.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}h1{font-size:28px;line-height:1.25;margin:4px 0 8px}h2{font-size:20px;margin:30px 0 10px}h3{font-size:16px;margin:0 0 8px}.lead{max-width:980px;color:var(--muted)}.summary{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:24px 0}.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:15px}.stat b{display:block;font-size:23px;line-height:1.2}.stat span{color:var(--muted)}.callout{border-left:4px solid var(--accent);background:var(--panel);padding:12px 16px;margin:18px 0}.callout.warn{border-left-color:var(--manual)}.callout.clean{border-left-color:var(--clean)}.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:7px;background:var(--panel)}table{border-collapse:collapse;width:100%;min-width:840px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);z-index:1;font-size:12px;color:var(--muted)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.pill{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;font-weight:650}.pill.clean{color:var(--clean);background:color-mix(in srgb,var(--clean) 10%,transparent)}.pill.manual{color:var(--manual);background:color-mix(in srgb,var(--manual) 12%,transparent)}.pill.observe{color:var(--observe);background:color-mix(in srgb,var(--observe) 10%,transparent)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.toolbar input,.toolbar select{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:8px 10px}.toolbar input{min-width:260px}.muted{color:var(--muted)}.small{font-size:12px}.metric{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.links a{color:var(--accent);text-decoration:none;margin-right:8px}.links a:hover{text-decoration:underline}.legend{display:flex;gap:14px;flex-wrap:wrap}.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}.footer{margin:28px 0 6px;color:var(--muted);font-size:12px}@media(max-width:980px){.summary{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.wrap{padding:18px}}@media(prefers-color-scheme:dark){:root{--bg:#171918;--panel:#222523;--text:#eef1ee;--muted:#abb2ad;--line:#3a403c;--accent:#6eb6a8;--clean:#ef8585;--manual:#e7ac68;--observe:#91afc5;--ok:#77bd91}}
</style>
</head>
<body><main class="wrap">
<div class="eyebrow">RoboTwin clean · train split · 15 Hz</div>
<h1>机器人轨迹清洗审计</h1>
<p class="lead">覆盖 50 个任务、2,250 条 episode。检查 state/action 的速度、加速度、jerk、逐帧连续性、时间戳和静止比例；点级 Z-score 用于定位，最终决策使用同任务轨迹级 robust Z-score 与实际关节跳变量。</p>

<div class="summary">
  <div class="stat"><b>4</b><span>可直接清洗</span></div>
  <div class="stat"><b>13</b><span>必须人工复核</span></div>
  <div class="stat"><b>190</b><span>宽松观察池</span></div>
  <div class="stat"><b>0</b><span>静止 &gt;95% 删除项</span></div>
  <div class="stat"><b>0</b><span>结构/时间戳异常</span></div>
</div>

<div class="callout clean"><b>可清洗：</b>rotate_qrcode episode 10、27、30、31。四条均在右臂 J2（dim 9）出现 0.72–0.83 rad/帧跳变，轨迹级 robust Z 为 51.47–52.93。</div>
<div class="callout warn"><b>人工复核：</b>另外 13 条达到 Z ≥ 3.5 且关节步长 ≥ 0.25 rad/帧。它们可能是碰撞、遥操作跳变，也可能是任务本身的快速接触动作；必须结合三相机视频判断。</div>

<h2>按 state/action 与导数维度审计</h2>
<p class="muted small">同一 episode 可能同时在多个维度触发，因此本表各行不可相加。单位按 15 Hz 时间戳换算。</p>
<div class="table-wrap" style="max-height:none"><table id="matrix"><thead><tr><th>维度</th><th>单位</th><th class="num">可清洗</th><th class="num">人工复核</th><th class="num">宽松观察</th><th>解释</th></tr></thead><tbody></tbody></table></div>

<div class="grid">
  <section>
    <h2>17 条重点候选</h2>
    <div class="toolbar"><input id="search" placeholder="搜索任务或 episode"><select id="stage"><option value="focus">可清洗 + 人工复核</option><option value="clean">只看可清洗</option><option value="manual">只看人工复核</option><option value="observe">宽松观察池</option><option value="all">全部 207 条</option></select><select id="dimension"><option value="all">全部导数维度</option><option value="state_velocity">state 速度</option><option value="state_acceleration">state 加速度</option><option value="state_jerk">state jerk</option><option value="action_velocity">action 速度</option><option value="action_acceleration">action 加速度</option><option value="action_jerk">action jerk</option></select><span class="muted" id="count"></span></div>
  </section>
</div>
<div class="table-wrap"><table><thead><tr><th>处置</th><th>任务</th><th>Episode</th><th>主导维度</th><th class="num">robust Z</th><th class="num">最大步长<br>rad/帧</th><th>异常位置</th><th class="num">静止比例</th><th>证据</th></tr></thead><tbody id="candidates"></tbody></table></div>

<div class="grid">
  <section class="panel"><h2 style="margin-top:0">关节维度分布</h2><p class="muted small">仅统计 4 条可清洗 + 13 条人工复核候选的主异常关节。</p><div id="joints"></div></section>
  <section class="panel"><h2 style="margin-top:0">结构完整性</h2><ul><li>state/action 非有限值：0</li><li>frame_index 不连续：0</li><li>时间戳非单调：0</li><li>时间间隔异常：0</li><li>夹爪 dim 6/13 已从平滑性判定排除</li></ul><p class="small muted">原始数据没有 forbidden/invalid_frame 字段；action_is_pad 由训练 loader 在动作块越过 episode 末尾时动态生成，并已从 loss 中屏蔽。</p></section>
</div>

<h2>高静止比例观察项</h2>
<p class="muted">删除标准要求 state 和 action 的相邻帧不变比例都超过 95%。实际没有命中；下表展示超过 50% 的 53 条观察项，最高为 69.62%。</p>
<div class="table-wrap" style="max-height:420px"><table><thead><tr><th>任务</th><th>Episode</th><th class="num">state 静止</th><th class="num">action 静止</th><th class="num">帧数</th><th>结论</th></tr></thead><tbody id="staticRows"></tbody></table></div>

<div class="callout"><b>建议执行顺序：</b>先把 4 条高置信断跳加入 blacklist；再用三相机页面复核 13 条重点候选。190 条宽松观察项只用于排序抽查，不能批量删除。清洗应用前应生成新数据版本，保留当前 active 数据以兼容 global_step_12000 的 dataloader 断点。</div>
<div class="links"><a href="summary.json">summary.json</a><a href="episode_motion_metrics.csv">完整 CSV</a><a href="remove_candidates.txt">4 条清洗清单</a><a href="priority_review_candidates.txt">17 条优先清单</a><a href="../manual_review_20261013/index.html">三相机人工复核页</a></div>
<div class="footer">来源：wsq /home/wangsf/lingbot_competition/data/lerobot_clean_v3 · 生成时间 2026-10-14</div>
</main>
<script>
const data=__AUDIT_DATA__;
const stageLabel={clean:'可清洗',manual:'人工复核',observe:'宽松观察'};
const stageExplain={clean:'高置信跳变，可进入 blacklist',manual:'达到优先阈值，需看视频',observe:'低阈值统计异常，仅抽查'};
const metricExplain={velocity:'检测遥操作/编码器位置瞬时跳变',acceleration:'检测速度突变、丢帧或碰撞',jerk:'检测加速度尖峰和高频抖动'};
document.querySelector('#matrix tbody').innerHTML=data.matrix.map(x=>`<tr><td><b>${x.label}</b></td><td>${x.unit}</td><td class="num">${x.clean}</td><td class="num">${x.manual}</td><td class="num">${x.observe}</td><td class="muted">${metricExplain[x.derivative]}</td></tr>`).join('');
function pct(x){return (100*x).toFixed(1)+'%'}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function renderCandidates(){const q=document.querySelector('#search').value.trim().toLowerCase(),stage=document.querySelector('#stage').value,dim=document.querySelector('#dimension').value;const rows=data.candidates.filter(x=>{const stageOk=stage==='all'||stage==='focus'&&x.stage!=='observe'||x.stage===stage;const dimOk=dim==='all'||x.dimension_z[dim]>=3.5;const qOk=!q||x.task.toLowerCase().includes(q)||String(x.episode).includes(q);return stageOk&&dimOk&&qOk});document.querySelector('#count').textContent=`显示 ${rows.length} 条`;document.querySelector('#candidates').innerHTML=rows.map(x=>{const links=x.video?`<span class="links"><a href="../manual_review_20261013/clips/${esc(x.video)}">候选视频</a><a href="../manual_review_20261013/clips/${esc(x.normal_video)}">正常参考</a></span>`:'<span class="muted">未生成视频</span>';return `<tr><td><span class="pill ${x.stage}" title="${stageExplain[x.stage]}">${stageLabel[x.stage]}</span></td><td>${esc(x.task)}</td><td>${x.episode}</td><td><span class="metric">${esc(x.metric_label)}</span></td><td class="num">${x.z.toFixed(2)}</td><td class="num">${x.step.toFixed(6)}</td><td>frame ${x.frame}<br><span class="small muted">${esc(x.joint_label)}</span></td><td class="num">${pct(Math.min(x.state_static,x.action_static))}</td><td>${links}</td></tr>`}).join('')}
['search','stage','dimension'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',renderCandidates));renderCandidates();
const maxJoint=Math.max(...data.joints.map(x=>x.count));document.querySelector('#joints').innerHTML=data.joints.map(x=>`<div style="display:grid;grid-template-columns:170px 1fr 28px;gap:8px;align-items:center;margin:9px 0"><span>${esc(x.label)}</span><div class="bar"><i style="width:${100*x.count/maxJoint}%"></i></div><b class="num">${x.count}</b></div>`).join('');
document.querySelector('#staticRows').innerHTML=data.static.map(x=>`<tr><td>${esc(x.task)}</td><td>${x.episode}</td><td class="num">${pct(x.state_static)}</td><td class="num">${pct(x.action_static)}</td><td class="num">${x.length}</td><td><span class="pill observe">保留观察</span></td></tr>`).join('');
</script></body></html>'''


if __name__ == "__main__":
    main()
