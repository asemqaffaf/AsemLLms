#!/usr/bin/env python3
"""Builds validation-loss.html (interactive) and validation-loss.svg (static) from the training logs.
Standard library only. Run from the repository root:  python3 -I docs/results/stage1/make_chart.py
Data: data/runs/final/{1m,3m,10m}/log.jsonl, the "eval" events (loss on 128 fixed validation windows, every 500 steps).
Palette: slots 1-3 of the reference categorical palette (blue, orange, aqua), validated with validate_palette.js --pairs all in
both modes (light worst CVD dE 9.2, dark 9.4). Aqua is 2.74:1 on the light surface, so direct labels and a table view are included."""
import json, html, sys

RUNS = [("1m", "1.16M parameters", "#2a78d6", "#3987e5"), ("3m", "2.90M parameters", "#eb6834", "#d95926"), ("10m", "10.52M parameters", "#1baf7a", "#199e70")]
TOK_PER_STEP = 32 * 128
data = {}
for key, *_ in RUNS:
    pts = [(d["step"], d["val_loss"]) for d in (json.loads(l) for l in open(f"data/runs/final/{key}/log.jsonl")) if d["event"] == "eval"]
    data[key] = pts
steps_all = sorted({s for pts in data.values() for s, _ in pts})
plot_steps = [s for s in steps_all if s >= 1000]                     # step 0 (9.2) and 500 would flatten the scale; they are in the table
W, H, ML, MR, MT, MB = 920, 460, 64, 150, 28, 54
PW, PH = W - ML - MR, H - MT - MB
YMIN, YMAX = 4.4, 5.8
X = lambda s: ML + (s - plot_steps[0]) / (plot_steps[-1] - plot_steps[0]) * PW
Y = lambda v: MT + (YMAX - v) / (YMAX - YMIN) * PH

def svg(colors, ink, grid, surface, interactive):
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d" width="100%" style="max-width:{W}px" font-family="system-ui, -apple-system, Helvetica, Arial, sans-serif">')
    o.append('<title id="t">Sampled validation loss versus training step for three AsemLLM sizes</title>')
    o.append('<desc id="d">Loss falls for all three models. The 10.52M model passes the 2.90M model near step 4000 and ends lowest at about 4.47; the 1.16M model ends highest at about 4.82.</desc>')
    o.append(f'<rect width="{W}" height="{H}" fill="{surface}"/>')
    for v in (4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8):
        o.append(f'<line x1="{ML}" x2="{ML+PW}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{grid}" stroke-width="1" stroke-dasharray="2 4"/>')
        o.append(f'<text x="{ML-10}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="12" fill="{ink[1]}">{v:.1f}</text>')
    for s in range(1000, 6001, 1000):                                   # 7,000 is skipped: it would collide with the 7,300 label
        o.append(f'<text x="{X(s):.1f}" y="{MT+PH+22}" text-anchor="middle" font-size="12" fill="{ink[1]}">{s:,}</text>')
    o.append(f'<text x="{X(7300):.1f}" y="{MT+PH+22}" text-anchor="middle" font-size="12" fill="{ink[1]}">7,300</text>')
    o.append(f'<text x="{ML+PW/2}" y="{H-8}" text-anchor="middle" font-size="12" fill="{ink[1]}">training step (4,096 tokens per step; 7,300 steps = 29.9M tokens)</text>')
    o.append(f'<text transform="translate(16 {MT+PH/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="{ink[1]}">validation loss (lower is better)</text>')
    o.append(f'<line x1="{ML}" x2="{ML+PW}" y1="{MT+PH}" y2="{MT+PH}" stroke="{ink[2]}" stroke-width="1"/>')
    label_y = {}
    for (key, name, *_), col in zip(RUNS, colors):
        pts = [(s, v) for s, v in data[key] if s >= 1000]
        path = " ".join(("M" if i == 0 else "L") + f"{X(s):.1f} {Y(v):.1f}" for i, (s, v) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in pts:
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="{surface}" stroke-width="2"/>')
        s, v = pts[-1]; label_y[key] = Y(v)
    for (key, name, *_), col in zip(RUNS, colors):                      # direct labels: swatch carries identity, text stays in ink
        v = data[key][-1][1]; y = label_y[key]
        o.append(f'<circle cx="{ML+PW+16}" cy="{y:.1f}" r="4" fill="{col}"/>')
        o.append(f'<text x="{ML+PW+26}" y="{y+4:.1f}" font-size="12" fill="{ink[0]}">{name.split()[0]} <tspan fill="{ink[1]}">{v:.3f}</tspan></text>')
    if interactive:
        o.append(f'<line id="cross" x1="0" x2="0" y1="{MT}" y2="{MT+PH}" stroke="{ink[2]}" stroke-width="1" visibility="hidden"/>')
        o.append(f'<rect id="hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="transparent"/>')
    o.append('</svg>')
    return "\n".join(o)

LIGHT = dict(colors=[c[2] for c in RUNS], ink=("#0b0b0b", "#52514e", "#9a998f"), grid="#d9d8d2", surface="#fcfcfb")
DARK = dict(colors=[c[3] for c in RUNS], ink=("#f0efec", "#c3c2b7", "#7a7970"), grid="#33332f", surface="#1a1a19")
open("docs/results/stage1/validation-loss.svg", "w").write(svg(interactive=False, **LIGHT) + "\n")

rows = "".join(f"<tr><td>{s:,}</td><td>{s*TOK_PER_STEP/1e6:.1f}M</td>" + "".join(f"<td>{dict(data[k]).get(s, float('nan')):.3f}</td>" for k, *_ in RUNS) + "</tr>" for s in steps_all)
js_data = json.dumps({"steps": plot_steps, "series": [{"name": n, "values": [dict(data[k]).get(s) for s in plot_steps]} for k, n, *_ in RUNS]})
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AsemLLM: validation loss by model size</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.viz-root {{ color-scheme: light; --surface:{LIGHT["surface"]}; --ink:{LIGHT["ink"][0]}; --ink2:{LIGHT["ink"][1]}; --rule:{LIGHT["grid"]};
  --s1:{RUNS[0][2]}; --s2:{RUNS[1][2]}; --s3:{RUNS[2][2]}; font:15px/1.5 system-ui,sans-serif; background:var(--surface); color:var(--ink); max-width:960px; margin:24px auto; padding:16px; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark; --surface:{DARK["surface"]}; --ink:{DARK["ink"][0]}; --ink2:{DARK["ink"][1]}; --rule:{DARK["grid"]}; --s1:{RUNS[0][3]}; --s2:{RUNS[1][3]}; --s3:{RUNS[2][3]}; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme: dark; --surface:{DARK["surface"]}; --ink:{DARK["ink"][0]}; --ink2:{DARK["ink"][1]}; --rule:{DARK["grid"]}; --s1:{RUNS[0][3]}; --s2:{RUNS[1][3]}; --s3:{RUNS[2][3]}; }}
body {{ background:#8882; margin:0 }} h1 {{ font-size:20px; margin:0 0 4px }} .sub {{ color:var(--ink2); margin:0 0 12px }}
.legend {{ display:flex; gap:18px; margin:8px 0; color:var(--ink2); font-size:13px }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px }}
.wrap {{ position:relative }} #tip {{ position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:8px 10px; font-size:13px; display:none; box-shadow:0 2px 8px #0003 }}
#tip b {{ display:block; margin-bottom:4px }} #tip span.sw {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px }}
button {{ font:inherit; background:none; color:var(--ink); border:1px solid var(--rule); border-radius:6px; padding:4px 10px; cursor:pointer }} table {{ border-collapse:collapse; margin-top:12px; font-size:13px; display:none }} td,th {{ padding:4px 12px; border-bottom:1px solid var(--rule); text-align:right }}
.note {{ color:var(--ink2); font-size:13px; margin-top:10px }}
</style></head><body><div class="viz-root">
<h1>AsemLLM: validation loss by model size</h1>
<p class="sub">Same corpus, tokenizer, seed and token budget (29.9M tokens). Loss on 128 fixed validation windows, every 500 steps. Steps 0 and 500 are in the table.</p>
<div class="legend"><span><i style="background:var(--s1)"></i>1.16M parameters (lr 0.01)</span><span><i style="background:var(--s2)"></i>2.90M (lr 0.003)</span><span><i style="background:var(--s3)"></i>10.52M (lr 0.003)</span>
<button id="tb" aria-expanded="false">Table view</button></div>
<div class="wrap" id="wrap">{svg(interactive=True, colors=["var(--s1)","var(--s2)","var(--s3)"], ink=("var(--ink)","var(--ink2)","var(--ink2)"), grid="var(--rule)", surface="var(--surface)")}<div id="tip"></div></div>
<table id="tbl"><thead><tr><th>step</th><th>tokens</th><th>1.16M</th><th>2.90M</th><th>10.52M</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Sampled validation loss, not the full-split number. Final full-validation loss: 4.820 (1.16M), 4.554 (2.90M), 4.469 (10.52M). The 10.52M model first drops below the 2.90M model at step 4,000.</p>
</div>
<script>
const D={js_data}, ML={ML}, PW={PW}, W={W}, X0=D.steps[0], X1=D.steps[D.steps.length-1];
const svg=document.querySelector("svg"), cross=document.getElementById("cross"), hit=document.getElementById("hit"), tip=document.getElementById("tip"), wrap=document.getElementById("wrap");
const cols=["var(--s1)","var(--s2)","var(--s3)"];
hit.addEventListener("pointermove", e=>{{
  const r=svg.getBoundingClientRect(), k=W/r.width, x=(e.clientX-r.left)*k, s=X0+(x-ML)/PW*(X1-X0);
  let i=0,b=1e9; D.steps.forEach((t,j)=>{{const d=Math.abs(t-s); if(d<b){{b=d;i=j}}}});
  const cx=ML+(D.steps[i]-X0)/(X1-X0)*PW; cross.setAttribute("x1",cx); cross.setAttribute("x2",cx); cross.setAttribute("visibility","visible");
  tip.innerHTML="<b>step "+D.steps[i].toLocaleString()+" ("+(D.steps[i]*4096/1e6).toFixed(1)+"M tokens)</b>"+D.series.map((s,j)=>"<div><span class=sw style='background:"+cols[j]+"'></span>"+s.name+": "+s.values[i].toFixed(3)+"</div>").join("");
  tip.style.display="block"; const left=(cx/k)+12; tip.style.left=Math.min(left, r.width-210)+"px"; tip.style.top="24px";
}});
hit.addEventListener("pointerleave",()=>{{cross.setAttribute("visibility","hidden"); tip.style.display="none";}});
document.getElementById("tb").onclick=function(){{const t=document.getElementById("tbl"), on=t.style.display!=="table"; t.style.display=on?"table":"none"; this.setAttribute("aria-expanded",on); this.textContent=on?"Hide table":"Table view";}};
</script></body></html>'''
open("docs/results/stage1/validation-loss.html", "w").write(page)
print("wrote validation-loss.html and validation-loss.svg;", len(plot_steps), "plotted steps, last step", plot_steps[-1])
