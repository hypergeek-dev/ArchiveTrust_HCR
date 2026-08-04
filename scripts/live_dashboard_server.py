"""A small, dependency-free local web server that serves a genuinely live, auto-refreshing dashboard
for an Experiment 1 (single continuous epoch) training run.

Deliberately NOT a published claude.ai Artifact: those run in a sandboxed browser tab with a strict
content policy and no local-filesystem access, so they can only ever show a snapshot someone rebuilds
and redeploys by hand. This server runs on the same machine as the training run, reads its state files
directly on every request, and the page it serves polls that over plain `fetch()` -- genuinely live,
no redeploying.

Usage:
    python scripts/live_dashboard_server.py --run-dir training/experiment-1-continuous-epoch-<stamp>Z
    # then open http://localhost:8420 in a browser on this machine
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_0_DIR = REPO_ROOT / "training" / "full-corpus-20260802T044250Z"

RUN_DIR: Path
EXPERIMENT_0_DIR: Path


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _progress_rows() -> list[dict]:
    candidates = sorted(RUN_DIR.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))
    if not candidates:
        return []
    text = candidates[-1].read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _experiment_0_reference() -> dict:
    state = _load_json(EXPERIMENT_0_DIR / "run-state" / "run_state.json") or {}
    best_cer = (state.get("best_metrics") or {}).get("val_cer")
    eval_candidates = sorted((EXPERIMENT_0_DIR / "lap-evaluation").glob("lap*_best_val/lap_evaluation.json")) \
        if (EXPERIMENT_0_DIR / "lap-evaluation").exists() else []
    corpus_cer = None
    if eval_candidates:
        ev = _load_json(eval_candidates[-1])
        if ev:
            corpus_cer = (ev.get("overall") or {}).get("corpus_cer")
    return {"best_val_cer_in_training": best_cer, "held_out_corpus_cer": corpus_cer}


def build_state() -> dict:
    run_state = _load_json(RUN_DIR / "run-state" / "run_state.json") or {}
    provenance = _load_json(RUN_DIR / "provenance_check.json") or {}
    manifest = _load_json(RUN_DIR / "launch_manifest.json") or {}
    telemetry = _tail_jsonl(RUN_DIR / "run-state" / "telemetry" / "gpu_samples.jsonl", 300)
    telemetry_status = _load_json(RUN_DIR / "run-state" / "telemetry" / "status.json") or {}
    progress = _progress_rows()
    checkpoints = _load_json(RUN_DIR / "run-state" / "checkpoint_index.json") or {"entries": []}

    started_at = run_state.get("started_at")
    elapsed_seconds = None
    if started_at:
        try:
            started_struct = time.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ")
            # `started_at` is UTC. `time.mktime` assumes its input is LOCAL time and applies the
            # local UTC offset when converting to epoch seconds -- silently wrong here, and wrong by
            # exactly the host's timezone offset (this is what produced the bogus "running for an
            # hour" reading). `calendar.timegm` is the correct inverse of `time.gmtime`: it treats the
            # struct as UTC with no offset applied.
            elapsed_seconds = time.time() - calendar.timegm(started_struct)
        except ValueError:
            elapsed_seconds = None

    return {
        "snapshot_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(RUN_DIR),
        "run": run_state,
        "provenance": provenance,
        "manifest": {
            k: manifest.get(k) for k in (
                "run_id", "batch_size", "optimizer", "scheduler", "learning_rate",
                "train_line_count", "validation_line_count", "container_image_name",
                "container_image_digest", "wall_clock_policy", "random_seed",
            )
        },
        "telemetry_samples": telemetry,
        "telemetry_status": telemetry_status,
        "progress_rows": progress,
        "checkpoint_count": len(checkpoints.get("entries", [])),
        "checkpoints": checkpoints.get("entries", [])[-5:],
        "experiment_0_reference": _experiment_0_reference(),
        "elapsed_seconds": elapsed_seconds,
    }


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Experiment 1 — Live Monitor</title>
<style>
  :root {
    --paper: #f6f3ec; --panel: #fffdf8; --panel-recessed: #efeadd; --ink: #24211b; --ink-dim: #6b6558;
    --line: #ddd4bf; --line-strong: #c7bca0; --accent: #3a6b63; --running: #b5732a; --running-bg: #f6e6cf;
    --good: #3f7d53; --good-bg: #e2efe2; --critical: #b4432f; --critical-bg: #f6e0da;
    --shadow: 0 1px 2px rgba(36,33,27,.06), 0 6px 20px -10px rgba(36,33,27,.18);
    --font-display: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --paper:#14171a; --panel:#1b1f22; --panel-recessed:#14171a; --ink:#e9e4d6; --ink-dim:#9a9384;
      --line:#2c3033; --line-strong:#3a3f43; --accent:#63a89b; --running:#d99a52; --running-bg:#3a2c17;
      --good:#6fbf8b; --good-bg:#1c2e21; --critical:#e08170; --critical-bg:#33201c;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 30px -12px rgba(0,0,0,.6); }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--paper); color:var(--ink); font-family:var(--font-body); line-height:1.45; }
  .wrap { max-width:1080px; margin:0 auto; padding:2.5rem 1.5rem 4rem; }
  header.top { display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:.75rem 1.5rem;
    border-bottom:1px solid var(--line-strong); padding-bottom:1.25rem; margin-bottom:.4rem; }
  .title-block h1 { font-family:var(--font-display); font-weight:400; font-size:1.9rem; margin:0 0 .2rem; }
  .title-block .subtitle { font-size:.92rem; color:var(--ink-dim); }
  .subtitle code, footer code { font-family:var(--font-mono); font-size:.85em; }
  .status-pill { display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .9rem; border-radius:999px;
    font-family:var(--font-mono); font-size:.82rem; letter-spacing:.04em; text-transform:uppercase;
    background:var(--running-bg); color:var(--running); white-space:nowrap; }
  .status-pill.good { background:var(--good-bg); color:var(--good); }
  .status-pill.critical { background:var(--critical-bg); color:var(--critical); }
  .status-pill .dot { width:.5rem; height:.5rem; border-radius:50%; background:currentColor; animation:pulse 1.8s ease-in-out infinite; }
  @media (prefers-reduced-motion: reduce) { .status-pill .dot { animation:none; } }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .snapshot-note { font-size:.8rem; color:var(--ink-dim); margin:.9rem 0 2rem; font-family:var(--font-mono); }
  .snapshot-note strong { color:var(--ink); font-weight:600; }
  .headline-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
    background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden;
    margin-bottom:2rem; box-shadow:var(--shadow); }
  .headline-stat { background:var(--panel); padding:1rem 1.15rem; }
  .headline-stat .label { font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-dim); margin-bottom:.35rem; }
  .headline-stat .value { font-family:var(--font-mono); font-variant-numeric:tabular-nums; font-size:1.35rem; font-weight:600; }
  .headline-stat .value small { font-size:.6em; font-weight:400; color:var(--ink-dim); }
  h2.section { font-family:var(--font-display); font-weight:400; font-size:1.15rem; margin:2.4rem 0 .9rem;
    display:flex; align-items:baseline; gap:.6rem; }
  h2.section .eyebrow { font-family:var(--font-mono); font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
    color:var(--ink-dim); font-weight:600; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:1rem; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1.15rem 1.25rem 1.3rem; box-shadow:var(--shadow); }
  .card h3 { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-dim); margin:0 0 .9rem;
    font-weight:600; display:flex; justify-content:space-between; align-items:center; }
  .card h3 .unit { font-weight:400; text-transform:none; letter-spacing:0; font-family:var(--font-mono); }
  .instrument-value { font-family:var(--font-mono); font-variant-numeric:tabular-nums; font-size:2rem; font-weight:600; line-height:1; margin-bottom:.15rem; }
  .instrument-sub { font-size:.78rem; color:var(--ink-dim); margin-bottom:.7rem; }
  canvas.spark { display:block; width:100%; height:46px; }
  dl.factsheet { margin:0; display:grid; gap:.55rem; }
  dl.factsheet div { display:flex; justify-content:space-between; gap:1rem; font-size:.86rem; }
  dl.factsheet dt { color:var(--ink-dim); flex-shrink:0; }
  dl.factsheet dd { margin:0; font-family:var(--font-mono); font-size:.82rem; text-align:right; word-break:break-word; }
  .check-row { display:flex; align-items:flex-start; gap:.55rem; font-size:.86rem; margin-bottom:.6rem; }
  .check-row:last-child { margin-bottom:0; }
  .check-mark { flex-shrink:0; width:1.1rem; height:1.1rem; border-radius:50%; background:var(--good-bg); color:var(--good);
    display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:700; margin-top:.1rem; }
  .check-mark.critical { background:var(--critical-bg); color:var(--critical); }
  .check-row .text strong { display:block; }
  .check-row .text span { color:var(--ink-dim); font-family:var(--font-mono); font-size:.78rem; }
  .empty-state { font-size:.86rem; color:var(--ink-dim); background:var(--panel-recessed); border:1px dashed var(--line-strong);
    border-radius:8px; padding:.9rem 1rem; }
  .empty-state strong { color:var(--ink); }
  .compare-bar-row { display:grid; gap:.7rem; }
  .compare-bar { display:grid; grid-template-columns:7rem 1fr 3.6rem; align-items:center; gap:.6rem; font-size:.82rem; }
  .compare-bar .track { height:10px; background:var(--panel-recessed); border-radius:5px; overflow:hidden; border:1px solid var(--line); }
  .compare-bar .fill { height:100%; border-radius:5px 0 0 5px; }
  .compare-bar .fill.exp0 { background:var(--ink-dim); }
  .compare-bar .fill.exp1 { background:var(--accent); }
  .compare-bar .num { font-family:var(--font-mono); text-align:right; font-variant-numeric:tabular-nums; }
  footer { margin-top:3rem; padding-top:1.25rem; border-top:1px solid var(--line); font-size:.78rem; color:var(--ink-dim); }
  #conn-warning { display:none; background:var(--critical-bg); color:var(--critical); font-family:var(--font-mono);
    font-size:.8rem; padding:.6rem 1rem; border-radius:8px; margin-bottom:1.5rem; }
</style>
</head>
<body>
<div class="wrap">
  <div id="conn-warning">Lost contact with the local server &mdash; is live_dashboard_server.py still running?</div>
  <header class="top">
    <div class="title-block">
      <h1>Experiment 1 &mdash; Live Monitor</h1>
      <div class="subtitle">Swedish HTR fine-tune, single uninterrupted pass &middot; <code id="run-id">&hellip;</code></div>
    </div>
    <span class="status-pill" id="status-pill"><span class="dot"></span><span id="status-text">&hellip;</span></span>
  </header>
  <p class="snapshot-note">Live &mdash; polling this machine every 4s. Last update <strong id="snapshot-time">&hellip;</strong></p>

  <div class="headline-row">
    <div class="headline-stat"><div class="label">Elapsed</div><div class="value" id="h-elapsed">&hellip;</div></div>
    <div class="headline-stat"><div class="label">Optimizer step</div><div class="value" id="h-step">&hellip;</div></div>
    <div class="headline-stat"><div class="label">Corpus</div><div class="value" id="h-corpus">&hellip;</div></div>
    <div class="headline-stat"><div class="label">Batch size</div><div class="value" id="h-batch">&hellip;</div></div>
    <div class="headline-stat"><div class="label">Checkpoints</div><div class="value" id="h-checkpoints">&hellip;</div></div>
  </div>

  <h2 class="section">Training progress <span class="eyebrow">live, from Keras's own running metrics</span></h2>
  <div class="grid">
    <div class="card"><h3>Train loss</h3><div class="instrument-value" id="v-loss">&mdash;</div>
      <div class="instrument-sub" id="s-loss">waiting for first sample</div><canvas class="spark" id="spark-loss"></canvas></div>
    <div class="card"><h3>Train CER <span class="unit">running</span></h3><div class="instrument-value" id="v-cer">&mdash;</div>
      <div class="instrument-sub" id="s-cer">waiting for first sample</div><canvas class="spark" id="spark-cer"></canvas></div>
    <div class="card"><h3>Train WER <span class="unit">running</span></h3><div class="instrument-value" id="v-wer">&mdash;</div>
      <div class="instrument-sub" id="s-wer">waiting for first sample</div><canvas class="spark" id="spark-wer"></canvas></div>
  </div>

  <h2 class="section">Optimizer &amp; LR continuity <span class="eyebrow">the point of this run</span></h2>
  <div class="grid">
    <div class="card">
      <h3>optimizer.iterations</h3>
      <div class="instrument-value" id="v-iter">&mdash;</div>
      <div class="instrument-sub" id="s-iter">no rows yet</div>
    </div>
    <div class="card">
      <h3>learning rate</h3>
      <div class="instrument-value" id="v-lr">&mdash;</div>
      <div class="instrument-sub" id="s-lr">no rows yet</div>
    </div>
  </div>

  <h2 class="section">Instruments <span class="eyebrow">GPU &amp; system</span></h2>
  <div class="grid">
    <div class="card"><h3>GPU utilization <span class="unit">%</span></h3><div class="instrument-value" id="v-util">&mdash;</div>
      <div class="instrument-sub" id="s-util">&nbsp;</div><canvas class="spark" id="spark-util"></canvas></div>
    <div class="card"><h3>VRAM used <span class="unit">MB / 8192</span></h3><div class="instrument-value" id="v-mem">&mdash;</div>
      <div class="instrument-sub" id="s-mem">&nbsp;</div><canvas class="spark" id="spark-mem"></canvas></div>
    <div class="card"><h3>GPU temperature <span class="unit">&deg;C</span></h3><div class="instrument-value" id="v-temp">&mdash;</div>
      <div class="instrument-sub" id="s-temp">&nbsp;</div><canvas class="spark" id="spark-temp"></canvas></div>
  </div>

  <h2 class="section">Provenance <span class="eyebrow">verified before launch</span></h2>
  <div class="grid" id="provenance-grid"></div>

  <h2 class="section">Reference point <span class="eyebrow">Experiment 0, completed</span></h2>
  <div class="grid">
    <div class="card" style="grid-column:1/-1;">
      <h3>Held-out validation CER</h3>
      <div class="compare-bar-row" id="compare-bars"></div>
      <p class="instrument-sub" style="margin-top:.9rem;margin-bottom:0;">Experiment 0 (57 shard-container invocations, optimizer reset every shard). Experiment 1 validates once, at the end of the single continuous epoch. Same corpus, same validation set, same optimizer/LR hyperparameters.</p>
    </div>
  </div>

  <footer><span id="footer-info">&hellip;</span></footer>
</div>

<script>
function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return isNaN(v) ? "—" : v.toFixed(digits === undefined ? 4 : digits);
  return v;
}
function fmtInt(v) { return (v === null || v === undefined || v === "") ? "—" : Number(v).toLocaleString(); }
function fmtElapsed(s) {
  if (s === null || s === undefined) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + "h " + m + "m";
}
function drawSpark(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data || data.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width, 260), h = 46;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1,0,0,1,0,0); ctx.scale(dpr, dpr);
  const min = Math.min(...data), max = Math.max(...data), range = (max - min) || 1;
  const pad = 4;
  const stepX = (w - pad * 2) / (data.length - 1 || 1);
  const y = v => pad + (h - pad * 2) * (1 - (v - min) / range);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
  ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, y(data[0]));
  data.forEach((v, i) => ctx.lineTo(pad + i * stepX, y(v)));
  ctx.lineTo(pad + (data.length - 1) * stepX, h - pad); ctx.lineTo(pad, h - pad); ctx.closePath();
  ctx.fillStyle = color + "22"; ctx.fill();
  ctx.beginPath(); ctx.moveTo(pad, y(data[0]));
  data.forEach((v, i) => ctx.lineTo(pad + i * stepX, y(v)));
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.lineJoin = "round"; ctx.stroke();
  const lastX = pad + (data.length - 1) * stepX, lastY = y(data[data.length - 1]);
  ctx.beginPath(); ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

let lastGoodUpdate = Date.now();

// Elapsed time ticks locally every second from the last known server value, rather than only
// updating once per fetch -- a background/inactive browser tab gets its `setInterval` throttled
// (sometimes to once a minute or less) by the browser itself to save power, which otherwise makes
// the page look frozen ("stalled") even though the real run behind it is progressing fine.
let baseElapsedSeconds = null;
let baseElapsedAtClientMs = null;

function renderElapsedLocally() {
  if (baseElapsedSeconds === null) return;
  const projected = baseElapsedSeconds + (Date.now() - baseElapsedAtClientMs) / 1000;
  document.getElementById("h-elapsed").textContent = fmtElapsed(projected);
}
setInterval(renderElapsedLocally, 1000);

async function tick() {
  let state;
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    state = await res.json();
    lastGoodUpdate = Date.now();
    document.getElementById("conn-warning").style.display = "none";
  } catch (e) {
    if (Date.now() - lastGoodUpdate > 15000) document.getElementById("conn-warning").style.display = "block";
    return;
  }

  try {
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const running = getComputedStyle(document.documentElement).getPropertyValue("--running").trim();

  document.getElementById("snapshot-time").textContent = state.snapshot_at;
  document.getElementById("run-id").textContent = (state.manifest.run_id || "").slice(0, 28) + "...";

  const status = (state.run.status || "unknown").toLowerCase();
  const pill = document.getElementById("status-pill");
  pill.className = "status-pill" + (status === "completed" ? " good" : (status === "failed" ? " critical" : ""));
  document.getElementById("status-text").textContent = status;

  if (state.elapsed_seconds !== null && state.elapsed_seconds !== undefined) {
    baseElapsedSeconds = state.elapsed_seconds;
    baseElapsedAtClientMs = Date.now();
    renderElapsedLocally();
  }
  document.getElementById("h-corpus").innerHTML = fmtInt(state.manifest.train_line_count) + " <small>lines</small>";
  document.getElementById("h-batch").textContent = state.manifest.batch_size ?? "—";
  document.getElementById("h-checkpoints").textContent = state.checkpoint_count;

  const rows = state.progress_rows || [];
  // `train_begin` is always logged with blank loss/CER/WER -- Keras hasn't run a batch yet at that
  // point, so it never carries real metric data. Only rows logged from `on_train_batch_end` (every
  // 500 optimizer steps) have real numbers; treat those as "samples" for display purposes.
  const dataRows = rows.filter(r => r.loss !== "" && r.loss !== undefined && r.loss !== null);
  const last = dataRows.length ? dataRows[dataRows.length - 1] : null;
  const latestRow = rows.length ? rows[rows.length - 1] : null;
  document.getElementById("h-step").textContent = latestRow ? fmtInt(latestRow.optimizer_iterations) : "—";

  if (last) {
    document.getElementById("v-loss").textContent = fmt(parseFloat(last.loss));
    document.getElementById("v-cer").textContent = fmt(parseFloat(last.CER_metric));
    document.getElementById("v-wer").textContent = fmt(parseFloat(last.WER_metric));
    document.getElementById("v-iter").textContent = fmtInt(last.optimizer_iterations);
    document.getElementById("v-lr").textContent = fmt(parseFloat(last.learning_rate), 7);
    document.getElementById("s-iter").textContent = "as of tag " + last.tag;
    document.getElementById("s-lr").textContent = "as of tag " + last.tag;
    const n = dataRows.length;
    document.getElementById("s-loss").textContent = n + " sample" + (n === 1 ? "" : "s") + " so far";
    document.getElementById("s-cer").textContent = n + " sample" + (n === 1 ? "" : "s") + " so far";
    document.getElementById("s-wer").textContent = n + " sample" + (n === 1 ? "" : "s") + " so far";

    const lossData = dataRows.map(r => parseFloat(r.loss)).filter(v => !isNaN(v));
    const cerData = dataRows.map(r => parseFloat(r.CER_metric)).filter(v => !isNaN(v));
    const werData = dataRows.map(r => parseFloat(r.WER_metric)).filter(v => !isNaN(v));
    drawSpark("spark-loss", lossData, running);
    drawSpark("spark-cer", cerData, accent);
    drawSpark("spark-wer", werData, accent);
  } else {
    const waitingMsg = "no batches logged yet — first row appears at optimizer step 500";
    document.getElementById("s-loss").textContent = waitingMsg;
    document.getElementById("s-cer").textContent = waitingMsg;
    document.getElementById("s-wer").textContent = waitingMsg;
  }

  const tel = state.telemetry_samples || [];
  if (tel.length) {
    const lastT = tel[tel.length - 1];
    document.getElementById("v-util").textContent = fmt(lastT.gpu.utilization_pct, 0);
    document.getElementById("v-mem").textContent = fmtInt(lastT.gpu.memory_used_mb);
    document.getElementById("v-temp").textContent = fmt(lastT.gpu.temperature_c, 0);
    const util = tel.map(t => t.gpu.utilization_pct).filter(v => v !== null && v !== undefined);
    const mem = tel.map(t => t.gpu.memory_used_mb).filter(v => v !== null && v !== undefined);
    const temp = tel.map(t => t.gpu.temperature_c).filter(v => v !== null && v !== undefined);
    document.getElementById("s-util").textContent = tel.length + " samples · range " + Math.min(...util) + "–" + Math.max(...util);
    document.getElementById("s-mem").textContent = "peak " + fmtInt(Math.max(...mem)) + " MB";
    document.getElementById("s-temp").textContent = "peak " + Math.max(...temp) + "°C";
    drawSpark("spark-util", util, running);
    drawSpark("spark-mem", mem, accent);
    drawSpark("spark-temp", temp, accent);
  }

  const prov = state.provenance || {};
  const provGrid = document.getElementById("provenance-grid");
  provGrid.innerHTML = `
    <div class="card">
      <h3>Parent checkpoint</h3>
      <div class="check-row"><div class="check-mark${prov.hash_matches_pin ? "" : " critical"}">${prov.hash_matches_pin ? "&check;" : "!"}</div>
        <div class="text"><strong>${prov.hash_matches_pin ? "Pristine, hash matches pin" : "Hash mismatch"}</strong><span>${(prov.checkpoint_sha256 || "").slice(0,16)}&hellip;</span></div></div>
      <div class="check-row"><div class="check-mark${prov.not_derived_from_prior_run ? "" : " critical"}">${prov.not_derived_from_prior_run ? "&check;" : "!"}</div>
        <div class="text"><strong>${prov.not_derived_from_prior_run ? "Not derived from a prior run" : "DERIVED FROM A PRIOR RUN"}</strong><span>checked against Experiment 0 &amp; pilot</span></div></div>
    </div>
    <div class="card">
      <h3>Configuration</h3>
      <dl class="factsheet">
        <div><dt>Optimizer</dt><dd>${state.manifest.optimizer || "—"}</dd></div>
        <div><dt>Learning rate</dt><dd>${state.manifest.learning_rate ?? "—"}</dd></div>
        <div><dt>Validation lines</dt><dd>${fmtInt(state.manifest.validation_line_count)}</dd></div>
        <div><dt>Seed</dt><dd>${state.manifest.random_seed ?? "—"}</dd></div>
      </dl>
    </div>`;

  const ref = state.experiment_0_reference || {};
  const exp1Cer = (state.run.best_metrics || {}).val_cer;
  const maxCer = Math.max(ref.held_out_corpus_cer || 0.3, exp1Cer || 0, 0.3);
  document.getElementById("compare-bars").innerHTML = `
    <div class="compare-bar"><span>Experiment&nbsp;0</span><span class="track"><span class="fill exp0" style="width:${((ref.held_out_corpus_cer||0)/maxCer*100)}%"></span></span><span class="num">${fmt(ref.held_out_corpus_cer)}</span></div>
    <div class="compare-bar"><span>Experiment&nbsp;1</span><span class="track"><span class="fill exp1" style="width:${((exp1Cer||0)/maxCer*100)}%"></span></span><span class="num">${exp1Cer ? fmt(exp1Cer) : "not yet validated"}</span></div>`;

  document.getElementById("footer-info").innerHTML = `Run directory <code>${state.run_dir}</code> &middot; config hash <code>${(state.run.configuration_hash||"").slice(0,16)}&hellip;</code>`;
  } catch (e) {
    console.error("dashboard render error:", e);
  }
}

tick();
setInterval(tick, 4000);
// Browsers throttle/pause setInterval in a backgrounded tab; force an immediate re-poll the moment
// this tab becomes visible again so the page catches up instantly instead of waiting out whatever
// throttled interval the browser was using while it was hidden.
document.addEventListener("visibilitychange", () => { if (!document.hidden) tick(); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
        pass  # quiet; state polling every 4s would otherwise spam the console

    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                payload = json.dumps(build_state()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode("utf-8"))


def main() -> int:
    global RUN_DIR, EXPERIMENT_0_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-0-dir", default=str(DEFAULT_EXPERIMENT_0_DIR))
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    RUN_DIR = Path(args.run_dir).resolve()
    EXPERIMENT_0_DIR = Path(args.experiment_0_dir).resolve()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving live dashboard for {RUN_DIR} at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
