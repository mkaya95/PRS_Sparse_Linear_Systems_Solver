// The page: three tabs over the server's JSON API. No framework; the DOM is the state's mirror.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const STORE = "p08-ui-settings";
const SINGLE_STORE = "p08-ui-single";
const ALL_STEPS = ["build", "test", "download", "speed", "accuracy", "scipy"];
const STATE_TEXT = { waiting: "waiting", running: "running", done: "done", failed: "failed", "gate-failed": "gate failed", stopped: "stopped" };

const ui = {
  settings: null, preview: [], selected: "build",
  single: { matrix: null, test: "speed" }, singleInfo: null,
  job: null, offset: 0, lines: [], polling: false,
  machines: [], machine: 0, pair: 0, kind: "real",
  precond: "jacobi", lastDemo: null, demoShown: false, demoNow: null,
  matrices: [], drawTool: false, matrixSel: null,
};

// ---------- server ----------
async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json", "X-P08": "1" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  let data = {};
  try { data = await response.json(); } catch (_) { /* an empty or broken reply keeps data empty */ }
  if (!response.ok) {
    const error = new Error(data.error || `server answered ${response.status}`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function clock(seconds) {
  if (seconds == null) return "";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function duration(seconds) {
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}

// ---------- tabs ----------
function showTab() {
  const wanted = (location.hash || "#results").slice(1);
  const tab = ["run", "results", "matrices", "demo"].includes(wanted) ? wanted : "results";
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${tab}`; });
}

// ---------- settings ----------
function readSettings() {
  const form = $("#settings");
  return Object.fromEntries(["machine", "build", "data", "compiler"].map((k) => [k, form.elements[k].value.trim()]));
}

function saveSettings() {
  ui.settings = readSettings();
  try { localStorage.setItem(STORE, JSON.stringify(ui.settings)); } catch (_) { /* private mode: keep it in memory */ }
  $("#machine-tag").textContent = ui.settings.machine ? `results name: ${ui.settings.machine}` : "";
}

let previewTimer = null;
function settingsChanged(event) {
  const field = event && event.target ? event.target.name : "";
  saveSettings();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    refreshPreview();
    if (field === "data") loadDemoChoices();
    if (field === "data" || field === "build") loadMatrices();
  }, 300);
}

async function refreshPreview() {
  try {
    const data = await api("/api/preview", { settings: ui.settings, single: ui.single });
    ui.preview = data.steps;
    ui.singleInfo = data.single;
    $("#settings-error").textContent = "";
    if (fixSingle()) return refreshPreview();
  } catch (error) {
    $("#settings-error").textContent = error.message;
    ui.preview = [];
    ui.singleInfo = null;
  }
  renderSingleChoices();
  renderSteps();
}

// ---------- run tab ----------
function jobStep(id) {
  if (!ui.job || ui.job.kind !== "run") return null;
  return ui.job.steps.find((s) => s.id === id) || null;
}

function renderSteps() {
  const busy = ui.job && ui.job.state === "running";
  const list = $("#steps");
  list.innerHTML = ui.preview.map((step) => {
    const js = jobStep(step.id);
    const state = js ? js.state : (step.existing.length ? "measured" : "");
    const text = js ? STATE_TEXT[js.state] : (step.existing.length ? "has a result" : "");
    return `<li class="step ${state}${step.id === ui.selected ? " selected" : ""}" data-id="${step.id}" tabindex="0">
      <span class="title">${esc(step.title)}</span><span class="lead"></span>
      <span class="state">${esc(text)}</span>
      <span class="time">${js && js.elapsed != null ? clock(js.elapsed) : `~${step.minutes} min`}</span>
      <button class="go-one" data-run="${step.id}" ${busy ? "disabled" : ""} aria-label="Run ${esc(step.title)} alone">Run</button>
      <span class="about">${esc(step.about)}</span>
    </li>`;
  }).join("");
  if (ui.selected === "single") {
    $("#cmd-label").textContent = "One matrix · the exact commands";
    $("#cmd").textContent = ui.singleInfo ? ui.singleInfo.commands.map((c) => `$ ${c}`).join("\n") : "";
  } else {
    const chosen = ui.preview.find((s) => s.id === ui.selected);
    const index = ui.preview.indexOf(chosen) + 1;
    $("#cmd-label").textContent = chosen ? `Step ${index} · the exact command${chosen.commands.length > 1 ? "s" : ""}` : "The exact command";
    $("#cmd").textContent = chosen ? chosen.commands.map((c) => `$ ${c}`).join("\n") : "";
  }
  $("#run-all").disabled = busy || !ui.preview.length;
  $("#stop").disabled = !busy;
  $("#solve-live").disabled = busy;
  renderSingleState(busy);
  renderDrawStatus();
}

// ---------- one matrix ----------
function loadSingle() {
  try {
    const saved = JSON.parse(localStorage.getItem(SINGLE_STORE) || "{}");
    if (typeof saved.matrix === "string") ui.single.matrix = saved.matrix;
    if (typeof saved.test === "string") ui.single.test = saved.test;
  } catch (_) { /* nothing saved, or storage is blocked */ }
}

function saveSingle() {
  try { localStorage.setItem(SINGLE_STORE, JSON.stringify(ui.single)); } catch (_) { /* keep it in memory */ }
}

// Real matrices first, then generated, as the list shows them.
function singleOrder(choices) {
  return [...choices.filter((c) => !c.generated), ...choices.filter((c) => c.generated)];
}

// Moves the choice to a downloaded matrix that offers the chosen test. Returns true if it changed.
function fixSingle() {
  const choices = ui.singleInfo ? ui.singleInfo.choices : [];
  const was = `${ui.single.matrix}|${ui.single.test}`;
  const pick = choices.find((c) => c.name === ui.single.matrix && c.available) || singleOrder(choices).find((c) => c.available);
  if (!pick) return false;
  ui.single.matrix = pick.name;
  if (!pick.tests.includes(ui.single.test)) ui.single.test = "speed";
  saveSingle();
  return `${ui.single.matrix}|${ui.single.test}` !== was;
}

function renderSingleChoices() {
  const info = ui.singleInfo;
  const select = $("#single-matrix");
  if (!info) {
    select.innerHTML = "";
    $("#single-note").textContent = "";
    return;
  }
  const option = (c) => `<option value="${esc(c.name)}"${c.available ? "" : " disabled"}>${esc(c.name)}` +
    `${c.available ? "" : " · not downloaded"}</option>`;
  const ordered = singleOrder(info.choices);
  select.innerHTML = `<optgroup label="Real matrices">${ordered.filter((c) => !c.generated).map(option).join("")}</optgroup>` +
    `<optgroup label="Generated">${ordered.filter((c) => c.generated).map(option).join("")}</optgroup>`;
  select.value = ui.single.matrix || "";
  const chosen = info.choices.find((c) => c.name === ui.single.matrix);
  $$("#single-test button").forEach((b) => {
    b.classList.toggle("on", b.dataset.test === ui.single.test);
    b.disabled = !chosen || !chosen.tests.includes(b.dataset.test);
  });
  const noAccuracy = chosen && !chosen.tests.includes("accuracy") ? " Accuracy runs only on the six matrices the Accuracy step uses." : "";
  $("#single-note").textContent = info.out
    ? `The full step's test and gate, on this matrix only. Saved as ${info.out}${info.exists ? " (has a result)" : ""}; the full Speed and Accuracy files are not touched.${noAccuracy}`
    : (info.error || "");
}

function renderSingleState(busy) {
  $("#single").classList.toggle("selected", ui.selected === "single");
  $("#single-run").disabled = busy || !(ui.singleInfo && ui.singleInfo.commands.length);
  const job = ui.job && ui.job.kind === "single" ? ui.job : null;
  const last = $("#single-last");
  last.className = `single-last${job ? ` ${job.state}` : ""}`;
  last.textContent = job ? `${job.steps[0].title}: ${STATE_TEXT[job.state] || job.state} · ${clock(job.elapsed)}` : "";
}

function selectSingle() {
  if (ui.selected === "single") return;
  ui.selected = "single";
  renderSteps();
}

function singleChanged() {
  ui.selected = "single";
  saveSingle();
  refreshPreview();
}

function startSingle() {
  return startJob("/api/single", { settings: ui.settings, matrix: ui.single.matrix, test: ui.single.test });
}

// Starts a job; if earlier results would be replaced, asks first and retries with replace set.
async function startJob(path, body, replace = false) {
  try {
    const data = await api(path, { ...body, replace });
    if (path === "/api/single") ui.selected = "single";
    takeJob(data.job, true);
  } catch (error) {
    if (error.status === 409 && error.data.existing) {
      $("#confirm-files").innerHTML = error.data.existing.map((f) => `<li><code>${esc(f)}</code></li>`).join("");
      const dialog = $("#confirm");
      dialog.returnValue = "";
      dialog.showModal();
      dialog.addEventListener("close", () => { if (dialog.returnValue === "replace") startJob(path, body, true); }, { once: true });
      return;
    }
    $("#run-status").textContent = error.message;
  }
}

function startRun(ids) {
  return startJob("/api/run", { settings: ui.settings, steps: ids });
}

// ---------- jobs ----------
function takeJob(job, fresh) {
  if (fresh) {
    ui.lines = [];
    ui.offset = 0;
  }
  applySnapshot(job);
  if (job.state === "running") poll();
}

function applySnapshot(job) {
  ui.job = job;
  if (job.lines && job.lines.length) {
    ui.lines.push(...job.lines);
    if (ui.lines.length > 4000) ui.lines.splice(0, ui.lines.length - 4000);
    const out = $("#out");
    const atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    out.textContent = ui.lines.join("\n");
    if (atEnd) out.scrollTop = out.scrollHeight;
  }
  ui.offset = job.offset;
  const running = job.state === "running";
  const label = { demo: "Demo", single: "One-matrix run", drawings: "Drawing" }[job.kind] || "Run";
  $("#run-status").textContent = running ? `${label} running · ${clock(job.elapsed)}` : `${label} ${STATE_TEXT[job.state] || job.state} after ${clock(job.elapsed)}`;
  renderSteps();
  if (job.kind === "demo") {
    readProgress(job.lines || []);
    renderDemoProgress(job);
  }
}

async function poll() {
  if (ui.polling) return;
  ui.polling = true;
  try {
    while (ui.job && ui.job.state === "running") {
      // The demo switches solver every second or so, so it is followed more closely.
      await new Promise((r) => setTimeout(r, ui.job.kind === "demo" ? 250 : 700));
      const data = await api(`/api/job?offset=${ui.offset}`);
      if (!data.job) break;
      applySnapshot(data.job);
    }
  } catch (error) {
    $("#run-status").textContent = `Lost contact with the server: ${error.message}`;
  } finally {
    ui.polling = false;
  }
  if (ui.job) jobEnded(ui.job);
}

function jobEnded(job) {
  refreshPreview();
  if (job.kind === "run" || job.kind === "single") loadResults();
  if (job.kind === "drawings" || job.kind === "run") loadMatrices();
  if (job.kind === "demo") demoEnded(job);
}

// ---------- results tab ----------
async function loadResults() {
  try {
    const data = await api("/api/results");
    ui.machines = data.machines;
    ui.singleRuns = data.single || [];
  } catch (error) {
    ui.machines = [];
    ui.singleRuns = [];
    $("#gates-note").textContent = `Could not read results: ${error.message}`;
  }
  renderResults();
  renderSingleRuns();
}

function renderSingleRuns() {
  const runs = ui.singleRuns || [];
  $("#single-figure").hidden = !runs.length;
  $("#single-runs").innerHTML = runs.length ? `<tr><th>Results name</th><th>Matrix</th><th>Test</th><th>Our time / Eigen's</th><th></th></tr>` +
    runs.map((r) => {
      const ratios = r.error ? esc(r.error) : r.pairs.map((p) => `${esc(p.name)} ${p.ratio == null ? "-" : p.ratio.toFixed(3)}`).join(" · ");
      const verdict = r.passed ? (r.test === "accuracy" ? "verified" : "pass") : `<b style="color:var(--red)">fail</b>`;
      return `<tr><td>${esc(r.results)}</td><td>${esc(r.matrix)}</td><td>${esc(r.test)}</td><td class="num">${ratios}</td><td>${verdict}</td></tr>`;
    }).join("") : "";
}

function stamp(section) {
  if (!section) return `<span class="stamp grey">Not run</span>`;
  if (section.error) return `<span class="stamp grey" title="${esc(section.error)}">Unreadable</span>`;
  return section.passed ? `<span class="stamp">Passed</span>` : `<span class="stamp solid">Failed</span>`;
}

function renderResults() {
  const machines = ui.machines;
  const gates = $("#gates");
  if (!machines.length) {
    gates.innerHTML = "";
    $("#gates-note").textContent = "No results yet. Run the Speed and Accuracy steps on the Run tab, or pull a teammate's results/ folder.";
    $("#result-controls").hidden = true;
    $$(".chart-grid, #machines-figure").forEach((e) => { e.hidden = true; });
    return;
  }
  $("#result-controls").hidden = false;
  $$(".chart-grid").forEach((e) => { e.hidden = false; });
  gates.innerHTML = `<tr><th>Machine</th><th>Commit</th><th>CPU</th><th>Speed gate</th><th>Accuracy gate</th></tr>` +
    machines.map((m) => {
      const info = (m.speed && m.speed.manifest) || (m.accuracy && m.accuracy.manifest) || {};
      return `<tr><td>${esc(m.name)}</td><td class="num">${esc((info.git_commit || "").slice(0, 7))}</td>
        <td>${esc(info.cpu_model || "")}</td><td>${stamp(m.speed)}</td><td>${stamp(m.accuracy)}</td></tr>`;
    }).join("");
  const done = machines.filter((m) => m.speed && m.speed.passed && m.accuracy && m.accuracy.passed).length;
  const battery = machines.filter((m) => [m.speed, m.accuracy].some((s) => s && s.power && s.power.includes("battery")));
  $("#gates-note").textContent = `G1 closes when every team machine passes both gates. ${done} of ${machines.length} machine${machines.length === 1 ? "" : "s"} listed here ${done === 1 ? "has" : "have"} passed.` +
    (battery.length ? ` Warning: ${battery.map((m) => m.name).join(", ")} measured on battery, so those timings are not reliable.` : "");

  const withSpeed = machines.filter((m) => m.speed && m.speed.pairs);
  const pick = $("#machine-pick");
  pick.innerHTML = withSpeed.map((m, i) => `<option value="${i}">${esc(m.name)}</option>`).join("");
  ui.machine = Math.min(ui.machine, Math.max(0, withSpeed.length - 1));
  pick.value = String(ui.machine);
  $("#machine-pick-wrap").hidden = withSpeed.length <= 1;
  renderCharts(withSpeed);
}

function renderCharts(withSpeed) {
  const m = withSpeed[ui.machine];
  $$("#pair-seg button").forEach((b) => b.classList.toggle("on", Number(b.dataset.pair) === ui.pair));
  if (!m) {
    $("#ratio-chart").innerHTML = `<p class="empty">No speed results yet.</p>`;
    $("#time-chart").innerHTML = "";
  } else {
    const pair = m.speed.pairs[ui.pair];
    const hasGenerated = pair.rows.some((r) => r.generated);
    const hasReal = pair.rows.some((r) => !r.generated);
    $("#kind-seg").hidden = !(hasGenerated && hasReal);
    if (!hasGenerated) ui.kind = "real";
    if (!hasReal) ui.kind = "generated";
    $$("#kind-seg button").forEach((b) => b.classList.toggle("on", b.dataset.kind === ui.kind));
    const rows = pair.rows.filter((r) => r.generated === (ui.kind === "generated"));
    const name = `${pair.ours} against ${pair.base} on ${m.name}`;
    $("#ratio-chart").innerHTML = rows.length ? Charts.ratioDots(rows, pair.limit, name) : `<p class="empty">No rows.</p>`;
    $("#time-chart").innerHTML = rows.length ? Charts.timeBars(rows, `time per iteration, ${name}`) : "";
  }

  const base = ui.pair === 0 ? "eigen-E0" : "eigen-E1";
  const ladderOwner = [m, ...ui.machines].find((x) => x && x.ladder && x.ladder.ladders && x.ladder.ladders.some((l) => l.base === base));
  const ladder = ladderOwner && ladderOwner.ladder.ladders.find((l) => l.base === base);
  $("#ladder-chart").innerHTML = ladder ? Charts.ladder(ladder.points, base.replace("eigen-", "Eigen "), "optimization ladder")
    : `<p class="empty">No ladder measurements (g1-ladder.csv) yet.</p>`;
  $("#ladder-note").textContent = ladder ? `Each step adds one change. Geometric mean over the generated matrices, our time / ${base.replace("eigen-", "Eigen ")}, measured on ${ladderOwner.name}.` : "";

  const accOwner = (m && m.accuracy && m.accuracy.rows) ? m : ui.machines.find((x) => x.accuracy && x.accuracy.rows);
  const acc = accOwner && accOwner.accuracy;
  $("#accuracy").innerHTML = acc ? `<tr><th>Matrix</th><th>Iterations, ours / Eigen</th><th>Time / Eigen</th><th></th></tr>` +
    acc.rows.map((r) => `<tr><td>${esc(r.matrix)}</td>
      <td class="num">${r.missing ? "-" : `${r.our_spmv.toLocaleString()} / ${r.base_spmv.toLocaleString()}`}</td>
      <td class="num">${r.ratio == null ? "-" : r.ratio.toFixed(3)}</td>
      <td>${r.ok ? "verified" : `<b style="color:var(--red)">fail</b>`}</td></tr>`).join("") +
    `<tr><td colspan="4" class="note">Jacobi CG against Eigen E1 on ${esc(accOwner.name)}. Verified: same iteration count within 1%, every true residual within 1e-8.</td></tr>`
    : `<tr><td class="empty">No accuracy results yet.</td></tr>`;

  const series = withSpeed.map((x) => ({ name: x.name, rows: x.speed.pairs[ui.pair].rows.filter((r) => r.generated === (ui.kind === "generated")) }));
  const figure = $("#machines-figure");
  figure.hidden = series.length < 2;
  $("#machines-chart").innerHTML = series.length >= 2 ? Charts.machines(series, withSpeed[0].speed.pairs[ui.pair].limit, "machines side by side") : "";
}

// ---------- matrices tab ----------
const big = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : v.toLocaleString());
const bytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);

async function loadMatrices() {
  try {
    const data = await api(`/api/matrices?${new URLSearchParams(ui.settings).toString()}`);
    ui.matrices = data.matrices;
    ui.drawTool = data.tool;
  } catch (error) {
    ui.matrices = [];
    ui.drawTool = false;
    $("#matrices-status").textContent = error.message;
  }
  renderMatrices();
}

function matrixCard(m) {
  const d = m.drawing;
  const state = !m.available ? "not downloaded" : (d ? `n = ${big(d.n)} · ${big(d.nnz)} non-zeros` : "not drawn yet");
  return `<button class="matrix-card${m.name === ui.matrixSel ? " selected" : ""}" data-name="${esc(m.name)}"${d ? "" : " disabled"}>
    <span class="thumb">${d ? `<canvas data-thumb="${esc(m.name)}"></canvas>` : ""}</span>
    <b>${esc(m.name)}</b><small>${esc(state)}</small></button>`;
}

function renderMatrices() {
  const list = ui.matrices || [];
  if (!list.some((m) => m.name === ui.matrixSel && m.drawing)) {
    ui.matrixSel = (list.find((m) => !m.generated && m.drawing) || list.find((m) => m.drawing) || {}).name || null;
  }
  const group = (title, items) => (items.length ? `<div class="label">${title}</div><div class="cards">${items.map(matrixCard).join("")}</div>` : "");
  $("#matrix-cards").innerHTML = group("Real matrices, from SuiteSparse", list.filter((m) => !m.generated)) +
    group("Generated by our code", list.filter((m) => m.generated));
  $$("#matrix-cards canvas[data-thumb]").forEach((c) => Charts.paintDensity(c, list.find((m) => m.name === c.dataset.thumb).drawing));
  renderMatrixDetail();
  renderDrawStatus();
}

function selectedMatrix() {
  return (ui.matrices || []).find((m) => m.name === ui.matrixSel && m.drawing) || null;
}

function renderMatrixDetail() {
  const m = selectedMatrix();
  $("#matrix-detail").hidden = !m;
  if (!m) return;
  const d = m.drawing;
  $("#matrix-name").textContent = m.name;
  $("#matrix-about").innerHTML = esc(m.about) +
    (m.page ? ` · <a href="${esc(m.page)}" target="_blank" rel="noopener">its SuiteSparse page</a>` : "");
  Charts.paintDensity($("#density"), d);
  $("#density").removeAttribute("data-tip");
  $("#density-max").textContent = `${d.cells.reduce((a, c) => Math.max(a, c), 0).toLocaleString()} per block`;
  $("#zoom").innerHTML = Charts.zoom(d, `top-left ${d.zoom.size} by ${d.zoom.size} corner of ${m.name}`);
  $("#zoom-caption").textContent = `Top-left ${d.zoom.size} × ${d.zoom.size} corner, entry by entry`;
  const csr = d.nnz * 12 + (d.n + 1) * 8;
  const oneIn = Math.round((d.n * d.n) / d.nnz);
  const facts = [
    ["Rows and columns", d.n.toLocaleString()],
    ["Non-zeros", `${d.nnz.toLocaleString()}: ${oneIn <= 1 ? "every entry" : `1 in ${oneIn.toLocaleString()} entries`}`],
    ["Per row", `${d.row_nnz.min} fewest · ${d.row_nnz.mean.toFixed(1)} on average · ${d.row_nnz.max} most`],
    ["Bandwidth", `${d.bandwidth.toLocaleString()}: the farthest entry from the diagonal`],
    ["Size in CSR", `${bytes(csr)}, read once per CG iteration`],
    ["Used for", m.tests.length > 1 ? "the speed and accuracy tests" : "the speed test"],
  ];
  $("#matrix-facts").innerHTML = facts.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
}

// Names the block under the pointer; the page's tooltip then shows it.
function densityHover(e) {
  const m = selectedMatrix();
  if (!m) return;
  const d = m.drawing, box = e.currentTarget.getBoundingClientRect();
  const col = Math.min(d.grid - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * d.grid)));
  const row = Math.min(d.grid - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * d.grid)));
  const [r0, r1] = Charts.blockRange(row, d.grid, d.n), [c0, c1] = Charts.blockRange(col, d.grid, d.n);
  const count = d.cells[row * d.grid + col];
  const span = (a, b) => (a === b ? a.toLocaleString() : `${a.toLocaleString()}–${b.toLocaleString()}`);
  e.currentTarget.setAttribute("data-tip", `rows ${span(r0, r1)}, columns ${span(c0, c1)}: ${count.toLocaleString()} non-zero${count === 1 ? "" : "s"}`);
}

function renderDrawStatus() {
  const list = ui.matrices || [];
  const missing = list.filter((m) => m.available && !m.drawing).length;
  const busy = ui.job && ui.job.state === "running";
  const button = $("#draw-all");
  button.disabled = busy || !ui.drawTool || !missing;
  button.textContent = missing ? `Draw ${missing} matri${missing === 1 ? "x" : "ces"}` : "All drawn";
  const job = ui.job && ui.job.kind === "drawings" ? ui.job : null;
  let text = "";
  if (job && job.state === "running") {
    const at = job.steps.findIndex((s) => s.state === "running");
    text = at >= 0 ? `Drawing ${job.steps[at].id} · ${at + 1} of ${job.steps.length} · ${clock(job.elapsed)}` : `Drawing · ${clock(job.elapsed)}`;
  } else if (!ui.drawTool) {
    text = "Run the Build step on the Run tab first; it builds the drawing tool.";
  } else if (job && job.state !== "done") {
    text = `Drawing ${STATE_TEXT[job.state] || job.state}; the Run tab shows the output.`;
  } else if (missing) {
    text = "Each matrix is read once, under a minute for all of them. Matrices you have not downloaded are greyed out.";
  }
  $("#matrices-status").textContent = text;
}

async function startDrawings() {
  try {
    takeJob((await api("/api/drawings", { settings: ui.settings })).job, true);
  } catch (error) {
    $("#matrices-status").textContent = error.message;
  }
}

// ---------- demo tab ----------
async function loadDemoChoices() {
  const select = $("#demo-matrix");
  const keep = select.value;
  try {
    const data = await api(`/api/demo/choices?data=${encodeURIComponent(ui.settings.data || "")}`);
    select.innerHTML = data.choices.map((c) => `<option value="${esc(c.spec)}">${esc(c.label)} · ${esc(c.takes)}</option>`).join("");
    if (keep && data.choices.some((c) => c.spec === keep)) select.value = keep;
  } catch (error) {
    $("#demo-status").textContent = error.message;
  }
}

async function loadLastDemo() {
  try {
    const query = new URLSearchParams(ui.settings).toString();
    ui.lastDemo = (await api(`/api/demo/last?${query}`)).last;
  } catch (_) {
    ui.lastDemo = null;
  }
  // Only fill an empty tab: a live result or a running demo on screen is newer than the saved one.
  if (ui.lastDemo && !ui.demoShown && !(ui.job && ui.job.kind === "demo" && ui.job.state === "running")) {
    renderDemo(ui.lastDemo, "Recorded");
  }
}

// bench_cg --progress names each stage as it starts: "progress: read", then
// "progress: solve ours-jacobi warmup", "progress: solve eigen-E1 round 1 of 2", and so on.
function readProgress(lines) {
  for (const line of lines) {
    const m = /^progress: (?:(read)|solve (\S+) (.+))$/.exec(line);
    if (!m) continue;
    if (m[1]) {
      ui.demoNow = { stage: "read", who: null, text: "reading the matrix and storing it as CSR" };
    } else {
      const who = m[2].startsWith("ours") ? "ours" : "eigen";
      ui.demoNow = { stage: "solve", who, text: `${who === "ours" ? "our solver" : "Eigen"} (${m[2]}) · ${m[3] === "warmup" ? "warm-up, not timed" : `timed ${m[3]}`}` };
    }
  }
}

// Lights the stages happening now in ink, with a pulsing ring: orange on the solve stages while our
// solver runs, blue while Eigen's does, ink elsewhere. Earlier stages get a double border. null clears it.
function setFlow(now) {
  const order = ["read", "solve", "check"];
  const at = now ? order.indexOf(now.stage) : -1;
  $$("#flow .box").forEach((box) => {
    const phase = order.indexOf(box.dataset.phase);
    // Recording the curve runs our solver again, so the solve stages stay lit during the check.
    const active = at >= 0 && !box.classList.contains("skip") && (phase === at || (now.stage === "check" && phase === 1));
    const accent = active && phase === 1 ? now.who : null;
    box.classList.toggle("on", active);
    box.classList.toggle("ours", accent === "ours");
    box.classList.toggle("eigen", accent === "eigen");
    box.classList.toggle("done", Boolean(now) && (now.stage === "done" || (at >= 0 && phase < at && !active)));
  });
  const live = at >= 0;
  $("#flow-now").innerHTML = live ? `<span class="sw ${now.who || "read"}"></span>Now: ${esc(now.text)}` : "";
  ["ours", "eigen"].forEach((who) => {
    if (live) $(`#lane-${who}`).classList.toggle("waiting", now.stage === "solve" && now.who === who);
  });
}

function clearDemo() {
  ui.demoShown = false;
  ui.demoNow = null;
  ["#lane-ours", "#lane-eigen"].forEach((id) => { const lane = $(id); lane.classList.add("waiting"); lane.style.width = "0"; });
  $("#time-ours").textContent = "";
  $("#time-eigen").textContent = "";
  $("#verdict").innerHTML = "&nbsp;";
  $("#demo-stamps").innerHTML = "";
  $("#demo-facts").textContent = "";
  $("#curve").innerHTML = "";
}

async function solveLive() {
  clearDemo();
  setFlow({ stage: "read", who: null, text: "starting" });
  $("#demo-status").textContent = "Starting…";
  try {
    const data = await api("/api/demo", { settings: ui.settings, matrix: $("#demo-matrix").value, precond: ui.precond });
    takeJob(data.job, true);
  } catch (error) {
    demoFailed(error.message);
  }
}

function renderDemoProgress(job) {
  if (job.state !== "running") return;
  const race = job.steps.find((s) => s.id === "race");
  const curve = job.steps.find((s) => s.id === "curve");
  if (race && race.state === "running") {
    setFlow(ui.demoNow || { stage: "read", who: null, text: "starting" });
    $("#demo-status").textContent = `Warm-up, then two timed rounds each · ${clock(race.elapsed)}`;
  } else if (curve && curve.state === "running") {
    ["ours", "eigen"].forEach((who) => $(`#lane-${who}`).classList.remove("waiting"));
    setFlow({ stage: "check", who: "ours", text: "our solver again, recording the error curve and checking the answer, untimed" });
    $("#demo-status").textContent = `Timed. Now recording the error curve · ${clock(curve.elapsed)}`;
  }
}

function demoEnded(job) {
  if (job.state === "done" && job.result) {
    ui.lastDemo = job.result;
    $("#demo-status").textContent = `Finished in ${clock(job.elapsed)}.`;
    renderDemo(job.result, null);
  } else {
    const reason = job.error || (job.state === "stopped" ? "stopped" : "the run failed; see the output on the Run tab");
    demoFailed(reason);
  }
}

function demoFailed(reason) {
  $("#demo-status").textContent = `Live run did not finish: ${reason}`;
  if (ui.lastDemo) renderDemo(ui.lastDemo, "Recorded");
  else { clearDemo(); setFlow(null); ["#lane-ours", "#lane-eigen"].forEach((id) => $(id).classList.remove("waiting")); }
}

function renderDemo(r, recorded) {
  ui.demoShown = true;
  setFlow({ stage: "done" });
  const slow = Math.max(r.ours.time, r.eigen.time);
  [["#lane-ours", "#time-ours", r.ours.time], ["#lane-eigen", "#time-eigen", r.eigen.time]].forEach(([lane, label, t]) => {
    const el = $(lane);
    el.classList.remove("waiting");
    el.style.width = "0";
    void el.offsetWidth;  // commit the empty bar, so the change below animates
    el.style.width = `${(100 * t) / slow}%`;
    $(label).textContent = duration(t);
  });
  if (r.speedup == null) $("#verdict").innerHTML = "&nbsp;";
  else if (r.speedup >= 1) $("#verdict").innerHTML = `${r.speedup.toFixed(2)}&times; faster <small>than Eigen</small>`;
  else $("#verdict").innerHTML = `${(1 / r.speedup).toFixed(2)}&times; slower <small>than Eigen</small>`;
  const stamps = [r.same_answer ? `<span class="stamp big">Same answer</span>` : `<span class="stamp big solid">Answers differ</span>`];
  if (recorded) stamps.push(`<span class="stamp grey">${esc(recorded)}${r.when ? ` ${esc(r.when)}` : ""}</span>`);
  $("#demo-stamps").innerHTML = stamps.join("");
  const its = r.ours.spmv === r.eigen.spmv ? `${r.ours.spmv.toLocaleString()} iterations each` : `ours ${r.ours.spmv.toLocaleString()} iterations, Eigen ${r.eigen.spmv.toLocaleString()}`;
  $("#demo-facts").textContent = `${r.label || r.matrix} · n = ${r.n.toLocaleString()}, ${r.nnz.toLocaleString()} non-zeros · ${its} · ` +
    (r.ours.converged && r.eigen.converged ? `both errors below ${r.tol}` : "not every solve met the tolerance") +
    ` · ${r.precond === "jacobi" ? "Jacobi CG against Eigen E1" : "plain CG against Eigen E0"}` +
    (r.power === "battery" ? " · ON BATTERY: these times are not reliable, plug in the charger" : "");
  if (r.history && r.history.length > 1) {
    $("#curve").innerHTML = Charts.residual(r.history, r.tol, "relative residual per iteration");
    const path = $("#curve .curve");
    if (path) path.style.setProperty("--len", String(Math.ceil(path.getTotalLength())));
  } else {
    $("#curve").innerHTML = `<p class="empty">No curve recorded.</p>`;
  }
}

// ---------- tooltip ----------
function tooltip() {
  const tip = $("#tip");
  document.addEventListener("mousemove", (e) => {
    const target = e.target.closest && e.target.closest("[data-tip]");
    if (!target) { tip.hidden = true; return; }
    tip.textContent = target.getAttribute("data-tip");
    tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.min(e.clientY + 16, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
  document.addEventListener("mouseleave", () => { tip.hidden = true; });
}

// ---------- start ----------
async function init() {
  window.addEventListener("hashchange", showTab);
  showTab();
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => { location.hash = b.dataset.tab; }));
  tooltip();

  let state = { suggested: {} };
  try { state = await api("/api/state"); } catch (error) { $("#run-status").textContent = `Server not reachable: ${error.message}`; }
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (_) { saved = {}; }
  const form = $("#settings");
  for (const key of ["machine", "build", "data", "compiler"]) form.elements[key].value = saved[key] || state.suggested[key] || "";
  saveSettings();
  form.addEventListener("input", settingsChanged);
  form.addEventListener("submit", (e) => e.preventDefault());

  $("#steps").addEventListener("click", (e) => {
    const run = e.target.closest("[data-run]");
    if (run) { startRun([run.dataset.run]); return; }
    const row = e.target.closest(".step");
    if (row) { ui.selected = row.dataset.id; renderSteps(); }
  });
  $("#steps").addEventListener("keydown", (e) => {
    const row = e.target.closest(".step");
    if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); ui.selected = row.dataset.id; renderSteps(); }
  });
  $("#run-all").addEventListener("click", () => startRun(ALL_STEPS));
  loadSingle();
  $("#single").addEventListener("click", selectSingle);
  $("#single").addEventListener("focusin", selectSingle);
  $("#single-matrix").addEventListener("change", (e) => { ui.single.matrix = e.target.value; singleChanged(); });
  $("#single-test").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b && !b.disabled) { ui.single.test = b.dataset.test; singleChanged(); }
  });
  $("#single-run").addEventListener("click", startSingle);
  $("#stop").addEventListener("click", () => api("/api/stop", {}).catch((error) => { $("#run-status").textContent = error.message; }));

  $("#machine-pick").addEventListener("change", (e) => { ui.machine = Number(e.target.value); renderResults(); });
  $("#pair-seg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { ui.pair = Number(b.dataset.pair); renderResults(); } });
  $("#kind-seg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { ui.kind = b.dataset.kind; renderResults(); } });

  $("#precond-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    ui.precond = b.dataset.precond;
    $$("#precond-seg button").forEach((x) => x.classList.toggle("on", x === b));
    $("#jacobi-box").classList.toggle("skip", ui.precond === "none");
  });
  $("#solve-live").addEventListener("click", solveLive);

  $("#draw-all").addEventListener("click", startDrawings);
  $("#density").addEventListener("mousemove", densityHover);
  $("#matrix-cards").addEventListener("click", (e) => {
    const card = e.target.closest(".matrix-card");
    if (!card || card.disabled) return;
    ui.matrixSel = card.dataset.name;
    $$("#matrix-cards .matrix-card").forEach((c) => c.classList.toggle("selected", c === card));
    renderMatrixDetail();
    $("#matrix-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  refreshPreview();
  loadResults();
  loadMatrices();
  loadDemoChoices();
  if (state.job) {
    if (state.job.kind === "single") ui.selected = "single";
    takeJob(state.job, true);
    if (state.job.state !== "running") jobEndedQuietly(state.job);
  }
  loadLastDemo();
}

// A job that finished before this page opened: show its output and state, without re-running effects.
function jobEndedQuietly(job) {
  if (job.kind === "demo" && job.state === "done" && job.result) renderDemo(job.result, null);
}

init();
