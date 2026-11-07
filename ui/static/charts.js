// Hand-drawn SVG charts for the training log. Each function returns an SVG string.
// Marks carry data-tip text, which the page shows as a tooltip on hover.
"use strict";

const Charts = (() => {
  const C = {
    ink: "#1f1b16", mute: "#6b6152", rule: "rgba(31,27,22,.16)", paper: "#ece4d2",
    ours: "#d9480f", eigen: "#1767c7", red: "#b3261e",
    machines: ["#d9480f", "#1767c7", "#c2187a", "#2b8a3e"],
  };
  const SHAPES = ["circle", "square", "diamond", "triangle"];

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const f3 = (v) => (v == null ? "-" : v.toFixed(3));
  const ms = (s) => (s * 1e3).toFixed(s < 0.01 ? 2 : 1);

  function svg(width, height, body, label) {
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}">${body}</svg>`;
  }

  function text(x, y, s, attrs = "") {
    return `<text x="${x}" y="${y}" ${attrs}>${esc(s)}</text>`;
  }

  function shape(kind, x, y, r, fill, tip) {
    const t = tip ? ` data-tip="${esc(tip)}"` : "";
    const ring = `fill="${fill}" stroke="${C.paper}" stroke-width="2"`;
    if (kind === "square") return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" rx="1" ${ring}${t}/>`;
    if (kind === "diamond") return `<path d="M${x},${y - r - 1} L${x + r + 1},${y} L${x},${y + r + 1} L${x - r - 1},${y} Z" ${ring}${t}/>`;
    if (kind === "triangle") return `<path d="M${x},${y - r - 1} L${x + r + 1},${y + r} L${x - r - 1},${y + r} Z" ${ring}${t}/>`;
    return `<circle cx="${x}" cy="${y}" r="${r}" ${ring}${t}/>`;
  }

  // Rounded end away from the baseline, square at the baseline.
  function hbar(x0, y, w, h, fill, tip) {
    if (w <= 0) return "";
    const r = Math.min(4, w, h / 2);
    const x1 = x0 + w;
    return `<path d="M${x0},${y} H${x1 - r} Q${x1},${y} ${x1},${y + r} V${y + h - r} Q${x1},${y + h} ${x1 - r},${y + h} H${x0} Z" fill="${fill}" data-tip="${esc(tip)}"/>`;
  }

  // A round tick step giving about `count` ticks up to v, and the axis end it implies.
  function niceAxis(v, count = 5) {
    const raw = v / count;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
    return { step, top: Math.ceil(v / step) * step };
  }

  function ratioDomain(rows) {
    let lo = 0.5, hi = 1.1;
    for (const r of rows) {
      lo = Math.min(lo, (r.ci_low ?? r.ratio) - 0.05);
      hi = Math.max(hi, (r.ci_high ?? r.ratio) + 0.05);
    }
    return [Math.floor(lo * 10) / 10, Math.ceil(hi * 10) / 10];
  }

  // Our time over Eigen's per matrix: dot = median paired ratio, line = 95% interval.
  function ratioDots(rows, limit, label) {
    rows = rows.filter((r) => !r.missing && r.ratio != null).sort((a, b) => a.ratio - b.ratio);
    const W = 600, L = 104, R = 540, T = 30, rowH = 24, H = T + rows.length * rowH + 44;
    const [lo, hi] = ratioDomain(rows);
    const x = (v) => L + ((v - lo) / (hi - lo)) * (R - L);
    const yb = T + rows.length * rowH;
    let s = "";
    for (let t = Math.round(lo * 10); t <= Math.round(hi * 10); t++) {
      const v = t / 10;
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${T - 4}" y2="${yb}" stroke="${C.rule}"/>`;
      s += text(x(v), yb + 16, v.toFixed(1), `class="tick" text-anchor="middle"`);
    }
    s += `<line x1="${x(1)}" x2="${x(1)}" y1="${T - 12}" y2="${yb}" stroke="${C.eigen}" stroke-width="2"/>`;
    s += text(x(1), T - 16, "Eigen", `class="lbl" text-anchor="end" dx="-4"`);
    if (limit) {
      s += `<line x1="${x(limit)}" x2="${x(limit)}" y1="${T - 12}" y2="${yb}" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
      s += text(x(limit), T - 16, `limit ${limit}`, `class="lbl red" dx="4"`);
    }
    rows.forEach((r, i) => {
      const y = T + i * rowH + rowH / 2;
      const ci = r.ci_low != null ? `95% interval ${f3(r.ci_low)} to ${f3(r.ci_high)}` : "rounds not paired";
      const tip = `${r.matrix}: ${f3(r.ratio)} (${ci}). Ours ${ms(r.our_time)} ms, Eigen ${ms(r.base_time)} ms per iteration.`;
      s += text(L - 10, y + 4, r.matrix, `class="name" text-anchor="end"`);
      if (r.ci_low != null) {
        s += `<line x1="${x(r.ci_low)}" x2="${x(r.ci_high)}" y1="${y}" y2="${y}" stroke="${C.ours}" stroke-width="2" stroke-linecap="round"/>`;
      }
      s += shape("circle", x(r.ratio), y, 5, C.ours);
      if (!r.ok) s += text(W - 4, y + 4, "FAIL", `class="lbl red" text-anchor="end"`);
      s += `<rect x="${L}" y="${y - rowH / 2}" width="${R - L}" height="${rowH}" fill="transparent" data-tip="${esc(tip)}"/>`;
    });
    s += text((L + R) / 2, yb + 34, "our time / Eigen's  ·  lower is faster", `class="axis" text-anchor="middle"`);
    return svg(W, H, s, label);
  }

  // Milliseconds per iteration, ours and Eigen's, one pair of bars per matrix.
  function timeBars(rows, label) {
    rows = rows.filter((r) => !r.missing && r.our_time != null).sort((a, b) => a.base_time - b.base_time);
    const W = 560, L = 104, R = 520, T = 34, rowH = 26, barH = 9, H = T + rows.length * rowH + 40;
    const { step, top } = niceAxis(Math.max(...rows.map((r) => Math.max(r.base_time, r.our_time))) * 1e3);
    const x = (v) => L + (v / top) * (R - L);
    const yb = T + rows.length * rowH;
    let s = "";
    for (let i = 0; i * step <= top + 1e-9; i++) {
      const v = i * step;
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${T - 4}" y2="${yb}" stroke="${C.rule}"/>`;
      s += text(x(v), yb + 16, String(+v.toFixed(2)), `class="tick" text-anchor="middle"`);
    }
    s += `<rect x="${L}" y="4" width="10" height="10" rx="2" fill="${C.ours}"/>` + text(L + 15, 13, "ours", `class="lbl"`);
    s += `<rect x="${L + 64}" y="4" width="10" height="10" rx="2" fill="${C.eigen}"/>` + text(L + 79, 13, "Eigen", `class="lbl"`);
    rows.forEach((r, i) => {
      const y = T + i * rowH + (rowH - 2 * barH - 2) / 2;
      const ours = r.our_time * 1e3, eig = r.base_time * 1e3;
      s += text(L - 10, y + barH + 5, r.matrix, `class="name" text-anchor="end"`);
      s += hbar(L, y, x(ours) - L, barH, C.ours, `${r.matrix}: ours ${ours.toFixed(2)} ms per iteration`);
      s += hbar(L, y + barH + 2, x(eig) - L, barH, C.eigen, `${r.matrix}: Eigen ${eig.toFixed(2)} ms per iteration`);
      s += text(x(ours) + 5, y + barH - 1, ours.toFixed(1), `class="val"`);
    });
    s += text((L + R) / 2, yb + 34, "milliseconds per iteration", `class="axis" text-anchor="middle"`);
    return svg(W, H, s, label);
  }

  // One rung per change, each as our time over Eigen's.
  function ladder(points, baseName, label) {
    const W = 420, L = 46, R = 404, T = 24, B = 214, H = 262;
    const values = points.map((p) => p.ratio).filter((v) => v != null);
    const lo = Math.min(Math.floor((Math.min(...values) - 0.02) * 20) / 20, 0.95), hi = 1.0;
    const y = (v) => B - ((v - lo) / (hi - lo)) * (B - T);
    const n = points.length;
    const xs = (i) => (n === 1 ? (L + R) / 2 : L + 20 + (i * (R - L - 40)) / (n - 1));
    let s = "";
    for (let k = Math.round(lo * 20); k <= Math.round(hi * 20); k++) {
      const v = k / 20;
      s += `<line x1="${L}" x2="${R}" y1="${y(v)}" y2="${y(v)}" stroke="${C.rule}"/>`;
      s += text(L - 6, y(v) + 4, v.toFixed(2), `class="tick" text-anchor="end"`);
    }
    s += `<line x1="${L}" x2="${R}" y1="${y(1)}" y2="${y(1)}" stroke="${C.eigen}" stroke-width="2"/>`;
    s += text(R, y(1) - 6, baseName, `class="lbl" text-anchor="end"`);
    const pts = points.map((p, i) => [xs(i), y(p.ratio ?? 1)]);
    s += `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${C.ours}" stroke-width="2" stroke-linejoin="round"/>`;
    points.forEach((p, i) => {
      const [px, py] = pts[i];
      s += shape("circle", px, py, 5, C.ours);
      s += `<circle cx="${px}" cy="${py}" r="13" fill="transparent" data-tip="${esc(`${p.rung}, ${p.label}: ${f3(p.ratio)} of ${baseName}'s time, geometric mean over ${p.matrices} matrices`)}"/>`;
      s += text(px, B + 16, p.rung.split("-")[0], `class="name" text-anchor="middle"`);
      s += text(px, B + 30, p.short || "", `class="tick" text-anchor="middle"`);
      if (i === 0) s += text(px + 9, py - 9, f3(p.ratio), `class="val"`);
      if (i === n - 1 && n > 1) s += text(px, py + 20, f3(p.ratio), `class="val" text-anchor="middle"`);
    });
    return svg(W, H, s, label);
  }

  // Every machine's ratio per matrix, so the team can compare laptops.
  function machines(series, limit, label) {
    const names = [...new Set(series.flatMap((m) => m.rows.filter((r) => r.ratio != null).map((r) => r.matrix)))].sort();
    const all = series.flatMap((m) => m.rows.filter((r) => r.ratio != null));
    const W = 560, L = 104, R = 548, T = 30 + 18 * Math.ceil(series.length / 2), rowH = 24;
    const H = T + names.length * rowH + 44;
    const [lo, hi] = ratioDomain(all.map((r) => ({ ratio: r.ratio })));
    const x = (v) => L + ((v - lo) / (hi - lo)) * (R - L);
    const yb = T + names.length * rowH;
    let s = "";
    series.forEach((m, i) => {
      const lx = L + (i % 2) * 220, ly = 12 + Math.floor(i / 2) * 18;
      s += shape(SHAPES[i % 4], lx + 5, ly - 4, 5, C.machines[i % 4]) + text(lx + 16, ly, m.name, `class="lbl"`);
    });
    for (let t = Math.round(lo * 10); t <= Math.round(hi * 10); t++) {
      const v = t / 10;
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${T - 4}" y2="${yb}" stroke="${C.rule}"/>`;
      s += text(x(v), yb + 16, v.toFixed(1), `class="tick" text-anchor="middle"`);
    }
    s += `<line x1="${x(1)}" x2="${x(1)}" y1="${T - 6}" y2="${yb}" stroke="${C.eigen}" stroke-width="2"/>`;
    if (limit) s += `<line x1="${x(limit)}" x2="${x(limit)}" y1="${T - 6}" y2="${yb}" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    names.forEach((name, j) => {
      const y = T + j * rowH + rowH / 2;
      s += text(L - 10, y + 4, name, `class="name" text-anchor="end"`);
      series.forEach((m, i) => {
        const r = m.rows.find((row) => row.matrix === name);
        if (r && r.ratio != null) {
          s += shape(SHAPES[i % 4], x(r.ratio), y, 5, C.machines[i % 4], `${m.name}, ${name}: ${f3(r.ratio)}`);
        }
      });
    });
    s += text((L + R) / 2, yb + 34, "our time / Eigen's  ·  lower is faster", `class="axis" text-anchor="middle"`);
    return svg(W, H, s, label);
  }

  // ‖r‖/‖b‖ per iteration on a log scale, down to the tolerance line.
  function residual(history, tol, label) {
    const W = 460, L = 52, R = 446, T = 14, B = 196, H = 228;
    const last = history[history.length - 1];
    const maxIt = Math.max(1, last[0]);
    const minLog = Math.min(Math.floor(Math.log10(tol)) - 1, Math.floor(Math.log10(Math.min(...history.map((p) => p[1])))));
    const y = (v) => T + ((0 - Math.log10(v)) / (0 - minLog)) * (B - T);
    const x = (i) => L + (i / maxIt) * (R - L);
    let s = "";
    for (let e = 0; e >= minLog; e -= 2) {
      s += `<line x1="${L}" x2="${R}" y1="${y(10 ** e)}" y2="${y(10 ** e)}" stroke="${C.rule}"/>`;
      s += text(L - 6, y(10 ** e) + 4, e === 0 ? "1" : `1e${e}`, `class="tick" text-anchor="end"`);
    }
    s += `<line x1="${L}" x2="${R}" y1="${y(tol)}" y2="${y(tol)}" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += text(L + 6, y(tol) - 6, `target ${tol}`, `class="lbl red"`);
    const d = history.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
    s += `<path class="curve" d="${d}" fill="none" stroke="${C.ours}" stroke-width="2" stroke-linejoin="round"/>`;
    s += shape("circle", x(last[0]), y(last[1]), 5, C.ours, `iteration ${last[0]}: ${last[1].toExponential(2)}`);
    for (const t of [0, maxIt]) s += text(x(t), B + 16, String(t), `class="tick" text-anchor="middle"`);
    s += text((L + R) / 2, B + 28, "iteration", `class="axis" text-anchor="middle"`);
    return svg(W, H, s, label);
  }

  return { ratioDots, timeBars, ladder, machines, residual, colors: C };
})();
