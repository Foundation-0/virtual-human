/* Virtual Human demo: wires the two tested engines (vh-md.js, vh-pbpk.js) to the page. */
(function () {
  "use strict";
  const DATA = window.VH_DATA;
  const $ = (s) => document.querySelector(s);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- theme
  let T = {};
  function readTokens() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["ink", "muted", "line", "panel", "accent", "accent-soft", "drug", "signal", "scope", "scope-ink", "scope-line", "bg"])
      T[k] = cs.getPropertyValue("--" + k).trim();
  }
  readTokens();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTokens);
  new MutationObserver(readTokens).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function parseColor(c) {
    c = c.trim();
    if (c[0] === "#") {
      const h = c.length === 4 ? c.slice(1).split("").map((x) => x + x).join("") : c.slice(1, 7);
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    }
    const m = c.match(/[\d.]+/g) || [0, 0, 0];
    return [+m[0], +m[1], +m[2]];
  }
  function mix(a, b, t) {
    const A = parseColor(a), B = parseColor(b);
    return `rgb(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)})`;
  }
  function fitCanvas(c) {
    const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }
  const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x > 0 ? x.toPrecision(2) : "0");

  // ---------------------------------------------------------------- ruler
  (function buildRuler() {
    const el = $("#ruler"), lo = -15, hi = 5.6, pos = (l) => ((l - lo) / (hi - lo)) * 100;
    const band = (cls, a, b) => { const d = document.createElement("div"); d.className = "ruler-band " + cls; d.style.left = pos(a) + "%"; d.style.width = pos(b) - pos(a) + "%"; el.appendChild(d); };
    band("atoms", -15, -8.5);
    band("body", Math.log10(60), Math.log10(3 * 86400));
    for (const [l, t] of [[-15, "fs"], [-12, "ps"], [-9, "ns"], [-6, "µs"], [-3, "ms"], [0, "1 s"], [Math.log10(3600), "1 h"], [Math.log10(86400), "1 day"]]) {
      const d = document.createElement("span"); d.className = "tick"; d.style.left = pos(l) + "%"; d.textContent = t; el.appendChild(d);
    }
  })();

  // ================================================================ ATOMS
  const CPK = { H: "#EEF1F6", C: "#8E97A8", N: "#4A6BFF", O: "#F0493D", S: "#F2C230" };
  const RAD = { H: 0.28, C: 0.46, N: 0.44, O: 0.43, S: 0.55 };
  const sprites = {};
  function sprite(el) {
    if (sprites[el]) return sprites[el];
    const s = 128, c = document.createElement("canvas"); c.width = c.height = s;
    const g = c.getContext("2d"), base = CPK[el] || "#B8BFCC";
    const grd = g.createRadialGradient(s * 0.37, s * 0.33, s * 0.04, s * 0.5, s * 0.5, s * 0.5);
    grd.addColorStop(0, mix(base, "#FFFFFF", 0.7)); grd.addColorStop(0.45, base); grd.addColorStop(1, mix(base, "#000000", 0.62));
    g.fillStyle = grd; g.beginPath(); g.arc(s / 2, s / 2, s / 2 - 1, 0, 2 * Math.PI); g.fill();
    return (sprites[el] = c);
  }
  const rotX = (t) => { const c = Math.cos(t), s = Math.sin(t); return [1, 0, 0, 0, c, -s, 0, s, c]; };
  const rotY = (t) => { const c = Math.cos(t), s = Math.sin(t); return [c, 0, s, 0, 1, 0, -s, 0, c]; };
  const mul3 = (A, B) => { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]; return C; };

  const view = { R: mul3(rotX(-0.35), rotY(0.6)), zoom: 1.3, auto: !reduceMotion, dragging: false, lx: 0, ly: 0 };
  let mol = null, molPlaying = true, showH = true, tempK = 310, molSteps = 0, molClock = performance.now(), molSps = 0;

  function loadMol(key) {
    const sys = DATA[key];
    const engine = new VHMD.Engine(sys);
    const dyn = new VHMD.Dynamics(engine, sys.x0, { temperature: tempK, friction: 2.0, seed: (Math.random() * 1e9) | 0, dt: 0.001 });
    const N = engine.N, x0 = Float64Array.from(sys.x0.flat()), ca = sys.ca_indices, contacts = [];
    for (let a = 0; a < ca.length; a++) for (let b = a + 3; b < ca.length; b++) {
      const i = 3 * ca[a], j = 3 * ca[b];
      const d = Math.hypot(x0[i] - x0[j], x0[i + 1] - x0[j + 1], x0[i + 2] - x0[j + 2]);
      if (d < 0.8) contacts.push([ca[a], ca[b], d]);
    }
    let cx = 0, cy = 0, cz = 0;
    for (let a = 0; a < N; a++) { cx += x0[3 * a]; cy += x0[3 * a + 1]; cz += x0[3 * a + 2]; }
    cx /= N; cy /= N; cz /= N;
    let ext = 0;
    for (let a = 0; a < N; a++) ext = Math.max(ext, Math.hypot(x0[3 * a] - cx, x0[3 * a + 1] - cy, x0[3 * a + 2] - cz));
    mol = { key, sys, engine, dyn, N, ca, contacts, extent: ext * 10 + 2.0, hist: [], tHist: [], lastSample: -1, tAvg: tempK };
    sampleMol();
    molSteps = 0; molClock = performance.now();
    $("#sys-chignolin").setAttribute("aria-pressed", key === "chignolin");
    $("#sys-trpcage").setAttribute("aria-pressed", key === "trpcage");
  }

  function radiusOfGyration(m) {
    const x = m.dyn.x, ca = m.ca; let cx = 0, cy = 0, cz = 0;
    for (const i of ca) { cx += x[3 * i]; cy += x[3 * i + 1]; cz += x[3 * i + 2]; }
    const n = ca.length; cx /= n; cy /= n; cz /= n;
    let s = 0;
    for (const i of ca) s += (x[3 * i] - cx) ** 2 + (x[3 * i + 1] - cy) ** 2 + (x[3 * i + 2] - cz) ** 2;
    return Math.sqrt(s / n) * 10;
  }
  function nativeContacts(m) {
    const x = m.dyn.x; let k = 0;
    for (const [i, j, d0] of m.contacts) if (Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]) < 1.2 * d0) k++;
    return m.contacts.length ? k / m.contacts.length : 1;
  }
  function sampleMol() {
    mol.lastSample = mol.dyn.time;
    mol.hist.push(nativeContacts(mol)); mol.tHist.push(mol.dyn.time);
    if (mol.hist.length > 800) { mol.hist.shift(); mol.tHist.shift(); }
  }

  function stepMol() {
    if (!molPlaying) return;
    const t0 = performance.now(), budget = 9, chunk = mol.N > 200 ? 1 : 2;
    let n = 0;
    do { mol.dyn.step(chunk); n += chunk; } while (performance.now() - t0 < budget);
    molSteps += n;
    mol.tAvg += 0.04 * (mol.dyn.temperature() - mol.tAvg);
    if (mol.dyn.time - mol.lastSample >= 0.05) sampleMol();
  }

  const molCanvas = $("#mol");
  function drawMol() {
    const { ctx, w, h } = fitCanvas(molCanvas);
    ctx.fillStyle = T.scope; ctx.fillRect(0, 0, w, h);
    const m = mol, x = m.dyn.x, N = m.N, R = view.R, el = m.sys.elements;
    let cx = 0, cy = 0, cz = 0;
    for (let a = 0; a < N; a++) { cx += x[3 * a]; cy += x[3 * a + 1]; cz += x[3 * a + 2]; }
    cx /= N; cy /= N; cz /= N;
    const scale = (Math.min(w, h) / (2 * m.extent)) * view.zoom * 0.9, F = m.extent * 4;
    const sx = new Float32Array(N), sy = new Float32Array(N), sz = new Float32Array(N), sp = new Float32Array(N);
    for (let a = 0; a < N; a++) {
      const px = (x[3 * a] - cx) * 10, py = (x[3 * a + 1] - cy) * 10, pz = (x[3 * a + 2] - cz) * 10;
      const qx = R[0] * px + R[1] * py + R[2] * pz, qy = R[3] * px + R[4] * py + R[5] * pz, qz = R[6] * px + R[7] * py + R[8] * pz;
      const p = F / (F - qz);
      sx[a] = w / 2 + qx * scale * p; sy[a] = h / 2 - qy * scale * p; sz[a] = qz; sp[a] = p;
    }
    const items = [];
    for (const b of m.sys.bonds) {
      const i = b[0], j = b[1];
      if (!showH && (el[i] === "H" || el[j] === "H")) continue;
      items.push([0.5 * (sz[i] + sz[j]), 1, i, j]);
    }
    for (let a = 0; a < N; a++) if (showH || el[a] !== "H") items.push([sz[a], 0, a, 0]);
    items.sort((u, v) => u[0] - v[0]);
    ctx.lineCap = "round";
    ctx.strokeStyle = T["scope-line"];
    const ext = m.extent;
    for (const it of items) {
      ctx.globalAlpha = Math.max(0.35, Math.min(1, 0.6 + 0.4 * (it[0] / ext)));
      if (it[1]) {
        const i = it[2], j = it[3];
        ctx.lineWidth = Math.max(1, 0.2 * scale * 0.5 * (sp[i] + sp[j]));
        ctx.beginPath(); ctx.moveTo(sx[i], sy[i]); ctx.lineTo(sx[j], sy[j]); ctx.stroke();
      } else {
        const a = it[2], r = (RAD[el[a]] || 0.45) * scale * sp[a];
        ctx.drawImage(sprite(el[a]), sx[a] - r, sy[a] - r, 2 * r, 2 * r);
      }
    }
    ctx.globalAlpha = 1;
  }

  const scope = $("#scope");
  scope.addEventListener("pointerdown", (e) => { view.dragging = true; view.auto = false; view.lx = e.clientX; view.ly = e.clientY; scope.setPointerCapture(e.pointerId); });
  scope.addEventListener("pointermove", (e) => {
    if (!view.dragging) return;
    const dx = e.clientX - view.lx, dy = e.clientY - view.ly; view.lx = e.clientX; view.ly = e.clientY;
    view.R = mul3(rotY(dx * 0.01), mul3(rotX(dy * 0.01), view.R));
  });
  const endDrag = () => { view.dragging = false; };
  scope.addEventListener("pointerup", endDrag);
  scope.addEventListener("pointercancel", endDrag);
  scope.addEventListener("wheel", (e) => { e.preventDefault(); view.zoom = Math.min(4, Math.max(0.5, view.zoom * Math.exp(-e.deltaY * 0.0015))); }, { passive: false });

  $("#sys-chignolin").addEventListener("click", () => loadMol("chignolin"));
  $("#sys-trpcage").addEventListener("click", () => loadMol("trpcage"));
  $("#mol-reset").addEventListener("click", () => loadMol(mol.key));
  $("#mol-play").addEventListener("click", () => {
    molPlaying = !molPlaying;
    $("#mol-play").textContent = molPlaying ? "Pause" : "Play";
    $("#live").classList.toggle("paused", !molPlaying);
    $("#live").textContent = molPlaying ? "LIVE" : "PAUSED";
  });
  $("#temp").addEventListener("input", (e) => { tempK = +e.target.value; mol.dyn.T = tempK; $("#temp-out").textContent = tempK + " K"; });
  $("#show-h").addEventListener("change", (e) => { showH = e.target.checked; });

  const qCanvas = $("#qchart");
  function drawSpark() {
    const { ctx, w, h } = fitCanvas(qCanvas);
    ctx.clearRect(0, 0, w, h);
    const t = mol.tHist, q = mol.hist, padL = 34, padB = 16, padT = 6;
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--font-data");
    ctx.fillStyle = T.muted; ctx.strokeStyle = T.line; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      const y = padT + (1 - v) * (h - padT - padB);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillText((v * 100).toFixed(0) + "%", 2, y + 4);
    }
    if (t.length < 2) return;
    const t1 = Math.max(t[t.length - 1], 5), t0 = Math.max(0, t1 - 40);
    const X = (tt) => padL + ((tt - t0) / (t1 - t0)) * (w - padL - 4), Y = (v) => padT + (1 - v) * (h - padT - padB);
    ctx.fillText(t0.toFixed(0) + " ps", padL, h - 2);
    ctx.textAlign = "right"; ctx.fillText(t1.toFixed(1) + " ps", w - 2, h - 2); ctx.textAlign = "left";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < t.length; i++) { if (t[i] < t0) continue; const px = X(t[i]), py = Y(q[i]); if (!started) { ctx.moveTo(px, Y(0)); ctx.lineTo(px, py); started = true; } else ctx.lineTo(px, py); }
    ctx.lineTo(X(t[t.length - 1]), Y(0)); ctx.closePath();
    ctx.fillStyle = T["accent-soft"]; ctx.fill();
    ctx.beginPath(); started = false;
    for (let i = 0; i < t.length; i++) { if (t[i] < t0) continue; const px = X(t[i]), py = Y(q[i]); if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py); }
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(X(t[t.length - 1]), Y(q[q.length - 1]), 3, 0, 2 * Math.PI); ctx.fill();
  }

  function updateMolReadouts(now) {
    const el = (now - molClock) / 1000;
    if (el > 0.5) { molSps = molSteps / el; molSteps = 0; molClock = now; }
    const d = mol.dyn;
    $("#hud-time").textContent = d.time.toFixed(2) + " ps";
    $("#r-time").innerHTML = d.time.toFixed(2) + "<small>ps</small>";
    $("#r-speed").innerHTML = (molSps * d.dt).toFixed(2) + "<small>ps/s · " + Math.round(molSps) + " steps/s</small>";
    $("#r-temp").innerHTML = mol.tAvg.toFixed(0) + "<small>K</small>";
    $("#r-energy").innerHTML = mol.engine.energy.toFixed(0) + "<small>kJ/mol</small>";
    $("#r-rg").innerHTML = radiusOfGyration(mol).toFixed(2) + "<small>Å</small>";
    $("#r-q").innerHTML = (nativeContacts(mol) * 100).toFixed(0) + "<small>%</small>";
  }

  // ================================================================ BODY
  const BODY = DATA.body, H_STEP = 1 / 60;
  let pk = null, pat = null, bodyPlaying = true, speed = 1, enzyme = 1, kdNm = BODY.compound.target.kd_nm;
  let hist = null, overlay = null, given = 0, debt = 0, cMaxSeen = 1;

  const patientSel = $("#patient");
  for (const [k, v] of Object.entries(BODY.individuals)) {
    const o = document.createElement("option"); o.value = k; o.textContent = `${v.label} (${v.body_mass.toFixed(0)} kg)`; patientSel.appendChild(o);
  }
  const studySel = $("#study");
  BODY.observed.forEach((d, i) => {
    const o = document.createElement("option"); o.value = String(i);
    o.textContent = `${d.study} · ${d.route === "IV" ? "IV" : "by mouth"} ${d.dose} · ${d.n} people`;
    studySel.appendChild(o);
  });
  const none = document.createElement("option"); none.value = ""; none.textContent = "None (free play)"; studySel.prepend(none);

  function resetPatient(key) {
    pk = new VHPBPK.Model(BODY, key, { enzymeActivity: enzyme });
    pat = new VHPBPK.Patient(pk); pat.kd = kdNm;
    hist = { t: [0], c: [0], occ: [0] }; given = 0; debt = 0; cMaxSeen = 1;
    patientSel.value = key;
    updateSummary();
  }
  function giveDose(mg, route) {
    pat.dose(mg, route, route === "iv" ? 2 / 60 : 0);
    given += mg;
  }
  function advanceBody(hours) {
    debt += hours;
    while (debt >= H_STEP) {
      pat.advance(H_STEP); debt -= H_STEP;
      const o = pat.observe();
      hist.t.push(o.time_h); hist.c.push(o.plasma * 1000); hist.occ.push(o.occupancy * 100);
      cMaxSeen = Math.max(cMaxSeen, o.plasma * 1000);
    }
    if (hist.t.length > 7200) { const k = hist.t.length - 7200; hist.t.splice(0, k); hist.c.splice(0, k); hist.occ.splice(0, k); }
  }
  function parseDose(s, mass) {
    const m = s.match(/([\d.]+)\s*(mg\/kg|mg|µg|ug)/);
    const v = parseFloat(m[1]);
    return m[2] === "mg/kg" ? v * mass : m[2] === "mg" ? v : v / 1000;
  }
  function replay(idx) {
    const d = BODY.observed[idx];
    enzyme = 1; $("#cyp").value = 0; updateCypLabel();
    resetPatient("ref_male");
    giveDose(parseDose(d.dose, BODY.individuals.ref_male.body_mass), d.route === "IV" ? "iv" : "oral");
    overlay = { t: d.time, c: d.conc_ng_ml };
    $("#obs-legend").textContent = `${d.study}, mean of ${d.n} people`;
    $("#obs-legend").hidden = false;
  }
  function clearOverlay() { overlay = null; $("#obs-legend").hidden = true; studySel.value = ""; }

  patientSel.addEventListener("change", () => { clearOverlay(); resetPatient(patientSel.value); });
  studySel.addEventListener("change", () => { if (studySel.value === "") clearOverlay(); else replay(+studySel.value); });
  $("#dose-iv").addEventListener("click", () => { if (overlay) clearOverlay(); giveDose(2, "iv"); });
  $("#dose-po").addEventListener("click", () => { if (overlay) clearOverlay(); giveDose(7.5, "oral"); });
  $("#dose-po15").addEventListener("click", () => { if (overlay) clearOverlay(); giveDose(15, "oral"); });
  document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
    speed = +b.dataset.speed;
    document.querySelectorAll("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", x === b));
  }));
  $("#body-play").addEventListener("click", () => { bodyPlaying = !bodyPlaying; $("#body-play").textContent = bodyPlaying ? "Pause" : "Play"; });

  const RT = 0.00831446261815324 * 310.15;
  function updateKd() {
    kdNm = Math.pow(10, +$("#kd").value);
    if (pat) pat.kd = kdNm;
    $("#kd-out").textContent = (kdNm < 10 ? kdNm.toFixed(1) : kdNm.toFixed(0)) + " nM";
    const dg = RT * Math.log(kdNm * 1e-9);
    $("#dg-out").textContent = `ΔG bind = ${dg.toFixed(1)} kJ/mol (${(dg / 4.184).toFixed(1)} kcal/mol)`;
  }
  function updateCypLabel() {
    enzyme = Math.pow(10, +$("#cyp").value);
    $("#cyp-out").textContent = enzyme.toFixed(enzyme < 1 ? 2 : 1) + "×";
    $("#cyp-label").textContent = enzyme < 0.3 ? "Strong inhibitor (like ketoconazole)" : enzyme < 0.75 ? "Moderate inhibitor" : enzyme <= 1.4 ? "Normal" : "Inducer (like rifampicin)";
  }
  $("#kd").addEventListener("input", updateKd);
  $("#cyp").addEventListener("input", () => { updateCypLabel(); pk.setEnzymeActivity(enzyme); updateSummary(); });

  function updateSummary() {
    const ind = pk.ind, cl = pk.clPlasma, t12 = (Math.LN2 * ind.vss) / cl;
    const F = ind.fa * pk.fg * (1 - pk.clHepBlood / ind.liver_flow_total);
    $("#model-summary").textContent = `${ind.label}, ${ind.body_mass.toFixed(0)} kg: volume of distribution ${ind.vss.toFixed(0)} L · clearance ${cl.toFixed(1)} L/h · half-life ${t12.toFixed(1)} h · oral bioavailability ${(F * 100).toFixed(0)}%`;
  }

  // --- charts
  function xWindow() {
    const t = pat.t; let x1 = Math.max(12, Math.ceil((t + 0.5) / 6) * 6), x0 = 0;
    if (x1 > 48) x0 = x1 - 48;
    return [x0, x1];
  }
  function axisFont() { return "11px " + getComputedStyle(document.body).getPropertyValue("--font-data"); }
  function drawTimeAxis(ctx, w, h, padL, padB, x0, x1) {
    const step = x1 - x0 > 30 ? 12 : x1 - x0 > 14 ? 6 : 2;
    ctx.fillStyle = T.muted; ctx.textAlign = "center";
    for (let v = Math.ceil(x0 / step) * step; v <= x1 + 1e-9; v += step) {
      const x = padL + ((v - x0) / (x1 - x0)) * (w - padL - 8);
      ctx.textAlign = x > w - 24 ? "right" : "center";
      ctx.fillText(v + (v === x1 ? " h" : ""), x > w - 24 ? w - 2 : x, h - 3);
      ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(x, h - padB); ctx.lineTo(x, h - padB + 4); ctx.stroke();
    }
    ctx.textAlign = "left";
  }
  const pkCanvas = $("#pkchart"), occCanvas = $("#occchart");
  function drawPK() {
    const { ctx, w, h } = fitCanvas(pkCanvas);
    ctx.clearRect(0, 0, w, h); ctx.font = axisFont();
    const padL = 40, padB = 18, padT = 6, [x0, x1] = xWindow();
    let lo = Infinity, hi = 0;
    for (let i = 0; i < hist.t.length; i++) if (hist.t[i] >= x0 && hist.c[i] > 1e-6) { lo = Math.min(lo, hist.c[i]); hi = Math.max(hi, hist.c[i]); }
    if (overlay) for (let i = 0; i < overlay.t.length; i++) if (overlay.c[i] > 0) { lo = Math.min(lo, overlay.c[i]); hi = Math.max(hi, overlay.c[i]); }
    if (!isFinite(lo)) { lo = 0.1; hi = 100; }
    const dHi = Math.ceil(Math.log10(Math.max(hi, 1e-3) * 1.1)), dLo = Math.max(Math.floor(Math.log10(lo)), dHi - 4);
    const yLo = Math.min(dLo, dHi - 3);
    const X = (t) => padL + ((t - x0) / (x1 - x0)) * (w - padL - 8);
    const Y = (c) => padT + (1 - (Math.log10(Math.max(c, 1e-12)) - yLo) / (dHi - yLo)) * (h - padT - padB);
    ctx.strokeStyle = T.line; ctx.fillStyle = T.muted; ctx.lineWidth = 1;
    for (let d = yLo; d <= dHi; d++) {
      const y = Y(Math.pow(10, d));
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 8, y); ctx.stroke();
      const lab = d >= 0 ? String(Math.pow(10, d)) : Math.pow(10, d).toPrecision(1);
      ctx.fillText(lab, 2, y + 4);
    }
    drawTimeAxis(ctx, w, h, padL, padB, x0, x1);
    ctx.save(); ctx.beginPath(); ctx.rect(padL, padT, w - padL - 8, h - padT - padB); ctx.clip();
    ctx.beginPath(); let on = false;
    for (let i = 0; i < hist.t.length; i++) {
      if (hist.t[i] < x0 || hist.c[i] <= 1e-6) { on = false; continue; }
      const px = X(hist.t[i]), py = Y(hist.c[i]);
      if (!on) { ctx.moveTo(px, py); on = true; } else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = T.drug; ctx.lineWidth = 2; ctx.stroke();
    if (overlay) {
      for (let i = 0; i < overlay.t.length; i++) {
        if (overlay.t[i] < x0 || overlay.c[i] <= 0) continue;
        ctx.beginPath(); ctx.arc(X(overlay.t[i]), Y(overlay.c[i]), 3.6, 0, 2 * Math.PI);
        ctx.fillStyle = T.ink; ctx.fill(); ctx.strokeStyle = T.panel; ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    const c = hist.c[hist.c.length - 1];
    if (c > 1e-6) { ctx.beginPath(); ctx.arc(X(pat.t), Y(c), 4, 0, 2 * Math.PI); ctx.fillStyle = T.drug; ctx.fill(); }
    ctx.restore();
  }
  function drawOcc() {
    const { ctx, w, h } = fitCanvas(occCanvas);
    ctx.clearRect(0, 0, w, h); ctx.font = axisFont();
    const padL = 40, padB = 18, padT = 6, [x0, x1] = xWindow();
    const X = (t) => padL + ((t - x0) / (x1 - x0)) * (w - padL - 8), Y = (v) => padT + (1 - v / 100) * (h - padT - padB);
    ctx.strokeStyle = T.line; ctx.fillStyle = T.muted; ctx.lineWidth = 1;
    for (const v of [0, 50, 100]) { const y = Y(v); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 8, y); ctx.stroke(); ctx.fillText(v + "%", 2, y + 4); }
    drawTimeAxis(ctx, w, h, padL, padB, x0, x1);
    ctx.beginPath(); let first = -1;
    for (let i = 0; i < hist.t.length; i++) { if (hist.t[i] < x0) continue; const px = X(hist.t[i]), py = Y(hist.occ[i]); if (first < 0) { ctx.moveTo(px, Y(0)); first = i; } ctx.lineTo(px, py); }
    if (first >= 0) {
      ctx.lineTo(X(hist.t[hist.t.length - 1]), Y(0)); ctx.closePath();
      ctx.globalAlpha = 0.18; ctx.fillStyle = T.signal; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); let on = false;
      for (let i = first; i < hist.t.length; i++) { const px = X(hist.t[i]), py = Y(hist.occ[i]); if (!on) { ctx.moveTo(px, py); on = true; } else ctx.lineTo(px, py); }
      ctx.strokeStyle = T.signal; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  // --- circulation diagram (the PBPK model's actual structure)
  const ORGANS = [["brain", "Brain"], ["heart", "Heart"], ["muscle", "Muscle"], ["adipose", "Fat"], ["skin", "Skin"], ["bone", "Bone"],
    ["kidney", "Kidney"], ["rest", "Other tissue"], ["spleen", "Spleen"], ["pancreas", "Pancreas"], ["gut", "Gut wall"], ["liver", "Liver"]];
  const PORTAL = new Set(["spleen", "pancreas", "gut"]);
  const mapCanvas = $("#bodymap");
  function heat(c, cmax) {
    const lo = Math.log10(cmax / 1000), t = Math.max(0, Math.min(1, (Math.log10(Math.max(c, 1e-9)) - lo) / 3));
    return t;
  }
  function drawBodyMap(now) {
    const { ctx, w, h } = fitCanvas(mapCanvas);
    ctx.clearRect(0, 0, w, h);
    const o = pat.observe(), cmax = Math.max(1, cMaxSeen * 1.5);
    const pad = 4, colW = Math.max(22, w * 0.085), gap = 8, legendH = 30;
    const top = pad, lungH = 26, bodyTop = top + lungH + 8, bodyBot = h - legendH - 4;
    const rows = ORGANS.length, rowGap = 4, rowH = (bodyBot - bodyTop - rowGap * (rows - 1)) / rows;
    const xa = pad, xv = w - pad - colW, xo = xa + colW + gap, wo = xv - gap - xo;
    const fill = (c) => mix(T.panel, T.drug, 0.05 + 0.95 * heat(c, cmax));
    const textOn = (c) => (heat(c, cmax) > 0.62 ? T.panel : T.ink);
    const font = (sz, wt) => `${wt || 500} ${sz}px ` + getComputedStyle(document.body).getPropertyValue("--font-body");
    const dataFont = (sz) => `${sz}px ` + getComputedStyle(document.body).getPropertyValue("--font-data");
    const rr = (x, y, ww, hh, r) => { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, ww, hh, r) : ctx.rect(x, y, ww, hh); };
    // blood columns
    const art = o.plasma * 1000 * pk.ind.bp, ven = o.plasma * 1000 * pk.ind.bp;
    rr(xa, bodyTop, colW, bodyBot - bodyTop, 6); ctx.fillStyle = fill(art); ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
    rr(xv, bodyTop, colW, bodyBot - bodyTop, 6); ctx.fillStyle = fill(ven); ctx.fill(); ctx.stroke();
    // circulation flow marks
    const phase = reduceMotion ? 0 : (now / 1000) * 18 * Math.sqrt(speed);
    ctx.save(); ctx.setLineDash([3, 9]); ctx.lineWidth = 2; ctx.strokeStyle = textOn(art);
    ctx.globalAlpha = 0.5;
    ctx.lineDashOffset = -phase; ctx.beginPath(); ctx.moveTo(xa + colW / 2, bodyTop + 6); ctx.lineTo(xa + colW / 2, bodyBot - 6); ctx.stroke();
    ctx.lineDashOffset = phase; ctx.beginPath(); ctx.moveTo(xv + colW / 2, bodyTop + 6); ctx.lineTo(xv + colW / 2, bodyBot - 6); ctx.stroke();
    ctx.restore();
    ctx.save(); ctx.fillStyle = textOn(art); ctx.font = font(11, 600);
    ctx.translate(xa + colW / 2 + 4, (bodyTop + bodyBot) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("Arterial blood", 0, 0); ctx.restore();
    ctx.save(); ctx.fillStyle = textOn(ven); ctx.font = font(11, 600);
    ctx.translate(xv + colW / 2 + 4, (bodyTop + bodyBot) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("Venous blood", 0, 0); ctx.restore();
    // lung across the top
    rr(xa, top, w - 2 * pad, lungH, 6); ctx.fillStyle = fill(o.lung * 1000); ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
    ctx.fillStyle = textOn(o.lung * 1000); ctx.font = font(12, 600); ctx.fillText("Lung", xa + 10, top + lungH / 2 + 4);
    ctx.font = dataFont(11); ctx.textAlign = "right"; ctx.fillText(fmt(o.lung * 1000) + " ng/mL", w - pad - 10, top + lungH / 2 + 4); ctx.textAlign = "left";
    // organs
    ORGANS.forEach(([k, label], i) => {
      const y = bodyTop + i * (rowH + rowGap), c = o[k] * 1000, indent = PORTAL.has(k) ? 14 : 0;
      rr(xo + indent, y, wo - indent, rowH, 5); ctx.fillStyle = fill(c); ctx.fill(); ctx.strokeStyle = T.line; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = textOn(c); ctx.font = font(rowH > 20 ? 12 : 11, 500);
      ctx.fillText(label, xo + indent + (k === "liver" ? 20 : 8), y + rowH / 2 + 4);
      ctx.font = dataFont(rowH > 20 ? 11 : 10); ctx.textAlign = "right"; ctx.fillText(fmt(c), xo + wo - 8, y + rowH / 2 + 4); ctx.textAlign = "left";
    });
    // portal vein bracket: spleen/pancreas/gut drain into liver
    const yS = bodyTop + 8 * (rowH + rowGap) + rowH / 2, yL = bodyTop + 11 * (rowH + rowGap) + rowH / 2;
    ctx.strokeStyle = T.muted; ctx.lineWidth = 1.2; ctx.beginPath();
    ctx.moveTo(xo + 14, yS); ctx.lineTo(xo + 5, yS); ctx.lineTo(xo + 5, yL); ctx.lineTo(xo + 10, yL); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xo + 10, yL - 3); ctx.lineTo(xo + 14, yL); ctx.lineTo(xo + 10, yL + 3); ctx.fillStyle = T.muted; ctx.fill();
    // legend
    const ly = h - legendH + 8, lw = Math.min(160, w * 0.45);
    for (let i = 0; i < 40; i++) { ctx.fillStyle = mix(T.panel, T.drug, 0.05 + 0.95 * (i / 39)); ctx.fillRect(pad + (i * lw) / 40, ly, lw / 40 + 0.5, 8); }
    ctx.strokeStyle = T.line; ctx.strokeRect(pad, ly, lw, 8);
    ctx.fillStyle = T.muted; ctx.font = dataFont(10);
    ctx.fillText(fmt(cmax / 1000), pad, ly + 20); ctx.textAlign = "right"; ctx.fillText(fmt(cmax) + " ng/mL", pad + lw, ly + 20);
    ctx.textAlign = "left";
    ctx.fillText(`gut lumen ${o.gut_lumen.toFixed(1)} mg`, pad + lw + 12, ly + 8);
  }

  function updateBodyReadouts() {
    const o = pat.observe(), t = o.time_h, hh = Math.floor(t), mm = Math.floor((t - hh) * 60);
    $("#b-time").innerHTML = `${hh}<small>h</small> ${String(mm).padStart(2, "0")}<small>min</small>`;
    $("#b-plasma").innerHTML = fmt(o.plasma * 1000) + "<small>ng/mL</small>";
    $("#b-brain").innerHTML = fmt(o.target_unbound_nm) + "<small>nM</small>";
    $("#b-occ").innerHTML = (o.occupancy * 100).toFixed(0) + "<small>%</small>";
    $("#b-given").innerHTML = given.toFixed(given < 1 ? 3 : 1) + "<small>mg</small>";
    $("#b-cleared").innerHTML = given > 0 ? ((o.eliminated / given) * 100).toFixed(0) + "<small>%</small>" : "–";
  }

  // ================================================================ evidence
  (function fillEvidence() {
    const ev = DATA.evidence, m = ev.mol, b = ev.body;
    const sci = (x) => { const e = Math.floor(Math.log10(x)); return `${(x / Math.pow(10, e)).toFixed(0)} × 10<sup>${e}</sup>`; };
    const pct = (x) => (x * 100).toFixed(0) + "%";
    const rows = (el, items) => { el.innerHTML = items.map(([k, v, good]) => `<dt>${k}</dt><dd class="${good ? "good" : ""}">${v}</dd>`).join(""); };
    rows($("#ev-mol"), [
      ["Energy, every term (worst relative error)", sci(m.worst_energy_rel), true],
      ["Forces (worst relative error)", sci(m.worst_force_rel), true],
      ["Same start, after 1,000 steps", sci(m.traj_1000_dx_nm) + " nm apart", true],
      ["Energy drift with no thermostat", Math.abs(m.nve_drift).toFixed(3) + " kJ/mol/ps"],
      ["Mean energy at 300 K, ours", `${m.u_ours.toFixed(0)} ± ${m.u_ours_se.toFixed(0)}`],
      ["Mean energy at 300 K, OpenMM", `${m.u_omm.toFixed(0)} ± ${m.u_omm_se.toFixed(0)}`],
      ["This page's JavaScript vs the Python engine", "≤ 1 × 10<sup>-13</sup>", true],
    ]);
    rows($("#ev-body"), [
      ["Clinical datasets", `${b.n_iv} IV · ${b.n_po} oral`],
      ["No fitting: IV exposure within 2×", pct(b.bu_iv_auc_2x), true],
      ["No fitting: oral exposure within 2×", pct(b.bu_po_auc_2x), true],
      ["No fitting: oral peak within 2×", pct(b.bu_po_cmax_2x), true],
      [`One factor fitted on IV (×${b.clint_scale.toFixed(2)}), oral within 2×`, pct(b.mo_po_auc_2x), true],
      ["Typical error, oral (fold)", b.mo_po_gmfe.toFixed(2) + "×"],
      ["This page's JavaScript vs the Python engine", "≤ 1 × 10<sup>-12</sup>", true],
    ]);
  })();

  // ================================================================ start
  loadMol("chignolin");
  updateKd(); updateCypLabel();
  const defaultStudy = BODY.observed.findIndex((d) => d.route === "PO");
  studySel.value = String(defaultStudy);
  replay(defaultStudy);
  advanceBody(2.0);           // open on a realistic, already-running state
  for (let i = 0; i < 40; i++) mol.dyn.step(1);
  updateMolReadouts(performance.now()); updateBodyReadouts();
  drawMol(); drawSpark(); drawPK(); drawOcc(); drawBodyMap(performance.now());

  let last = performance.now(), lastUI = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    stepMol();
    if (view.auto && !view.dragging) view.R = mul3(rotY(0.0035), view.R);
    drawMol();
    if (bodyPlaying) advanceBody(dt * speed);
    drawPK(); drawOcc(); drawBodyMap(now);
    if (now - lastUI > 200) { updateMolReadouts(now); updateBodyReadouts(); drawSpark(); lastUI = now; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
