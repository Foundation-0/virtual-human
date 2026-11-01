/* Virtual Human — real-time whole-body engine (browser / Node).
 *
 * Mirrors vh.body.pbpk: the same linear system dx/dt = M x + u, assembled from
 * the same exported physiology and drug parameters, advanced EXACTLY with a
 * matrix exponential (Pade(6) with scaling and squaring). Tested against the
 * Python engine in web/test_engines.mjs.
 *
 * Units: mg, L, L/h, mg/L, h.
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------------------
  // small dense linear algebra
  // ---------------------------------------------------------------------------
  function matmul(A, B, n) {
    const C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
      const a = A[i * n + k]; if (a === 0) continue;
      for (let j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
    }
    return C;
  }
  function solve(D, Nm, n) {  // solve D X = Nm (both n x n), partial pivoting
    const A = Float64Array.from(D), B = Float64Array.from(Nm);
    for (let c = 0; c < n; c++) {
      let p = c, best = Math.abs(A[c * n + c]);
      for (let r = c + 1; r < n; r++) { const v = Math.abs(A[r * n + c]); if (v > best) { best = v; p = r; } }
      if (p !== c) for (let j = 0; j < n; j++) {
        let t = A[c * n + j]; A[c * n + j] = A[p * n + j]; A[p * n + j] = t;
        t = B[c * n + j]; B[c * n + j] = B[p * n + j]; B[p * n + j] = t;
      }
      const piv = A[c * n + c];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const fct = A[r * n + c] / piv; if (fct === 0) continue;
        for (let j = 0; j < n; j++) { A[r * n + j] -= fct * A[c * n + j]; B[r * n + j] -= fct * B[c * n + j]; }
      }
    }
    for (let r = 0; r < n; r++) { const piv = A[r * n + r]; for (let j = 0; j < n; j++) B[r * n + j] /= piv; }
    return B;
  }
  function expm(Ain, n) {
    let norm = 0;
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += Math.abs(Ain[i * n + j]); norm = Math.max(norm, s); }
    const s = Math.max(0, Math.ceil(Math.log2(norm / 0.5)));
    const X = Float64Array.from(Ain, v => v / Math.pow(2, s));
    const p = 6, c = [1];
    for (let k = 1; k <= p; k++) c.push(c[k - 1] * (p - k + 1) / (k * (2 * p - k + 1)));
    const Nm = new Float64Array(n * n), D = new Float64Array(n * n);
    let P = new Float64Array(n * n); for (let i = 0; i < n; i++) P[i * n + i] = 1;
    for (let k = 0; k <= p; k++) {
      if (k > 0) P = matmul(P, X, n);
      const sg = k % 2 ? -1 : 1;
      for (let i = 0; i < n * n; i++) { Nm[i] += c[k] * P[i]; D[i] += sg * c[k] * P[i]; }
    }
    let E = solve(D, Nm, n);
    for (let k = 0; k < s; k++) E = matmul(E, E, n);
    return E;
  }

  // ---------------------------------------------------------------------------
  // model
  // ---------------------------------------------------------------------------
  function Model(body, individualKey, opts) {
    opts = opts || {};
    const ind = body.individuals[individualKey], C = body.constants;
    this.body = body; this.ind = ind; this.key = individualKey;
    this.tissues = C.tissues; this.portal = C.portal;
    this.enzymeActivity = opts.enzymeActivity || 1.0;
    const names = [...C.tissues, "arterial", "venous", "stomach"];
    for (let i = 1; i <= C.n_transit; i++) names.push("si" + i);
    names.push("elim_hepatic", "elim_renal", "elim_gut_wall", "feces");
    this.names = names; this.n = names.length;
    this.ix = {}; names.forEach((nm, i) => (this.ix[nm] = i));
    this.V = Object.assign({}, ind.volumes, { arterial: ind.v_arterial, venous: ind.v_venous });
    this.build();
  }

  Model.prototype.build = function () {
    const ind = this.ind, C = this.body.constants, ix = this.ix, n = this.n, V = this.V;
    const M = new Float64Array(n * n);
    const flow = (src, dst, rate) => { M[ix[src] * n + ix[src]] -= rate; M[ix[dst] * n + ix[src]] += rate; };
    const bp = ind.bp, kp = ind.kp, Q = ind.flows;
    const outk = {}; for (const t of this.tissues) outk[t] = bp / kp[t] / V[t];
    const s = this.enzymeActivity;
    this.clint = ind.clint_hep * s;
    this.fg = ind.clint_gut > 0 ? ind.q_gut / (ind.q_gut + ind.clint_gut * s) : ind.fg;
    flow("venous", "lung", ind.cardiac_output / V.venous);
    flow("lung", "arterial", ind.cardiac_output * outk.lung);
    for (const t of this.tissues) if (t !== "lung") flow("arterial", t, Q[t] / V.arterial);
    for (const t of this.tissues) {
      if (t === "lung" || t === "liver") continue;
      if (this.portal.includes(t)) flow(t, "liver", Q[t] * outk[t]); else flow(t, "venous", Q[t] * outk[t]);
    }
    flow("liver", "venous", ind.liver_flow_total * outk.liver);
    flow("liver", "elim_hepatic", this.clint * ind.fu / (kp.liver * V.liver));
    flow("kidney", "elim_renal", ind.cl_renal / (kp.kidney * V.kidney));
    flow("stomach", "si1", Math.LN2 / C.gastric_half_life_h);
    const kt = C.n_transit / C.si_transit_h;
    for (let i = 1; i <= C.n_transit; i++) {
      const seg = "si" + i, nxt = i < C.n_transit ? "si" + (i + 1) : "feces";
      flow(seg, nxt, kt);
      if (ind.ka > 0) { flow(seg, "gut", ind.ka * this.fg); flow(seg, "elim_gut_wall", ind.ka * (1 - this.fg)); }
    }
    this.M = M; this.cache = new Map();
    const fub = ind.fu / bp, qh = ind.liver_flow_total;
    this.clHepBlood = qh * fub * this.clint / (qh + fub * this.clint);
    this.clPlasma = this.clHepBlood * bp + ind.cl_renal;
  };

  Model.prototype.setEnzymeActivity = function (s) { this.enzymeActivity = s; this.build(); };

  // exact step: x(t+h) = E x + f*rate  (augmented matrix for constant IV infusion)
  Model.prototype.propagator = function (h, rate) {
    const key = h.toPrecision(12) + "|" + rate.toPrecision(12);
    let P = this.cache.get(key);
    if (!P) {
      const n = this.n, m = n + 1, A = new Float64Array(m * m);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * m + j] = this.M[i * n + j] * h;
      A[this.ix.venous * m + n] = rate * h;
      const E = expm(A, m);
      const En = new Float64Array(n * n), f = new Float64Array(n);
      for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) En[i * n + j] = E[i * m + j]; f[i] = E[i * m + n]; }
      P = { E: En, f };
      if (this.cache.size > 64) this.cache.delete(this.cache.keys().next().value);
      this.cache.set(key, P);
    }
    return P;
  };

  Model.prototype.observe = function (x) {
    const ind = this.ind, V = this.V, ix = this.ix, o = {};
    o.plasma = x[ix.venous] / V.venous / ind.bp;
    for (const t of this.tissues) o[t] = x[ix[t]] / V[t];
    o.brain_unbound = o.brain * ind.fu / ind.kp.brain;
    o.eliminated = x[ix.elim_hepatic] + x[ix.elim_renal] + x[ix.elim_gut_wall];
    o.gut_lumen = x[ix.stomach]; for (let i = 1; i <= this.body.constants.n_transit; i++) o.gut_lumen += x[ix["si" + i]];
    o.total = x.reduce((a, b) => a + b, 0);
    return o;
  };

  // ---------------------------------------------------------------------------
  // real-time patient with live dosing and receptor binding
  // ---------------------------------------------------------------------------
  function Patient(model) {
    this.model = model;
    this.x = new Float64Array(model.n);
    this.t = 0; this.infusions = []; this.occupancy = 0;
    const tgt = model.body.compound.target;
    this.kd = tgt.kd_nm; this.koff = tgt.koff_per_h;
  }

  Patient.prototype.dose = function (mg, route, durationH) {
    const ix = this.model.ix;
    if (route === "oral") this.x[ix.stomach] += mg;
    else if (durationH > 0) this.infusions.push({ end: this.t + durationH, rate: mg / durationH });
    else this.x[ix.venous] += mg;
  };

  Patient.prototype.advance = function (dt) {
    const tEnd = this.t + dt, n = this.model.n;
    while (this.t < tEnd - 1e-12) {
      this.infusions = this.infusions.filter(f => f.end > this.t + 1e-12);
      const stop = Math.min(tEnd, ...this.infusions.map(f => f.end));
      const rate = this.infusions.reduce((a, f) => a + f.rate, 0);
      const h = stop - this.t;
      const cuBefore = this.unboundTargetNm();
      const P = this.model.propagator(h, rate), E = P.E, x = this.x, y = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = P.f[i]; for (let j = 0; j < n; j++) s += E[i * n + j] * x[j]; y[i] = s; }
      this.x = y; this.t = stop;
      // receptor binding: exact for Cu held at the interval mean
      const cu = 0.5 * (cuBefore + this.unboundTargetNm());
      const kon = this.koff / this.kd, k = kon * cu + this.koff, oinf = kon * cu / k;
      this.occupancy = oinf + (this.occupancy - oinf) * Math.exp(-k * h);
    }
    return this.observe();
  };

  Patient.prototype.unboundTargetNm = function () {
    const m = this.model, tgt = m.body.compound.target, ind = m.ind;
    const c = this.x[m.ix[tgt.tissue]] / m.V[tgt.tissue];
    return c * ind.fu / ind.kp[tgt.tissue] / m.body.compound.mw * 1e6;
  };

  Patient.prototype.observe = function () {
    const o = this.model.observe(this.x);
    o.time_h = this.t; o.target_unbound_nm = this.unboundTargetNm(); o.occupancy = this.occupancy;
    return o;
  };

  const api = { Model, Patient, expm };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VHPBPK = api;
})(typeof window !== "undefined" ? window : globalThis);
